import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def int8_shared_store_kernel(Q, K, Scores, Output, STRIDE: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                             ACTIVE_M: tl.constexpr, ACTIVE_N: tl.constexpr, PAIRED: tl.constexpr,
                             VALUE_DIM: tl.constexpr):
    rows = tl.arange(0, M)
    tokens = tl.arange(0, N)
    features = tl.arange(0, 512)
    value_features = tl.arange(0, VALUE_DIM)
    query = tl.load(Q + rows[:, None] * STRIDE + features[None, :], rows[:, None] < ACTIVE_M, 0)
    keys = tl.load(K + tokens[:, None] * STRIDE + features[None, :], tokens[:, None] < ACTIVE_N, 0)
    scores = tl.dot(query, tl.trans(keys), out_dtype=tl.int32)
    tl.store(Scores + rows[:, None] * N + tokens[None, :], scores)
    if PAIRED:
        probabilities = (scores & 32767).to(tl.float32) * (127.0 / 32768.0)
    else:
        probabilities = tl.trans((scores & 32767).to(tl.float32)) * (127.0 / 32768.0)
    high = tl.floor(probabilities + 0.5)
    low = tl.floor((probabilities - high) * 127.0 + 0.5)
    if PAIRED:
        high = high.to(tl.int8)
        low = low.to(tl.int8)
    else:
        high = tl.trans(high.to(tl.int8))
        low = tl.trans(low.to(tl.int8))
    values = tl.load(K + tokens[:, None] * STRIDE + value_features[None, :], tokens[:, None] < ACTIVE_N, 0)
    if PAIRED:
        packed = tl.reshape(tl.permute(tl.join(high, low), (2, 0, 1)), (2 * M, N))
        packed = tl.inline_asm_elementwise('ppu.mov.b32 $0, $1;', '=r,r', [packed.to(tl.int32)], dtype=tl.int32,
                                           is_pure=False, pack=1).to(tl.int8)
        products = tl.dot(packed, values, out_dtype=tl.int32)
        upper, lower = tl.split(tl.permute(tl.reshape(products, (2, M, VALUE_DIM)), (1, 2, 0)))
        output = upper * 127 + lower
    else:
        upper = tl.dot(high, values, out_dtype=tl.int32)
        output = tl.dot(low, values, upper * 127, out_dtype=tl.int32)
    tl.store(Output + rows[:, None] * VALUE_DIM + value_features[None, :], output)


@pytest.mark.parametrize('vector_store', [False, True])
@pytest.mark.parametrize('m,n,active_m,active_n,stride,warps,paired,value_dim,vectorizable', [
    (16, 128, 13, 97, 520, 8, True, 512, False),
    (32, 256, 32, 256, 512, 16, False, 128, False),
    (32, 256, 32, 256, 512, 16, False, 512, True),
    (32, 256, 17, 129, 512, 16, False, 512, True),
    (32, 256, 17, 129, 520, 16, False, 512, False),
    (32, 128, 17, 127, 528, 16, False, 128, False),
])
def test_int8_shared_store(m, n, active_m, active_n, stride, warps, paired, value_dim, vectorizable, vector_store,
                           device):
    generator = torch.Generator().manual_seed(283)
    query = torch.randint(-128, 128, (m, stride), dtype=torch.int8, generator=generator)
    keys = torch.randint(-128, 128, (n, stride), dtype=torch.int8, generator=generator)
    masked_query = query[:, :512].int()
    masked_keys = keys[:, :512].int()
    masked_query[active_m:] = 0
    masked_keys[active_n:] = 0
    scores = masked_query @ masked_keys.T
    probabilities = (scores & 32767).float() * (127.0 / 32768.0)
    high = torch.floor(probabilities + 0.5)
    low = torch.floor((probabilities - high) * 127.0 + 0.5)
    expected = 127 * (high.int() @ masked_keys[:, :value_dim]) + low.int() @ masked_keys[:, :value_dim]
    actual_scores = torch.empty((m, n), dtype=torch.int32, device=device)
    actual = torch.empty((m, value_dim), dtype=torch.int32, device=device)
    compiled = int8_shared_store_kernel[(1, )](query.to(device), keys.to(device), actual_scores, actual, stride, m, n,
                                               active_m, active_n, paired, value_dim, num_warps=warps, num_stages=1,
                                               ppu_vectorize_i8_shared_store=vector_store)
    assert torch.equal(actual_scores.cpu(), scores)
    assert torch.equal(actual.cpu(), expected)

    # The 520-byte stride selects eight consecutive register bytes and uses the fallback.
    assert bool(compiled.metadata.vector_i8_shared_store_applied) == bool(vector_store and vectorizable)
    assert bool(compiled.metadata.vector_i8_shared_store_grouped) == bool(vector_store and vectorizable)
