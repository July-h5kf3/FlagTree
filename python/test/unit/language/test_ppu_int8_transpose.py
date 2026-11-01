import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires a PPU")


@triton.jit
def _int8_qk_pv(Query, Value, Output, M: tl.constexpr, TOKENS: tl.constexpr, DIM: tl.constexpr,
                QUERY_STRIDE: tl.constexpr, VALUE_STRIDE: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                BK: tl.constexpr):
    rows = tl.arange(0, BM)
    dims = tl.arange(0, BD)
    token_offsets = tl.arange(0, BK)
    query = tl.load(Query + rows[:, None] * QUERY_STRIDE + dims[None, :], (rows[:, None] < M) & (dims[None, :] < DIM),
                    0)
    accumulator = tl.zeros((BM, BD), tl.int32)
    for start in range(0, TOKENS, BK):
        tokens = start + token_offsets
        value = tl.load(Value + tokens[:, None] * VALUE_STRIDE + dims[None, :],
                        (tokens[:, None] < TOKENS) & (dims[None, :] < DIM), 0)
        score = tl.dot(query, tl.trans(value), out_dtype=tl.int32)
        accumulator = tl.dot(score.to(tl.int8), value, accumulator, out_dtype=tl.int32)
    tl.store(Output + rows[:, None] * DIM + dims[None, :], accumulator, (rows[:, None] < M) & (dims[None, :] < DIM))


@pytest.mark.parametrize("m,tokens,dim,warps,strided", [
    (16, 32, 32, 4, False),
    (17, 37, 33, 4, True),
    (32, 64, 128, 8, True),
    (16, 129, 256, 8, False),
    (32, 129, 512, 16, True),
    (32, 256, 512, 16, False),
])
def test_int8_transposed_rhs(m, tokens, dim, warps, strided, device):
    generator = torch.Generator().manual_seed(0)
    query_cpu = torch.randint(-128, 128, (m, dim), dtype=torch.int8, generator=generator)
    value_cpu = torch.randint(-128, 128, (tokens, dim), dtype=torch.int8, generator=generator)
    stride = dim * (2 if strided else 1)
    query = torch.empty_strided((m, dim), (stride, 1), dtype=torch.int8, device=device)
    value = torch.empty_strided((tokens, dim), (stride, 1), dtype=torch.int8, device=device)
    query.copy_(query_cpu)
    value.copy_(value_cpu)
    output = torch.empty((m, dim), dtype=torch.int32, device=device)
    _int8_qk_pv[(1, )](query, value, output, m, tokens, dim, stride, stride, triton.next_power_of_2(m),
                       triton.next_power_of_2(dim), 64, num_warps=warps, num_stages=1)
    expected = ((query_cpu.int() @ value_cpu.int().T).to(torch.int8).int() @ value_cpu.int())
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
