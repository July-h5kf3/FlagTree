import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import get_current_target

target = get_current_target()
pytestmark = pytest.mark.skipif(target is None or target.backend != "maca", reason="requires the MetaX backend")


@triton.jit(do_not_specialize=["USED"])
def attention_full_v_kernel(Q, K, V, O, USED, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.arange(0, BM)
    dims = tl.arange(0, 128)
    query = tl.load(Q + rows[:, None] * 128 + dims[None, :])
    maximum = tl.full((BM, ), float("-inf"), tl.float32)
    denominator = tl.zeros((BM, ), tl.float32)
    accumulator = tl.zeros((BM, 128), tl.float32)
    for start in range(tl.cdiv(USED, BN)):
        columns = start * BN + tl.arange(0, BN)
        key = tl.load(K + columns[None, :] * 128 + dims[:, None], columns[None, :] < USED, 0)
        raw = tl.trans(tl.dot(tl.trans(key), tl.trans(query), out_dtype=tl.int32))
        scores = tl.where(columns[None, :] < USED, raw.to(tl.float32) * 0.125, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_maximum)
        probability = tl.exp2(scores - new_maximum[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        value = tl.load(V + dims[None, :] * N + columns[:, None], columns[:, None] < USED, 0)
        accumulator = tl.trans(
            tl.dot(tl.trans(value), tl.trans(probability.to(tl.float16)), tl.trans(accumulator * alpha[:, None])))
        maximum = new_maximum
    tl.store(O + rows[:, None] * 128 + dims[None, :], accumulator / denominator[:, None])


@pytest.mark.parametrize("key_length", [128, 137])
@pytest.mark.parametrize("block_n", [32, 64])
def test_attention_full_v_scratch(key_length, block_n, device, fresh_triton_cache):
    torch.manual_seed(0)
    query = torch.randint(-2, 3, (64, 128), dtype=torch.int8, device=device)
    key = torch.randint(-2, 3, (key_length, 128), dtype=torch.int8, device=device)
    value = torch.randint(-4, 5, (128, key_length), dtype=torch.int8, device=device).to(torch.float16)
    scores = query.float() @ key.float().T * (0.125 * 0.6931471805599453)
    reference = torch.softmax(scores, dim=-1) @ value.float().T
    outputs, kernels = [], []
    for scenario in ("attention-query-warps", "attention-query-warps;attention-full-v-scratch"):
        # FP16 output keeps its conversion from hiding the V scratch peak.
        output = torch.empty((64, 128), dtype=torch.float16, device=device)
        kernel = attention_full_v_kernel[(1, )](query, key, value, output, key_length, key_length, 64, block_n,
                                                num_warps=4, num_stages=1, pipeline="basic", scenario=scenario)
        torch.testing.assert_close(output.float(), reference, atol=2e-3, rtol=2e-3)
        outputs.append(output)
        kernels.append(kernel)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    assert kernels[0].hash != kernels[1].hash
    if block_n == 64 and key_length == 128:
        ttgir = kernels[1].asm["ttgir"]
        assert "scf.for" in ttgir
        value_load = re.search(r"(%\w+) = tt.load [^\n]*tensor<64x128x!tt.ptr<f16>, (#\w+)>", ttgir)
        assert value_load is not None
        conversion = re.search(
            rf"ttg.convert_layout {re.escape(value_load[1])} : "
            r"tensor<64x128xf16, (#\w+)> -> tensor<64x128xf16, (#\w+)>", ttgir)
        assert conversion is not None and conversion[1] == value_load[2]
        layouts = dict(re.findall(r"^(#\w+) = (.+)$", ttgir, re.MULTILINE))
        assert layouts[conversion[1]] == ("#ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], "
                                          "warpsPerCTA = [1, 4], order = [0, 1]}>")
        assert layouts[conversion[2]].startswith("#ttg.linear<")
        assert "warp = [[0, 0], [0, 0]]" in layouts[conversion[2]]
        assert kernels[1].metadata.shared == kernels[0].metadata.shared + 8192
    elif block_n == 32:
        assert kernels[1].metadata.shared == kernels[0].metadata.shared
        assert kernels[0].asm["mcfatbin"] == kernels[1].asm["mcfatbin"]
