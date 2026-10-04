import re

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import get_current_target

target = get_current_target()
pytestmark = pytest.mark.skipif(target is None or target.backend != "maca", reason="requires the MetaX backend")


@triton.jit(do_not_specialize=["NK"])
def attention_query_warps_kernel(Q, K, V, O, NK, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
                                 SCALE: tl.constexpr):
    rows = tl.arange(0, BM)
    dimensions = tl.arange(0, D)
    queries = tl.load(Q + rows[:, None] * D + dimensions[None, :])
    maximum = tl.full((BM, ), float("-inf"), tl.float32)
    denominator = tl.zeros((BM, ), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    for start in range(0, NK, BN):
        columns = start + tl.arange(0, BN)
        keys = tl.load(K + dimensions[:, None] + columns[None, :] * D, columns[None, :] < NK, 0)
        scores = tl.trans(tl.dot(tl.trans(keys), tl.trans(queries), out_dtype=tl.int32)).to(tl.float32)
        scores = tl.where(columns[None, :] < NK, scores * (SCALE * 1.4426950408889634), float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_maximum)
        probabilities = tl.exp2(scores - new_maximum[:, None])
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        values = tl.load(V + columns[:, None] * D + dimensions[None, :], columns[:, None] < NK, 0)
        accumulator = tl.trans(
            tl.dot(tl.trans(values.to(tl.float16)), tl.trans(probabilities.to(tl.float16)),
                   tl.trans(accumulator * alpha[:, None])))
        maximum = new_maximum
    tl.store(O + rows[:, None] * D + dimensions[None, :], accumulator / denominator[:, None])


@pytest.mark.parametrize("block_m, num_warps, selected_scenario", [
    (32, 2, "attention-query-warps"),
    (64, 4, "attention-query-warps"),
    (64, 4, "attention-query-warps;attention-short-query-scales"),
    (128, 8, "attention-query-warps"),
    (128, 4, "attention-query-warps;attention-wide-query-warps"),
])
def test_attention_query_warps(block_m, num_warps, selected_scenario, device, fresh_triton_cache):
    torch.manual_seed(0)
    head_dim, key_length = 128, 128
    query = torch.randint(-2, 3, (block_m, head_dim), dtype=torch.int8, device=device)
    key = torch.randint(-2, 3, (key_length, head_dim), dtype=torch.int8, device=device)
    value = torch.randint(-4, 5, (key_length, head_dim), dtype=torch.int8, device=device)
    scale = head_dim**-0.5
    reference = torch.softmax(query.float() @ key.float().T * scale, dim=-1) @ value.float()
    compiled_ir = []
    for scenario in ("", selected_scenario):
        output = torch.empty((block_m, head_dim), dtype=torch.float32, device=device)
        kernel = attention_query_warps_kernel[(1, )](query, key, value, output, key_length, BM=block_m, BN=64,
                                                     D=head_dim, SCALE=scale, num_warps=num_warps, num_stages=1,
                                                     pipeline="basic", scenario=scenario)
        torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)
        compiled_ir.append(kernel.asm["ttgir"])
    ttgir = compiled_ir[1]
    assert "scf.for" in ttgir
    encodings = {
        name: (int(warp_m), int(warp_n))
        for name, warp_m, warp_n in re.findall(r"^(#\w+) = #ttg.maca_mma<\{[^\n]*warpsPerCTA = \[(\d+), (\d+)\]", ttgir,
                                               re.MULTILINE)
    }
    dot_encodings = [
        re.search(r"-> tensor<[^>]*, (#\w+)>", line).group(1) for line in ttgir.splitlines() if " = tt.dot " in line
    ]
    assert len(dot_encodings) == 2
    assert [encodings[name] for name in dot_encodings] == [(1, num_warps)] * 2
    assert (num_warps, 1) not in encodings.values()


@triton.jit(do_not_specialize=["LOOPS"])
def independent_matmul_kernel(A, B, C, LOOPS):
    rows = tl.arange(0, 64)
    columns = tl.arange(0, 64)
    inner = tl.arange(0, 128)
    left = tl.load(A + rows[:, None] * 128 + inner[None, :])
    right = tl.load(B + inner[:, None] * 64 + columns[None, :])
    accumulator = tl.zeros((64, 64), C.dtype.element_ty)
    for _ in range(LOOPS):
        accumulator = tl.dot(left, right, accumulator, out_dtype=C.dtype.element_ty)
    tl.store(C + rows[:, None] * 64 + columns[None, :], accumulator)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float16])
@pytest.mark.parametrize("selected_scenario", [
    "attention-query-warps", "attention-query-warps;attention-short-query-scales",
    "attention-query-warps;attention-wide-query-warps",
    "attention-query-warps;attention-wide-query-warps;attention-wide-4g-address"
])
def test_attention_scenario_preserves_independent_matmul(dtype, selected_scenario, device, fresh_triton_cache):
    torch.manual_seed(0)
    left = torch.randint(-2, 3, (64, 128), dtype=torch.int8, device=device).to(dtype)
    right = torch.randint(-2, 3, (128, 64), dtype=torch.int8, device=device).to(dtype)
    output_dtype = torch.int32 if dtype == torch.int8 else torch.float32
    reference = (left.cpu().long() @ right.cpu().long()) * 2
    atol = 0 if dtype == torch.int8 else 1e-4
    compiled_ir = []
    outputs = []
    for scenario in ("", selected_scenario):
        output = torch.empty((64, 64), dtype=output_dtype, device=device)
        kernel = independent_matmul_kernel[(1, )](left, right, output, 2, num_warps=4, num_stages=1, pipeline="basic",
                                                  scenario=scenario)
        outputs.append(output.cpu())
        torch.testing.assert_close(outputs[-1].double(), reference.double(), atol=atol, rtol=0)
        compiled_ir.append(kernel.asm["ttgir"])
    # The FP16 baseline has small roundoff, but this scenario must not change it.
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    normalized_ir = []
    for text in compiled_ir:
        for attribute in ("mxg.attention_query_warps = false", "mxg.attention_query_warps = true",
                          "mxg.attention_wide_query_warps", "mxg.attention_short_query_warps"):
            text = text.replace(", " + attribute, "").replace(attribute + ", ", "")
        normalized_ir.append(text)
    assert normalized_ir[0] == normalized_ir[1]


@pytest.mark.parametrize("block_n, head_dim, stages, expected", [(64, 128, 1, 1), (32, 128, 1, 0), (64, 64, 1, 0),
                                                                 (64, 128, 2, 0)])
def test_short_attention_scope(block_n, head_dim, stages, expected, device, fresh_triton_cache):
    torch.manual_seed(1)
    q = torch.randint(-2, 3, (64, head_dim), dtype=torch.int8, device=device)
    k = torch.randint(-2, 3, (128, head_dim), dtype=torch.int8, device=device)
    v = torch.randint(-4, 5, (128, head_dim), dtype=torch.int8, device=device)
    out = torch.empty((64, head_dim), dtype=torch.float32, device=device)
    scale = head_dim**-0.5
    kernel = attention_query_warps_kernel[(1, )](q, k, v, out, 128, BM=64, BN=block_n, D=head_dim, SCALE=scale,
                                                 num_warps=4, num_stages=stages, pipeline="basic",
                                                 scenario="attention-query-warps;attention-short-query-scales")
    assert kernel.metadata.attention_short_query_warps == expected
    ref = torch.softmax(q.float() @ k.float().T * scale, -1) @ v.float()
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
