"""SQMMA instruction selection follows proven row-reduction consumers."""

import re

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_warp_specialize import _compile_ws_ir


@triton.jit
def consumer_probe(Out, Row, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, DTYPE: tl.constexpr, MODE: tl.constexpr,
                   LAYOUT: tl.constexpr):
    q = tle.gpu.alloc((M, K), DTYPE, init_value=tl.full((M, K), 1.0, DTYPE))
    k = tle.gpu.alloc((N, K), DTYPE, init_value=tl.full((N, K), 1.0, DTYPE))
    tl.debug_barrier()
    acc = tl.zeros((M, N), tl.float32)
    if MODE == 5:
        acc = tle.gpu.set_layout(acc, LAYOUT)
    scores = tle.gpu.wgmma(q, k, acc=acc, trans_b=True)
    scores = tle.gpu.wgmma_wait(0, scores)
    if MODE == 8:
        positions = tl.broadcast_to((tl.arange(0, N) ^ 1)[None, :], (M, N))
        scores = tl.gather(scores, positions, axis=1)
    if MODE == 0 or MODE == 4:
        probabilities = scores
    elif MODE == 3 or MODE == 10:
        if MODE == 10:
            column_sum = tl.sum(tl.trans(scores), axis=1)
        else:
            column_sum = tl.sum(scores, axis=0)
        probabilities = scores - column_sum[None, :]
    else:
        if MODE == 9:
            row_max = tl.max(tl.trans(scores), axis=0)
        else:
            row_max = tl.max(scores, axis=1)
        probabilities = tl.exp2(scores - row_max[:, None])
        probabilities = probabilities / tl.sum(probabilities, axis=1)[:, None]
        if MODE == 6:
            probabilities = probabilities + tl.sum(scores, axis=0)[None, :]
    if MODE != 0 and MODE != 5:
        tl.store(Row + tl.arange(0, M), tl.sum(scores, axis=1))
    if MODE != 2:
        if MODE == 7:
            p = tle.gpu.alloc((M, N), DTYPE)
            if tl.program_id(0) == 0:
                tl.store(tle.gpu.local_ptr(p), probabilities.to(DTYPE))
            else:
                tl.store(tle.gpu.local_ptr(p), (probabilities + 1.0).to(DTYPE))
        else:
            p = tle.gpu.alloc((M, N), DTYPE, init_value=probabilities.to(DTYPE))
        v = tle.gpu.alloc((N, 128), DTYPE, init_value=tl.full((N, 128), 1.0, DTYPE))
        tl.debug_barrier()
        result = tle.gpu.wgmma(p, v)
        result = tle.gpu.wgmma_wait(0, result)
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, 128)[None, :]
        tl.store(Out + rows * 128 + cols, result)
    else:
        rows = tl.arange(0, M)[:, None]
        cols = tl.arange(0, N)[None, :]
        tl.store(Out + rows * N + cols, scores)


@triton.jit
def ssa_consumer_probe(Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    dtype: tl.constexpr = tl.float8e4nv
    q = tle.gpu.alloc((M, K), dtype, init_value=tl.full((M, K), 1.0, dtype))
    k = tle.gpu.alloc((N, K), dtype, init_value=tl.full((N, K), 1.0, dtype))
    tl.debug_barrier()
    scores = tle.gpu.wgmma(q, k, trans_b=True)
    scores = tle.gpu.wgmma_wait(0, scores)
    p = tl.exp2(scores - tl.max(scores, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    v = tl.full((N, 128), 1.0, dtype)
    result = tl.dot(p.to(dtype), v)
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, 128)[None, :]
    tl.store(Out + rows * 128 + cols, result)


@pytest.mark.parametrize("dtype", [tl.float8e4nv, tl.bfloat16])
@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
def test_row_reduction_consumer_selection(dtype, mode):
    _, ttgir = _compile_ws_ir(
        consumer_probe,
        {"Out": "*fp32", "Row": "*fp32"},
        {"M": 32, "N": 64, "K": 256 if dtype == tl.float8e4nv else 64, "DTYPE": dtype, "MODE": mode, "LAYOUT": None},
        consumer_warps=8,
    )
    expected = "warpsPerCTA = [8, 1], instrShape = [16, 64," if mode in (
        1, 9) else "warpsPerCTA = [4, 2], instrShape = [32, 32,"
    dot = next(line for line in ttgir.splitlines() if " = ttmg.squad_dot " in line)
    encoding = re.search(r"-> tensor<32x64xf32, (#[^>]+)>", dot).group(1)
    assert expected in next(line for line in ttgir.splitlines() if line.startswith(encoding + " = "))


@pytest.mark.parametrize("m,n,warps,explicit,expected", [
    (32, 64, 8, True, "warpsPerCTA = [4, 2], instrShape = [32, 32,"),
    (32, 64, 4, False, "warpsPerCTA = [4, 1], instrShape = [32, 64,"),
    (64, 64, 8, False, "warpsPerCTA = [8, 1], instrShape = [32, 64,"),
    (32, 128, 8, False, "warpsPerCTA = [4, 2], instrShape = [32, 64,"),
])
def test_row_reduction_preserves_other_selection_priorities(m, n, warps, explicit, expected):
    _, ttgir = _compile_ws_ir(
        consumer_probe,
        {"Out": "*fp32", "Row": "*fp32"},
        {
            "M": m, "N": n, "K": 256, "DTYPE": tl.float8e4nv, "MODE": 5 if explicit else 1, "LAYOUT":
            tle.gpu.BlockEncoding((1, 1), (1, 32), (4, 2), (1, 0)) if explicit else None
        },
        consumer_warps=warps,
    )
    dot = next(line for line in ttgir.splitlines() if " = ttmg.squad_dot " in line)
    encoding = re.search(rf"-> tensor<{m}x{n}xf32, (#[^>]+)>", dot).group(1)
    assert expected in next(line for line in ttgir.splitlines() if line.startswith(encoding + " = "))


def test_row_reduction_direct_ssa_consumer():
    _, ttgir = _compile_ws_ir(ssa_consumer_probe, {"Out": "*fp32"}, {"M": 32, "N": 64, "K": 256}, consumer_warps=8)
    assert "warpsPerCTA = [8, 1], instrShape = [16, 64, 128]" in ttgir
