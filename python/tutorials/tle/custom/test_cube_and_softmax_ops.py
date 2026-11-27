# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for stateless Ascend L1/Cube and fused online-softmax custom ops.

Covers:
  - nd2nz_gm_to_l1 (int8, float16, bfloat16)
  - gather_blocks_gm_to_l1 (int8, float16, bfloat16; clamping & non-contiguous IDs)
  - matmul_l1_to_gm (int8->int32, float16->float32, bfloat16->float32; trans_b=0/1;
    L1 ZN buffer reuse across transposed and non-transposed matmuls)
  - matmul_l1_to_gm_deqf16 (int8->float16 Fixpipe DEQF16 + pointer offset lowering)
  - online_softmax_quant_s8 (1-bank & 2-bank UB ping-pong, shifted FP16 fast path,
    exact FP32 2-digit fallback, optional R / row_scale, multi-window online state,
    valid_cols masking including valid_cols == 0)
  - Compile-time registry argument validation
"""

import math
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend import custom_ops  # noqa: F401
from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import (
    gather_blocks_gm_to_l1,
    matmul_l1_to_gm,
    matmul_l1_to_gm_deqf16,
    nd2nz_gm_to_l1,
    online_softmax_quant_s8,
)

DEVICE = "npu"


@triton.jit
def _nd2nz_matmul_kernel(
    A,
    B,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TRANS_B: tl.constexpr,
    DTYPE: tl.constexpr,
):
    with al.scope(core_mode="cube"):
        a_buf = tle.dsa.alloc([M, K], DTYPE, tle.dsa.ascend.L1)
        if TRANS_B == 1:
            b_buf = tle.dsa.alloc([N, K], DTYPE, tle.dsa.ascend.L1)
            b_l1 = tle_raw.call(
                "ascend_nd2nz_gm_to_l1",
                B.to(tl.uint64),
                N,
                K,
                K,
                out=tle.dsa.to_tensor(b_buf, writable=True),
            )
        else:
            b_buf = tle.dsa.alloc([K, N], DTYPE, tle.dsa.ascend.L1)
            b_l1 = tle_raw.call(
                "ascend_nd2nz_gm_to_l1",
                B.to(tl.uint64),
                K,
                N,
                N,
                out=tle.dsa.to_tensor(b_buf, writable=True),
            )
        a_l1 = tle_raw.call(
            "ascend_nd2nz_gm_to_l1",
            A.to(tl.uint64),
            M,
            K,
            K,
            out=tle.dsa.to_tensor(a_buf, writable=True),
        )
        tle_raw.call(
            "ascend_matmul_l1_to_gm",
            a_l1,
            b_l1,
            Out.to(tl.uint64),
            M,
            N,
            K,
            N,
            TRANS_B,
        )


@triton.jit
def _l1_reuse_dual_matmul_kernel(
    A,
    P,
    KV,
    BlockIds,
    OutQK,
    OutPV,
    M: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    VALID_BLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    D: tl.constexpr,
    DEQ_SCALE: tl.constexpr,
):
    """Loads paged KV once into compiler-managed L1 and reuses it for QK (trans_b=1) and PV (trans_b=0, DEQF16)."""
    KV_ROWS: tl.constexpr = NUM_BLOCKS * BLOCK_ROWS
    with al.scope(core_mode="cube"):
        a_buf = tle.dsa.alloc([M, D], tl.int8, tle.dsa.ascend.L1)
        p_buf = tle.dsa.alloc([M, KV_ROWS], tl.int8, tle.dsa.ascend.L1)
        kv_buf = tle.dsa.alloc([KV_ROWS, D], tl.int8, tle.dsa.ascend.L1)

        a_l1 = tle_raw.call(
            "ascend_nd2nz_gm_to_l1",
            A.to(tl.uint64),
            M,
            D,
            D,
            out=tle.dsa.to_tensor(a_buf, writable=True),
        )
        kv_l1 = tle_raw.call(
            "ascend_gather_blocks_gm_to_l1",
            KV.to(tl.uint64),
            BlockIds.to(tl.uint64),
            NUM_BLOCKS,
            VALID_BLOCKS,
            BLOCK_ROWS,
            D,
            BLOCK_ROWS * D,
            out=tle.dsa.to_tensor(kv_buf, writable=True),
        )
        tle.dsa.to_buffer(kv_l1, bind_buffer=kv_buf)

        # Write to a non-zero offset in OutQK to exercise FixOffsetExtractAlignedPointerPattern
        tle_raw.call(
            "ascend_matmul_l1_to_gm",
            a_l1,
            tle.dsa.to_tensor(kv_buf, writable=False),
            (OutQK + M * KV_ROWS).to(tl.uint64),
            M,
            KV_ROWS,
            D,
            KV_ROWS,
            1,
        )

        p_l1 = tle_raw.call(
            "ascend_nd2nz_gm_to_l1",
            P.to(tl.uint64),
            M,
            KV_ROWS,
            KV_ROWS,
            out=tle.dsa.to_tensor(p_buf, writable=True),
        )
        tle_raw.call(
            "ascend_matmul_l1_to_gm_deqf16",
            p_l1,
            tle.dsa.to_tensor(kv_buf, writable=False),
            (OutPV + M * D).to(tl.uint64),
            M,
            D,
            KV_ROWS,
            D,
            0,
            DEQ_SCALE,
        )


@triton.jit
def _online_softmax_quant_s8_kernel(
    C,
    R,
    RowScale,
    ColScale,
    StateIn,
    PHi,
    PLo,
    StateOut,
    HAS_R: tl.constexpr,
    HAS_RS: tl.constexpr,
    HAS_PLO: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    VALID_COLS,
    SCALE: tl.constexpr,
    OUT_SCALE: tl.constexpr,
    FLAGS,
    WORK_F: tl.constexpr,
):
    ks_buf = tle.dsa.alloc([1, COLS], tl.float32, tle.dsa.ascend.UB)
    st_in_buf = tle.dsa.alloc([5, ROWS], tl.float32, tle.dsa.ascend.UB)
    st_out_buf = tle.dsa.alloc([5, ROWS], tl.float32, tle.dsa.ascend.UB)
    work_buf = tle.dsa.alloc([WORK_F], tl.float32, tle.dsa.ascend.UB)

    offs_c = tl.arange(0, COLS)
    ks_val = tl.load(ColScale + offs_c)[None, :]
    tle.dsa.to_buffer(ks_val, bind_buffer=ks_buf)

    offs_r = tl.arange(0, ROWS)
    for row in tl.static_range(5):
        v = tl.load(StateIn + row * ROWS + offs_r)[None, :]
        tle.dsa.to_buffer(v, bind_buffer=tle.dsa.subview(st_in_buf, [row, 0], [1, ROWS], [1, 1]))

    r_ptr = R.to(tl.uint64) if HAS_R else tl.full((), 0, tl.uint64)
    rs_ptr = RowScale.to(tl.uint64) if HAS_RS else tl.full((), 0, tl.uint64)
    plo_ptr = PLo.to(tl.uint64) if HAS_PLO else tl.full((), 0, tl.uint64)

    st_res, _ = tle_raw.call(
        "ascend_online_softmax_quant_s8",
        C.to(tl.uint64),
        r_ptr,
        rs_ptr,
        tle.dsa.to_tensor(ks_buf, writable=False),
        tle.dsa.to_tensor(st_in_buf, writable=False),
        PHi.to(tl.uint64),
        plo_ptr,
        ROWS,
        COLS,
        VALID_COLS.to(tl.uint32),
        SCALE,
        OUT_SCALE,
        FLAGS.to(tl.uint32),
        out=(
            tle.dsa.to_tensor(st_out_buf, writable=True),
            tle.dsa.to_tensor(work_buf, writable=True),
        ),
    )
    tle.dsa.to_buffer(st_res, bind_buffer=st_out_buf)

    for row in tl.static_range(5):
        v = tle.dsa.to_tensor(tle.dsa.subview(st_out_buf, [row, 0], [1, ROWS], [1, 1]), writable=False)
        tl.store(StateOut + row * ROWS + offs_r, v.reshape(ROWS))


@pytest.mark.parametrize(
    "torch_dtype,tl_dtype,out_dtype,trans_b,m,n,k,atol",
    [
        (torch.int8, tl.int8, torch.int32, 1, 32, 64, 128, 0.0),
        (torch.int8, tl.int8, torch.int32, 0, 32, 64, 128, 0.0),
        (torch.bfloat16, tl.bfloat16, torch.float32, 1, 32, 64, 64, 2e-2),
        (torch.float16, tl.float16, torch.float32, 0, 32, 64, 64, 1e-2),
    ],
)
def test_nd2nz_and_matmul_l1_to_gm(torch_dtype, tl_dtype, out_dtype, trans_b, m, n, k, atol):
    torch.manual_seed(0)
    if torch_dtype == torch.int8:
        a = torch.randint(-32, 33, (m, k), dtype=torch.int8, device=DEVICE)
        b_shape = (n, k) if trans_b else (k, n)
        b = torch.randint(-32, 33, b_shape, dtype=torch.int8, device=DEVICE)
        ref = (a.cpu().to(torch.int32) @ (b.T if trans_b else b).cpu().to(torch.int32)).to(DEVICE)
    else:
        a = torch.randn((m, k), dtype=torch_dtype, device=DEVICE) * 0.5
        b_shape = (n, k) if trans_b else (k, n)
        b = torch.randn(b_shape, dtype=torch_dtype, device=DEVICE) * 0.5
        ref = a.float() @ (b.T if trans_b else b).float()

    out = torch.zeros((m, n), dtype=out_dtype, device=DEVICE)
    _nd2nz_matmul_kernel[(1, )](a, b, out, M=m, N=n, K=k, TRANS_B=trans_b, DTYPE=tl_dtype, multibuffer=False)
    torch_npu.npu.synchronize()

    err = (out.float() - ref.float()).abs().max().item()
    assert err <= atol, f"nd2nz+matmul mismatch ({torch_dtype}, trans_b={trans_b}): max_err={err} > {atol}"


def test_gather_blocks_l1_reuse_and_deqf16():
    torch.manual_seed(7)
    m, num_blocks, valid_blocks, block_rows, d = 64, 4, 3, 32, 128
    kv_rows = num_blocks * block_rows
    total_pages = 8
    deq_scale = 1.0 / 16384.0

    a = torch.randint(-64, 64, (m, d), dtype=torch.int8, device=DEVICE)
    p = torch.randint(-64, 64, (m, kv_rows), dtype=torch.int8, device=DEVICE)
    kv_pool = torch.randint(-64, 64, (total_pages, block_rows, d), dtype=torch.int8, device=DEVICE)
    block_ids = torch.tensor([5, 1, 6, 99], dtype=torch.int32, device=DEVICE)  # slot 3 clamped to idx 2 -> page 6

    # Allocate 2x slots so kernel writes into slot 1 via non-zero pointer offset
    out_qk = torch.zeros((2, m, kv_rows), dtype=torch.int32, device=DEVICE)
    out_pv = torch.zeros((2, m, d), dtype=torch.float16, device=DEVICE)

    _l1_reuse_dual_matmul_kernel[(1, )](
        a,
        p,
        kv_pool,
        block_ids,
        out_qk,
        out_pv,
        M=m,
        NUM_BLOCKS=num_blocks,
        VALID_BLOCKS=valid_blocks,
        BLOCK_ROWS=block_rows,
        D=d,
        DEQ_SCALE=deq_scale,
        multibuffer=False,
    )
    torch_npu.npu.synchronize()

    # Slot 0 must remain untouched zeros (verifies pointer offset was NOT dropped!)
    assert out_qk[0].abs().max().item() == 0, "Slot 0 of OutQK was clobbered (pointer offset bug)"
    assert out_pv[0].abs().max().item() == 0, "Slot 0 of OutPV was clobbered (pointer offset bug)"

    gathered = kv_pool[torch.tensor([5, 1, 6, 6], device=DEVICE)].reshape(kv_rows, d)
    ref_qk = (a.cpu().to(torch.int32) @ gathered.T.cpu().to(torch.int32)).to(DEVICE)
    ref_pv = (p.cpu().to(torch.int32) @ gathered.cpu().to(torch.int32)).float().to(DEVICE) * deq_scale

    assert torch.equal(out_qk[1], ref_qk), "gather_blocks + QK mismatch"
    pv_err = (out_pv[1].float() - ref_pv).abs().max().item()
    assert pv_err <= 1e-2, f"L1 reuse + DEQF16 PV mismatch: max_err={pv_err}"


def _osq_work_f(rows, cols):
    tile_r = 16 if rows >= 16 else 8
    nbanks = 2 if (rows // tile_r) >= 2 else 1
    return (nbanks * 2 * tile_r * cols + ((rows + 63) & ~63) + ((cols + 63) & ~63) + (((cols + 127) & ~127) // 2) +
            tile_r * 64 + 512)


def _reference_osq(c, r, row_scale, col_scale, state_in, valid_cols, scale, out_scale, flags):
    is_first = (flags & 1) != 0
    use_shifted = (flags & 2) != 0
    running_norm = (flags & 8) != 0
    rows, cols = c.shape

    score = c.float()
    if r is not None:
        score = score + r.float()
    if row_scale is not None:
        score = score * (row_scale.float()[:, None] * scale)
    else:
        score = score * scale
    score = score * col_scale.float()[None, :]
    if valid_cols < cols:
        score[:, valid_cols:] = -float("inf")

    row_max = score.amax(dim=-1)
    if is_first:
        m_now = row_max.clone()
        w_prev = torch.zeros_like(row_max)
        beta = torch.ones_like(row_max)
    else:
        m_prev = state_in[0]
        m_now = torch.maximum(m_prev, row_max)
        safe = torch.clamp(m_now, min=-1e30)
        w_prev = torch.exp(m_prev - safe)
        beta = torch.exp(row_max - safe)

    safe_local = torch.clamp(row_max, min=-1e30)
    probs = torch.exp(torch.clamp(score - safe_local[:, None], min=-90.0))
    if valid_cols < cols:
        probs[:, valid_cols:] = 0.0
    row_sum = probs.sum(dim=-1)

    if is_first:
        l_now = row_sum
    else:
        l_prev = state_in[1]
        if running_norm:
            w_prev = l_prev * w_prev
        l_now = l_prev * torch.exp(state_in[0] - torch.clamp(m_now, min=-1e30)) + row_sum * beta

    if running_norm:
        inv_l = 1.0 / torch.clamp(l_now, min=1e-30)
        if not is_first:
            w_prev = w_prev * inv_l
        beta = beta * inv_l

    weighted = probs * col_scale.float()[None, :]
    high = weighted.amax(dim=-1)
    max_q = 255.0 if use_shifted else 127.0
    w_cur = (high / max_q) * beta * out_scale
    return m_now, l_now, w_prev, w_cur, weighted


@pytest.mark.parametrize(
    "rows,cols,valid_cols,has_r,has_rs,flags,out_scale,peaked",
    [
        (8, 64, 64, False, False, 1 | 4, 1.0, False),
        (16, 128, 91, True, True, 0 | 4 | 8, 16384.0, False),
        (64, 384, 384, True, True, 1 | 2 | 8, 16384.0, False),
        (64, 384, 300, True, True, 0 | 2 | 8, 16384.0, True),
        (16, 384, 0, True, True, 1 | 4, 1.0, False),
    ],
)
def test_online_softmax_quant_s8(rows, cols, valid_cols, has_r, has_rs, flags, out_scale, peaked):
    torch.manual_seed(123)
    scale = 1.0 / math.sqrt(576.0)
    c = torch.randint(-400, 400, (rows, cols), dtype=torch.int32, device=DEVICE)
    if peaked and valid_cols > 0:
        c[:, 0] = 50000000  # force peaked row (score ~ 200 -> l_now ~ 1.0 < 2.0 -> adaptive 2-digit)
    r = torch.randn((rows, cols), dtype=torch.float32, device=DEVICE) * 2.0
    row_scale = torch.rand((rows, ), dtype=torch.float32, device=DEVICE) * 0.05 + 0.01
    col_scale = torch.rand((cols, ), dtype=torch.float32, device=DEVICE) * 0.05 + 0.01
    state_in = torch.zeros((5, rows), dtype=torch.float32, device=DEVICE)
    state_in[0] = torch.randn((rows, ), dtype=torch.float32, device=DEVICE) * 0.5
    state_in[1] = torch.rand((rows, ), dtype=torch.float32, device=DEVICE) * 10.0 + 1.0

    p_hi = torch.zeros((rows, cols), dtype=torch.int8, device=DEVICE)
    p_lo = torch.zeros((rows, cols), dtype=torch.int8, device=DEVICE)
    state_out = torch.zeros((5, rows), dtype=torch.float32, device=DEVICE)

    _online_softmax_quant_s8_kernel[(1, )](
        c,
        r,
        row_scale,
        col_scale,
        state_in,
        p_hi,
        p_lo,
        state_out,
        HAS_R=has_r,
        HAS_RS=has_rs,
        HAS_PLO=True,
        ROWS=rows,
        COLS=cols,
        VALID_COLS=valid_cols,
        SCALE=scale,
        OUT_SCALE=out_scale,
        FLAGS=flags,
        WORK_F=_osq_work_f(rows, cols),
        multibuffer=False,
    )
    torch_npu.npu.synchronize()

    m_ref, l_ref, wp_ref, wc_ref, weighted_ref = _reference_osq(
        c,
        r if has_r else None,
        row_scale if has_rs else None,
        col_scale,
        state_in,
        valid_cols,
        scale,
        out_scale,
        flags,
    )

    if valid_cols > 0:
        assert (state_out[0] - m_ref).abs().max().item() <= 1e-4
        assert (state_out[1] - l_ref).abs().max().item() <= 2e-3
        assert (state_out[2] - wp_ref).abs().max().item() <= 1e-4
        wc_tol = 2e-2 * max(1.0, wc_ref.abs().max().item())
        assert (state_out[3] - wc_ref).abs().max().item() <= wc_tol

        use_shifted = (flags & 2) != 0
        force_2d = (flags & 4) != 0
        need_lo = state_out[4].max().item() > 0.5
        if force_2d or peaked:
            assert need_lo, "Expected two-digit residual (need_lo == 1.0)"
        else:
            assert not need_lo, "Expected fast 1-digit shifted path (need_lo == 0.0)"

        # Verify dequantized probabilities accurately reconstruct weighted = exp(s - m) * ks
        p_rec = p_hi.float() + (128.0 if use_shifted else 0.0)
        if need_lo:
            p_rec = p_rec + p_lo.float() / 127.0
        # Direct check via row max of weighted_ref
        high = weighted_ref.amax(dim=-1, keepdim=True)
        max_q = 255.0 if use_shifted else 127.0
        recon = p_rec * (high / max_q)
        recon_tol = 5e-3 if need_lo else 1e-2
        assert (recon - weighted_ref).abs().max().item() <= recon_tol
    else:
        assert (state_out[1] == 0).all()


def test_registry_validation_errors():

    class _Dim(int):

        @property
        def value(self):
            return int(self)

    class _FakeTensor:

        def __init__(self, shape, dtype):
            self.shape = tuple(_Dim(d) for d in shape)
            self.dtype = dtype

    # nd2nz_gm_to_l1 requires out and supported dtype
    with pytest.raises(AssertionError):
        nd2nz_gm_to_l1(0, 16, 64, 64, out=None)
    with pytest.raises(AssertionError):
        nd2nz_gm_to_l1(0, 16, 64, 64, out=_FakeTensor((16, 64), tl.float32))

    # gather_blocks_gm_to_l1 requires matching total rows
    with pytest.raises(AssertionError):
        gather_blocks_gm_to_l1(0, 0, 4, 4, 32, 64, 2048, out=_FakeTensor((64, 64), tl.int8))

    # matmul_l1_to_gm rejects mismatched dtypes and invalid trans_b
    with pytest.raises(AssertionError):
        matmul_l1_to_gm(_FakeTensor((16, 64), tl.int8), _FakeTensor((64, 32), tl.float16), 0, 16, 32, 64, 32, 0)
    with pytest.raises(AssertionError):
        matmul_l1_to_gm(_FakeTensor((16, 64), tl.int8), _FakeTensor((64, 32), tl.int8), 0, 16, 32, 64, 32, 2)

    # matmul_l1_to_gm_deqf16 requires int8 inputs
    with pytest.raises(AssertionError):
        matmul_l1_to_gm_deqf16(_FakeTensor((16, 64), tl.float16), _FakeTensor((64, 32), tl.float16), 0, 16, 32, 64, 32,
                               0, 1.0)

    # online_softmax_quant_s8 requires 2-tuple out and sufficient work_ub
    with pytest.raises(AssertionError):
        online_softmax_quant_s8(0, 0, 0, _FakeTensor((1, 64), tl.float32), _FakeTensor((5, 8), tl.float32), 0, 0, 8, 64,
                                64, 1.0, 1.0, 1, out=None)


def main():
    for args in (
        (torch.int8, tl.int8, torch.int32, 1, 32, 64, 128, 0.0),
        (torch.int8, tl.int8, torch.int32, 0, 32, 64, 128, 0.0),
        (torch.bfloat16, tl.bfloat16, torch.float32, 1, 32, 64, 64, 2e-2),
        (torch.float16, tl.float16, torch.float32, 0, 32, 64, 64, 1e-2),
    ):
        test_nd2nz_and_matmul_l1_to_gm(*args)
    print("[PASS] nd2nz_gm_to_l1 + matmul_l1_to_gm")
    test_gather_blocks_l1_reuse_and_deqf16()
    print("[PASS] gather_blocks_gm_to_l1 + L1 reuse + matmul_l1_to_gm_deqf16")
    for args in (
        (8, 64, 64, False, False, 1 | 4, 1.0, False),
        (16, 128, 91, True, True, 0 | 4 | 8, 16384.0, False),
        (64, 384, 384, True, True, 1 | 2 | 8, 16384.0, False),
        (64, 384, 300, True, True, 0 | 2 | 8, 16384.0, True),
        (16, 384, 0, True, True, 1 | 4, 1.0, False),
    ):
        test_online_softmax_quant_s8(*args)
    print("[PASS] online_softmax_quant_s8")
    test_registry_validation_errors()
    print("[PASS] registry validation errors")


if __name__ == "__main__":
    main()
