# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for stateless Ascend L1/Cube and fused Vector custom ops.

Covers:
  - nd2nz_gm_to_l1 (int8, float16, bfloat16)
  - gather_blocks_gm_to_l1 (int8, float16, bfloat16; clamping & non-contiguous IDs)
  - gather_blocks_matmul_pair_l1_to_gm (chunk-pipelined int8+bf16 gather + dual
    trans_b=1 matmuls; preloaded_rhs0_chunks=0 & 1)
  - matmul_l1_to_gm (int8->int32, float16->float32, bfloat16->float32; trans_b=0/1;
    L1 ZN buffer reuse across transposed and non-transposed matmuls)
  - matmul_l1_to_gm_deqf16 (int8->float16 Fixpipe DEQF16 + pointer offset lowering)
  - nd2nz_matmul_l1_to_gm_deqf16 (k-chunked ND2NZ + DEQF16 matmul, with and
    without overlapped chunk-0 block-gather prefetch into rhs_l1)
  - online_softmax_quant_s8 (1-bank & 2-bank UB ping-pong, hybrid FP16 vexp +
    FP32 row_sum, group_scale_fast & per-row FP16 fast path, direct GM paged
    col_scale gather & flag_gm write, exact FP32 2-digit fallback, peaked &
    outlier fallback, valid_cols masking including valid_cols == 0)
  - online_rescaled_accum (FP16 & FP32 2-bank UB ping-pong, bias/lo_scale,
    state_ub[4,0] need_lo gating, optional final BF16 GM writeback)
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
    gather_blocks_matmul_pair_l1_to_gm,
    matmul_l1_to_gm,
    matmul_l1_to_gm_deqf16,
    nd2nz_gm_to_l1,
    nd2nz_matmul_l1_to_gm_deqf16,
    online_rescaled_accum,
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
def _pipelined_cube_pair_and_pv_kernel(
    Q,
    QR,
    K,
    KR,
    P,
    Ids0,
    Ids1,
    OutC0,
    OutR0,
    OutPV0,
    OutC1,
    OutR1,
    M: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    VALID0: tl.constexpr,
    VALID1: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    DEQ_SCALE: tl.constexpr,
):
    """Exercises gather_blocks_matmul_pair_l1_to_gm + nd2nz_matmul_l1_to_gm_deqf16 chunk-0 prefetch."""
    KV_ROWS: tl.constexpr = NUM_BLOCKS * BLOCK_ROWS
    with al.scope(core_mode="cube"):
        q_buf = tle.dsa.alloc([M, K0], tl.int8, tle.dsa.ascend.L1)
        qr_buf = tle.dsa.alloc([M, K1], tl.bfloat16, tle.dsa.ascend.L1)
        k_buf = tle.dsa.alloc([KV_ROWS, K0], tl.int8, tle.dsa.ascend.L1)
        kr_buf = tle.dsa.alloc([KV_ROWS, K1], tl.bfloat16, tle.dsa.ascend.L1)
        p_buf = tle.dsa.alloc([M, KV_ROWS], tl.int8, tle.dsa.ascend.L1)

        q_l1 = tle_raw.call(
            "ascend_nd2nz_gm_to_l1",
            Q.to(tl.uint64),
            M,
            K0,
            K0,
            out=tle.dsa.to_tensor(q_buf, writable=True),
        )
        qr_l1 = tle_raw.call(
            "ascend_nd2nz_gm_to_l1",
            QR.to(tl.uint64),
            M,
            K1,
            K1,
            out=tle.dsa.to_tensor(qr_buf, writable=True),
        )
        # Step 0: gather + QK pair without preloaded chunk 0
        k0_l1, _ = tle_raw.call(
            "ascend_gather_blocks_matmul_pair_l1_to_gm",
            q_l1,
            K.to(tl.uint64),
            qr_l1,
            KR.to(tl.uint64),
            Ids0.to(tl.uint64),
            OutC0.to(tl.uint64),
            OutR0.to(tl.uint64),
            NUM_BLOCKS,
            VALID0,
            BLOCK_ROWS,
            M,
            K0,
            K1,
            BLOCK_ROWS * K0,
            BLOCK_ROWS * K1,
            KV_ROWS,
            tl.cast(0, tl.uint32),
            out=(
                tle.dsa.to_tensor(k_buf, writable=True),
                tle.dsa.to_tensor(kr_buf, writable=True),
            ),
        )
        tle.dsa.to_buffer(k0_l1, bind_buffer=k_buf)

        # Step 1: chunked ND2NZ + DEQF16 PV, overlapping chunk-0 prefetch of Ids1 into k_buf
        _, k_pf = tle_raw.call(
            "ascend_nd2nz_matmul_l1_to_gm_deqf16",
            P.to(tl.uint64),
            OutPV0.to(tl.uint64),
            M,
            K0,
            KV_ROWS,
            K0,
            DEQ_SCALE,
            K.to(tl.uint64),
            Ids1.to(tl.uint64),
            tl.cast(2, tl.uint32),
            tl.cast(VALID1, tl.uint32),
            tl.cast(BLOCK_ROWS, tl.uint32),
            tl.cast(BLOCK_ROWS * K0, tl.uint32),
            out=(
                tle.dsa.to_tensor(p_buf, writable=True),
                tle.dsa.to_tensor(k_buf, writable=True),
            ),
        )
        tle.dsa.to_buffer(k_pf, bind_buffer=k_buf)

        # Step 2: gather + QK pair consuming preloaded chunk 0 of Ids1
        k1_l1, _ = tle_raw.call(
            "ascend_gather_blocks_matmul_pair_l1_to_gm",
            q_l1,
            K.to(tl.uint64),
            qr_l1,
            KR.to(tl.uint64),
            Ids1.to(tl.uint64),
            OutC1.to(tl.uint64),
            OutR1.to(tl.uint64),
            NUM_BLOCKS,
            VALID1,
            BLOCK_ROWS,
            M,
            K0,
            K1,
            BLOCK_ROWS * K0,
            BLOCK_ROWS * K1,
            KV_ROWS,
            tl.cast(1, tl.uint32),
            out=(
                tle.dsa.to_tensor(k_buf, writable=True),
                tle.dsa.to_tensor(kr_buf, writable=True),
            ),
        )
        tle.dsa.to_buffer(k1_l1, bind_buffer=k_buf)


@triton.jit
def _online_softmax_quant_s8_kernel(
    C,
    R,
    RowScale,
    ColScale,
    ColBlockIds,
    FlagOut,
    StateIn,
    PHi,
    PLo,
    StateOut,
    HAS_R: tl.constexpr,
    HAS_RS: tl.constexpr,
    GATHER_KS: tl.constexpr,
    HAS_FLAG: tl.constexpr,
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
    if GATHER_KS:
        tle.dsa.to_buffer(tl.zeros((1, COLS), dtype=tl.float32), bind_buffer=ks_buf)
    else:
        ks_val = tl.load(ColScale + offs_c)[None, :]
        tle.dsa.to_buffer(ks_val, bind_buffer=ks_buf)

    offs_r = tl.arange(0, ROWS)
    for row in tl.static_range(5):
        v = tl.load(StateIn + row * ROWS + offs_r)[None, :]
        tle.dsa.to_buffer(v, bind_buffer=tle.dsa.subview(st_in_buf, [row, 0], [1, ROWS], [1, 1]))

    r_ptr = R.to(tl.uint64) if HAS_R else tl.full((), 0, tl.uint64)
    rs_ptr = RowScale.to(tl.uint64) if HAS_RS else tl.full((), 0, tl.uint64)
    ks_gm_ptr = ColScale.to(tl.uint64) if GATHER_KS else tl.full((), 0, tl.uint64)
    ids_ptr = ColBlockIds.to(tl.uint64) if GATHER_KS else tl.full((), 0, tl.uint64)
    flag_ptr = FlagOut.to(tl.uint64) if HAS_FLAG else tl.full((), 0, tl.uint64)
    plo_ptr = PLo.to(tl.uint64) if HAS_PLO else tl.full((), 0, tl.uint64)

    st_res, _ = tle_raw.call(
        "ascend_online_softmax_quant_s8",
        C.to(tl.uint64),
        r_ptr,
        rs_ptr,
        ks_gm_ptr,
        ids_ptr,
        flag_ptr,
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


@triton.jit
def _online_rescaled_accum_kernel(
    Src0,
    SrcLo0,
    Bias0,
    State0,
    Src1,
    SrcLo1,
    Bias1,
    State1,
    Out,
    HAS_LO0: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_FP16: tl.constexpr,
    CHECK_NEED_LO: tl.constexpr,
    ROWS: tl.constexpr,
    DIM: tl.constexpr,
    BIAS_SCALE: tl.constexpr,
    LO_SCALE: tl.constexpr,
    WORK_F: tl.constexpr,
):
    acc_dtype: tl.constexpr = tl.float16 if USE_FP16 else tl.float32
    acc_buf = tle.dsa.alloc([ROWS, DIM], acc_dtype, tle.dsa.ascend.UB)
    st0_buf = tle.dsa.alloc([5, ROWS], tl.float32, tle.dsa.ascend.UB)
    st1_buf = tle.dsa.alloc([5, ROWS], tl.float32, tle.dsa.ascend.UB)
    work_buf = tle.dsa.alloc([WORK_F], tl.float32, tle.dsa.ascend.UB)

    offs_r = tl.arange(0, ROWS)
    for row in tl.static_range(5):
        v0 = tl.load(State0 + row * ROWS + offs_r)[None, :]
        tle.dsa.to_buffer(v0, bind_buffer=tle.dsa.subview(st0_buf, [row, 0], [1, ROWS], [1, 1]))
        v1 = tl.load(State1 + row * ROWS + offs_r)[None, :]
        tle.dsa.to_buffer(v1, bind_buffer=tle.dsa.subview(st1_buf, [row, 0], [1, ROWS], [1, 1]))

    lo0_ptr = SrcLo0.to(tl.uint64) if HAS_LO0 else tl.full((), 0, tl.uint64)
    b0_ptr = Bias0.to(tl.uint64) if HAS_BIAS else tl.full((), 0, tl.uint64)
    f0 = (1 | (16 if CHECK_NEED_LO else 0))
    acc_res, _ = tle_raw.call(
        "ascend_online_rescaled_accum",
        Src0.to(tl.uint64),
        lo0_ptr,
        b0_ptr,
        tle.dsa.to_tensor(st0_buf, writable=False),
        tl.cast(0, tl.uint64),
        ROWS,
        DIM,
        BIAS_SCALE,
        LO_SCALE,
        tl.cast(f0, tl.uint32),
        out=(
            tle.dsa.to_tensor(acc_buf, writable=True),
            tle.dsa.to_tensor(work_buf, writable=True),
        ),
    )
    tle.dsa.to_buffer(acc_res, bind_buffer=acc_buf)

    b1_ptr = Bias1.to(tl.uint64) if HAS_BIAS else tl.full((), 0, tl.uint64)
    f1 = (2 | (0 if USE_FP16 else 8) | (16 if CHECK_NEED_LO else 0))
    acc_res1, _ = tle_raw.call(
        "ascend_online_rescaled_accum",
        Src1.to(tl.uint64),
        SrcLo1.to(tl.uint64),
        b1_ptr,
        tle.dsa.to_tensor(st1_buf, writable=False),
        Out.to(tl.uint64),
        ROWS,
        DIM,
        BIAS_SCALE,
        LO_SCALE,
        tl.cast(f1, tl.uint32),
        out=(
            tle.dsa.to_tensor(acc_buf, writable=True),
            tle.dsa.to_tensor(work_buf, writable=True),
        ),
    )
    tle.dsa.to_buffer(acc_res1, bind_buffer=acc_buf)


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
    tile_r = 8 if (rows >= 128 or cols > 384) else (16 if rows >= 16 else 8)
    nbanks = 2 if (rows // tile_r) >= 2 else 1
    return (nbanks * 2 * tile_r * cols + ((rows + 63) & ~63) + 2 * ((cols + 63) & ~63) + (((cols + 127) & ~127) // 2) +
            tile_r * 64 + 512)


def test_pipelined_cube_pair_and_nd2nz_deqf16_prefetch():
    torch.manual_seed(11)
    m, num_blocks, block_rows, k0, k1 = 128, 6, 64, 512, 64
    kv_rows = num_blocks * block_rows
    total_pages = 16
    deq_scale = 1.0 / 16384.0

    q = torch.randint(-64, 64, (m, k0), dtype=torch.int8, device=DEVICE)
    qr = torch.randn((m, k1), dtype=torch.bfloat16, device=DEVICE) * 0.5
    k_pool = torch.randint(-64, 64, (total_pages, block_rows, k0), dtype=torch.int8, device=DEVICE)
    kr_pool = torch.randn((total_pages, block_rows, k1), dtype=torch.bfloat16, device=DEVICE) * 0.5
    p = torch.randint(0, 128, (m, kv_rows), dtype=torch.int8, device=DEVICE)
    ids0 = torch.tensor([12, 3, 7, 1, 14, 5], dtype=torch.int32, device=DEVICE)
    ids1 = torch.tensor([9, 2, 11, 4, 99, 99], dtype=torch.int32, device=DEVICE)  # valid1=4 -> clamp to idx 3 (page 4)

    out_c0 = torch.zeros((m, kv_rows), dtype=torch.int32, device=DEVICE)
    out_r0 = torch.zeros((m, kv_rows), dtype=torch.float32, device=DEVICE)
    out_pv0 = torch.zeros((m, k0), dtype=torch.float16, device=DEVICE)
    out_c1 = torch.zeros((m, kv_rows), dtype=torch.int32, device=DEVICE)
    out_r1 = torch.zeros((m, kv_rows), dtype=torch.float32, device=DEVICE)

    _pipelined_cube_pair_and_pv_kernel[(1, )](
        q,
        qr,
        k_pool,
        kr_pool,
        p,
        ids0,
        ids1,
        out_c0,
        out_r0,
        out_pv0,
        out_c1,
        out_r1,
        M=m,
        NUM_BLOCKS=num_blocks,
        VALID0=6,
        VALID1=4,
        BLOCK_ROWS=block_rows,
        K0=k0,
        K1=k1,
        DEQ_SCALE=deq_scale,
        multibuffer=False,
    )
    torch_npu.npu.synchronize()

    k0_ref = k_pool[ids0.long()].reshape(kv_rows, k0)
    kr0_ref = kr_pool[ids0.long()].reshape(kv_rows, k1)
    assert torch.equal(out_c0, (q.cpu().to(torch.int32) @ k0_ref.T.cpu().to(torch.int32)).to(DEVICE))
    assert (out_r0 - (qr.float() @ kr0_ref.T.float())).abs().max().item() <= 3e-2
    pv0_ref = (p.cpu().to(torch.int32) @ k0_ref.cpu().to(torch.int32)).float().to(DEVICE) * deq_scale
    assert (out_pv0.float() - pv0_ref).abs().max().item() <= 5e-2

    ids1_clamped = torch.tensor([9, 2, 11, 4, 4, 4], dtype=torch.long, device=DEVICE)
    k1_ref = k_pool[ids1_clamped].reshape(kv_rows, k0)
    kr1_ref = kr_pool[ids1_clamped].reshape(kv_rows, k1)
    assert torch.equal(out_c1, (q.cpu().to(torch.int32) @ k1_ref.T.cpu().to(torch.int32)).to(DEVICE))
    assert (out_r1 - (qr.float() @ kr1_ref.T.float())).abs().max().item() <= 3e-2


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
    "rows,cols,valid_cols,has_r,has_rs,gather_ks,flags,out_scale,peaked",
    [
        (8, 64, 64, False, False, False, 1 | 4, 1.0, False),
        (16, 128, 91, True, True, False, 0 | 4 | 8, 16384.0, False),
        (64, 384, 384, True, True, False, 1 | 2 | 8, 16384.0, False),
        (64, 384, 384, True, True, True, 1 | 8 | 16, 16384.0, False),
        (64, 384, 300, True, True, True, 0 | 2 | 8, 16384.0, True),
        (16, 384, 0, True, True, True, 1 | 4, 1.0, False),
    ],
)
def test_online_softmax_quant_s8(rows, cols, valid_cols, has_r, has_rs, gather_ks, flags, out_scale, peaked):
    torch.manual_seed(123)
    scale = 1.0 / math.sqrt(576.0)
    c = torch.randint(-400, 400, (rows, cols), dtype=torch.int32, device=DEVICE)
    if peaked and valid_cols > 0:
        c[:, 0] = 50000000  # force peaked row (score ~ 200 -> l_now ~ 1.0 < 2.0 -> adaptive 2-digit)
    r = torch.randn((rows, cols), dtype=torch.float32, device=DEVICE) * 2.0
    row_scale = torch.rand((rows, ), dtype=torch.float32, device=DEVICE) * 0.05 + 0.01
    if gather_ks:
        num_b = cols // 64
        ks_pool = torch.rand((12, 64), dtype=torch.float32, device=DEVICE) * 0.02 + 0.03
        col_block_ids = torch.tensor([(i * 3 + 1) % 12 for i in range(num_b)], dtype=torch.int32, device=DEVICE)
        valid_b = (valid_cols + 63) // 64
        eff_ids = [col_block_ids[i if i < valid_b else max(valid_b - 1, 0)].item() for i in range(num_b)]
        col_scale = ks_pool[torch.tensor(eff_ids, device=DEVICE)].reshape(cols).contiguous()
        if valid_cols < cols:
            col_scale[valid_cols:] = 0.0
        ks_arg = ks_pool
    else:
        col_scale = torch.rand((cols, ), dtype=torch.float32, device=DEVICE) * 0.02 + 0.03
        col_block_ids = torch.zeros((1, ), dtype=torch.int32, device=DEVICE)
        ks_arg = col_scale
    flag_out = torch.full((8, ), -1, dtype=torch.int32, device=DEVICE)
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
        ks_arg,
        col_block_ids,
        flag_out,
        state_in,
        p_hi,
        p_lo,
        state_out,
        HAS_R=has_r,
        HAS_RS=has_rs,
        GATHER_KS=gather_ks,
        HAS_FLAG=gather_ks,
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
        assert (state_out[1] - l_ref).abs().max().item() <= 5e-2
        assert (state_out[2] - wp_ref).abs().max().item() <= 1e-4

        use_shifted = (flags & 2) != 0
        force_2d = (flags & 4) != 0
        group_fast = (flags & 16) != 0 and not use_shifted and not force_2d and valid_cols >= cols and cols == 384
        need_lo = state_out[4].max().item() > 0.5
        if force_2d or peaked:
            assert need_lo, "Expected two-digit residual (need_lo == 1.0)"
            if gather_ks:
                assert flag_out[0].item() == 1
        else:
            assert not need_lo, "Expected fast 1-digit path (need_lo == 0.0)"
            if gather_ks:
                assert flag_out[0].item() == 0

        if not group_fast:
            wc_tol = 2e-2 * max(1.0, wc_ref.abs().max().item())
            assert (state_out[3] - wc_ref).abs().max().item() <= wc_tol

        # Verify dequantized probabilities accurately reconstruct weighted = exp(s - m) * ks
        p_rec = p_hi.float() + (128.0 if use_shifted else 0.0)
        if need_lo:
            p_rec = p_rec + p_lo.float() / 127.0
        if group_fast and not need_lo:
            scale_step = col_scale.abs().amax() / 127.0
            recon = p_rec * scale_step
        else:
            high = weighted_ref.amax(dim=-1, keepdim=True)
            max_q = 255.0 if use_shifted else 127.0
            recon = p_rec * (high / max_q)
        recon_tol = 5e-3 if need_lo else 1e-2
        assert (recon - weighted_ref).abs().max().item() <= recon_tol
    else:
        assert (state_out[1] == 0).all()


@pytest.mark.parametrize(
    "rows,use_fp16,has_lo0,has_bias,check_need_lo,need_lo1",
    [
        (64, True, True, False, True, 0.0),
        (64, True, False, True, True, 1.0),
        (16, False, True, True, True, 1.0),
        (16, False, True, False, True, 0.0),
    ],
)
def test_online_rescaled_accum(rows, use_fp16, has_lo0, has_bias, check_need_lo, need_lo1):
    torch.manual_seed(77)
    dim = 512
    bias_scale = 0.25
    lo_scale = 1.0 / 127.0
    src_dtype = torch.float16 if use_fp16 else torch.int32

    if use_fp16:
        s0 = torch.randn((rows, dim), dtype=torch.float16, device=DEVICE) * 0.5
        s0_lo = torch.randn((rows, dim), dtype=torch.float16, device=DEVICE) * 0.5
        b0 = torch.randn((dim, ), dtype=torch.float16, device=DEVICE) * 0.5
        s1 = torch.randn((rows, dim), dtype=torch.float16, device=DEVICE) * 0.5
        s1_lo = torch.randn((rows, dim), dtype=torch.float16, device=DEVICE) * 0.5
        b1 = torch.randn((dim, ), dtype=torch.float16, device=DEVICE) * 0.5
    else:
        s0 = torch.randint(-200, 200, (rows, dim), dtype=src_dtype, device=DEVICE)
        s0_lo = torch.randint(-200, 200, (rows, dim), dtype=src_dtype, device=DEVICE)
        b0 = torch.randint(-100, 100, (dim, ), dtype=src_dtype, device=DEVICE)
        s1 = torch.randint(-200, 200, (rows, dim), dtype=src_dtype, device=DEVICE)
        s1_lo = torch.randint(-200, 200, (rows, dim), dtype=src_dtype, device=DEVICE)
        b1 = torch.randint(-100, 100, (dim, ), dtype=src_dtype, device=DEVICE)

    st0 = torch.zeros((5, rows), dtype=torch.float32, device=DEVICE)
    st0[1] = torch.rand((rows, ), device=DEVICE) * 5.0 + 1.0
    st0[2] = 0.0
    st0[3] = torch.rand((rows, ), device=DEVICE) * 0.5 + 0.1
    st0[4] = 1.0 if has_lo0 else 0.0

    st1 = torch.zeros((5, rows), dtype=torch.float32, device=DEVICE)
    st1[1] = torch.rand((rows, ), device=DEVICE) * 5.0 + 2.0
    st1[2] = torch.rand((rows, ), device=DEVICE) * 0.5 + 0.2
    st1[3] = torch.rand((rows, ), device=DEVICE) * 0.5 + 0.1
    st1[4] = need_lo1

    if use_fp16:
        tr = 16 if rows >= 128 else (32 if rows >= 32 else rows)
        nb = 2 if (rows // tr) >= 2 else 1
        lo_r = 8 if tr >= 8 else tr
        work_f = (nb * tr + lo_r) * (dim // 2) + dim // 2 + tr * 16 + 128
    else:
        tr = 8 if rows >= 8 else rows
        nb = 2 if (rows // tr) >= 2 else 1
        work_f = (nb + 1) * tr * dim + dim + tr * 16 + 128

    out = torch.zeros((rows, dim), dtype=torch.bfloat16, device=DEVICE)
    _online_rescaled_accum_kernel[(1, )](
        s0,
        s0_lo,
        b0,
        st0,
        s1,
        s1_lo,
        b1,
        st1,
        out,
        HAS_LO0=has_lo0,
        HAS_BIAS=has_bias,
        USE_FP16=use_fp16,
        CHECK_NEED_LO=check_need_lo,
        ROWS=rows,
        DIM=dim,
        BIAS_SCALE=bias_scale,
        LO_SCALE=lo_scale,
        WORK_F=work_f,
        multibuffer=False,
    )
    torch_npu.npu.synchronize()

    u0 = s0.float()
    if has_bias:
        u0 = u0 + b0.float()[None, :] * bias_scale
    if has_lo0:
        u0 = u0 + s0_lo.float() * lo_scale
    acc = u0 * st0[3][:, None]

    u1 = s1.float()
    if has_bias:
        u1 = u1 + b1.float()[None, :] * bias_scale
    if need_lo1 > 0.5:
        u1 = u1 + s1_lo.float() * lo_scale
    acc = acc * st1[2][:, None] + u1 * st1[3][:, None]
    if not use_fp16:
        acc = acc / st1[1][:, None].clamp_min(1e-30)
    ref = acc.to(torch.bfloat16)

    tol = 2e-2 if use_fp16 else 2e-1
    assert (out.float() - ref.float()).abs().max().item() <= tol


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

    # gather_blocks_matmul_pair_l1_to_gm requires 2-tuple out and matching dtypes
    with pytest.raises(AssertionError):
        gather_blocks_matmul_pair_l1_to_gm(
            _FakeTensor((64, 512), tl.int8),
            0,
            _FakeTensor((64, 64), tl.bfloat16),
            0,
            0,
            0,
            0,
            6,
            6,
            64,
            64,
            512,
            64,
            64 * 512,
            64 * 64,
            384,
            0,
            out=None,
        )

    # matmul_l1_to_gm rejects mismatched dtypes and invalid trans_b
    with pytest.raises(AssertionError):
        matmul_l1_to_gm(_FakeTensor((16, 64), tl.int8), _FakeTensor((64, 32), tl.float16), 0, 16, 32, 64, 32, 0)
    with pytest.raises(AssertionError):
        matmul_l1_to_gm(_FakeTensor((16, 64), tl.int8), _FakeTensor((64, 32), tl.int8), 0, 16, 32, 64, 32, 2)

    # matmul_l1_to_gm_deqf16 requires int8 inputs
    with pytest.raises(AssertionError):
        matmul_l1_to_gm_deqf16(_FakeTensor((16, 64), tl.float16), _FakeTensor((64, 32), tl.float16), 0, 16, 32, 64, 32,
                               0, 1.0)

    # nd2nz_matmul_l1_to_gm_deqf16 requires 2-tuple int8 out
    with pytest.raises(AssertionError):
        nd2nz_matmul_l1_to_gm_deqf16(0, 0, 64, 512, 384, 512, 1.0, 0, 0, 0, 0, 0, 0, out=None)

    # online_softmax_quant_s8 requires 2-tuple out and sufficient work_ub
    with pytest.raises(AssertionError):
        online_softmax_quant_s8(0, 0, 0, 0, 0, 0, _FakeTensor((1, 64), tl.float32), _FakeTensor((5, 8), tl.float32), 0,
                                0, 8, 64, 64, 1.0, 1.0, 1, out=None)

    # online_rescaled_accum requires 2-tuple out and float16/float32 acc_ub
    with pytest.raises(AssertionError):
        online_rescaled_accum(0, 0, 0, _FakeTensor((5, 16), tl.float32), 0, 16, 512, 0.0, 0.0, 1, out=None)
    with pytest.raises(AssertionError):
        online_rescaled_accum(
            0,
            0,
            0,
            _FakeTensor((5, 16), tl.float32),
            0,
            16,
            512,
            0.0,
            0.0,
            1,
            out=(_FakeTensor((16, 512), tl.int8), _FakeTensor((32768, ), tl.float32)),
        )


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
    test_pipelined_cube_pair_and_nd2nz_deqf16_prefetch()
    print("[PASS] gather_blocks_matmul_pair_l1_to_gm + nd2nz_matmul_l1_to_gm_deqf16 (prefetch)")
    for args in (
        (8, 64, 64, False, False, False, 1 | 4, 1.0, False),
        (16, 128, 91, True, True, False, 0 | 4 | 8, 16384.0, False),
        (64, 384, 384, True, True, False, 1 | 2 | 8, 16384.0, False),
        (64, 384, 384, True, True, True, 1 | 8 | 16, 16384.0, False),
        (64, 384, 300, True, True, True, 0 | 2 | 8, 16384.0, True),
        (16, 384, 0, True, True, True, 1 | 4, 1.0, False),
    ):
        test_online_softmax_quant_s8(*args)
    print("[PASS] online_softmax_quant_s8")
    for args in (
        (64, True, True, False, True, 0.0),
        (64, True, False, True, True, 1.0),
        (16, False, True, True, True, 1.0),
        (16, False, True, False, True, 0.0),
    ):
        test_online_rescaled_accum(*args)
    print("[PASS] online_rescaled_accum")
    test_registry_validation_errors()
    print("[PASS] registry validation errors")


if __name__ == "__main__":
    main()
