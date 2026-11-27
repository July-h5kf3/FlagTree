# Copyright 2026- FlagOS Contributors
# Copyright 2026- Xcoresigma Technology Co., Ltd
"""
Ascend custom op per-op correctness tests
===================================================

Minimal standalone correctness checks for every op registered in
custom_ops.bc, one group per op:

  - gather_gm_to_l1      (fp16 / bf16): verified through a following tl.dot
  - gather_gm_to_ub      (fp16 / bf16): verified by storing the result to GM
  - sort_1d_pack         (all three sort paths BASE / S4096_K129_512 /
                          S4096_K1_128_K2048, plus an index_offset case)
  - merge_exhaust_sort4  (4-way / 2-way / 3-way with a hole way)
  - unpack_sort          (split proposals into value / index)

Correctness only, no benchmarking.
"""

import numpy as np
import torch
import torch_npu
import triton
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend.custom_ops import (
    SORT_IMPL_BASE,
    SORT_IMPL_S4096_K1_128_K2048,
    SORT_IMPL_S4096_K129_512,
)

DEVICE = "npu"

SMALLK_SORT_CHUNK = 1024
SMALLK_SORT_WAYS = 4

# ══════════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════════


@triton.jit
def duplicate_bitwise_mask_kernel(src, dst):
    ONE_REPEAT_SORT_NUM: tl.constexpr = 32
    DEFAULT_BLOCK_SIZE: tl.constexpr = 256
    BUFFER_SIZE: tl.constexpr = DEFAULT_BLOCK_SIZE // 4
    NUM_GROUPS: tl.constexpr = 8
    src_offs = tl.arange(0, NUM_GROUPS)
    dst_offs = tl.arange(0, BUFFER_SIZE)
    src_ub = tle.dsa.alloc([BUFFER_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(src + src_offs, tle.dsa.subview(src_ub, offsets=[0], sizes=[NUM_GROUPS], strides=[1]), [NUM_GROUPS])
    src_tensor = tle.dsa.to_tensor(src_ub, writable=True)
    neg_inf = tl.full([1], float("-inf"), dtype=tl.float32)
    mask = tl.zeros([2], dtype=tl.int64)
    mask = tl.where(tl.arange(0, 2) == 0, (-1) << NUM_GROUPS, mask)
    src_tensor = tle_raw.call("ascend_duplicate_bitwise_mask", neg_inf, mask, 1, 1, 8, out=src_tensor)
    #out_tensor = tle.dsa.to_tensor(tle.dsa.subview(src_ub, offsets=[0], sizes=[ONE_REPEAT_SORT_NUM], strides=[1]))
    tl.store(dst + dst_offs, src_tensor, mask=dst_offs < ONE_REPEAT_SORT_NUM)


@triton.jit
def gather_gm_to_l1_dot_kernel(
    src,
    src_index,
    query,
    output,
    NUM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    src_2d = tl.make_block_ptr(
        base=src,
        shape=(NUM_ROWS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    src_index_2d = tl.make_block_ptr(
        base=src_index,
        shape=(TILE_SIZE, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, 1),
        order=(1, 0),
    )
    tile_k = tl.full((TILE_SIZE, D), 0, DTYPE)
    tile_k = tle_raw.call(
        "ascend_gather_gm_to_l1",
        src_2d,
        src_index_2d,
        TILE_SIZE,
        D,
        out=tile_k,
    )

    query_2d = tl.make_block_ptr(
        base=query,
        shape=(TILE_SIZE, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    tile_q = tl.load(query_2d)
    if DTYPE == tl.int8:
        result = tl.dot(tile_q, tl.trans(tile_k), out_dtype=tl.int32).to(tl.float32)
    else:
        result = tl.dot(tile_q, tl.trans(tile_k))

    output_2d = tl.make_block_ptr(
        base=output,
        shape=(TILE_SIZE, TILE_SIZE),
        strides=(TILE_SIZE, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, TILE_SIZE),
        order=(1, 0),
    )
    tl.store(output_2d, result)


@triton.jit
def gather_gm_to_ub_store_kernel(
    src,
    src_index,
    output,
    NUM_ROWS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    D: tl.constexpr,
    DTYPE: tl.constexpr,
):
    src_2d = tl.make_block_ptr(
        base=src,
        shape=(NUM_ROWS, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    src_index_2d = tl.make_block_ptr(
        base=src_index,
        shape=(TILE_SIZE, 1),
        strides=(1, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, 1),
        order=(1, 0),
    )
    tile_v = tl.full((TILE_SIZE, D), 0, DTYPE)
    tile_v = tle_raw.call(
        "ascend_gather_gm_to_ub",
        src_2d,
        src_index_2d,
        TILE_SIZE,
        D,
        out=tile_v,
    )

    output_2d = tl.make_block_ptr(
        base=output,
        shape=(TILE_SIZE, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(output_2d, tile_v)


@triton.jit
def gather_mask_builtin_pattern_kernel(src, dst, rsvd_cnt):
    ONE_REPEAT_SORT_NUM: tl.constexpr = 32
    src_offs = tl.arange(0, ONE_REPEAT_SORT_NUM * 2)
    dst_offs = tl.arange(0, ONE_REPEAT_SORT_NUM)
    src_ub = tle.dsa.alloc([ONE_REPEAT_SORT_NUM * 2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(src + src_offs, src_ub, [ONE_REPEAT_SORT_NUM * 2])
    # src1_pattern for indices
    src1_pattern = 2  # 10101010...101010
    reduce_mode = False
    mask = 0
    src0_block_stride = 1
    repeat_times = 1
    src0_repeat_stride = 0
    src1_repeat_stride = 0
    out0 = tl.zeros([ONE_REPEAT_SORT_NUM], dtype=tl.float32)
    out1 = tl.zeros([1], dtype=tl.int64)
    out0, out1 = tle_raw.call("ascend_gather_mask_builtin_pattern", tle.dsa.to_tensor(src_ub), src1_pattern,
                              reduce_mode, mask, src0_block_stride, repeat_times, src0_repeat_stride,
                              src1_repeat_stride, out=[out0, out1])
    tl.store(dst + dst_offs, out0)
    tl.store(rsvd_cnt + tl.arange(0, 1), out1)


@triton.jit
def gather_mask_custom_pattern_kernel(src, dst, rsvd_cnt):
    NUM_EXPERTS: tl.constexpr = 256
    NUM_GROUPS: tl.constexpr = 8
    src_offs = tl.arange(0, NUM_EXPERTS * 2)
    dst_offs = tl.arange(0, NUM_GROUPS * 2)
    src_ub = tle.dsa.alloc([NUM_EXPERTS * 2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(src + src_offs, src_ub, [NUM_EXPERTS * 2])
    src1 = tl.zeros([2], dtype=tl.uint32)
    # src1_pattern for top2
    src1 = tl.where(tl.arange(0, 2) == 0, 5, src1)  # [0b0101, 0]
    reduce_mode = True
    mask = 64
    src0_block_stride = 1
    repeat_times = 8
    src0_repeat_stride = 8
    src1_repeat_stride = 0
    out0 = tl.zeros([NUM_GROUPS * 2], dtype=tl.float32)
    out1 = tl.zeros([1], dtype=tl.int64)
    out0, out1 = tle_raw.call("ascend_gather_mask_custom_pattern", tle.dsa.to_tensor(src_ub), src1, reduce_mode, mask,
                              src0_block_stride, repeat_times, src0_repeat_stride, src1_repeat_stride, out=[out0, out1])
    tl.store(dst + dst_offs, out0)
    tl.store(rsvd_cnt + tl.arange(0, 1), out1)


@triton.jit
def pair_reduce_sum_continuous_mask_kernel(src, dst):
    NUM_GROUPS: tl.constexpr = 8
    src_offs = tl.arange(0, NUM_GROUPS * 2)
    dst_offs = tl.arange(0, NUM_GROUPS)
    src_ub = tle.dsa.alloc([NUM_GROUPS * 2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(src + src_offs, src_ub, [NUM_GROUPS * 2])
    out = tl.zeros([NUM_GROUPS], dtype=tl.float32)
    out = tle_raw.call("ascend_pair_reduce_sum_continuous_mask", tle.dsa.to_tensor(src_ub), 1, NUM_GROUPS * 2, 1, 1, 1,
                       out=out)
    tl.store(dst + dst_offs, out)


@triton.jit
def sort32_kernel(src0, src1, repeat_times: tl.constexpr, dst, BLOCK_SIZE: tl.constexpr):
    PROPOSALS_BYTES: tl.constexpr = 2
    DST_SIZE: tl.constexpr = BLOCK_SIZE * PROPOSALS_BYTES

    src_offs = tl.arange(0, BLOCK_SIZE)
    dst_offs = tl.arange(0, DST_SIZE)
    src0_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    src1_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.uint32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(src0 + src_offs, src0_ub, [BLOCK_SIZE])
    tle.dsa.copy(src1 + src_offs, src1_ub, [BLOCK_SIZE])
    pair = tl.zeros([DST_SIZE], dtype=tl.float32)
    pair = tle_raw.call("ascend_sort32", tle.dsa.to_tensor(src0_ub), tle.dsa.to_tensor(src1_ub), repeat_times, out=pair)
    tl.store(dst + dst_offs, pair)


@triton.jit
def sort_pack_kernel(X, OutGM, N: tl.constexpr, K: tl.constexpr, INDEX_OFFSET: tl.constexpr, SORT_IMPL: tl.constexpr,
                     TMP_SIZE: tl.constexpr):
    """Single-segment sort_1d_pack: load N f32 values from GM into UB, sort,
    then write the K proposals (2K f32) back to OutGM."""
    NP2: tl.constexpr = triton.next_power_of_2(N)
    KP2: tl.constexpr = triton.next_power_of_2(K * 2)
    tmp = tl.zeros([TMP_SIZE], dtype=tl.float32)
    src_ub = tle.dsa.alloc([N], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(X + tl.arange(0, NP2), src_ub, [N])
    props = tl.zeros([KP2], dtype=tl.float32)
    props = tle_raw.call("ascend_sort_1d_pack", tle.dsa.to_tensor(src_ub), tmp, True, K, INDEX_OFFSET, SORT_IMPL,
                         out=props)
    tl.store(OutGM + tl.arange(0, KP2), props)


@triton.jit
def merge_exhaust_kernel(SrcGM, OutGM, ConsGM, WAY_CAP: tl.constexpr, WAYS: tl.constexpr, L0: tl.constexpr,
                         L1: tl.constexpr, L2: tl.constexpr, L3: tl.constexpr, OUT_CAP: tl.constexpr):
    """One merge_exhaust_sort4 call: the four proposal ways sit in a single
    UB buffer at fixed offsets 0/1/2/3 * WAY_CAP; merge once, then write the
    safe prefix and the per-way consumed counts back to GM."""
    IN_LEN: tl.constexpr = 4 * WAY_CAP * 2
    IN_P2: tl.constexpr = triton.next_power_of_2(IN_LEN)
    OUT_P2: tl.constexpr = triton.next_power_of_2(OUT_CAP * 2)
    in_ub = tle.dsa.alloc([IN_LEN], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    out_t = tl.zeros([OUT_P2], dtype=tl.float32)
    cons = tl.zeros([4], dtype=tl.int32)

    tle.dsa.copy(SrcGM + tl.arange(0, IN_P2), in_ub, [IN_LEN])
    out_t, cons = tle_raw.call("ascend_merge_exhaust_sort4", tle.dsa.to_tensor(in_ub), WAYS, 0 * WAY_CAP, 1 * WAY_CAP,
                               2 * WAY_CAP, 3 * WAY_CAP, L0, L1, L2, L3, out=[out_t, cons])

    tl.store(OutGM + tl.arange(0, OUT_P2), out_t)
    tl.store(ConsGM + tl.arange(0, 4), cons)


@triton.jit
def mrgsort_kernel(src, dst, TOP_K: tl.constexpr):
    GROUP_SIZE: tl.constexpr = 32
    PROPOSALS_BYTES: tl.constexpr = 2
    BLOCK_SIZE: tl.constexpr = GROUP_SIZE * 4 * PROPOSALS_BYTES
    IF_EXHAUSTED_SUSPENSION: tl.constexpr = 1
    VALID_BIT: tl.constexpr = 15
    REPEAT_TIMES: tl.constexpr = 1

    offs = tl.arange(0, BLOCK_SIZE)
    src_ub = tle.dsa.alloc([BLOCK_SIZE], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(src + offs, src_ub, [BLOCK_SIZE])
    pair = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    pair = tle_raw.call("ascend_mrgsort", tle.dsa.to_tensor(src_ub), 0, GROUP_SIZE, GROUP_SIZE * 2, GROUP_SIZE * 3,
                        TOP_K, TOP_K, TOP_K, TOP_K, IF_EXHAUSTED_SUSPENSION, VALID_BIT, REPEAT_TIMES, out=pair)
    tl.store(dst + offs, pair, mask=offs < TOP_K * 2)


@triton.jit
def unpack_sort_kernel(SrcGM, Yv, Yi, K: tl.constexpr):
    """unpack_sort: split K proposals into two GM outputs, value (f32) and
    index (i32)."""
    KP2: tl.constexpr = triton.next_power_of_2(K * 2)
    s_ub = tle.dsa.alloc([K * 2], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    dval = tl.zeros([K], dtype=tl.float32)
    didx = tl.zeros([K], dtype=tl.int32)
    offs = tl.arange(0, K)
    tle.dsa.copy(SrcGM + tl.arange(0, KP2), s_ub, [K * 2])
    dval, didx = tle_raw.call("ascend_unpack_sort", tle.dsa.to_tensor(s_ub), K, out=[dval, didx])
    tl.store(Yv + offs, dval)
    tl.store(Yi + offs, didx)


# ══════════════════════════════════════════════════════════════════════════
# Host-side helpers
# ══════════════════════════════════════════════════════════════════════════


def _encode_props(values, indices):
    """(value f32, index i32) lists -> interleaved f32 numpy array
    [v0, idx0_as_f32, v1, idx1_as_f32, ...]."""
    raw = np.empty(2 * len(values), dtype=np.int32)
    raw[0::2] = np.asarray(values, dtype=np.float32).view(np.int32)
    raw[1::2] = np.asarray(indices, dtype=np.int32)
    return raw.view(np.float32)


def _decode_props(props):
    """Interleaved proposal f32 array -> (value f32, index i32)."""
    raw = np.ascontiguousarray(props, dtype=np.float32).view(np.int32)
    return raw[0::2].view(np.float32), raw[1::2]


def _sort_tmp_size(seg_len: int, sort_run_len: int, sort_impl: int) -> int:
    if sort_impl == SORT_IMPL_BASE:
        return seg_len * 4
    if sort_impl == SORT_IMPL_S4096_K1_128_K2048:
        props_ab = seg_len * 2
        group_buf = 4 * 512 * 2
        return 2 * props_ab + 2 * group_buf + (2 * group_buf + 8)
    candidates = SMALLK_SORT_WAYS * sort_run_len * 2
    chunk_tmp = SMALLK_SORT_CHUNK * 4
    merge_out = SMALLK_SORT_WAYS * sort_run_len * 2
    return candidates + chunk_tmp + merge_out


# ══════════════════════════════════════════════════════════════════════════
# gather_gm_to_l1 / gather_gm_to_ub
# ══════════════════════════════════════════════════════════════════════════

NUM_ROWS = 32
TILE_SIZE = 16
D = 16
L1_INDEX = [3, 4, 9, 10, 2, 7, 0, 1, 12, 13, 5, 8, 11, 6, 15, 14]
UB_INDEX = [20, 21, 19, 30, 31, 25, 24, 18, 27, 28, 22, 29, 26, 23, 17, 16]

_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.int8: tl.int8}


def test_gather_gm_to_l1(torch_dtype, tol):
    torch.manual_seed(0)
    width = 2 * D if torch_dtype == torch.int8 else D
    if torch_dtype == torch.int8:
        src = torch.randint(-127, 128, (NUM_ROWS, width), dtype=torch.int8, device=DEVICE)
        query = torch.randint(-127, 128, (TILE_SIZE, width), dtype=torch.int8, device=DEVICE)
    else:
        src = torch.randn((NUM_ROWS, width), dtype=torch_dtype, device=DEVICE)
        query = torch.randn((TILE_SIZE, width), dtype=torch_dtype, device=DEVICE)
    output = torch.empty((TILE_SIZE, TILE_SIZE), dtype=torch.float32, device=DEVICE)
    src_index = torch.tensor(L1_INDEX, dtype=torch.int32, device=DEVICE)

    gather_gm_to_l1_dot_kernel[(1, )](
        src,
        src_index,
        query,
        output,
        NUM_ROWS=NUM_ROWS,
        TILE_SIZE=TILE_SIZE,
        D=width,
        DTYPE=_TL_DTYPE[torch_dtype],
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
    torch_npu.npu.synchronize()

    gathered_src = src[src_index.long(), :]
    expected = torch.matmul(query.float(), gathered_src.float().transpose(0, 1))
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=tol, atol=tol)
    print(f"[PASS] gather_gm_to_l1 dot correctness ({torch_dtype})")


def test_gather_gm_to_ub(torch_dtype):
    torch.manual_seed(0)
    src = torch.randn((NUM_ROWS, D), dtype=torch_dtype, device=DEVICE)
    output = torch.empty((TILE_SIZE, D), dtype=torch_dtype, device=DEVICE)
    src_index = torch.tensor(UB_INDEX, dtype=torch.int32, device=DEVICE)

    gather_gm_to_ub_store_kernel[(1, )](
        src,
        src_index,
        output,
        NUM_ROWS=NUM_ROWS,
        TILE_SIZE=TILE_SIZE,
        D=D,
        DTYPE=_TL_DTYPE[torch_dtype],
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
    torch_npu.npu.synchronize()

    expected = src[src_index.long(), :]
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    print(f"[PASS] gather_gm_to_ub store correctness ({torch_dtype})")


# ══════════════════════════════════════════════════════════════════════════
# sort_1d_pack
# ══════════════════════════════════════════════════════════════════════════


def _run_sort_case(n, k, index_offset, sort_impl, label):
    torch.manual_seed(0)
    x = torch.rand((n, ), dtype=torch.float32, device=DEVICE)
    out = torch.zeros((triton.next_power_of_2(k * 2), ), dtype=torch.float32, device=DEVICE)

    sort_pack_kernel[(1, )](
        x,
        out,
        N=n,
        K=k,
        INDEX_OFFSET=index_offset,
        SORT_IMPL=sort_impl,
        TMP_SIZE=_sort_tmp_size(n, k, sort_impl),
    )
    torch_npu.npu.synchronize()

    values, indices = _decode_props(out.cpu().numpy()[:k * 2])
    x_np = x.cpu().numpy()
    ref = np.sort(x_np)[::-1][:k]

    assert indices.min() >= index_offset and indices.max() < index_offset + n, (
        f"{label}: index out of range [{indices.min()}, {indices.max()}]")
    np.testing.assert_allclose(values, ref, rtol=1e-5, atol=1e-5,
                               err_msg=f"{label}: values are not the descending top-{k}")
    np.testing.assert_allclose(x_np[indices.astype(np.int64) - index_offset], values, rtol=1e-5, atol=1e-5,
                               err_msg=f"{label}: index does not match value")
    print(f"[PASS] sort_1d_pack {label} (N={n}, K={k}, offset={index_offset})")


def test_sort_1d_pack():
    # BASE generic path + a non-zero index_offset case
    _run_sort_case(2048, 100, 0, SORT_IMPL_BASE, "BASE")
    _run_sort_case(2048, 100, 500, SORT_IMPL_BASE, "BASE index_offset")
    # 4x1024 small-K path: 4096 inputs, 128 < K <= 512
    _run_sort_case(4096, 256, 0, SORT_IMPL_S4096_K129_512, "S4096_K129_512")
    # Layered path: 4096 inputs; K <= 128 can early-stop, K == 2048 uses the
    # fixed merge tree
    _run_sort_case(4096, 64, 0, SORT_IMPL_S4096_K1_128_K2048, "S4096_K1_128_K2048 early-stop")
    _run_sort_case(4096, 2048, 0, SORT_IMPL_S4096_K1_128_K2048, "S4096_K1_128_K2048 fixed-tree")


# ══════════════════════════════════════════════════════════════════════════
# merge_exhaust_sort4
# ══════════════════════════════════════════════════════════════════════════


def _make_way(rng, length, tag):
    """Generate one descending proposal way: globally unique values with a
    one-to-one value/index mapping."""
    values = np.sort(rng.random(length))[::-1].astype(np.float32)
    indices = np.arange(length, dtype=np.int32) + tag * 100000
    return list(zip(values.tolist(), indices.astype(np.int64).tolist()))


def _run_merge_case(lengths, label, seed=0):
    rng = np.random.default_rng(seed)
    way_cap = max(lengths)
    src = np.zeros(4 * way_cap * 2, dtype=np.float32)
    ways = []
    for tag, ln in enumerate(lengths):
        if ln > 0:
            way = _make_way(rng, ln, tag)
            ways.append((tag, way))
            src[tag * way_cap * 2:tag * way_cap * 2 + ln * 2] = _encode_props(*zip(*way))

    src_gm = torch.from_numpy(src).to(DEVICE)
    out_gm = torch.zeros((4 * way_cap * 2, ), dtype=torch.float32, device=DEVICE)
    cons_gm = torch.zeros((4, ), dtype=torch.int32, device=DEVICE)

    merge_exhaust_kernel[(1, )](
        src_gm,
        out_gm,
        cons_gm,
        WAY_CAP=way_cap,
        WAYS=sum(1 for ln in lengths if ln > 0),
        L0=lengths[0],
        L1=lengths[1],
        L2=lengths[2],
        L3=lengths[3],
        OUT_CAP=4 * way_cap,
    )
    torch_npu.npu.synchronize()

    cons = cons_gm.cpu().numpy()
    total = int(cons[:len(lengths)].sum())
    assert total > 0, f"{label}: merge produced no output"
    for tag, ln in enumerate(lengths):
        assert 0 <= cons[tag] <= ln, (f"{label}: way{tag} consumed count {cons[tag]} exceeds length {ln}")

    out_v, out_i = _decode_props(out_gm.cpu().numpy()[:total * 2])
    assert np.all(out_v[:-1] >= out_v[1:]), f"{label}: output prefix is not descending"

    out_pairs = list(zip(out_v.tolist(), out_i.tolist()))
    full_merge = sorted((p for _, way in ways for p in way), key=lambda p: -p[0])
    assert out_pairs == full_merge[:total], (f"{label}: output is not a safe prefix of the full merge")

    value_to_way = {p[0]: tag for tag, way in ways for p in way}
    for tag, way in ways:
        used = [p for p in out_pairs if value_to_way[p[0]] == tag]
        assert used == way[:cons[tag]], (
            f"{label}: way{tag} proposals actually consumed do not match the reported consumed count")
    print(f"[PASS] merge_exhaust_sort4 {label} "
          f"(lens={lengths}, consumed={cons[:len(lengths)].tolist()}, total={total})")


def test_merge_exhaust_sort4():
    _run_merge_case((64, 64, 64, 64), "4-way")
    _run_merge_case((64, 64, 0, 0), "2-way")
    _run_merge_case((64, 0, 64, 32), "3-way with hole")


# ══════════════════════════════════════════════════════════════════════════
# unpack_sort
# ══════════════════════════════════════════════════════════════════════════


def test_unpack_sort():
    rng = np.random.default_rng(0)
    k = 128
    values = np.sort(rng.random(k))[::-1].astype(np.float32)
    indices = rng.integers(0, 10000, size=k).astype(np.int32)
    src = torch.from_numpy(_encode_props(values, indices)).to(DEVICE)
    yv = torch.zeros((k, ), dtype=torch.float32, device=DEVICE)
    yi = torch.zeros((k, ), dtype=torch.int32, device=DEVICE)

    unpack_sort_kernel[(1, )](src, yv, yi, K=k)
    torch_npu.npu.synchronize()

    np.testing.assert_array_equal(yv.cpu().numpy(), values, err_msg="unpack_sort values mismatch")
    np.testing.assert_array_equal(yi.cpu().numpy(), indices, err_msg="unpack_sort indices mismatch")
    print("[PASS] unpack_sort correctness")


def test_sort32():
    torch.manual_seed(0)
    n_experts = 256
    n_groups = 256 // 32
    val = torch.rand([n_experts], dtype=torch.float32, device=DEVICE)
    idx = torch.arange(0, n_experts, dtype=torch.int32, device=DEVICE).view(torch.uint32)
    out = torch.zeros([n_experts * 2], dtype=torch.float32, device=DEVICE)

    sort32_kernel[(1, )](val, idx, n_groups, out, BLOCK_SIZE=n_experts)
    torch_npu.npu.synchronize()

    out_v, out_i = _decode_props(out.cpu().numpy()[:n_experts * 2])
    val_2d = val.cpu().numpy().reshape(n_groups, 32)
    idx_2d = idx.cpu().numpy().reshape(n_groups, 32)
    local_idx_2d = np.argsort(-val_2d, axis=1, kind="stable")
    global_val = np.take_along_axis(val_2d, local_idx_2d, axis=1).reshape(n_experts)
    global_idx = np.take_along_axis(idx_2d, local_idx_2d, axis=1).reshape(n_experts)

    np.testing.assert_array_equal(global_val, out_v, err_msg="sort32 values mismatch")
    np.testing.assert_array_equal(global_idx, out_i, err_msg="sort32 indices mismatch")
    print("[PASS] sort32 correctness")


def test_mrgsort():
    torch.manual_seed(0)
    n_experts = 128
    n_groups = 4
    top_k = 8

    val = torch.rand([n_experts], dtype=torch.float32, device="cpu")
    idx = torch.arange(0, n_experts, dtype=torch.int32, device="cpu")
    val_2d = val.numpy().reshape(n_groups, 32)
    idx_2d = idx.numpy().reshape(n_groups, 32)
    local_idx_2d = np.argsort(-val_2d, axis=1, kind="stable")
    global_val = np.take_along_axis(val_2d, local_idx_2d, axis=1).reshape(n_experts)
    global_idx = np.take_along_axis(idx_2d, local_idx_2d, axis=1).reshape(n_experts)
    pair = torch.from_numpy(_encode_props(global_val, global_idx)).to(DEVICE)
    out = torch.zeros([top_k * 2], dtype=torch.float32, device=DEVICE)

    mrgsort_kernel[(1, )](pair, out, TOP_K=top_k)
    torch_npu.npu.synchronize()

    out_v, out_i = _decode_props(out.cpu().numpy()[:top_k * 2])
    local_idx = np.argsort(-val.numpy(), axis=0, kind="stable")
    ref_val = np.take_along_axis(val.numpy(), local_idx[:top_k], axis=0)
    ref_idx = np.take_along_axis(idx.numpy(), local_idx[:top_k], axis=0)

    np.testing.assert_array_equal(ref_val, out_v, err_msg="mrgsort values mismatch")
    np.testing.assert_array_equal(ref_idx, out_i, err_msg="mrgsort indices mismatch")
    print("[PASS] mrgsort correctness")


def test_gather_mask():
    ONE_REPEAT_SORT_NUM = 32
    torch.manual_seed(0)
    n_experts = 256
    n_groups = 256 // 32
    val = torch.rand([n_experts], dtype=torch.float32, device="cpu")
    idx = torch.arange(0, n_experts, dtype=torch.int32, device="cpu")
    # sorted by group
    val_2d = val.numpy().reshape(n_groups, 32)
    idx_2d = idx.numpy().reshape(n_groups, 32)
    local_idx_2d = np.argsort(-val_2d, axis=1, kind="stable")
    global_val = np.take_along_axis(val_2d, local_idx_2d, axis=1).reshape(n_experts)
    global_idx = np.take_along_axis(idx_2d, local_idx_2d, axis=1).reshape(n_experts)
    pair = torch.from_numpy(_encode_props(global_val, global_idx)).to(DEVICE)
    out = torch.zeros([n_groups * 2], dtype=torch.float32, device=DEVICE)
    rsvd_cnt = torch.zeros([1], dtype=torch.int64, device=DEVICE)

    # top2 value per group
    gather_mask_custom_pattern_kernel[(1, )](pair, out, rsvd_cnt)
    torch_npu.npu.synchronize()

    assert rsvd_cnt.cpu()[0] == n_groups * 2, "gather_mask_custom_pattern rsvd_cnt mismatch"
    ref = global_val.reshape(n_groups, 32)[:, :2].reshape(n_groups * 2)
    np.testing.assert_array_equal(ref, out.cpu().numpy(), err_msg="gather_mask_custom_pattern dst mismatch")

    # top2 sum
    top2_sum = np.sum(ref.reshape(n_groups, 2), axis=1)
    top2_sum_pad = np.pad(top2_sum, pad_width=(0, ONE_REPEAT_SORT_NUM - n_groups), mode='constant',
                          constant_values=-np.inf)
    # sort32
    group_idx_pad = torch.arange(0, ONE_REPEAT_SORT_NUM, dtype=torch.int32, device="cpu")
    local_group_idx = np.argsort(-top2_sum_pad, axis=0, kind="stable")
    global_group_val = np.take_along_axis(top2_sum_pad, local_group_idx, axis=0)
    global_group_idx = np.take_along_axis(group_idx_pad.numpy(), local_group_idx, axis=0)
    pair2 = torch.from_numpy(_encode_props(global_group_val, global_group_idx)).to(DEVICE)

    out2 = torch.zeros([ONE_REPEAT_SORT_NUM], dtype=torch.float32, device=DEVICE)
    rsvd_cnt2 = torch.zeros([1], dtype=torch.int64, device=DEVICE)

    # gather group indices
    gather_mask_builtin_pattern_kernel[(1, )](pair2, out2, rsvd_cnt2)
    torch_npu.npu.synchronize()

    assert rsvd_cnt2.cpu()[0] == ONE_REPEAT_SORT_NUM, "gather_mask_builtin_pattern rsvd_cnt mismatch"
    np.testing.assert_array_equal(global_group_idx,
                                  out2.cpu().numpy().view(np.int32), err_msg="gather_mask_builtin_pattern dst mismatch")
    print("[PASS] gather_mask correctness")


def test_pair_reduce_sum():
    torch.manual_seed(0)
    n_groups = 8
    top2 = torch.rand([n_groups * 2], dtype=torch.float32, device=DEVICE)
    out = torch.zeros([n_groups], dtype=torch.float32, device=DEVICE)

    # top2 sum
    pair_reduce_sum_continuous_mask_kernel[(1, )](top2, out)
    torch_npu.npu.synchronize()

    ref = np.sum(top2.cpu().numpy().reshape(n_groups, 2), axis=1)
    np.testing.assert_array_equal(ref, out.cpu().numpy(), err_msg="pair_reduce_sum_continuous_mask dst mismatch")
    print("[PASS] pair_reduce_sum correctness")


def test_duplicate():
    ONE_REPEAT_SORT_NUM = 32
    torch.manual_seed(0)
    n_groups = 8
    top2_sum = torch.rand([n_groups], dtype=torch.float32, device=DEVICE)
    out = torch.zeros([ONE_REPEAT_SORT_NUM], dtype=torch.float32, device=DEVICE)

    duplicate_bitwise_mask_kernel[(1, )](top2_sum, out)
    torch_npu.npu.synchronize()

    top2_sum_pad = np.pad(top2_sum.cpu().numpy(), pad_width=(0, ONE_REPEAT_SORT_NUM - n_groups), mode='constant',
                          constant_values=-np.inf)
    np.testing.assert_array_equal(top2_sum_pad, out.cpu().numpy(), err_msg="duplicate_bitwise_mask dst mismatch")
    print("[PASS] duplicate correctness")


@triton.jit
def paged_int8_qk_pv_kernel(
    Q,
    QR,
    K,
    KR,
    Table,
    P,
    C,
    R,
    PV,
    M: tl.constexpr,
):
    with al.scope(core_mode="cube"):
        tle.dsa.ascend.raw(
            "paged_int8_qk",
            M,
            Q.to(tl.uint64),
            QR.to(tl.uint64),
            K.to(tl.uint64),
            KR.to(tl.uint64),
            Table.to(tl.uint64),
            C.to(tl.uint64),
            R.to(tl.uint64),
            6,
            1,
            0,
            M // 4,
        )
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
        tle.dsa.ascend.raw(
            "paged_int8_pv",
            M,
            P.to(tl.uint64),
            PV.to(tl.uint64),
            K.to(tl.uint64),
            KR.to(tl.uint64),
            Table.to(tl.uint64),
            0,
            0,
            0,
            0,
            1,
        )
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
        if al.sub_vec_id() == 0:
            cols = tl.arange(0, 64)
            r_head = tl.load(R + cols)
            tl.store(R + cols, r_head)
        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)


@triton.jit
def triton_paged_int8_qk_pv_bench_kernel(Q, QR, K, KR, Table, P, C, R, PV, M: tl.constexpr, GROUPS: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, M)
    cols512 = tl.arange(0, 512)
    cols64 = tl.arange(0, 64)
    q = tl.load(Q + rows[:, None] * 512 + cols512[None, :])
    qr = tl.load(QR + rows[:, None] * 64 + cols64[None, :])
    c_base = C + pid * M * 384
    r_base = R + pid * M * 384
    pv_base = PV + pid * 2 * M * 512
    for group in range(0, GROUPS):
        for p_idx in range(6):
            page = tl.load(Table + (group % 2) * 6 + p_idx)
            tok = page * 64 + cols64
            k_page = tl.load(K + tok[:, None] * 512 + cols512[None, :])
            kr_page = tl.load(KR + tok[:, None] * 64 + cols64[None, :])
            c_tile = tl.dot(q, tl.trans(k_page), out_dtype=tl.int32)
            r_tile = tl.dot(qr, tl.trans(kr_page), out_dtype=tl.float32)
            tl.store(c_base + rows[:, None] * 384 + (p_idx * 64 + cols64)[None, :], c_tile)
            tl.store(r_base + rows[:, None] * 384 + (p_idx * 64 + cols64)[None, :], r_tile)
        for c_blk in range(4):
            cols128 = c_blk * 128 + tl.arange(0, 128)
            acc_hi = tl.zeros((M, 128), dtype=tl.int32)
            acc_lo = tl.zeros((M, 128), dtype=tl.int32)
            for p_idx in range(6):
                page = tl.load(Table + (group % 2) * 6 + p_idx)
                tok = page * 64 + cols64
                v_slice = tl.load(K + tok[:, None] * 512 + cols128[None, :])
                p_hi = tl.load(P + rows[:, None] * 384 + (p_idx * 64 + cols64)[None, :])
                p_lo = tl.load(P + (M + rows)[:, None] * 384 + (p_idx * 64 + cols64)[None, :])
                acc_hi += tl.dot(p_hi, v_slice, out_dtype=tl.int32)
                acc_lo += tl.dot(p_lo, v_slice, out_dtype=tl.int32)
            tl.store(pv_base + rows[:, None] * 512 + cols128[None, :], acc_hi)
            tl.store(pv_base + (M + rows)[:, None] * 512 + cols128[None, :], acc_lo)


@triton.jit
def custom_paged_int8_qk_pv_bench_kernel(Q, QR, K, KR, Table, P, C, R, PV, M: tl.constexpr, GROUPS: tl.constexpr):
    pid = tl.program_id(0)
    pages = GROUPS * 6
    with al.scope(core_mode="cube"):
        tle.dsa.ascend.raw(
            "paged_int8_qk",
            M,
            Q.to(tl.uint64),
            QR.to(tl.uint64),
            K.to(tl.uint64),
            KR.to(tl.uint64),
            Table.to(tl.uint64),
            C.to(tl.uint64) + pid * M * 384 * 4,
            R.to(tl.uint64) + pid * M * 384 * 4,
            pages,
            1,
            0,
            M // 4,
        )
        for group in tl.range(0, GROUPS, disable_licm=True):
            slot = (group % 2).to(tl.uint32)
            if group + 1 < GROUPS:
                tle.dsa.ascend.raw(
                    "paged_int8_qk",
                    M,
                    Q.to(tl.uint64),
                    QR.to(tl.uint64),
                    K.to(tl.uint64),
                    KR.to(tl.uint64),
                    (Table + ((group + 1) % 2) * 6).to(tl.uint64),
                    C.to(tl.uint64) + pid * M * 384 * 4,
                    R.to(tl.uint64) + pid * M * 384 * 4,
                    (pages - (group + 1) * 6).to(tl.uint32),
                    0,
                    1 - slot,
                    M // 4,
                )
            rem_pv = tl.maximum(0, pages - (group + 2) * 6).to(tl.uint32)
            is_last = tl.cast(group + 1 == GROUPS, tl.uint32)
            tle.dsa.ascend.raw(
                "paged_int8_pv",
                M,
                P.to(tl.uint64),
                PV.to(tl.uint64) + pid * 2 * M * 512 * 4,
                K.to(tl.uint64),
                KR.to(tl.uint64),
                (Table + ((group + 2) % 2) * 6).to(tl.uint64),
                0,
                0,
                slot,
                rem_pv,
                is_last,
            )
    with al.scope(core_mode="vector"):
        if al.sub_vec_id() == 0 and pid == 0:
            cols = tl.arange(0, 64)
            tl.store(R + cols, tl.load(R + cols))


def test_paged_int8_qk_pv():
    torch.manual_seed(0)
    k = torch.randint(-64, 64, (8, 64, 512), dtype=torch.int8, device=DEVICE)
    kr = torch.randn((8, 64, 64), dtype=torch.bfloat16, device=DEVICE)
    table = torch.tensor([5, 1, 7, 0, 3, 2], dtype=torch.int32, device=DEVICE)
    k_win = k[table.long()].reshape(384, 512)
    kr_win = kr[table.long()].reshape(384, 64)
    for m in (16, 32, 64, 128):
        q = torch.randint(-64, 64, (m, 512), dtype=torch.int8, device=DEVICE)
        qr = torch.randn((m, 64), dtype=torch.bfloat16, device=DEVICE)
        p_hi = torch.randint(-64, 64, (m, 384), dtype=torch.int8, device=DEVICE)
        p_lo = torch.randint(-64, 64, (m, 384), dtype=torch.int8, device=DEVICE)
        p = torch.cat((p_hi, p_lo), dim=0).contiguous()
        c = torch.empty((m, 384), dtype=torch.int32, device=DEVICE)
        r = torch.empty((m, 384), dtype=torch.float32, device=DEVICE)
        pv_dtype = torch.float16 if m >= 64 else torch.int32
        pv = torch.empty((2 * m, 512), dtype=pv_dtype, device=DEVICE)
        paged_int8_qk_pv_kernel[(1, )](
            q,
            qr,
            k,
            kr,
            table,
            p,
            c,
            r,
            pv,
            M=m,
            enable_auto_bind_sub_block=False,
            disable_auto_inject_block_sync=True,
            multibuffer=False,
        )
        torch_npu.npu.synchronize()
        ref_c = (q.cpu().int() @ k_win.cpu().int().T).to(torch.int32)
        ref_r = qr.cpu().float() @ kr_win.cpu().float().T
        ref_pv_hi = (p_hi.cpu().int() @ k_win.cpu().int()).to(torch.int32)
        ref_pv_lo = (p_lo.cpu().int() @ k_win.cpu().int()).to(torch.int32)
        ref_pv = torch.cat((ref_pv_hi, ref_pv_lo), dim=0)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=0, atol=0)
        torch.testing.assert_close(r.cpu(), ref_r, rtol=1e-2, atol=1e-1)
        if m >= 64:
            torch.testing.assert_close(pv.cpu().float(), ref_pv.float() / 16384.0, rtol=1e-2, atol=2e-2)
        else:
            torch.testing.assert_close(pv.cpu(), ref_pv, rtol=0, atol=0)
        print(f"[PASS] paged_int8_qk + paged_int8_pv match (M={m})")

    # Minimal benchmark comparing L1-resident paged_int8_qk/pv vs ordinary Triton tl.dot.
    m, groups, tiles = 64, 8, 160
    q = torch.randint(-64, 64, (m, 512), dtype=torch.int8, device=DEVICE)
    qr = torch.randn((m, 64), dtype=torch.bfloat16, device=DEVICE)
    table12 = torch.tensor([5, 1, 7, 0, 3, 2, 4, 6, 2, 1, 0, 7], dtype=torch.int32, device=DEVICE)
    p = torch.randint(-64, 64, (2 * m, 384), dtype=torch.int8, device=DEVICE)
    c = torch.empty((20, m, 384), dtype=torch.int32, device=DEVICE)
    r = torch.empty((20, m, 384), dtype=torch.float32, device=DEVICE)
    pv = torch.empty((20, 2 * m, 512), dtype=torch.int32, device=DEVICE)
    custom_ms = triton.testing.do_bench(
        lambda: custom_paged_int8_qk_pv_bench_kernel[(20, )]
        (q, qr, k, kr, table12, p, c, r, pv, M=m, GROUPS=groups, enable_auto_bind_sub_block=False,
         disable_auto_inject_block_sync=True, multibuffer=False),
        return_mode="median",
    )
    triton_ms = triton.testing.do_bench(
        lambda: triton_paged_int8_qk_pv_bench_kernel[(20, )](q, qr, k, kr, table12, p, c, r, pv, M=m, GROUPS=groups),
        return_mode="median",
    )
    print(f"[BENCH] paged_int8_qk+pv (M={m}, tiles={tiles}): "
          f"custom {custom_ms * 1000:.1f} us, triton {triton_ms * 1000:.1f} us, "
          f"triton/custom {triton_ms / custom_ms:.2f}x")


@triton.jit
def window_stage_sync(slot, stage: tl.constexpr, wait: tl.constexpr):
    base: tl.constexpr = stage * 2
    if wait:
        if stage == 1:
            if slot == 0:
                al.sync_block_wait("vector", "cube", base, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            else:
                al.sync_block_wait("vector", "cube", base + 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
        else:
            if slot == 0:
                al.sync_block_wait("cube", "vector", base, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
            else:
                al.sync_block_wait("cube", "vector", base + 1, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
    else:
        if stage == 1:
            if slot == 0:
                al.sync_block_set("vector", "cube", base, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
            else:
                al.sync_block_set("vector", "cube", base + 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
        else:
            if slot == 0:
                al.sync_block_set("cube", "vector", base, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
            else:
                al.sync_block_set("cube", "vector", base + 1, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)


@triton.jit
def paged_int8_full_window_kernel(
    Q,
    QR,
    K,
    KR,
    QS,
    KS,
    Table,
    C,
    R,
    P,
    PV,
    Out,
    Lse,
    N,
    SCALE: tl.constexpr,
    M: tl.constexpr,
):
    pages = tl.cdiv(N, 64)
    groups = tl.cdiv(N, 384)
    stride: tl.constexpr = 8 if M == 16 else M // 4
    table = Table.to(tl.uint64)
    with al.scope(core_mode="cube"):
        tle.dsa.ascend.raw(
            "paged_int8_qk",
            M,
            Q.to(tl.uint64),
            QR.to(tl.uint64),
            K.to(tl.uint64),
            KR.to(tl.uint64),
            table,
            C.to(tl.uint64),
            R.to(tl.uint64),
            pages,
            1,
            0,
            stride,
        )
        window_stage_sync(0, 0, False)
        for group in tl.range(0, groups, disable_licm=True):
            slot = group % 2
            if group + 1 < groups:
                byte = ((1 - slot) * M * 384 * 4).to(tl.uint64)
                tle.dsa.ascend.raw(
                    "paged_int8_qk",
                    M,
                    Q.to(tl.uint64),
                    QR.to(tl.uint64),
                    K.to(tl.uint64),
                    KR.to(tl.uint64),
                    table + (group + 1) * 24,
                    C.to(tl.uint64) + byte,
                    R.to(tl.uint64) + byte,
                    pages - (group + 1) * 6,
                    0,
                    1 - slot,
                    stride,
                )
                window_stage_sync(1 - slot, 0, False)
            window_stage_sync(slot, 1, True)
            tle.dsa.ascend.raw(
                "paged_int8_pv",
                M,
                P.to(tl.uint64) + slot.to(tl.uint64) * (2 * M * 384),
                PV.to(tl.uint64) + slot.to(tl.uint64) * (2 * M * 2048),
                K.to(tl.uint64),
                KR.to(tl.uint64),
                table + (group + 2) * 24,
                0,
                0,
                slot,
                tl.maximum(0, pages - (group + 2) * 6),
                tl.cast(group + 1 == groups, tl.uint32),
            )
            window_stage_sync(slot, 2, False)
    with al.scope(core_mode="vector"):
        sub = al.sub_vec_id().to(tl.uint64) * (M // 2)
        dst = Out.to(tl.uint64) + sub * 1024
        lse = Lse.to(tl.uint64) + sub * 4
        pv_row_bytes: tl.constexpr = 1024 if M >= 64 else 2048
        for group in tl.range(0, groups, disable_licm=True):
            slot = group % 2
            buf = slot.to(tl.uint64) * M + sub
            p_sub = slot.to(tl.uint64) * (2 * M * 384) + sub * 384
            window_stage_sync(slot, 0, True)
            prior_pv = PV.to(tl.uint64) + (1 - slot).to(tl.uint64) * (2 * M * 2048) + sub * pv_row_bytes
            for phase in tl.static_range(2):
                if phase == 1 and group > 0:
                    window_stage_sync(1 - slot, 2, True)
                tle.dsa.ascend.raw(
                    "paged_int8_prob",
                    M,
                    phase,
                    C.to(tl.uint64) + buf * 1536,
                    R.to(tl.uint64) + buf * 1536,
                    P.to(tl.uint64) + p_sub,
                    QS.to(tl.uint64) + sub * 4,
                    KS.to(tl.uint64),
                    table + group * 24,
                    prior_pv,
                    0,
                    N - group * 384,
                    tl.cast(group == 0, tl.uint32),
                    slot,
                    SCALE,
                    stride,
                    0,
                )
            window_stage_sync(slot, 1, False)
            if group > 0:
                tle.dsa.ascend.raw(
                    "paged_int8_update",
                    M,
                    prior_pv,
                    0,
                    dst,
                    lse,
                    1 - slot,
                    tl.cast(group == 1, tl.uint32),
                    0,
                    0,
                    stride,
                    0,
                )
        last = (groups - 1) % 2
        window_stage_sync(last, 2, True)
        tle.dsa.ascend.raw(
            "paged_int8_update",
            M,
            PV.to(tl.uint64) + last.to(tl.uint64) * (2 * M * 2048) + sub * pv_row_bytes,
            0,
            dst,
            lse,
            last,
            tl.cast(groups == 1, tl.uint32),
            1,
            0,
            stride,
            0,
        )


@triton.jit
def custom_paged_int8_vec_bench_kernel(
    C,
    R,
    P,
    QS,
    KS,
    Table,
    PV,
    Out,
    Lse,
    M: tl.constexpr,
    GROUPS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.uint64)
    with al.scope(core_mode="vector"):
        sub = al.sub_vec_id().to(tl.uint64) * (M // 2)
        stride: tl.constexpr = 8 if M == 16 else M // 4
        c_base = C.to(tl.uint64) + (pid * M + sub) * 1536
        r_base = R.to(tl.uint64) + (pid * M + sub) * 1536
        p_base = P.to(tl.uint64) + (pid * 2 * M + sub) * 384
        pv_base = PV.to(tl.uint64) + (pid * 2 * M + sub) * 2048
        dst = Out.to(tl.uint64) + (pid * M + sub) * 1024
        lse = Lse.to(tl.uint64) + (pid * M + sub) * 4
        for group in tl.range(0, GROUPS, disable_licm=True):
            slot = (group % 2).to(tl.uint32)
            for phase in tl.static_range(2):
                tle.dsa.ascend.raw(
                    "paged_int8_prob",
                    M,
                    phase,
                    c_base,
                    r_base,
                    p_base,
                    QS.to(tl.uint64) + sub * 4,
                    KS.to(tl.uint64),
                    Table.to(tl.uint64) + (group % 2) * 24,
                    pv_base,
                    0,
                    384,
                    tl.cast(group == 0, tl.uint32),
                    slot,
                    0.0416666679,
                    stride,
                    0,
                )
            if group > 0:
                tle.dsa.ascend.raw(
                    "paged_int8_update",
                    M,
                    pv_base,
                    0,
                    dst,
                    lse,
                    1 - slot,
                    tl.cast(group == 1, tl.uint32),
                    0,
                    0,
                    stride,
                    0,
                )
        tle.dsa.ascend.raw(
            "paged_int8_update",
            M,
            pv_base,
            0,
            dst,
            lse,
            (GROUPS - 1) % 2,
            0,
            1,
            0,
            stride,
            0,
        )


@triton.jit
def triton_store_prob_chunk(rk, inv_step, P, pid, rows, cols128, M: tl.constexpr, CHUNK: tl.constexpr):
    hi = tl.maximum(-127.0, tl.minimum(127.0, tl.floor(rk * inv_step[:, None] + 0.5))).to(tl.int8)
    lo = tl.maximum(-127.0,
                    tl.minimum(127.0, tl.floor((rk * inv_step[:, None] - hi.to(tl.float32)) * 127.0 + 0.5))).to(tl.int8)
    tl.store(P + (pid * 2 * M + rows)[:, None] * 384 + (CHUNK * 128 + cols128)[None, :], hi)
    tl.store(P + (pid * 2 * M + M + rows)[:, None] * 384 + (CHUNK * 128 + cols128)[None, :], lo)


@triton.jit
def triton_paged_int8_vec_bench_kernel(
    C,
    R,
    P,
    QS,
    KS,
    Table,
    PV,
    Out,
    Lse,
    M: tl.constexpr,
    GROUPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, M)
    cols128 = tl.arange(0, 128)
    cols256 = tl.arange(0, 256)
    qs = tl.load(QS + rows) * 0.0416666679
    anchor = tl.full((M, ), -float("inf"), tl.float32)
    norm = tl.zeros((M, ), tl.float32)
    acc0 = tl.zeros((M, 256), tl.float32)
    acc1 = tl.zeros((M, 256), tl.float32)
    for group in range(0, GROUPS):
        row_base = (pid * M + rows)[:, None] * 384
        p0 = tl.load(Table + (group % 2) * 6 + 0)
        p1 = tl.load(Table + (group % 2) * 6 + 1)
        ks0 = tl.load(KS + tl.where(cols128 < 64, p0 * 64 + cols128, p1 * 64 + cols128 - 64))
        s0 = (tl.load(C + row_base + cols128[None, :]).to(tl.float32) +
              tl.load(R + row_base + cols128[None, :])) * qs[:, None] * ks0[None, :]
        p2 = tl.load(Table + (group % 2) * 6 + 2)
        p3 = tl.load(Table + (group % 2) * 6 + 3)
        ks1 = tl.load(KS + tl.where(cols128 < 64, p2 * 64 + cols128, p3 * 64 + cols128 - 64))
        s1 = (tl.load(C + row_base + (128 + cols128)[None, :]).to(tl.float32) +
              tl.load(R + row_base + (128 + cols128)[None, :])) * qs[:, None] * ks1[None, :]
        p4 = tl.load(Table + (group % 2) * 6 + 4)
        p5 = tl.load(Table + (group % 2) * 6 + 5)
        ks2 = tl.load(KS + tl.where(cols128 < 64, p4 * 64 + cols128, p5 * 64 + cols128 - 64))
        s2 = (tl.load(C + row_base + (256 + cols128)[None, :]).to(tl.float32) +
              tl.load(R + row_base + (256 + cols128)[None, :])) * qs[:, None] * ks2[None, :]

        next_anchor = tl.maximum(anchor, tl.maximum(tl.maximum(tl.max(s0, 1), tl.max(s1, 1)), tl.max(s2, 1)))
        corr = tl.exp(anchor - next_anchor)
        w0 = tl.exp(s0 - next_anchor[:, None])
        w1 = tl.exp(s1 - next_anchor[:, None])
        w2 = tl.exp(s2 - next_anchor[:, None])
        norm = norm * corr + tl.sum(w0, 1) + tl.sum(w1, 1) + tl.sum(w2, 1)
        anchor = next_anchor
        rk0 = w0 * ks0[None, :]
        rk1 = w1 * ks1[None, :]
        rk2 = w2 * ks2[None, :]
        peak = tl.maximum(tl.maximum(tl.max(tl.abs(rk0), 1), tl.max(tl.abs(rk1), 1)), tl.max(tl.abs(rk2), 1))
        step = peak * (1.0 / 127.0)
        inv_step = 127.0 / tl.maximum(peak, 1e-30)
        triton_store_prob_chunk(rk0, inv_step, P, pid, rows, cols128, M, 0)
        triton_store_prob_chunk(rk1, inv_step, P, pid, rows, cols128, M, 1)
        triton_store_prob_chunk(rk2, inv_step, P, pid, rows, cols128, M, 2)
        pv_hi0 = tl.load(PV + (pid * 2 * M + rows)[:, None] * 512 + cols256[None, :]).to(tl.float32)
        pv_lo0 = tl.load(PV + (pid * 2 * M + M + rows)[:, None] * 512 + cols256[None, :]).to(tl.float32)
        acc0 = acc0 * corr[:, None] + (pv_hi0 + pv_lo0 * (1.0 / 127.0)) * step[:, None]
        pv_hi1 = tl.load(PV + (pid * 2 * M + rows)[:, None] * 512 + (256 + cols256)[None, :]).to(tl.float32)
        pv_lo1 = tl.load(PV + (pid * 2 * M + M + rows)[:, None] * 512 + (256 + cols256)[None, :]).to(tl.float32)
        acc1 = acc1 * corr[:, None] + (pv_hi1 + pv_lo1 * (1.0 / 127.0)) * step[:, None]
    tl.store(Out + (pid * M + rows)[:, None] * 512 + cols256[None, :], (acc0 / norm[:, None]).to(tl.bfloat16))
    tl.store(Out + (pid * M + rows)[:, None] * 512 + (256 + cols256)[None, :], (acc1 / norm[:, None]).to(tl.bfloat16))
    tl.store(Lse + pid * M + rows, anchor + tl.log(norm))


def test_paged_int8_prob_update():
    torch.manual_seed(7)
    scale = 576**-0.5
    k = torch.randint(-64, 64, (16, 64, 512), dtype=torch.int8, device=DEVICE)
    kr = (torch.randn((16, 64, 64), dtype=torch.bfloat16, device=DEVICE) * 0.2).bfloat16()
    ks = (torch.rand((16, 64), dtype=torch.float32, device=DEVICE) * 0.03 + 0.01)
    table = torch.tensor([5, 1, 7, 0, 3, 2, 11, 4, 9, 6, 13, 8], dtype=torch.int32, device=DEVICE)
    n = 700
    pages = table[:(n + 63) // 64].long()
    ks_flat = ks[pages].reshape(-1, 1)[:n].cpu().float()
    k_deq = k[pages].reshape(-1, 512)[:n].cpu().float() * ks_flat
    kr_deq = kr[pages].reshape(-1, 64)[:n].cpu().float() * ks_flat

    for m in (16, 32, 64, 128):
        q = torch.randint(-64, 64, (m, 512), dtype=torch.int8, device=DEVICE)
        qr = (torch.randn((m, 64), dtype=torch.bfloat16, device=DEVICE) * 0.2).bfloat16()
        qs = torch.rand((m, ), dtype=torch.float32, device=DEVICE) * 0.03 + 0.01
        c = torch.empty((2, m, 384), dtype=torch.int32, device=DEVICE)
        r = torch.empty((2, m, 384), dtype=torch.float32, device=DEVICE)
        p = torch.empty((2, 2 * m, 384), dtype=torch.int8, device=DEVICE)
        pv = torch.empty((2, 2 * m, 512), dtype=torch.int32, device=DEVICE)
        out = torch.empty((m, 512), dtype=torch.bfloat16, device=DEVICE)
        lse = torch.empty((m, ), dtype=torch.float32, device=DEVICE)
        paged_int8_full_window_kernel[(1, )](
            q,
            qr,
            k,
            kr,
            qs,
            ks,
            table,
            c,
            r,
            p,
            pv,
            out,
            lse,
            n,
            SCALE=scale,
            M=m,
            enable_auto_bind_sub_block=False,
            disable_auto_inject_block_sync=True,
            multibuffer=False,
        )
        torch_npu.npu.synchronize()
        q_cpu = q.cpu().float() * qs.cpu().unsqueeze(1)
        qr_cpu = qr.cpu().float() * qs.cpu().unsqueeze(1)
        scores = (q_cpu @ k_deq.T + qr_cpu @ kr_deq.T) * scale
        ref_out = (scores.softmax(-1) @ k_deq).bfloat16()
        ref_lse = scores.logsumexp(-1)
        torch.testing.assert_close(out.cpu(), ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-3, atol=2e-3)
        print(f"[PASS] paged_int8_prob + paged_int8_update window pipeline match (M={m})")

    m_b, groups_b = 16, 8
    c_b = torch.randint(-2000, 2000, (20, m_b, 384), dtype=torch.int32, device=DEVICE)
    r_b = torch.randn((20, m_b, 384), dtype=torch.float32, device=DEVICE)
    p_b = torch.empty((20, 2 * m_b, 384), dtype=torch.int8, device=DEVICE)
    qs_b = torch.rand((m_b, ), dtype=torch.float32, device=DEVICE) * 0.02 + 0.01
    pv_b = torch.randint(-5000, 5000, (20, 2 * m_b, 512), dtype=torch.int32, device=DEVICE)
    out_b = torch.empty((20, m_b, 512), dtype=torch.bfloat16, device=DEVICE)
    lse_b = torch.empty((20, m_b), dtype=torch.float32, device=DEVICE)
    custom_vec_ms = triton.testing.do_bench(
        lambda: custom_paged_int8_vec_bench_kernel[(20, )]
        (c_b, r_b, p_b, qs_b, ks, table, pv_b, out_b, lse_b, M=m_b, GROUPS=groups_b, enable_auto_bind_sub_block=False,
         disable_auto_inject_block_sync=True, multibuffer=False),
        return_mode="median",
    )
    triton_vec_ms = triton.testing.do_bench(
        lambda: triton_paged_int8_vec_bench_kernel[(20, )]
        (c_b, r_b, p_b, qs_b, ks, table, pv_b, out_b, lse_b, M=m_b, GROUPS=groups_b),
        return_mode="median",
    )
    print(f"[BENCH] paged_int8_prob+update (M={m_b}, tiles={20 * groups_b}): "
          f"custom {custom_vec_ms * 1000:.1f} us, triton {triton_vec_ms * 1000:.1f} us, "
          f"triton/custom {triton_vec_ms / custom_vec_ms:.2f}x")


def test_paged_int8_registration():
    from triton.experimental.tle.language.dsa.ascend.custom_ops import registry as reg

    # Valid configurations succeed.
    reg.paged_int8_qk(16, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 8)
    reg.paged_int8_qk(128, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 32)
    reg.paged_int8_pv(64, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1)
    reg.paged_int8_prob(128, 1, 0, 0, 0, 0, 0, 0, 0, 0, 384, 0, 1, 0.04, 32, 0)
    reg.paged_int8_update(128, 0, 0, 0, 0, 1, 0, 1, 0, 32, 0)

    # Invalid m, flags, phase, or head_stride are rejected at compile time.
    bad_calls = [
        lambda: reg.paged_int8_qk(48, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 16),
        lambda: reg.paged_int8_qk(64, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0, 16),
        lambda: reg.paged_int8_qk(64, 0, 0, 0, 0, 0, 0, 0, 6, 1, 0, 8),
        lambda: reg.paged_int8_pv(96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        lambda: reg.paged_int8_pv(64, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1),
        lambda: reg.paged_int8_prob(64, 2, 0, 0, 0, 0, 0, 0, 0, 0, 384, 1, 0, 0.04, 16, 0),
        lambda: reg.paged_int8_update(64, 0, 0, 0, 0, 0, 0, 1, 2, 16, 0),
    ]
    for call in bad_calls:
        try:
            call()
        except AssertionError:
            continue
        raise AssertionError("Expected paged_int8 registration check to fail")
    print("[PASS] paged_int8 registration validation (7 invalid configs rejected)")


def main():
    from test_cast_ops import main as test_cast_ops
    from test_compare_scalar import main as test_compare_scalar

    for torch_dtype, tol in ((torch.float16, 1e-3), (torch.bfloat16, 1e-2), (torch.int8, 0)):
        test_gather_gm_to_l1(torch_dtype, tol)
        if torch_dtype != torch.int8:
            test_gather_gm_to_ub(torch_dtype)
    test_paged_int8_registration()
    test_paged_int8_qk_pv()
    test_paged_int8_prob_update()
    test_sort_1d_pack()
    test_merge_exhaust_sort4()
    test_unpack_sort()
    test_compare_scalar()
    test_cast_ops()
    test_sort32()
    test_mrgsort()
    test_gather_mask()
    test_pair_reduce_sum()
    test_duplicate()
    print("\nAll custom op correctness tests passed.")


if __name__ == "__main__":
    main()
