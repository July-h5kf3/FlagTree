# Copyright 2026- Xcoresigma Technology Co., Ltd
# Copyright 2026 FlagOS Contributors
"""Custom-op registrations sharing custom_ops.bc.

The cube_* entries run in al.scope(core_mode="cube"). Their GM addresses are
uint64 byte addresses; local addresses are uint32 byte offsets.
Sizes and strides follow the corresponding CANN overload. The caller owns local
buffers, bounds, padding, ND writeback setup, and pipeline synchronization.
These primitives do not allocate buffers or insert barriers.
"""

from pathlib import Path

import triton.language as tl
import triton.language.extra.cann.extension as al

CUSTOM_OPS_BITCODE = str(Path(__file__).with_name("custom_ops.bc").resolve())

_GATHER_SUFFIX_BY_DTYPE = {tl.float16: "half", tl.bfloat16: "bf16"}


def _element_dtype(value):
    # Block-pointer tensors (tl.make_block_ptr) carry a nested dtype:
    # pointer<block<[shape], dtype>>. Unwrap element_ty down to the scalar.
    dtype = value.dtype
    while hasattr(dtype, "element_ty"):
        dtype = dtype.element_ty
    return dtype


def _gather_dtype_suffix(value, op_name):
    dtype = _element_dtype(value)
    suffix = _GATHER_SUFFIX_BY_DTYPE.get(dtype)
    assert suffix is not None, (f"{op_name} only supports fp16/bf16, got {dtype}")
    return suffix


@al.register_custom_op
class gather_gm_to_l1:
    """
    /*
     * Function:
     *   Gather fp16/bf16 rows from a row-contiguous 2D GM tensor by discrete
     *   indices, write them into an L1/CBUF tensor and perform the ND2NZ
     *   layout conversion for CUBE use. Output row i comes from source row
     *   index[i]; adjacent indices (index[i + 1] == index[i] + 1) are merged
     *   into a single two-row copy.
     *
     * Inputs:
     *   src: row-contiguous 2D fp16/bf16 source tensor in GM.
     *   index: 2D int32 index tensor in GM (shape (N, 1), stride (1, 1));
     *     output row i takes its data from source row index[i] (0-based row
     *     number); the starting offset of the indices is expressed through
     *     the block ptr offsets, so no extra parameter is needed.
     *   tile_size: number of data rows to gather.
     *   D: number of fp16/bf16 elements in each source row.
     *
     * Outputs:
     *   out: required 4D L1/CBUF fp16/bf16 destination tensor (same dtype as
     *     src), corresponding to dst in the C++ ABI; the gathered result is
     *     written into this tensor directly and it is returned on the Python
     *     side.
     */
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(self, src, index, tile_size, D, out=None):
        assert out is not None, "out buffer is required"
        assert _element_dtype(out) == _element_dtype(src), (
            f"gather_gm_to_l1 requires out dtype ({_element_dtype(out)}) "
            f"to match src dtype ({_element_dtype(src)})")
        assert _element_dtype(index) == tl.int32, (f"gather_gm_to_l1 requires int32 index, "
                                                   f"got {_element_dtype(index)}")
        self.symbol = ("custom_gather_gm_to_l1_" + _gather_dtype_suffix(src, "gather_gm_to_l1"))
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class gather_gm_to_ub:
    """
    /*
     * Function:
     *   On the VECTOR core, gather fp16/bf16 rows from a row-contiguous 2D
     *   GM tensor by index and write them into a UB tensor. Output row i
     *   comes from source row index[i]; adjacent indices
     *   (index[i + 1] == index[i] + 1) are merged into a single two-row DMA
     *   copy.
     *
     * Inputs:
     *   src: row-contiguous 2D fp16/bf16 source tensor in GM.
     *   index: 2D int32 index tensor in GM (shape (N, 1), stride (1, 1));
     *     output row i takes its data from source row index[i] (0-based row
     *     number); the starting offset of the indices is expressed through
     *     the block ptr offsets, so no extra parameter is needed.
     *   tile_size: number of data rows to gather.
     *   D: number of fp16/bf16 elements copied from each source row.
     *
     * Outputs:
     *   out: required 2D UB fp16/bf16 destination tensor (same dtype as
     *     src), corresponding to dst in the C++ ABI; the stride of the first
     *     dimension must be no smaller than D. The gathered result is
     *     written into this tensor directly and it is returned on the
     *     Python side.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(self, src, index, tile_size, D, out=None):
        assert out is not None, "out buffer is required"
        assert _element_dtype(out) == _element_dtype(src), (
            f"gather_gm_to_ub requires out dtype ({_element_dtype(out)}) "
            f"to match src dtype ({_element_dtype(src)})")
        assert _element_dtype(index) == tl.int32, (f"gather_gm_to_ub requires int32 index, "
                                                   f"got {_element_dtype(index)}")
        self.symbol = ("custom_gather_gm_to_ub_" + _gather_dtype_suffix(src, "gather_gm_to_ub"))
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class sort_1d_pack:
    """
    /*
     * Function:
     *   Sort a 1D float tensor in UB and, according to sort_impl, select the
     *   matching TopK sort implementation, emitting the best TOPK compact
     *   proposals. Each proposal occupies two float slots holding the value
     *   and, in the second slot, the encoded index.
     *
     * Inputs:
     *   src: 1D float input tensor in UB.
     *   tmp_buf: UB scratch tensor holding intermediate sort/merge results.
     *   descending: sort descending when True, ascending when False.
     *   TOPK: number of proposals to keep.
     *   index_offset: offset added to each raw index.
     *   sort_impl: sort implementation number; the device side picks the
     *     matching sort path from it based on the input size.
     *
     * Outputs:
     *   out: required UB float destination tensor, corresponding to
     *     dst_proposals in the C++ ABI; must hold at least 2 * TOPK float
     *     slots storing the TOPK proposals compactly as
     *     [value, encoded_index]. It is returned on the Python side.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, tmp_buf, descending, TOPK, index_offset, sort_impl, out=None):
        assert out
        assert _element_dtype(src) == tl.float32, (f"sort_1d_pack only supports fp32 src, got {_element_dtype(src)}")
        assert _element_dtype(tmp_buf) == tl.float32, (f"sort_1d_pack only supports fp32 tmp_buf, "
                                                       f"got {_element_dtype(tmp_buf)}")
        assert _element_dtype(out) == tl.float32, (f"sort_1d_pack only supports fp32 out, got {_element_dtype(out)}")
        self.symbol = "custom_sort_1d_pack_float"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class merge_exhaust_sort4:
    """
    /*
     * Function:
     *   Perform one exhaustion-mode merge over up to four sorted proposal
     *   ways in UB, emitting the safely determined sorted prefix and the
     *   number of proposals actually consumed from each way. All offsets
     *   and lengths are counted in proposals, not float slots.
     *
     * Inputs:
     *   src_proposals: UB tensor holding multiple sorted compact proposal
     *     ways.
     *   ways: number of active input ways; must match the number of
     *     non-zero values among len0..len3.
     *   off0, off1, off2, off3: starting offset of each of the four input
     *     ways, in proposals.
     *   len0, len1, len2, len3: number of proposals available in each of
     *     the four input ways; 0 marks the way as inactive.
     *
     * Outputs:
     *   out: required two-element output sequence [dst_proposals,
     *     consumed_out]:
     *     - out[0] / dst_proposals: UB float tensor receiving the safely
     *       determined sorted proposal prefix of the merge, corresponding
     *       to dst_proposals in the C++ ABI.
     *     - out[1] / consumed_out: UB tensor holding at least four int32
     *       elements, recording in order the number of proposals actually
     *       consumed this round from original ways 0 through 3,
     *       corresponding to consumed_out in the C++ ABI.
     *   The Python side returns (dst_proposals, consumed_out) in the same
     *   order.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src_proposals, ways, off0, off1, off2, off3, len0, len1, len2, len3, out=None):
        assert out
        assert len(out) == 2, ("merge_exhaust_sort4 requires out to be "
                               "[dst_proposals, consumed_out]")
        assert _element_dtype(src_proposals) == tl.float32, (f"merge_exhaust_sort4 only supports fp32 src_proposals, "
                                                             f"got {_element_dtype(src_proposals)}")
        assert _element_dtype(out[0]) == tl.float32, (f"merge_exhaust_sort4 only supports fp32 out[0], "
                                                      f"got {_element_dtype(out[0])}")
        assert _element_dtype(out[1]) == tl.int32, (f"merge_exhaust_sort4 only supports int32 out[1], "
                                                    f"got {_element_dtype(out[1])}")
        self.symbol = "custom_merge_exhaust_sort4_float"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class unpack_sort:
    """
    /*
     * Function:
     *   Split compactly stored TopK proposals in UB into separate float
     *   values and int32 indices. Each input proposal consists of two
     *   float-sized slots.
     *
     * Inputs:
     *   src_proposals: UB tensor holding at least topk compact proposals;
     *     its valid view must contain exactly 2 * topk float slots.
     *   topk: number of proposals to split.
     *
     * Outputs:
     *   out: required two-element output sequence [dst_value, dst_index]:
     *     - out[0] / dst_value: UB float tensor holding at least topk
     *       elements, receiving the value of each proposal, corresponding
     *       to dst_value in the C++ ABI.
     *     - out[1] / dst_index: UB int32 tensor holding at least topk
     *       elements, receiving the decoded index of each proposal,
     *       corresponding to dst_index in the C++ ABI.
     *   The Python side returns (dst_value, dst_index) in the same order.
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src_proposals, topk, out=None):
        assert out
        assert len(out) == 2, ("unpack_sort requires out to be [dst_value, dst_index]")
        assert _element_dtype(src_proposals) == tl.float32, (f"unpack_sort only supports fp32 src_proposals, "
                                                             f"got {_element_dtype(src_proposals)}")
        assert _element_dtype(out[0]) == tl.float32, (f"unpack_sort only supports fp32 out[0], "
                                                      f"got {_element_dtype(out[0])}")
        assert _element_dtype(out[1]) == tl.int32, (f"unpack_sort only supports int32 out[1], "
                                                    f"got {_element_dtype(out[1])}")
        self.symbol = "custom_unpack_sort_float"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class cube_mmad_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD

    def __init__(self, src_a: tl.uint32, src_b: tl.uint32, left_height, n_dim, right_width, unit_flag,
                 k_direction_align, c_matrix_source, c_matrix_init_val, dst: tl.uint32):
        self.symbol = "mmad_int8"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cube_load2d_b_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(self, src: tl.uint32, startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrMode,
                 dst: tl.uint32):
        self.symbol = "load_data_2d_int8_b"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cube_load3d_a_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(self, src: tl.uint32, padList, l1H, l1W, channelSize, kExtension, mExtension, kStartPt, mStartPt,
                 strideW, strideH, filterW, filterH, dilationFilterW, dilationFilterH, enTranspose, enSmallK, padValue,
                 filterSizeW, filterSizeH, fMatrixCtrl, isSetFMatrix, isSetPadding, dst: tl.uint32):
        self.symbol = "load_data_3d_int8_a"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cube_set_l0c_copy_params:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_S
    mode = al.MODE.SIMD

    def __init__(self, nd_num, src_nd_stride, dst_nd_stride):
        self.symbol = "set_l0c_copy_params"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cube_copy_l0c2gm_i32:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_FIX
    mode = al.MODE.SIMD

    def __init__(self, dst: tl.uint64, src: tl.uint32, n_size, m_size, dst_stride_dst_d, src_stride, unit_flag_mode,
                 quant_pre: tl.uint64, relu_pre, channel_split, nz2nd_en):
        self.symbol = "copy_l0c2gm_i32"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cube_nd2nz_i8:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(self, dst: tl.uint32, src: tl.uint64, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                 dstNzC0Stride, dstNzNStride, dstNzMatrixStride):
        self.symbol = "data_copy_nd2nz_i8"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cube_load_transpose_b_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(self, src: tl.uint32, startIndex, repeatTimes, srcStride, dstGap, dstFracGap, addrMode,
                 dst: tl.uint32):
        self.symbol = "load_data_transpose_int8_b"
        self.bitcode = CUSTOM_OPS_BITCODE


def validate_cast(src, out, src_dtype, dst_dtype, roundMode, count):
    assert src.dtype == src_dtype and len(src.shape) == 1, "Cast requires a 1D source of the declared dtype"
    assert out is not None, "Cast requires an output buffer"
    assert out.dtype == dst_dtype and len(out.shape) == 1, "Cast requires a 1D output of the declared dtype"
    capacity = src.numel.value
    assert capacity >= 32 and capacity & (capacity - 1) == 0, "Cast capacity must be a power of two of at least 32"
    assert out.numel.value == capacity, "Cast source and output capacities must match"
    assert isinstance(roundMode,
                      int) and roundMode in (1, 2, 3, 4, 5), "Cast supports RINT/FLOOR/CEIL/ROUND/TRUNC (1..5)"
    assert isinstance(count,
                      int) and 0 < count <= capacity, "Cast count must be a compile-time integer in [1, capacity]"


@al.register_custom_op
class cast_fp32_to_int16:
    """CANN Cast(dst, src, roundMode, count), float32 to int16.

    All scalar parameters of the count overload are exposed. Modes are
    1=RINT (ties to even), 2=FLOOR, 3=CEIL, 4=ROUND (ties away from zero),
    5=TRUNC. Source and required out are contiguous, 32-byte aligned,
    disjoint 1D UB tensors with equal power-of-two capacity of at least 32; UB fit is the caller's responsibility.
    Elements at or beyond count retain their previous out values.
    The caller must supply finite values representable after rounding.
    Mask count mode is restored to normal/all lanes; no barrier is added.
    """
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, roundMode, count, out=None):
        validate_cast(src, out, tl.float32, tl.int16, roundMode, count)
        self.arg_type["roundMode"] = tl.int32
        self.arg_type["count"] = tl.uint32
        self.symbol = "cast_fp32_to_int16"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cast_fp16_to_int8:
    """CANN Cast(dst, src, roundMode, count), float16 to int8.

    All scalar parameters of the count overload are exposed. Modes are
    1=RINT (ties to even), 2=FLOOR, 3=CEIL, 4=ROUND (ties away from zero),
    5=TRUNC. Source and required out are contiguous, 32-byte aligned,
    disjoint 1D UB tensors with equal power-of-two capacity of at least 32; UB fit is the caller's responsibility.
    Elements at or beyond count retain their previous out values.
    The caller must supply finite values representable after rounding.
    Mask count mode is restored to normal/all lanes; no barrier is added.
    """
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, roundMode, count, out=None):
        validate_cast(src, out, tl.float16, tl.int8, roundMode, count)
        self.arg_type["roundMode"] = tl.int32
        self.arg_type["count"] = tl.uint32
        self.symbol = "cast_fp16_to_int8"
        self.bitcode = CUSTOM_OPS_BITCODE
