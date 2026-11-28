# Copyright 2026- Xcoresigma Technology Co., Ltd
from pathlib import Path

import triton.language as tl
import triton.language.extra.cann.extension as al

CUSTOM_OPS_BITCODE = str(Path(__file__).with_name("custom_ops.bc").resolve())

_GATHER_SUFFIX_BY_DTYPE = {tl.float16: "half", tl.bfloat16: "bf16"}
_GATHER_L1_SUFFIX_BY_DTYPE = {**_GATHER_SUFFIX_BY_DTYPE, tl.int8: "s8"}


def _element_dtype(value):
    # Block-pointer tensors (tl.make_block_ptr) carry a nested dtype:
    # pointer<block<[shape], dtype>>. Unwrap element_ty down to the scalar.
    dtype = value.dtype
    while hasattr(dtype, "element_ty"):
        dtype = dtype.element_ty
    return dtype


def _gather_dtype_suffix(value, op_name, table=_GATHER_SUFFIX_BY_DTYPE):
    dtype = _element_dtype(value)
    suffix = table.get(dtype)
    assert suffix is not None, (f"{op_name} only supports "
                                f"{'/'.join(str(d) for d in table)}, got {dtype}")
    return suffix


@al.register_custom_op
class duplicate_bitwise_mask:
    """
    /*
     * Function:
     *   Copy scalar (variable or immediate) to fill vector in bitwise mask mode, corresponding to the Duplicate in Ascend C API.
     *
     * Inputs:
     *   scalar_value: source
     *   mask: controls the elements involved in computation per iteration
     *   repeat_times: repeat times
     *   dst_block_stride: data block address stride in one iteration
     *   dst_repeat_stride: address stride of dst between adjacent iterations
     *
     * Outputs:
     *   out: required UB destination tensor storing output
     *
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, scalar_value, mask, repeat_times, dst_block_stride, dst_repeat_stride, out=None):
        assert out
        assert _element_dtype(out) == _element_dtype(scalar_value), (
            f"duplicate_bitwise_mask requires dst dtype ({_element_dtype(out)}) "
            f"to match scalar_value dtype ({_element_dtype(scalar_value)})")
        if _element_dtype(out) == tl.float16:
            self.symbol = "custom_duplicate_bitwise_mask_half"
        elif _element_dtype(out) == tl.bfloat16:
            self.symbol = "custom_duplicate_bitwise_mask_bf16"
        elif _element_dtype(out) == tl.float32:
            self.symbol = "custom_duplicate_bitwise_mask_float"
        else:
            assert False, (f"duplicate_bitwise_mask does not support dtype {_element_dtype(out)} yet")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


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
        self.symbol = ("custom_gather_gm_to_l1_" +
                       _gather_dtype_suffix(src, "gather_gm_to_l1", _GATHER_L1_SUFFIX_BY_DTYPE))
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
class gather_mask_builtin_pattern:
    """
    /*
     * Function:
     *   Select elements from src operand with built-in fixed patterns, corresponding to the GatherMask in Ascend C API.
     *
     * Inputs:
     *   src0: 1D input tensor in UB storing source.
     *   src1_pattern: built-in fixed patterns.
     *   reduce_mode: selects mask parameter mode.
     *   mask: controls elements involved in computation per iteration.
     *   src0_block_stride: the address stride between different DataBlocks of src0 within the same iteration. (struct GatherMaskParams)
     *   repeat_times: number of iterations. (struct GatherMaskParams)
     *   src0_repeat_stride: address stride of src0 between adjacent iterations. (struct GatherMaskParams)
     *   src1_repeat_stride: address stride of src1 between adjacent iterations. (struct GatherMaskParams)
     *
     * Outputs:
     *   out: required two-element output sequence [dst, rsvd_cnt]
     *     - out[0] / dst: UB float tensor for gather result
     *     - out[1] / rsvd_cnt: number of valid elements in dst
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src0, src1_pattern, reduce_mode, mask, src0_block_stride, repeat_times, src0_repeat_stride,
                 src1_repeat_stride, out=None):
        assert out
        assert len(out) == 2, ("gather_mask_builtin_pattern requires out to be [dst, rsvd_cnt]")
        assert _element_dtype(out[0]) == _element_dtype(src0), (
            f"gather_mask_builtin_pattern requires dst dtype ({_element_dtype(out[0])}) "
            f"to match src0 dtype ({_element_dtype(src0)})")
        assert _element_dtype(out[1]) == tl.int64, (
            f"gather_mask_builtin_pattern only supports int64 rsvd_cnt, got {_element_dtype(out[1])}")
        if _element_dtype(src0) == tl.float16:
            self.symbol = "custom_gather_mask_builtin_pattern_half"
        elif _element_dtype(src0) == tl.bfloat16:
            self.symbol = "custom_gather_mask_builtin_pattern_bf16"
        elif _element_dtype(src0) == tl.uint16:
            self.symbol = "custom_gather_mask_builtin_pattern_ushort"
        elif _element_dtype(src0) == tl.int16:
            self.symbol = "custom_gather_mask_builtin_pattern_short"
        elif _element_dtype(src0) == tl.float32:
            self.symbol = "custom_gather_mask_builtin_pattern_float"
        elif _element_dtype(src0) == tl.uint32:
            self.symbol = "custom_gather_mask_builtin_pattern_uint"
        elif _element_dtype(src0) == tl.int32:
            self.symbol = "custom_gather_mask_builtin_pattern_int"
        else:
            assert False, (f"gather_mask_builtin_pattern src0 does not support dtype {_element_dtype(src0)}")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class gather_mask_custom_pattern:
    """
    /*
     * Function:
     *   Select elements from src operand with user-defined pattern, corresponding to the GatherMask in Ascend C API.
     *
     * Inputs:
     *   src0: 1D input tensor in UB storing source.
     *   src1_pattern: user-defined pattern.
     *   reduce_mode: selects mask parameter mode.
     *   mask: controls elements involved in computation per iteration.
     *   src0_block_stride: the address stride between different DataBlocks of src0 within the same iteration. (struct GatherMaskParams)
     *   repeat_times: number of iterations. (struct GatherMaskParams)
     *   src0_repeat_stride: address stride of src0 between adjacent iterations. (struct GatherMaskParams)
     *   src1_repeat_stride: address stride of src1 between adjacent iterations. (struct GatherMaskParams)
     *
     * Outputs:
     *   out: required two-element output sequence [dst, rsvd_cnt]
     *     - out[0] / dst: UB float tensor for gather result
     *     - out[1] / rsvd_cnt: number of valid elements in dst
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src0, src1_pattern, reduce_mode, mask, src0_block_stride, repeat_times, src0_repeat_stride,
                 src1_repeat_stride, out=None):
        assert out
        assert len(out) == 2, ("gather_mask_custom_pattern requires out to be [dst, rsvd_cnt]")
        assert _element_dtype(out[0]) == _element_dtype(src0), (
            f"gather_mask_custom_pattern requires dst dtype ({_element_dtype(out[0])}) "
            f"to match src0 dtype ({_element_dtype(src0)})")
        assert _element_dtype(out[1]) == tl.int64, (
            f"gather_mask_custom_pattern only supports int64 rsvd_cnt, got {_element_dtype(out[1])}")
        assert isinstance(src1_pattern, tl.tensor), "gather_mask_custom_pattern requires src1_pattern to be a tensor"
        if _element_dtype(src1_pattern) == tl.uint16:
            if _element_dtype(src0) == tl.float16:
                self.symbol = "custom_gather_mask_custom_pattern_half"
            elif _element_dtype(src0) == tl.bfloat16:
                self.symbol = "custom_gather_mask_custom_pattern_bf16"
            elif _element_dtype(src0) == tl.uint16:
                self.symbol = "custom_gather_mask_custom_pattern_ushort"
            elif _element_dtype(src0) == tl.int16:
                self.symbol = "custom_gather_mask_custom_pattern_short"
            else:
                assert False, "when src1_pattern is uint16_t tensor, src0 only supports dtype half / bfloat16 / uint16 / int16"
        elif _element_dtype(src1_pattern) == tl.uint32:
            if _element_dtype(src0) == tl.float32:
                self.symbol = "custom_gather_mask_custom_pattern_float"
            elif _element_dtype(src0) == tl.uint32:
                self.symbol = "custom_gather_mask_custom_pattern_uint"
            elif _element_dtype(src0) == tl.int32:
                self.symbol = "custom_gather_mask_custom_pattern_int"
            else:
                assert False, "when src1_pattern is uint32_t tensor, src0 only supports dtype float32 / uint32/ int32"
        else:
            assert False, (
                f"gather_mask_custom_pattern src1_pattern does not support dtype {_element_dtype(src1_pattern)}")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class pair_reduce_sum_continuous_mask:
    """
    /*
     * Function:
     *   Sum pairwise for odd and even elements in continuous mask mode, corresponding to the PairReduceSum in Ascend C API.
     *
     * Inputs:
     *   src: 1D input tensor in UB storing source
     *   repeat_times: repeat times
     *   mask: number of leading consecutive elements for computation
     *   dst_rep_stride: address stride of dst between adjacent iterations
     *   src_blk_stride: data block address stride in one iteration
     *   src_rep_stride: Address stride of src between adjacent iterations
     *
     * Outputs:
     *   out: required UB destination tensor storing output
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, repeat_times, mask, dst_rep_stride, src_blk_stride, src_rep_stride, out=None):
        assert out
        assert _element_dtype(out) == _element_dtype(src), (
            f"pair_reduce_sum_continuous_mask requires dst dtype ({_element_dtype(out)}) "
            f"to match src dtype ({_element_dtype(src)})")
        if _element_dtype(src) == tl.float16:
            self.symbol = "custom_pair_reduce_sum_continuous_mask_half"
        elif _element_dtype(src) == tl.float32:
            self.symbol = "custom_pair_reduce_sum_continuous_mask_float"
        else:
            assert False, (f"pair_reduce_sum_continuous_mask does not support dtype {_element_dtype(src)} yet")
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


@al.register_custom_op
class sort32:
    """
    /*
     * Function:
     *   Sort 32 elements in descending order in each repeat, corresponding to the Sort32 in Ascend C API.
     *
     * Inputs:
     *   src0: 1D float input tensor in UB storing value
     *   src1: 1D uint32 input tensor in UB storing index
     *   repeat_times: repeat times
     *
     * Outputs:
     *   out: required UB float destination tensor storing sorted pair [value, index]
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src0, src1, repeat_times, out=None):
        assert _element_dtype(src0) == tl.float32, (f"sort32 only supports fp32 value, got {_element_dtype(src0)}")
        assert _element_dtype(src1) == tl.uint32, (f"sort32 only supports uint32 index, got {_element_dtype(src1)}")
        assert out
        assert _element_dtype(out) == tl.float32, (f"sort32 only supports fp32 out, got {_element_dtype(out)}")
        self.symbol = "custom_sort32"
        self.bitcode = CUSTOM_OPS_BITCODE
        self.extra_buffers = [(tl.float16, 0)]


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
class mrgsort:
    """
    /*
     * Function:
     *   Arrange and merge up to four arranged potential queues into one queue, corresponding to the MrgSort in Ascend C API.
     *
     * Inputs:
     *   src_proposals: UB tensor holding multiple sorted compact proposal
     *     queues.
     *   off0, off1, off2, off3: starting offset of each of the four input
     *     queues, in proposals.
     *   len0, len1, len2, len3: number of proposals available in each of
     *     the four input queues. (struct MrgSort4Info)
     *   if_exhausted_suspension: judge whether to stop after a queue is exhausted. (struct MrgSort4Info)
     *   valid_bit: judge whether a queue is valid or not. (struct MrgSort4Info)
     *   repeat_times: repeat times. (struct MrgSort4Info)
     *
     * Outputs:
     *   out: UB float destination tensor for the merged proposals
     */
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src_proposals, off0, off1, off2, off3, len0, len1, len2, len3, if_exhausted_suspension,
                 valid_bit, repeat_times, out=None):
        assert _element_dtype(src_proposals) == tl.float32, (
            f"mrgsort only supports fp32 src, got {_element_dtype(src_proposals)}")
        assert out
        assert _element_dtype(out) == tl.float32, (f"mrgsort only supports fp32 out, got {_element_dtype(out)}")
        self.symbol = "custom_mrgsort"
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


# Mask buffers occupy at least one 32-byte UB block.
def _mask_source_size(src, op_name):
    assert src.dtype in (tl.float16, tl.float32), f"{op_name} requires an fp16/fp32 UB tensor"
    assert len(src.shape) == 1, f"{op_name} requires a 1D source"
    size = src.numel.value
    limit = 32768 if src.dtype == tl.float16 else 4096
    assert 256 <= size <= limit and size & (size - 1) == 0, (
        f"{op_name} requires a power-of-two source size in [256, {limit}]")
    return size


def _check_mask_buffer(mask, size, op_name):
    assert mask.dtype in (tl.uint16, tl.uint32), f"{op_name} requires a uint16/uint32 bitmask"
    width = 16 if mask.dtype == tl.uint16 else 32
    assert len(mask.shape) == 1 and mask.numel.value == size // width, (
        f"{op_name} requires a 1D bitmask with {size // width} elements")


@al.register_custom_op
class flip:
    """Reverse a 2D UB tile along its last dimension.

    Vectorized replacement for the meta-op library flip_1d scalar loop.
    Two NCHW-convert transpose passes (scatter_vnchwconv_b16/b32) run
    src -> tmp -> dst, each feeding src 32-byte rows in reverse order and
    dst rows in forward order; the composition is the full reversal of
    every 512B (2-byte dtypes) / 256B (4-byte dtypes) chunk, and repeat
    strides walk the chunks of a row (or a contiguous multi-row tile) in
    two instruction sequences. Shapes the transpose cannot cover
    (2-byte dtypes below 256 elements per row, 4-byte dtypes below 64,
    non-unit or unaligned strides, 1/8-byte dtypes) fall back to the same
    scalar loop as the library op. The op inserts no pipeline barrier;
    ordering is the caller's responsibility.

    Inputs:
      src: 2D UB tensor; the last dim must be fully covered by the tile.
      tmp: 2D UB scratch tensor of the same dtype as src providing at
        least 512 bytes (e.g. shape (1, 256) for 2-byte dtypes); used as
        the fixed intermediate of the per-chunk transpose pair and
        untouched by the scalar fallback.

    Outputs:
      out: required 2D UB destination tensor, same dtype and shape as src.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, tmp, out=None):
        assert out is not None, "flip requires an out buffer"
        assert _element_dtype(out) == _element_dtype(src) == _element_dtype(tmp), (
            f"flip requires out/tmp dtype ({_element_dtype(out)}/{_element_dtype(tmp)}) "
            f"to match src dtype ({_element_dtype(src)})")
        assert len(src.shape) == 2, f"flip requires a 2D tile, got shape {src.shape}"
        assert len(out.shape) == 2, f"flip requires a 2D out tile, got shape {out.shape}"
        assert out.shape == src.shape, (f"flip requires out shape ({out.shape}) to match src shape ({src.shape})")
        assert len(tmp.shape) == 2, f"flip requires a 2D tmp tile, got shape {tmp.shape}"
        tmp_bytes = tmp.numel.value * _element_dtype(tmp).primitive_bitwidth // 8
        assert tmp_bytes >= 512, (f"flip requires a tmp buffer of at least 512 bytes, "
                                  f"got {tmp_bytes}")
        dtype = _element_dtype(src)
        suffix_by_dtype = {
            tl.float16: "half",
            tl.bfloat16: "bf16",
            tl.float32: "float",
            tl.int16: "short",
            tl.uint16: "ushort",
            tl.int32: "int",
            tl.uint32: "uint",
            tl.int8: "char",
            tl.uint8: "uchar",
            tl.int64: "long",
            tl.uint64: "ulong",
        }
        suffix = suffix_by_dtype.get(dtype)
        assert suffix is not None, f"flip does not support dtype {dtype}"
        self.symbol = "custom_flip_" + suffix
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class compare_scalar:
    """Packed scalar comparison of contiguous FP16/FP32 UB values.

    Based on CANN 9.1 `dav_c220/kernel_operator_vec_cmp_impl.h`
    CompareScalarCompute / VcmpvsIntrinsicsImpl (CompareScalar Level 2). All
    parameters of that API are exposed: cmpMode is the CANN CMPMODE value
    0=LT, 1=GT, 2=EQ, 3=LE, 4=GE, 5=NE (utils/kernel_utils_mode.h); count is
    the number of compared elements and must equal the source size. scalar is
    passed as FP32 and rounded to the source dtype before comparison.
    Required out: uint16[N/16], or uint32[N/32] for FP32 sources. The uint32
    form feeds gather_mask_custom_pattern without repacking. FP32 sizes:
    powers of two 256..4096; FP16: 256..32768. Bit i records the predicate
    for element i, least-significant bit first. All bits are defined.
    Inputs/outputs must be 32-byte aligned and disjoint. The op inserts no
    pipeline barrier; ordering is the caller's responsibility.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, scalar, cmpMode, count, out=None):
        assert out is not None, "compare_scalar requires an output bitmask"
        size = _mask_source_size(src, "compare_scalar")
        assert isinstance(
            cmpMode, int) and 0 <= cmpMode <= 5, ("compare_scalar cmpMode must be a compile-time CANN CMPMODE value "
                                                  "0=LT, 1=GT, 2=EQ, 3=LE, 4=GE, 5=NE")
        assert isinstance(count, int) and count == size, (
            f"compare_scalar count must be a compile-time element count equal to the source size ({size})")
        assert src.dtype == tl.float32 or out.dtype == tl.uint16, "FP16 comparison requires a uint16 mask"
        _check_mask_buffer(out, count, "compare_scalar")
        self.arg_type["scalar"] = tl.float32
        self.arg_type["cmpMode"] = tl.int32
        self.arg_type["count"] = tl.int32
        suffix = "half" if src.dtype == tl.float16 else "float"
        if out.dtype == tl.uint32:
            suffix += "_mask32"
        self.symbol = "custom_compare_scalar_" + suffix
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cast_int4_to_fp16:
    """Unpack signed INT4 values from a 1D uint8 UB tensor into FP16.

    Based on CANN 9.1 `dav_c220/kernel_operator_vec_vconv_impl.h` CastImpl
    (Cast Level 2) and its int4b_t -> half specialization. All parameters of
    that API are exposed: roundMode must be CAST_NONE (0), the only mode
    supported from int4b_t to half on this device; count is the number of
    output FP16 elements and must equal 2 * N. src: uint8[N] in UB, N a
    power of two from 32 through 8192 bytes. Required out: float16[2*N] in
    UB. Each source byte encodes two signed two's-complement values in
    [-8, 7]; the low nibble is emitted first. All output elements are
    defined. No zero-point, scale or layout conversion is applied. Inputs
    and outputs must be contiguous, 32-byte aligned and disjoint. GM
    accesses stay in the caller. The op inserts no pipeline barrier beyond
    the mask set/restore of the reference implementation; ordering is the
    caller's responsibility.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, roundMode, count, out=None):
        assert out is not None, "cast_int4_to_fp16 requires an output buffer"
        assert src.dtype == tl.uint8 and len(src.shape) == 1, ("cast_int4_to_fp16 requires a 1D uint8 UB source")
        size = src.numel.value
        assert size in (32, 64, 128, 256, 512, 1024, 2048, 4096,
                        8192), ("cast_int4_to_fp16 requires a power-of-two byte count in [32, 8192]")
        assert isinstance(roundMode, int) and roundMode == 0, (
            "cast_int4_to_fp16 only supports RoundMode::CAST_NONE (0) from int4b_t to half on this device")
        assert isinstance(count, int) and count == 2 * size, (
            f"cast_int4_to_fp16 count must be a compile-time output element count equal to 2 * N ({2 * size})")
        assert out.dtype == tl.float16 and len(out.shape) == 1 and out.numel.value == count, (
            "cast_int4_to_fp16 requires a 1D float16 output with twice the source element count")
        self.arg_type["roundMode"] = tl.int32
        self.arg_type["count"] = tl.int32
        self.symbol = "custom_cast_int4_to_fp16"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cast_fp16_to_int8_rint:
    """Round-to-nearest-even saturating FP16 -> INT8 conversion on a 1D UB tensor.

    Wraps `vconv_f162s8r` in element-count mask mode (dst/src repeat strides
    4/8). src: float16[N] in UB, N a multiple of 32 in [32, 32640]. Required
    out: int8[N] in UB. Inputs and outputs must be contiguous, 32-byte aligned
    and disjoint. Pipeline barriers remain the caller's responsibility.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, src, count, out=None):
        assert out is not None, "cast_fp16_to_int8_rint requires an output buffer"
        assert src.dtype == tl.float16 and len(
            src.shape) == 1, ("cast_fp16_to_int8_rint requires a 1D float16 UB source")
        size = src.numel.value
        assert size % 32 == 0 and 32 <= size <= 32640, (
            f"cast_fp16_to_int8_rint requires a 32-element-aligned count in [32, 32640], got {size}")
        assert isinstance(count,
                          int) and count == size, (f"cast_fp16_to_int8_rint count must equal the source size ({size})")
        assert out.dtype == tl.int8 and len(out.shape) == 1 and out.numel.value == count, (
            "cast_fp16_to_int8_rint requires a 1D int8 output matching count")
        self.arg_type["count"] = tl.uint32
        self.symbol = "custom_cast_fp16_to_int8_rint"
        self.bitcode = CUSTOM_OPS_BITCODE


_L1_DTYPE_ELEM_BYTES = {tl.int8: 1, tl.float16: 2, tl.bfloat16: 2}
_MATMUL_L1_SUFFIX_BY_DTYPE = {
    tl.int8: "s8_s32",
    tl.float16: "f16_f32",
    tl.bfloat16: "bf16_f32",
}


def _validate_l1_zn_matrix(tensor, op_name, arg_name):
    dtype = _element_dtype(tensor)
    elem_bytes = _L1_DTYPE_ELEM_BYTES.get(dtype)
    assert elem_bytes is not None, (f"{op_name} {arg_name} only supports int8/float16/bfloat16, got {dtype}")
    assert len(tensor.shape) == 2, (f"{op_name} requires a 2D {arg_name} L1 tensor, got shape {tensor.shape}")
    rows, cols = tensor.shape[0].value, tensor.shape[1].value
    c0 = 32 // elem_bytes
    assert rows > 0 and rows % 16 == 0, (f"{op_name} {arg_name} rows must be a positive multiple of 16, got {rows}")
    assert cols > 0 and cols % c0 == 0, (f"{op_name} {arg_name} cols must be a positive multiple of {c0}, got {cols}")
    return dtype, elem_bytes, rows, cols


@al.register_custom_op
class nd2nz_gm_to_l1:
    """Copy a 2D GM matrix into a compiler-allocated L1/CBUF buffer in ZN layout.

    Supports int8, float16, and bfloat16 with arbitrary row stride `src_stride`.
    Uses `pipe=PIPE_MTE1` so BiShengIR allocates `out` in L1 (`#cbuf`).
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(
        self,
        src_gm: tl.uint64,
        rows: tl.uint32,
        cols: tl.uint32,
        src_stride: tl.uint32,
        out=None,
    ):
        assert out is not None, "nd2nz_gm_to_l1 requires an L1 output tensor"
        dtype, _, out_rows, out_cols = _validate_l1_zn_matrix(out, "nd2nz_gm_to_l1", "out")
        if isinstance(rows, int):
            assert rows == out_rows, (f"nd2nz_gm_to_l1 rows ({rows}) must match out.shape[0] ({out_rows})")
        if isinstance(cols, int):
            assert cols == out_cols, (f"nd2nz_gm_to_l1 cols ({cols}) must match out.shape[1] ({out_cols})")
        if isinstance(src_stride, int):
            assert src_stride >= out_cols, (f"nd2nz_gm_to_l1 src_stride ({src_stride}) must be >= cols ({out_cols})")
        self.symbol = ("custom_nd2nz_gm_to_l1_" +
                       _gather_dtype_suffix(out, "nd2nz_gm_to_l1", _GATHER_L1_SUFFIX_BY_DTYPE))
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class gather_blocks_gm_to_l1:
    """Gather `num_blocks` contiguous `[block_rows, cols]` GM blocks into L1 ZN.

    Block IDs are read from `block_ids_gm` (int32, clamped to `[0, valid_blocks - 1]`),
    issuing one ND2NZ DMA per block into a canonical `[num_blocks * block_rows, cols]`
    ZN layout in a compiler-allocated L1 buffer (`#cbuf`).
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(
        self,
        src_gm: tl.uint64,
        block_ids_gm: tl.uint64,
        num_blocks: tl.uint32,
        valid_blocks: tl.uint32,
        block_rows: tl.uint32,
        cols: tl.uint32,
        src_block_stride: tl.uint32,
        out=None,
    ):
        assert out is not None, "gather_blocks_gm_to_l1 requires an L1 output tensor"
        _, _, out_rows, out_cols = _validate_l1_zn_matrix(out, "gather_blocks_gm_to_l1", "out")
        if isinstance(block_rows, int):
            assert block_rows > 0 and block_rows % 16 == 0, (
                f"gather_blocks_gm_to_l1 block_rows must be a positive multiple of 16, got {block_rows}")
        if isinstance(num_blocks, int) and isinstance(block_rows, int):
            assert num_blocks > 0 and num_blocks * block_rows == out_rows, (
                f"gather_blocks_gm_to_l1 num_blocks * block_rows ({num_blocks * block_rows}) "
                f"must match out.shape[0] ({out_rows})")
        if isinstance(cols, int):
            assert cols == out_cols, (f"gather_blocks_gm_to_l1 cols ({cols}) must match out.shape[1] ({out_cols})")
        self.symbol = ("custom_gather_blocks_gm_to_l1_" +
                       _gather_dtype_suffix(out, "gather_blocks_gm_to_l1", _GATHER_L1_SUFFIX_BY_DTYPE))
        self.bitcode = CUSTOM_OPS_BITCODE


def _validate_matmul_l1_to_gm_args(op_name, lhs, rhs, m, n, k, dst_stride, trans_b, require_int8=False):
    lhs_dtype, elem_bytes, lhs_r, lhs_c = _validate_l1_zn_matrix(lhs, op_name, "lhs")
    rhs_dtype, _, rhs_r, rhs_c = _validate_l1_zn_matrix(rhs, op_name, "rhs")
    assert lhs_dtype == rhs_dtype, (f"{op_name} requires matching lhs ({lhs_dtype}) and rhs ({rhs_dtype}) dtypes")
    if require_int8:
        assert lhs_dtype == tl.int8, (f"{op_name} requires int8 inputs, got {lhs_dtype}")
    c0 = 32 // elem_bytes
    if isinstance(m, int):
        assert m > 0 and m % 16 == 0 and m == lhs_r, (
            f"{op_name} m ({m}) must be a positive multiple of 16 matching lhs.shape[0] ({lhs_r})")
    if isinstance(k, int):
        assert k > 0 and k % c0 == 0 and k == lhs_c, (
            f"{op_name} k ({k}) must be a positive multiple of {c0} matching lhs.shape[1] ({lhs_c})")
    if isinstance(n, int):
        assert n > 0 and n % 16 == 0, (f"{op_name} n ({n}) must be a positive multiple of 16, got {n}")
    if isinstance(dst_stride, int) and isinstance(n, int):
        assert dst_stride >= n, (f"{op_name} dst_stride ({dst_stride}) must be >= n ({n})")
    if isinstance(trans_b, int):
        assert trans_b in (0, 1), f"{op_name} trans_b must be 0 or 1, got {trans_b}"
        if trans_b == 1:
            if isinstance(n, int):
                assert rhs_r == n, (f"{op_name} trans_b=1 requires rhs.shape[0] ({rhs_r}) == n ({n})")
            if isinstance(k, int):
                assert rhs_c == k, (f"{op_name} trans_b=1 requires rhs.shape[1] ({rhs_c}) == k ({k})")
        else:
            if isinstance(k, int):
                assert rhs_r == k, (f"{op_name} trans_b=0 requires rhs.shape[0] ({rhs_r}) == k ({k})")
            if isinstance(n, int):
                assert rhs_c == n, (f"{op_name} trans_b=0 requires rhs.shape[1] ({rhs_c}) == n ({n})")
                if lhs_dtype == tl.int8:
                    assert n % 32 == 0, (f"{op_name} int8 trans_b=0 requires n to be a multiple of 32, got {n}")
    return lhs_dtype


@al.register_custom_op
class matmul_l1_to_gm:
    """General L1-ZN x L1-ZN -> GM matrix multiplication (int8->int32, fp16/bf16->fp32).

    Consumes two compiler-allocated L1 ZN tensors (as produced by `nd2nz_gm_to_l1`
    or `gather_blocks_gm_to_l1`) and writes the `[m, n]` product directly to GM
    `out_gm` with row stride `dst_stride`. Supports `trans_b` in `{0, 1}` so the
    same L1 ZN buffer can be reused across both transposed and non-transposed
    matmuls without GM reloads.
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD

    def __init__(
        self,
        lhs,
        rhs,
        out_gm: tl.uint64,
        m: tl.uint32,
        n: tl.uint32,
        k: tl.uint32,
        dst_stride: tl.uint32,
        trans_b: tl.uint32,
    ):
        lhs_dtype = _validate_matmul_l1_to_gm_args("matmul_l1_to_gm", lhs, rhs, m, n, k, dst_stride, trans_b)
        self.symbol = ("custom_matmul_l1_to_gm_" + _MATMUL_L1_SUFFIX_BY_DTYPE[lhs_dtype])
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class matmul_l1_to_gm_deqf16:
    """General INT8 L1-ZN x L1-ZN -> FP16 GM matmul with Fixpipe DEQF16 scaling.

    Accumulates `int8 x int8 -> int32` in L0C and dequantizes directly to FP16
    in GM via hardware Fixpipe `DEQF16` (`out = (lhs @ rhs) * deq_scale`).
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD

    def __init__(
        self,
        lhs,
        rhs,
        out_gm: tl.uint64,
        m: tl.uint32,
        n: tl.uint32,
        k: tl.uint32,
        dst_stride: tl.uint32,
        trans_b: tl.uint32,
        deq_scale: tl.float32,
    ):
        _validate_matmul_l1_to_gm_args(
            "matmul_l1_to_gm_deqf16",
            lhs,
            rhs,
            m,
            n,
            k,
            dst_stride,
            trans_b,
            require_int8=True,
        )
        self.symbol = "custom_matmul_l1_to_gm_s8_deqf16"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class online_softmax_quant_s8:
    """Stateless fused score-scale, FP32 online softmax, and INT8 quantization.

    Processes a `[total_rows, cols]` score tile (`total_rows % 8 == 0`,
    `cols % 64 == 0`) using strictly compiler-allocated UB buffers (`col_scale`,
    `state_in`, and `out=(state_out, work_ub)`). Updates online softmax state
    and writes dynamic per-row INT8 weights (`p_hi_gm`, and optional `p_lo_gm`).
    Flags bitmask: bit 0 (`is_first`), bit 1 (`use_shifted`), bit 2
    (`force_two_digit`), bit 3 (`running_norm`), bit 4 (`group_scale_fast`).
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(
        self,
        c_gm: tl.uint64,
        r_gm: tl.uint64,
        row_scale_gm: tl.uint64,
        col_scale_gm: tl.uint64,
        col_block_ids_gm: tl.uint64,
        flag_gm: tl.uint64,
        col_scale_ub,
        state_in,
        p_hi_gm: tl.uint64,
        p_lo_gm: tl.uint64,
        total_rows: tl.uint32,
        cols: tl.uint32,
        valid_cols: tl.uint32,
        scale: tl.float32,
        out_scale: tl.float32,
        flags: tl.uint32,
        out=None,
    ):
        assert out is not None and isinstance(
            out,
            (list, tuple, tl.tuple)) and len(out) == 2, ("online_softmax_quant_s8 requires out=(state_out, work_ub)")
        state_out, work_ub = out[0], out[1]
        assert isinstance(col_scale_ub, tl.tensor) and len(
            col_scale_ub.shape) == 2, ("online_softmax_quant_s8 requires 2D col_scale_ub [1, cols]")
        assert col_scale_ub.dtype == tl.float32, (
            f"online_softmax_quant_s8 requires float32 col_scale_ub, got {col_scale_ub.dtype}")
        assert isinstance(state_in, tl.tensor) and len(
            state_in.shape) == 2, ("online_softmax_quant_s8 requires 2D state_in [>=2, total_rows]")
        assert state_in.dtype == tl.float32 and state_in.shape[0] >= 2, (
            f"online_softmax_quant_s8 requires float32 state_in with >=2 rows, got {state_in.shape}, {state_in.dtype}")
        assert isinstance(state_out, tl.tensor) and len(
            state_out.shape) == 2, ("online_softmax_quant_s8 requires 2D state_out [5, total_rows]")
        assert state_out.dtype == tl.float32 and state_out.shape[0] == 5, (
            f"online_softmax_quant_s8 requires float32 state_out [5, total_rows], got {state_out.shape}, {state_out.dtype}"
        )
        assert isinstance(work_ub, tl.tensor) and len(
            work_ub.shape) == 1 and work_ub.dtype == tl.float32, ("online_softmax_quant_s8 requires 1D float32 work_ub")
        rows = int(state_out.shape[1])
        ncols = int(col_scale_ub.shape[1])
        assert rows >= 8 and rows % 8 == 0, (
            f"online_softmax_quant_s8 requires total_rows >= 8 and multiple of 8, got {rows}")
        assert ncols >= 64 and ncols % 64 == 0, (
            f"online_softmax_quant_s8 requires cols >= 64 and multiple of 64, got {ncols}")
        assert int(state_in.shape[1]) == rows, (
            f"online_softmax_quant_s8 state_in cols ({state_in.shape[1]}) must equal total_rows ({rows})")
        tile_r = 8 if (rows >= 128 or ncols > 384) else (16 if rows >= 16 else 8)
        nbanks = 2 if (rows // tile_r) >= 2 else 1
        min_work = (nbanks * 2 * tile_r * ncols + ((rows + 63) & ~63) + 2 * ((ncols + 63) & ~63) +
                    (((ncols + 127) & ~127) // 2) + tile_r * 64 + 512)
        assert int(work_ub.shape[0]) >= min_work, (
            f"online_softmax_quant_s8 work_ub too small: got {work_ub.shape[0]}, need >= {min_work}")
        self.symbol = "custom_online_softmax_quant_s8"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class gather_blocks_matmul_pair_l1_to_gm:
    """Stateless chunk-pipelined paired paged-gather + L1-ZN x L1-ZN^T -> GM matmul.

    Streams two block-indexed GM matrices (int8 + bf16 sharing `block_ids_gm`)
    into compiler-allocated L1 ZN outputs `out=(rhs0_l1, rhs1_l1)` while
    overlapping chunked `lhs0 @ rhs0^T -> out0_gm` (int32) and `lhs1 @ rhs1^T ->
    out1_gm` (fp32) matmuls on Cube. When `preloaded_rhs0_chunks != 0`, chunk 0
    of `rhs0` is assumed already resident in `rhs0_l1`.
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD

    def __init__(
        self,
        lhs0,
        rhs0_gm: tl.uint64,
        lhs1,
        rhs1_gm: tl.uint64,
        block_ids_gm: tl.uint64,
        out0_gm: tl.uint64,
        out1_gm: tl.uint64,
        num_blocks: tl.uint32,
        valid_blocks: tl.uint32,
        block_rows: tl.uint32,
        m: tl.uint32,
        k0: tl.uint32,
        k1: tl.uint32,
        src0_block_stride: tl.uint32,
        src1_block_stride: tl.uint32,
        dst_stride: tl.uint32,
        preloaded_rhs0_chunks: tl.uint32,
        out=None,
    ):
        assert out is not None and isinstance(out, (list, tuple, tl.tuple)) and len(out) == 2, (
            "gather_blocks_matmul_pair_l1_to_gm requires out=(rhs0_l1, rhs1_l1)")
        rhs0_l1, rhs1_l1 = out[0], out[1]
        d0, _, m0, c0 = _validate_l1_zn_matrix(lhs0, "gather_blocks_matmul_pair_l1_to_gm", "lhs0")
        d1, _, m1, c1 = _validate_l1_zn_matrix(lhs1, "gather_blocks_matmul_pair_l1_to_gm", "lhs1")
        rd0, _, rn0, rc0 = _validate_l1_zn_matrix(rhs0_l1, "gather_blocks_matmul_pair_l1_to_gm", "rhs0_l1")
        rd1, _, rn1, rc1 = _validate_l1_zn_matrix(rhs1_l1, "gather_blocks_matmul_pair_l1_to_gm", "rhs1_l1")
        assert d0 == tl.int8 and rd0 == tl.int8, "gather_blocks_matmul_pair_l1_to_gm requires int8 lhs0/rhs0_l1"
        assert d1 == tl.bfloat16 and rd1 == tl.bfloat16, (
            "gather_blocks_matmul_pair_l1_to_gm requires bfloat16 lhs1/rhs1_l1")
        assert m0 == m1 and rn0 == rn1 and c0 == rc0 and c1 == rc1
        self.symbol = "custom_gather_blocks_matmul_pair_s8s32_bf16f32"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class nd2nz_matmul_l1_to_gm_deqf16:
    """Stateless chunk-pipelined GM->L1 ND2NZ + INT8 L1-ZN x L1-ZN -> FP16 GM DEQF16 matmul.

    Streams `[m, k]` int8 `lhs_gm` into a compiler-allocated `[m, k]` L1 scratch
    buffer in k-chunks on PIPE_MTE2 overlapped with `lhs @ rhs -> out_gm`
    Fixpipe DEQF16 matmul on Cube. Optionally prefetches the next iteration's
    leading blocks into `rhs_l1` (passed via `out=(scratch_l1, rhs_l1)`) behind
    the final tile's MMA and Fixpipe stores when `prefetch_gm != 0`.
    """

    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD

    def __init__(
        self,
        lhs_gm: tl.uint64,
        out_gm: tl.uint64,
        m: tl.uint32,
        n: tl.uint32,
        k: tl.uint32,
        dst_stride: tl.uint32,
        deq_scale: tl.float32,
        prefetch_gm: tl.uint64,
        prefetch_ids_gm: tl.uint64,
        prefetch_blocks: tl.uint32,
        prefetch_valid_blocks: tl.uint32,
        prefetch_block_rows: tl.uint32,
        prefetch_block_stride: tl.uint32,
        out=None,
    ):
        assert out is not None and isinstance(out, (list, tuple, tl.tuple)) and len(out) == 2, (
            "nd2nz_matmul_l1_to_gm_deqf16 requires out=(scratch_l1, rhs_l1)")
        scratch_l1, rhs_l1 = out[0], out[1]
        sd, _, sm, sk = _validate_l1_zn_matrix(scratch_l1, "nd2nz_matmul_l1_to_gm_deqf16", "scratch_l1")
        rd, _, rk, rn = _validate_l1_zn_matrix(rhs_l1, "nd2nz_matmul_l1_to_gm_deqf16", "rhs_l1")
        assert sd == tl.int8 and rd == tl.int8, "nd2nz_matmul_l1_to_gm_deqf16 requires int8 rhs/scratch"
        assert sk == rk, f"nd2nz_matmul_l1_to_gm_deqf16 scratch k ({sk}) must match rhs k ({rk})"
        if isinstance(m, int):
            assert m == sm, f"nd2nz_matmul_l1_to_gm_deqf16 m ({m}) must match scratch rows ({sm})"
        if isinstance(n, int):
            assert n == rn and n % 32 == 0, f"nd2nz_matmul_l1_to_gm_deqf16 n ({n}) must match rhs cols ({rn})"
        if isinstance(k, int):
            assert k == sk and k % 32 == 0, f"nd2nz_matmul_l1_to_gm_deqf16 k ({k}) must match scratch cols ({sk})"
        self.symbol = "custom_nd2nz_matmul_l1_to_gm_s8_deqf16"
        self.bitcode = CUSTOM_OPS_BITCODE


@al.register_custom_op
class online_rescaled_accum:
    """Stateless 2-bank contiguous online rescaled accumulator update in UB.

    Updates compiler-allocated `acc_ub` (`[total_rows, dim]` FP16 or FP32) in
    place with `acc = alpha * acc + step * (src + bias_scale * bias + lo_scale *
    src_lo)` using 2-bank contiguous GM->UB DMA ping-pong and fused `vmla`, with
    optional final BF16 GM writeback when `(flags & 2) != 0` and optional
    `state_ub[4, 0]` gating of `src_lo_gm` when `(flags & 16) != 0`.
    """

    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(
        self,
        src_gm: tl.uint64,
        src_lo_gm: tl.uint64,
        bias_gm: tl.uint64,
        state_ub,
        out_gm: tl.uint64,
        total_rows: tl.uint32,
        dim: tl.uint32,
        bias_scale: tl.float32,
        lo_scale: tl.float32,
        flags: tl.uint32,
        out=None,
    ):
        assert out is not None and isinstance(
            out, (list, tuple, tl.tuple)) and len(out) == 2, ("online_rescaled_accum requires out=(acc_ub, work_ub)")
        acc_ub, work_ub = out[0], out[1]
        assert isinstance(state_ub, tl.tensor) and len(state_ub.shape) == 2 and state_ub.dtype == tl.float32
        assert state_ub.shape[0] >= 4
        assert isinstance(acc_ub, tl.tensor) and len(acc_ub.shape) == 2
        assert acc_ub.dtype in (tl.float16, tl.float32), (
            f"online_rescaled_accum requires float16 or float32 acc_ub, got {acc_ub.dtype}")
        assert isinstance(work_ub, tl.tensor) and len(work_ub.shape) == 1 and work_ub.dtype == tl.float32
        rows, ndim = int(acc_ub.shape[0]), int(acc_ub.shape[1])
        assert rows >= 8 and rows % 8 == 0 and int(state_ub.shape[1]) == rows
        assert ndim >= 128 and ndim % 128 == 0
        if acc_ub.dtype == tl.float16:
            tr = 16 if rows >= 128 else (32 if rows >= 32 else rows)
            nb = 2 if (rows // tr) >= 2 else 1
            lo_r = 8 if tr >= 8 else tr
            min_w = (nb * tr + lo_r) * (ndim // 2) + ndim // 2 + tr * 16 + 128
            assert int(work_ub.shape[0]) >= min_w, (
                f"online_rescaled_accum work_ub too small: got {work_ub.shape[0]}, need >= {min_w}")
            self.symbol = "custom_online_rescaled_accum_f16"
        else:
            tr = 8 if rows >= 8 else rows
            nb = 2 if (rows // tr) >= 2 else 1
            min_w = (nb + 1) * tr * ndim + ndim + tr * 16 + 128
            assert int(work_ub.shape[0]) >= min_w, (
                f"online_rescaled_accum work_ub too small: got {work_ub.shape[0]}, need >= {min_w}")
            self.symbol = "custom_online_rescaled_accum_f32"
        self.bitcode = CUSTOM_OPS_BITCODE
