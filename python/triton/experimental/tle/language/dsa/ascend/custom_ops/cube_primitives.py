# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton.language as tl
import triton.language.extra.cann.extension as al

from .registry import CUSTOM_OPS_BITCODE

CUBE_PRIMITIVES_BITCODE = CUSTOM_OPS_BITCODE


@al.register_custom_op
class cube_mmad_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_M
    mode = al.MODE.SIMD

    def __init__(self, src_a: tl.uint32, src_b: tl.uint32, left_height, n_dim, right_width, unit_flag,
                 k_direction_align, c_matrix_source, c_matrix_init_val, dst: tl.uint32):
        self.symbol = "mmad_int8"
        self.bitcode = CUBE_PRIMITIVES_BITCODE


@al.register_custom_op
class cube_load2d_b_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(self, src: tl.uint32, startIndex, repeatTimes, srcStride, sid, dstGap, ifTranspose, addrMode,
                 dst: tl.uint32):
        self.symbol = "load_data_2d_int8_b"
        self.bitcode = CUBE_PRIMITIVES_BITCODE


@al.register_custom_op
class cube_load3d_a_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(self, src: tl.uint32, padList, l1H, l1W, channelSize, kExtension, mExtension, kStartPt, mStartPt,
                 strideW, strideH, filterW, filterH, dilationFilterW, dilationFilterH, enTranspose, enSmallK, padValue,
                 filterSizeW, filterSizeH, fMatrixCtrl, isSetFMatrix, isSetPadding, dst: tl.uint32):
        self.symbol = "load_data_3d_int8_a"
        self.bitcode = CUBE_PRIMITIVES_BITCODE


@al.register_custom_op
class cube_set_l0c_copy_params:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_S
    mode = al.MODE.SIMD

    def __init__(self, nd_num, src_nd_stride, dst_nd_stride):
        self.symbol = "set_l0c_copy_params"
        self.bitcode = CUBE_PRIMITIVES_BITCODE


@al.register_custom_op
class cube_copy_l0c2gm_i32:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_FIX
    mode = al.MODE.SIMD

    def __init__(self, dst: tl.uint64, src: tl.uint32, n_size, m_size, dst_stride_dst_d, src_stride, unit_flag_mode,
                 quant_pre: tl.uint64, relu_pre, channel_split, nz2nd_en):
        self.symbol = "copy_l0c2gm_i32"
        self.bitcode = CUBE_PRIMITIVES_BITCODE


@al.register_custom_op
class cube_nd2nz_i8:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(self, dst: tl.uint32, src: tl.uint64, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
                 dstNzC0Stride, dstNzNStride, dstNzMatrixStride):
        self.symbol = "data_copy_nd2nz_i8"
        self.bitcode = CUBE_PRIMITIVES_BITCODE


@al.register_custom_op
class cube_load_transpose_b_into:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE1
    mode = al.MODE.SIMD

    def __init__(self, src: tl.uint32, startIndex, repeatTimes, srcStride, dstGap, dstFracGap, addrMode,
                 dst: tl.uint32):
        self.symbol = "load_data_transpose_int8_b"
        self.bitcode = CUBE_PRIMITIVES_BITCODE
