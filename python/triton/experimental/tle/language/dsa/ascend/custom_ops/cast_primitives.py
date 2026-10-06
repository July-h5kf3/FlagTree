# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CANN dav-c220 Cast count overloads; arithmetic and ordering stay in TLE."""

import triton.language as tl
import triton.language.extra.cann.extension as al

from .registry import CUSTOM_OPS_BITCODE

CAST_PRIMITIVES_BITCODE = CUSTOM_OPS_BITCODE


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
        self.bitcode = CAST_PRIMITIVES_BITCODE


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
        self.bitcode = CAST_PRIMITIVES_BITCODE
