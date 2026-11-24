// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// Saturating round-to-nearest-even FP16 -> INT8 conversion in UB via
// vconv_f162s8r in element-count mask mode (dst/src repeat strides 4/8).
// Pipeline barriers remain the caller's responsibility.
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_cast_fp16_to_int8_rint(memref_t<__ubuf__ half, 1> *src,
                                           uint32_t count,
                                           memref_t<__ubuf__ int8_t, 1> *dst) {
  set_mask_count();
  set_vector_mask(0, count);
  vconv_f162s8r(dst->aligned + dst->offset, src->aligned + src->offset, 1, 1, 1,
                4, 8);
  set_mask_norm();
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
