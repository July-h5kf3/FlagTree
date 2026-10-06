/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "Utils.h"

// CANN 9.1 dav_c220 CastImpl count overload and its half -> int8_t
// specialization.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_cast_fp16_to_int8(memref_t<__ubuf__ half, 1> *source,
                               int32_t round_mode, uint32_t count,
                               memref_t<__ubuf__ int8_t, 1> *destination) {
  auto *src = source->aligned + source->offset;
  auto *dst = destination->aligned + destination->offset;
  set_mask_count();
  set_vector_mask(0, count);
  switch (round_mode) {
  case 1:
    vconv_f162s8r(dst, src, 1, 1, 1, 4, 8);
    break;
  case 2:
    vconv_f162s8f(dst, src, 1, 1, 1, 4, 8);
    break;
  case 3:
    vconv_f162s8c(dst, src, 1, 1, 1, 4, 8);
    break;
  case 4:
    vconv_f162s8a(dst, src, 1, 1, 1, 4, 8);
    break;
  case 5:
    vconv_f162s8z(dst, src, 1, 1, 1, 4, 8);
    break;
  default:
    break;
  }
  set_mask_norm();
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
