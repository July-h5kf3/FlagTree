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

// Copied INT8 branch of CANN 9.1 dav_c220/kernel_operator_mm_impl.h:
// MmadCal (lines 341-362), specialized for isBias=false.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_mmad_int8(uint32_t a_address, uint32_t b_address,
                       int32_t left_height, int32_t n_dim, int32_t right_width,
                       int32_t unit_flag, int32_t k_direction_align,
                       int32_t c_matrix_source, int32_t c_matrix_init_val,
                       uint32_t c_address) {
  mad(reinterpret_cast<__cc__ int32_t *>((uint64_t)c_address),
      reinterpret_cast<__ca__ int8_t *>((uint64_t)a_address),
      reinterpret_cast<__cb__ int8_t *>((uint64_t)b_address),
      static_cast<uint16_t>(left_height), static_cast<uint16_t>(n_dim),
      static_cast<uint16_t>(right_width), static_cast<uint8_t>(unit_flag),
      static_cast<bool>(k_direction_align), static_cast<bool>(c_matrix_source),
      static_cast<bool>(c_matrix_init_val));
}
