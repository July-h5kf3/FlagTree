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

// Copied CANN 9.1 dav_c220/kernel_operator_fixpipe_impl.h:
// SetFixpipeNz2ndFlagImpl (lines 65-75).
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_set_l0c_copy_params(int32_t nd_num, int32_t src_nd_stride,
                                 int32_t dst_nd_stride) {
  uint64_t config = 0;
  config = config | (static_cast<uint64_t>((uint16_t)nd_num));
  config = config | (static_cast<uint64_t>((uint16_t)src_nd_stride) << 16);
  config = config | (static_cast<uint64_t>((uint16_t)dst_nd_stride) << 32);
  set_nd_para(config);
}

// CANN 9.1 asc_copy_l0c2gm_impl, INT32 to INT32 overload. Register packing
// is expanded from the dav-c220 compiler lowering; see PRIMITIVES.md.
// Full Fixpipe tiling, quantization setup and barriers are outside this ABI.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_copy_l0c2gm_i32(uint64_t dst_address, uint32_t src_address,
                             int32_t n_size, int32_t m_size,
                             int32_t dst_stride_dst_d, int32_t src_stride,
                             int32_t unit_flag_mode, uint64_t quant_pre,
                             int32_t relu_pre, int32_t channel_split,
                             int32_t nz2nd_en) {
  // sid occupies bits [3:0] and is fixed to zero by the CANN overload.
  uint64_t shape_config = ((uint64_t(n_size) & 0xFFF) << 4) |
                          ((uint64_t(m_size) & 0xFFFF) << 16) |
                          ((uint64_t(dst_stride_dst_d) & 0xFFFFFFFF) << 32);
  uint64_t control_config =
      (uint64_t(src_stride) & 0xFFFF) |
      ((uint64_t(unit_flag_mode) & 0x3) << 32) | ((quant_pre & 0x1F) << 34) |
      ((uint64_t(relu_pre) & 0x7) << 39) |
      (uint64_t(bool(channel_split)) << 42) | (uint64_t(bool(nz2nd_en)) << 43);
  copy_matrix_cc_to_gm(
      reinterpret_cast<__gm__ int32_t *>(dst_address),
      reinterpret_cast<__cc__ int32_t *>((uint64_t)src_address), shape_config,
      control_config);
}
