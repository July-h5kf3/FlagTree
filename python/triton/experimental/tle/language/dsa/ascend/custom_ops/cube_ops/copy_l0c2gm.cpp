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

// Copied CANN 9.1 npu_arch_2201/cube_datamove_impl/asc_copy_l0c2gm_impl.h:
// asc_copy_l0c2gm_impl, INT32 to INT32 overload (lines 174-182).
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_copy_l0c2gm_i32(uint64_t dst_address, uint32_t src_address,
                             int32_t n_size, int32_t m_size,
                             int32_t dst_stride_dst_d, int32_t src_stride,
                             int32_t unit_flag_mode, uint64_t quant_pre,
                             int32_t relu_pre, int32_t channel_split,
                             int32_t nz2nd_en) {
  copy_matrix_cc_to_gm(
      reinterpret_cast<__gm__ int32_t *>(dst_address),
      reinterpret_cast<__cc__ int32_t *>((uint64_t)src_address), 0,
      (uint16_t)n_size, (uint16_t)m_size, (uint32_t)dst_stride_dst_d,
      (uint16_t)src_stride, (uint8_t)unit_flag_mode,
      static_cast<QuantMode_t>(quant_pre), (uint8_t)relu_pre,
      (bool)channel_split, (bool)nz2nd_en);
}
