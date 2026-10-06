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

// CANN 9.1 DataCopyGM2L1ND2NZImplBase, INT8 branch. Register packing is
// expanded from the dav-c220 compiler lowering; see PRIMITIVES.md.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_data_copy_nd2nz_i8(uint32_t dst_address, uint64_t src_address,
                                int32_t ndNum, int32_t nValue, int32_t dValue,
                                int32_t srcNdMatrixStride, int32_t srcDValue,
                                int32_t dstNzC0Stride, int32_t dstNzNStride,
                                int32_t dstNzMatrixStride) {
  // sid occupies bits [3:0] and is fixed to zero by the CANN overload.
  uint64_t shape_config = ((uint64_t(ndNum) & 0xFFF) << 4) |
                          ((uint64_t(nValue) & 0xFFFF) << 16) |
                          ((uint64_t(dValue) & 0xFFFF) << 32) |
                          ((uint64_t(srcNdMatrixStride) & 0xFFFF) << 48);
  uint64_t stride_config = (uint64_t(srcDValue) & 0xFFFF) |
                           ((uint64_t(dstNzC0Stride) & 0xFFFF) << 16) |
                           ((uint64_t(dstNzNStride) & 0xFFFF) << 32) |
                           ((uint64_t(dstNzMatrixStride) & 0xFFFF) << 48);
  copy_gm_to_cbuf_multi_nd2nz_b8(
      reinterpret_cast<__cbuf__ int8_t *>((uint64_t)dst_address),
      reinterpret_cast<__gm__ int8_t *>(src_address), shape_config,
      stride_config);
}
