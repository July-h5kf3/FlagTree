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

// Copied INT8 branch of CANN 9.1 dav_c220/kernel_operator_data_copy_impl.h:
// DataCopyGM2L1ND2NZImplBase (lines 245-258).
// copy_gm_to_cbuf_multi_nd2nz_b8 is a compiler builtin alias (see
// PRIMITIVES.md).
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_data_copy_nd2nz_i8(uint32_t dst_address, uint64_t src_address,
                                int32_t ndNum, int32_t nValue, int32_t dValue,
                                int32_t srcNdMatrixStride, int32_t srcDValue,
                                int32_t dstNzC0Stride, int32_t dstNzNStride,
                                int32_t dstNzMatrixStride) {
  copy_gm_to_cbuf_multi_nd2nz_b8(
      reinterpret_cast<__cbuf__ int8_t *>((uint64_t)dst_address),
      reinterpret_cast<__gm__ int8_t *>(src_address), 0, (uint16_t)ndNum,
      (uint16_t)nValue, (uint16_t)dValue, (uint16_t)srcNdMatrixStride,
      (uint16_t)srcDValue, (uint16_t)dstNzC0Stride, (uint16_t)dstNzNStride,
      (uint16_t)dstNzMatrixStride);
}
