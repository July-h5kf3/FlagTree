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

// CANN 9.1 dav_c220/kernel_operator_fixpipe_v2_impl.h:
// FixpipeL0C2GMImpl and FixpipeL0cToOut, half/int32_t/CFG_ROW_MAJOR/DEQF16.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_fixpipe_int32_to_fp16(
    uint64_t dst, uint32_t src, uint16_t nSize, uint16_t mSize,
    uint16_t srcStride, uint32_t dstStride, int32_t quantPre,
    uint64_t deqScalar, uint16_t ndNum, uint16_t srcNdStride,
    uint16_t dstNdStride, bool reluEn, uint8_t unitFlag, bool isChannelSplit) {
  if (quantPre != static_cast<int32_t>(QuantMode_t::DEQF16) || isChannelSplit) {
    return;
  }
  uint64_t ndPara = static_cast<uint64_t>(dstNdStride) << 32;
  ndPara |= static_cast<uint64_t>(srcNdStride) << 16;
  ndPara |= static_cast<uint64_t>(ndNum);
  set_nd_para(ndPara);
  set_quant_pre(deqScalar);
  pipe_barrier(PIPE_FIX);
  copy_matrix_cc_to_gm(
      reinterpret_cast<__gm__ half *>(dst),
      reinterpret_cast<__cc__ int32_t *>(static_cast<uint64_t>(src)), 0, nSize,
      mSize, dstStride, srcStride, unitFlag, QuantMode_t::DEQF16,
      static_cast<uint8_t>(reluEn), false, true);
}
