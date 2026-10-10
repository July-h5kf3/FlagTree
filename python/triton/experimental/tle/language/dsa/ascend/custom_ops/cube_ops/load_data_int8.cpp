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
// LoadData2DL12L0BCal (lines 50-68).
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_data_2d_int8_b(uint32_t src_address, int32_t startIndex,
                                 int32_t repeatTimes, int32_t srcStride,
                                 int32_t sid, int32_t dstGap,
                                 int32_t ifTranspose, int32_t addrMode,
                                 uint32_t dst_address) {
  if (ifTranspose) {
    load_cbuf_to_cb(reinterpret_cast<__cb__ int8_t *>((uint64_t)dst_address),
                    reinterpret_cast<__cbuf__ int8_t *>((uint64_t)src_address),
                    (uint16_t)startIndex, (uint8_t)repeatTimes,
                    (uint16_t)srcStride, (uint16_t)dstGap, (uint8_t)sid, 1,
                    inc);
  } else {
    load_cbuf_to_cb(reinterpret_cast<__cb__ int8_t *>((uint64_t)dst_address),
                    reinterpret_cast<__cbuf__ int8_t *>((uint64_t)src_address),
                    (uint16_t)startIndex, (uint8_t)repeatTimes,
                    (uint16_t)srcStride, (uint16_t)dstGap, (uint8_t)sid, 0,
                    inc);
  }
}

// Copied INT8 branches of CANN 9.1 kernel_operator_mm_base_impl.h:173-178
// and dav_c220/kernel_operator_mm_impl.h:191-209,424-441,462-475
// (LoadDataImpl, LoadData3DV2L12L0ACal, Load3DSetFMatrixCal,
// Load3DSetPaddingCal).
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_data_3d_int8_a(
    uint32_t src_address, int32_t pad0, int32_t pad1, int32_t pad2,
    int32_t pad3, int32_t l1H, int32_t l1W, int32_t channelSize,
    int32_t kExtension, int32_t mExtension, int32_t kStartPt, int32_t mStartPt,
    int32_t strideW, int32_t strideH, int32_t filterW, int32_t filterH,
    int32_t dilationFilterW, int32_t dilationFilterH, int32_t enTranspose,
    int32_t enSmallK, int32_t padValue, int32_t filterSizeW,
    int32_t filterSizeH, int32_t fMatrixCtrl, int32_t isSetFMatrix,
    int32_t isSetPadding, uint32_t dst_address) {
  if (isSetFMatrix) {
    uint64_t regFMatrix = 0;
    regFMatrix |= uint64_t(l1W & 0xFFFF);
    uint32_t l1HShiftBit = 16;
    regFMatrix |= uint64_t(l1H & 0xFFFF) << l1HShiftBit;
    uint8_t padList[4] = {(uint8_t)pad0, (uint8_t)pad1, (uint8_t)pad2,
                          (uint8_t)pad3};
    uint32_t padNumber = 4;
    uint32_t padListShiftBit = 8;
    uint32_t padListShiftBase = 32;
    for (uint32_t i = 0; i < padNumber; i++) {
      regFMatrix |= uint64_t(padList[i] & 0xFF)
                    << (padListShiftBase + i * padListShiftBit);
    }
    set_fmatrix(regFMatrix);
  }
  if (isSetPadding) {
    uint64_t paddingValue = 0;
    uint64_t padValueShiftBit = 8;
    paddingValue =
        (static_cast<uint64_t>((int8_t)padValue) << padValueShiftBit) |
        (static_cast<uint64_t>((int8_t)padValue) & 0xFF);
    set_padding(paddingValue);
  }
  img2colv2_cbuf_to_ca(
      reinterpret_cast<__ca__ int8_t *>((uint64_t)dst_address),
      reinterpret_cast<__cbuf__ int8_t *>((uint64_t)src_address),
      (uint16_t)kExtension, (uint16_t)mExtension, (uint16_t)kStartPt,
      (uint16_t)mStartPt, (uint8_t)strideW, (uint8_t)strideH, (uint8_t)filterW,
      (uint8_t)filterH, (uint8_t)dilationFilterW, (uint8_t)dilationFilterH,
      (bool)filterSizeW, (bool)filterSizeH, (bool)enTranspose,
      (bool)fMatrixCtrl, (uint16_t)channelSize);
}

// Copied INT8 branch of CANN 9.1 dav_c220/kernel_operator_mm_impl.h:
// LoadData2DL12L0BTransposeCal (lines 163-176).
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_data_transpose_int8_b(uint32_t src_address,
                                        int32_t startIndex, int32_t repeatTimes,
                                        int32_t srcStride, int32_t dstGap,
                                        int32_t dstFracGap, int32_t addrMode,
                                        uint32_t dst_address) {
  load_cbuf_to_cb_transpose(
      reinterpret_cast<__cb__ int8_t *>((uint64_t)dst_address),
      reinterpret_cast<__cbuf__ int8_t *>((uint64_t)src_address),
      (uint16_t)startIndex, (uint8_t)repeatTimes, (uint16_t)srcStride,
      (uint16_t)dstGap, inc, (uint16_t)dstFracGap);
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_data_2d_f16_b(uint32_t src_address, int32_t startIndex,
                                int32_t repeatTimes, int32_t srcStride,
                                int32_t sid, int32_t dstGap,
                                int32_t ifTranspose, int32_t addrMode,
                                uint32_t dst_address) {
  if (ifTranspose) {
    load_cbuf_to_cb(reinterpret_cast<__cb__ half *>((uint64_t)dst_address),
                    reinterpret_cast<__cbuf__ half *>((uint64_t)src_address),
                    (uint16_t)startIndex, (uint8_t)repeatTimes,
                    (uint16_t)srcStride, (uint16_t)dstGap, (uint8_t)sid, 1,
                    inc);
  } else {
    load_cbuf_to_cb(reinterpret_cast<__cb__ half *>((uint64_t)dst_address),
                    reinterpret_cast<__cbuf__ half *>((uint64_t)src_address),
                    (uint16_t)startIndex, (uint8_t)repeatTimes,
                    (uint16_t)srcStride, (uint16_t)dstGap, (uint8_t)sid, 0,
                    inc);
  }
}

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_load_data_3d_f16_a(
    uint32_t src_address, int32_t pad0, int32_t pad1, int32_t pad2,
    int32_t pad3, int32_t l1H, int32_t l1W, int32_t channelSize,
    int32_t kExtension, int32_t mExtension, int32_t kStartPt, int32_t mStartPt,
    int32_t strideW, int32_t strideH, int32_t filterW, int32_t filterH,
    int32_t dilationFilterW, int32_t dilationFilterH, int32_t enTranspose,
    int32_t enSmallK, int32_t padValue, int32_t filterSizeW,
    int32_t filterSizeH, int32_t fMatrixCtrl, int32_t isSetFMatrix,
    int32_t isSetPadding, uint32_t dst_address) {
  if (isSetFMatrix) {
    uint64_t regFMatrix = 0;
    regFMatrix |= uint64_t(l1W & 0xFFFF);
    uint32_t l1HShiftBit = 16;
    regFMatrix |= uint64_t(l1H & 0xFFFF) << l1HShiftBit;
    uint8_t padList[4] = {(uint8_t)pad0, (uint8_t)pad1, (uint8_t)pad2,
                          (uint8_t)pad3};
    uint32_t padNumber = 4;
    uint32_t padListShiftBit = 8;
    uint32_t padListShiftBase = 32;
    for (uint32_t i = 0; i < padNumber; i++) {
      regFMatrix |= uint64_t(padList[i] & 0xFF)
                    << (padListShiftBase + i * padListShiftBit);
    }
    set_fmatrix(regFMatrix);
  }
  if (isSetPadding) {
    uint64_t bits = static_cast<uint64_t>(padValue) & 0xFFFF;
    uint64_t paddingValue = bits | (bits << 16);
    set_padding(paddingValue);
  }
  img2colv2_cbuf_to_ca(reinterpret_cast<__ca__ half *>((uint64_t)dst_address),
                       reinterpret_cast<__cbuf__ half *>((uint64_t)src_address),
                       (uint16_t)kExtension, (uint16_t)mExtension,
                       (uint16_t)kStartPt, (uint16_t)mStartPt, (uint8_t)strideW,
                       (uint8_t)strideH, (uint8_t)filterW, (uint8_t)filterH,
                       (uint8_t)dilationFilterW, (uint8_t)dilationFilterH,
                       (bool)filterSizeW, (bool)filterSizeH, (bool)enTranspose,
                       (bool)fMatrixCtrl, (uint16_t)channelSize);
}
