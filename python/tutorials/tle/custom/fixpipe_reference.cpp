// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// This fixture has no tiling-key dispatch or per-kernel tiling-key global.
#define TILING_KEY_VAR 0
#include "kernel_operator.h"

// Test-only producer: M=size, N=128, K=128. ND tests split its N panels.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_prepare_fixpipe(uint64_t aAddress, uint64_t bAddress, uint32_t src,
                             uint8_t unitFlag, uint16_t size) {
  auto *a = reinterpret_cast<__cbuf__ int8_t *>(0);
  auto *b = reinterpret_cast<__cbuf__ int8_t *>(16384);
  copy_gm_to_cbuf_multi_nd2nz_b8(a, reinterpret_cast<__gm__ int8_t *>(aAddress),
                                 0, 1, size, 128, 0, 128, size, 1, 0);
  copy_gm_to_cbuf_multi_nd2nz_b8(b, reinterpret_cast<__gm__ int8_t *>(bAddress),
                                 0, 1, 128, 128, 0, 128, 128, 1, 0);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  set_fmatrix(uint64_t(size) | (uint64_t(1) << 16) | (uint64_t(255) << 56));
  img2colv2_cbuf_to_ca(reinterpret_cast<__ca__ int8_t *>(0), a, 128, size, 0, 0,
                       1, 1, 1, 1, 1, 1, false, false, false, false, 128);
  load_cbuf_to_cb(reinterpret_cast<__cb__ int8_t *>(0), b, 0, 32, 1, 0, 0, 0,
                  inc);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  mad(reinterpret_cast<__cc__ int32_t *>(uint64_t(src)),
      reinterpret_cast<__ca__ int8_t *>(0),
      reinterpret_cast<__cb__ int8_t *>(0), size, 128, 128, unitFlag, false,
      false, true);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
}

// The SDK API itself is the reference, including its own field types.
using Params = AscendC::FixpipeParamsV220;
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_reference_fixpipe(
    uint64_t dst, uint32_t src, decltype(Params::nSize) nSize,
    decltype(Params::mSize) mSize, decltype(Params::srcStride) srcStride,
    decltype(Params::dstStride) dstStride, int32_t quantPre,
    decltype(Params::deqScalar) deqScalar, decltype(Params::ndNum) ndNum,
    decltype(Params::srcNdStride) srcNdStride,
    decltype(Params::dstNdStride) dstNdStride, decltype(Params::reluEn) reluEn,
    decltype(Params::unitFlag) unitFlag,
    decltype(Params::isChannelSplit) isChannelSplit) {
  AscendC::GlobalTensor<half> destination;
  destination.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(dst));
  AscendC::LocalTensor<int32_t> source(AscendC::TPosition::CO1, src,
                                       ndNum * srcStride *
                                           ((nSize + 15) / 16 * 16));
  Params params;
  params.nSize = nSize;
  params.mSize = mSize;
  params.srcStride = srcStride;
  params.dstStride = dstStride;
  params.quantPre = static_cast<decltype(Params::quantPre)>(quantPre);
  params.deqScalar = deqScalar;
  params.ndNum = ndNum;
  params.srcNdStride = srcNdStride;
  params.dstNdStride = dstNdStride;
  params.reluEn = reluEn;
  params.unitFlag = unitFlag;
  params.isChannelSplit = isChannelSplit;
  AscendC::Fixpipe<half, int32_t, AscendC::CFG_ROW_MAJOR>(destination, source,
                                                          params);
}
