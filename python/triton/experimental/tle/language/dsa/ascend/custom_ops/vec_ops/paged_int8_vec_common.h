// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef TLE_ASCEND_CUSTOM_OPS_PAGED_INT8_VEC_COMMON_H
#define TLE_ASCEND_CUSTOM_OPS_PAGED_INT8_VEC_COMMON_H

#include "Utils.h"

namespace {
template <typename T>
__aiv__ __attribute__((always_inline)) __ubuf__ T *ub_ptr(uint32_t bytes) {
  return reinterpret_cast<__ubuf__ T *>(
      reinterpret_cast<__ubuf__ uint8_t *>(get_imm(0)) + bytes);
}

__aiv__ __attribute__((always_inline)) event_t tile_event(int idx) {
  return (idx & 1) == 0 ? EVENT_ID3 : EVENT_ID4;
}

__aiv__ __attribute__((always_inline)) void mask_f32_full() {
  set_mask_norm();
  set_vector_mask(0, ~uint64_t(0));
}

__aiv__ __attribute__((always_inline)) void mask_b16_full() {
  set_mask_norm();
  set_vector_mask(~uint64_t(0), ~uint64_t(0));
}

template <int count> __aiv__ __attribute__((always_inline)) void mask_short() {
  static_assert(count > 0 && count <= 64);
  set_mask_norm();
  constexpr uint64_t lo =
      count == 64 ? ~uint64_t(0) : ((uint64_t(1) << (count & 63)) - 1);
  set_vector_mask(0, lo);
}

template <int rows, int cols>
__aiv__ __attribute__((always_inline)) void
mul_row_bcast(__ubuf__ float *dst, __ubuf__ float *src,
              __ubuf__ float *bcast_scalar) {
  for (int c = 0; c < cols; c += 64)
    vmul(dst + c, src + c, bcast_scalar, rows, 1, 1, 0, cols / 8, cols / 8, 1);
}

template <int rows, int cols>
__aiv__ __attribute__((always_inline)) void
mla_row_bcast(__ubuf__ float *dst, __ubuf__ float *src,
              __ubuf__ float *bcast_scalar) {
  for (int c = 0; c < cols; c += 64)
    vmla(dst + c, src + c, bcast_scalar, rows, 1, 1, 0, cols / 8, cols / 8, 1);
}

template <int rows, int cols>
__aiv__ __attribute__((always_inline)) void
mul_row_bcast_f16(__ubuf__ half *dst, __ubuf__ half *src,
                  __ubuf__ half *bcast_scalar) {
  for (int c = 0; c < cols; c += 128)
    vmul(dst + c, src + c, bcast_scalar, rows, 1, 1, 0, cols / 16, cols / 16,
         1);
}

template <int rows, int cols>
__aiv__ __attribute__((always_inline)) void
mla_row_bcast_f16(__ubuf__ half *dst, __ubuf__ half *src,
                  __ubuf__ half *bcast_scalar) {
  for (int c = 0; c < cols; c += 128)
    vmla(dst + c, src + c, bcast_scalar, rows, 1, 1, 0, cols / 16, cols / 16,
         1);
}

template <int rows, int cols>
__aiv__ __attribute__((always_inline)) void
sub_row_bcast_f16(__ubuf__ half *dst, __ubuf__ half *src,
                  __ubuf__ half *bcast_scalar) {
  for (int c = 0; c < cols; c += 128)
    vsub(dst + c, src + c, bcast_scalar, rows, 1, 1, 0, cols / 16, cols / 16,
         1);
}

template <int rows>
__aiv__ __attribute__((always_inline)) void
bcast_row_scalars(__ubuf__ float *dst, __ubuf__ float *src) {
  vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(dst),
        reinterpret_cast<__ubuf__ uint32_t *>(src), 1, 8, rows / 8);
}

template <int rows>
__aiv__ __attribute__((always_inline)) void
bcast_row_scalars_f16(__ubuf__ half *dst, __ubuf__ half *src) {
  vbrcb(reinterpret_cast<__ubuf__ uint16_t *>(dst),
        reinterpret_cast<__ubuf__ uint16_t *>(src), 1, 8, rows / 8);
}

template <int width>
__aiv__ __attribute__((always_inline)) void
fill_tail_cols(__ubuf__ float *dst, int valid, float fill, int num_rows) {
  for (int c = 0; c < width; c += 64) {
    if (valid >= c + 64)
      continue;
    uint64_t m = ~uint64_t(0);
    if (valid > c)
      m <<= (valid - c);
    set_vector_mask(0, m);
    vector_dup(dst + c, fill, num_rows, 1, 1, width / 8, 0);
  }
  mask_f32_full();
}

template <int width>
__aiv__ __attribute__((always_inline)) void
fill_tail_cols_f16(__ubuf__ half *dst, int valid, half fill, int num_rows) {
  for (int c = 0; c < width; c += 128) {
    if (valid >= c + 128)
      continue;
    uint64_t lo = ~uint64_t(0);
    uint64_t hi = ~uint64_t(0);
    if (valid > c) {
      int rem = valid - c;
      if (rem < 64) {
        lo <<= rem;
      } else {
        lo = 0;
        hi <<= (rem - 64);
      }
    }
    set_vector_mask(hi, lo);
    vector_dup(dst + c, fill, num_rows, 1, 1, width / 16, 0);
  }
  mask_b16_full();
}

template <int M> struct PagedInt8VecLayout {
  static constexpr int kRows = M == 16 ? 8 : M / 4;
  static constexpr int kHeads = M / 2;
  static constexpr int kWidth = 384;
  static constexpr int kDim = 512;
  static constexpr int kTiles = kHeads / kRows;
  static constexpr int kPvStageM16 = 32768;
  static_assert(M != 16 || (kHeads * kDim * 4 <= kPvStageM16 &&
                            kPvStageM16 + kHeads * kDim * 4 <= 65536));
  static constexpr int kAccum = 0;
  static constexpr int kContent = 65536;
  static constexpr int kRotary = 114688;
  static constexpr int kPvLoStage = 131072;
  static constexpr int kScratch = 163840;
  static constexpr int kScalesF16 = kScratch + 15360; // 768 B
  static constexpr int kScales = 180224;
  static constexpr int kQuery = kScales + kWidth * 4;
  static constexpr int kBcast = kQuery + kHeads * 4;
  static constexpr int kSumBase = kBcast + kHeads * 32;
  static constexpr int kMaxBase = kSumBase + kHeads * 2 * 4;
  static constexpr int kAlphaBase = kMaxBase + kHeads * 2 * 4;
  static constexpr int kStepBase = kAlphaBase + kHeads * 2 * 4;
  static constexpr int kTmpRows = kStepBase + kHeads * 2 * 4;
  static constexpr int kNeedLoBase = kTmpRows + 128;
  static constexpr int kKsOutlier = kNeedLoBase + 64;
  static constexpr int kTileBytes = 2 * kRows * kWidth * 4;
  static constexpr int kPHiStage = 184320;
  static constexpr int kPLoStage = 190464;
  static_assert(kKsOutlier + 32 <= kPHiStage);
  static_assert(kPLoStage + kRows * kWidth <= 192 * 1024);
};

// `M == 128` uses running-normalized FP16 `kAccum` (64 KiB), freeing 64 KiB UB
// so `Prob` runs 2-bank ping-pong at `kRows = 16` (`kTiles = 4`, 48 KiB/bank
// for FP32 `C` + FP32 `R`) and `Update` runs 2-bank ping-pong at `kUpdRows =
// 32` (`kUpdTiles = 2`, 32 KiB/bank for FP16 `PV`).
template <> struct PagedInt8VecLayout<128> {
  static constexpr int kRows = 16;
  static constexpr int kHeads = 64;
  static constexpr int kWidth = 384;
  static constexpr int kDim = 512;
  static constexpr int kTiles = 4;
  static constexpr int kUpdRows = 32;
  static constexpr int kUpdTiles = 2;
  static constexpr int kAccum = 0; // 64 x 512 FP16 = 64 KiB
  static constexpr int kContent =
      65536; // 2 banks x 48 KiB (Prob) / 2 banks x 32 KiB (Update)
  static constexpr int kTileBytes =
      2 * kRows * kWidth * 4; // 48 KiB (24 KiB C + 24 KiB R)
  static constexpr int kPvBankBytes =
      kUpdRows * kDim * 2; // 32 KiB (32 x 512 FP16)
  static constexpr int kPvLoStage =
      131072; // 32 x 512 FP16 = 32 KiB (Update rare 2-digit fallback)
  static constexpr int kScratch = 163840; // 15 KiB (3 x 4 KiB t0/t1/t2 + bcast)
  static constexpr int kScales = 179200;  // 384 FP32 = 1536 B
  static constexpr int kScalesF16 = kScales + kWidth * 4; // 384 FP16 = 768 B
  static constexpr int kQuery = kScalesF16 + kWidth * 2;  // 64 FP32 = 256 B
  static constexpr int kBcast = kQuery + kHeads * 4;    // 64 x 8 FP32 = 2048 B
  static constexpr int kSumBase = kBcast + kHeads * 32; // 512 B
  static constexpr int kMaxBase = kSumBase + kHeads * 2 * 4;    // 512 B
  static constexpr int kAlphaBase = kMaxBase + kHeads * 2 * 4;  // 512 B
  static constexpr int kStepBase = kAlphaBase + kHeads * 2 * 4; // 512 B
  static constexpr int kTmpRows = kStepBase + kHeads * 2 * 4;   // 128 B
  static constexpr int kNeedLoBase = kTmpRows + 128;            // 64 B
  static constexpr int kKsOutlier = kNeedLoBase + 64;           // 32 B
  static constexpr int kPHiStage = 186112; // 16 x 384 INT8 = 6144 B (fallback)
  static_assert(kContent + 2 * kTileBytes == kScratch);
  static_assert(kContent + 2 * kPvBankBytes == kPvLoStage);
  static_assert(kPvLoStage + kPvBankBytes == kScratch);
  static_assert(kScratch + 15360 == kScales);
  static_assert(kKsOutlier + 32 <= kPHiStage);
  static_assert(kPHiStage + kRows * kWidth <= 192 * 1024);
};
} // namespace

#endif // TLE_ASCEND_CUSTOM_OPS_PAGED_INT8_VEC_COMMON_H
