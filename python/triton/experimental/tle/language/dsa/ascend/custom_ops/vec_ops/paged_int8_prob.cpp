// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_int8_vec_common.h"

namespace {
// FP32 online softmax over `rows x 384` `score` using local row max so
// `exp(score - local_max)` always spans `[0, 1]` without underflow and `lse`
// retains full FP32 accuracy (`< 1e-5`). Writes `beta = exp(local_max -
// safe_max)` to `tmp_addr + 64` and `min(local_sum)` to `tmp_addr + 128`
// (signaling `PIPE_V -> PIPE_S` `EVENT_ID0` for hidden latency).
template <int rows, int width, bool norm_weights = false>
__aiv__ __attribute__((always_inline)) void
online_softmax_rows(int score_addr, int work_addr, int sum_now, int sum_prev,
                    int max_now, int max_prev, int alpha_now, int tmp_addr,
                    uint32_t valid, uint32_t first) {
  constexpr int kTBuf = rows * 64 * 4;
  auto score = ub_ptr<float>(score_addr);
  auto t0 = ub_ptr<float>(work_addr);
  auto t1 = ub_ptr<float>(work_addr + kTBuf);
  auto t2 = ub_ptr<float>(work_addr + 2 * kTBuf);
  auto row_max = ub_ptr<float>(tmp_addr);
  auto prev_max = ub_ptr<float>(max_prev);
  auto now_max = ub_ptr<float>(max_now);
  auto beta_f32 = ub_ptr<float>(tmp_addr + 64);
  auto safe_max = ub_ptr<float>(work_addr + 4096);
  auto now_alpha = ub_ptr<float>(alpha_now);

  mask_f32_full();
  vmax(t0, score, score + 64, rows, 1, 1, 1, 8, width / 8, width / 8);
  vmax(t1, score + 128, score + 192, rows, 1, 1, 1, 8, width / 8, width / 8);
  vmax(t2, score + 256, score + 320, rows, 1, 1, 1, 8, width / 8, width / 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vcmax(row_max, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);

  mask_short<rows>();
  if (first) {
    vadds(now_max, row_max, 0.0f, 1, 1, 1, 8, 8);
    vector_dup(beta_f32, 1.0f, 1, 1, 1, 8, 0);
  } else {
    vmax(now_max, prev_max, row_max, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmaxs(safe_max, now_max, -1.0e30f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(now_alpha, prev_max, safe_max, 1, 1, 1, 1, 8, 8, 8);
    vsub(beta_f32, row_max, safe_max, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vexp(now_alpha, now_alpha, 1, 1, 1, 8, 8);
    vexp(beta_f32, beta_f32, 1, 1, 1, 8, 8);
  }
  vmaxs(safe_max, row_max, -1.0e30f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);

  mask_f32_full();
  bcast_row_scalars<rows>(t0, safe_max);
  pipe_barrier(PIPE_V);
  for (int col = 0; col < width; col += 64)
    vsub(score + col, score + col, t0, rows, 1, 1, 0, width / 8, width / 8, 1);
  pipe_barrier(PIPE_V);
  if (valid < width) {
    vmaxs(score, score, -90.0f, rows * width / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
  }
  vexp(score, score, rows * width / 64, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  if (valid < width) {
    fill_tail_cols<width>(score, static_cast<int>(valid), 0.0f, rows);
    pipe_barrier(PIPE_V);
  }

  vadd(t0, score, score + 64, rows, 1, 1, 1, 8, width / 8, width / 8);
  vadd(t1, score + 128, score + 192, rows, 1, 1, 1, 8, width / 8, width / 8);
  vadd(t2, score + 256, score + 320, rows, 1, 1, 1, 8, width / 8, width / 8);
  pipe_barrier(PIPE_V);
  vadd(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vadd(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vcadd(row_max, t0, rows, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);

  auto now_sum = ub_ptr<float>(sum_now);
  auto prev_sum = ub_ptr<float>(sum_prev);
  mask_short<rows>();
  if (first) {
    vadds(now_sum, row_max, 0.0f, 1, 1, 1, 8, 8);
  } else {
    vmul(safe_max, row_max, beta_f32, 1, 1, 1, 1, 8, 8, 8);
    if constexpr (norm_weights) {
      vmul(now_alpha, prev_sum, now_alpha, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(now_sum, now_alpha, safe_max, 1, 1, 1, 1, 8, 8, 8);
    } else {
      vmul(now_sum, prev_sum, now_alpha, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(now_sum, now_sum, safe_max, 1, 1, 1, 1, 8, 8, 8);
    }
  }
  pipe_barrier(PIPE_V);
  vcmin(ub_ptr<float>(work_addr + 5120), now_sum, 1, 1, 1, 8,
        Order_t::ONLY_VALUE);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  if constexpr (norm_weights) {
    // Precompute `inv_sum = 1 / max(now_sum, 1e-30)` at `work_addr + 5184` and
    // fold `inv_sum` into `now_alpha` (`w_prev = prev_sum * alpha / now_sum`)
    // and `beta_f32` (`16384 * beta / now_sum`) while `prev_sum` is still valid
    // in this ping-pong slot.
    auto inv_sum = ub_ptr<float>(work_addr + 5184);
    vmaxs(safe_max, now_sum, 1.0e-30f, 1, 1, 1, 8, 8);
    vector_dup(inv_sum, 1.0f, 1, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    vdiv(inv_sum, inv_sum, safe_max, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmul(beta_f32, beta_f32, inv_sum, 1, 1, 1, 1, 8, 8, 8);
    if (!first)
      vmul(now_alpha, now_alpha, inv_sum, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(beta_f32, beta_f32, 16384.0f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
  }
  mask_f32_full();
}

// Fast FP16 shifted-8-bit `P_hi` quantization (`round(p * ks_norm * 255 / high)
// - 128 in [-128, 127]`) operating on FP32 `score = exp(score - local_max)` in
// `[0, 1]`. Leaves FP32 `score` intact so a rare peaked-row fallback can run
// directly without re-reading `C`/`R`.
template <int rows, int width>
__aiv__ __attribute__((always_inline)) void
quantize_p_hi_f16(int score_addr, int score_h_addr, int ks_f16_addr,
                  int work_addr, int tmp_addr, int step_now, int p_hi_addr,
                  uint64_t prob_hi_dst, float ks_max, bool wait_mte3) {
  auto score = ub_ptr<float>(score_addr);
  auto score_h = ub_ptr<half>(score_h_addr);
  auto ks_h = ub_ptr<half>(ks_f16_addr);
  auto t0_h = ub_ptr<half>(work_addr);
  auto high_h = ub_ptr<half>(work_addr + 4096);
  auto bcast_h = ub_ptr<half>(work_addr + 4160);
  auto high = ub_ptr<float>(tmp_addr);
  auto beta_f32 = ub_ptr<float>(tmp_addr + 64);

  mask_f32_full();
  vconv_f322f16(score_h, score, rows * width / 64, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);

  mask_b16_full();
  for (int col = 0; col < width; col += 128)
    vmul(score_h + col, score_h + col, ks_h + col, rows, 1, 1, 1, width / 16,
         width / 16, 0);
  pipe_barrier(PIPE_V);

  vmax(t0_h, score_h, score_h + 128, rows, 1, 1, 1, 8, width / 16, width / 16);
  pipe_barrier(PIPE_V);
  vmax(t0_h, t0_h, score_h + 256, rows, 1, 1, 1, 8, 8, width / 16);
  pipe_barrier(PIPE_V);
  vcmax(high_h, t0_h, rows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);

  mask_short<rows>();
  vconv_f162f32(high, high_h, 1, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);
  vmuls(ub_ptr<float>(step_now), high, ks_max / 255.0f, 1, 1, 1, 8, 8);
  vmaxs(high, high, 1.0e-30f, 1, 1, 1, 8, 8);
  auto num = ub_ptr<float>(work_addr + 4672);
  vector_dup(num, 255.0f, 1, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);
  vmul(ub_ptr<float>(step_now), ub_ptr<float>(step_now), beta_f32, 1, 1, 1, 1,
       8, 8, 8);
  vdiv(high, num, high, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322f16(high_h, high, 1, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);

  mask_b16_full();
  bcast_row_scalars_f16<rows>(bcast_h, high_h);
  pipe_barrier(PIPE_V);
  mul_row_bcast_f16<rows, width>(score_h, score_h, bcast_h);
  pipe_barrier(PIPE_V);
  vadds(score_h, score_h, static_cast<half>(-128.0f), rows * width / 128, 1, 1,
        8, 8);
  pipe_barrier(PIPE_V);

  auto p_hi = ub_ptr<int8_t>(p_hi_addr);
  if (wait_mte3) {
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  }
  vconv_f162s8r(p_hi, score_h, rows * width / 128, 1, 1, 4, 8);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(prob_hi_dst), p_hi, 0, 1,
                  rows * width / 32, 0, 0);
}

// Exact FP32 two-digit quantization directly on `score = exp(score -
// local_max)`. When `shifted == true`, uses `[-128, 127]` (`255` levels) to
// match `vsum` (`M >= 64`); when `shifted == false`, uses `[0, 127]` (`127`
// levels, zero `vsum` overhead for `M <= 32`).
template <int rows, int width, bool shifted>
__aiv__ __attribute__((always_inline)) void
quantize_two_digit_fp32(int score_addr, int rope_addr, int ks_f32_addr,
                        int work_addr, int tmp_addr, int step_now,
                        int p_hi_addr, int p_lo_addr, uint64_t prob_hi_dst,
                        uint64_t prob_lo_dst) {
  constexpr int kTBuf = rows * 64 * 4;
  auto score = ub_ptr<float>(score_addr);
  auto rope = ub_ptr<float>(rope_addr);
  auto h_tmp = ub_ptr<half>(work_addr);
  auto ks = ub_ptr<float>(ks_f32_addr);
  auto t0 = ub_ptr<float>(work_addr);
  auto t1 = ub_ptr<float>(work_addr + kTBuf);
  auto t2 = ub_ptr<float>(work_addr + 2 * kTBuf);
  auto high = ub_ptr<float>(tmp_addr);
  auto beta_f32 = ub_ptr<float>(tmp_addr + 64);

  mask_f32_full();
  for (int col = 0; col < width; col += 64)
    vmul(score + col, score + col, ks + col, rows, 1, 1, 1, width / 8,
         width / 8, 0);
  pipe_barrier(PIPE_V);
  vmax(t0, score, score + 64, rows, 1, 1, 1, 8, width / 8, width / 8);
  vmax(t1, score + 128, score + 192, rows, 1, 1, 1, 8, width / 8, width / 8);
  vmax(t2, score + 256, score + 320, rows, 1, 1, 1, 8, width / 8, width / 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vcmax(high, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);

  float max_q = shifted ? 255.0f : 127.0f;
  mask_short<rows>();
  vmuls(ub_ptr<float>(step_now), high, 1.0f / max_q, 1, 1, 1, 8, 8);
  vmaxs(high, high, 1.0e-30f, 1, 1, 1, 8, 8);
  auto num = ub_ptr<float>(work_addr + 4672);
  vector_dup(num, max_q, 1, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);
  vmul(ub_ptr<float>(step_now), ub_ptr<float>(step_now), beta_f32, 1, 1, 1, 1,
       8, 8, 8);
  vdiv(high, num, high, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);

  mask_f32_full();
  bcast_row_scalars<rows>(t0, high);
  pipe_barrier(PIPE_V);
  mul_row_bcast<rows, width>(score, score, t0);
  pipe_barrier(PIPE_V);
  if constexpr (shifted) {
    vadds(score, score, -128.0f, rows * width / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
  }
  vconv_f322f16(h_tmp, score, rows * width / 64, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);

  auto p_hi = ub_ptr<int8_t>(p_hi_addr);
  auto p_lo = ub_ptr<int8_t>(p_lo_addr);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  mask_b16_full();
  vconv_f162s8r(p_hi, h_tmp, rows * width / 128, 1, 1, 4, 8);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(prob_hi_dst), p_hi, 0, 1,
                  rows * width / 32, 0, 0);

  vconv_s82f16(h_tmp, p_hi, rows * width / 128, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);
  mask_f32_full();
  vconv_f162f32(rope, h_tmp, rows * width / 64, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);
  vsub(score, score, rope, rows * width / 64, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmuls(score, score, 127.0f, rows * width / 64, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322f16(h_tmp, score, rows * width / 64, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
  mask_b16_full();
  vconv_f162s8r(p_lo, h_tmp, rows * width / 128, 1, 1, 4, 8);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(prob_lo_dst), p_lo, 0, 1,
                  rows * width / 32, 0, 0);
  mask_f32_full();
  pipe_barrier(PIPE_V);
}

// Computes `ks_max`, `ks_outlier`, and normalized `ks_norm_h = half(ks /
// ks_max)` in `[0, 1]`. Stores `outlier` at `outlier_addr` and `ks_max` at
// `outlier_addr + 4`.
__aiv__ __attribute__((always_inline)) void
compute_ks_outlier_and_norm_f16(int ks_f32_addr, int ks_f16_addr,
                                int scratch_addr, int tmp_addr,
                                int outlier_addr, uint32_t visible) {
  auto ks = ub_ptr<float>(ks_f32_addr);
  auto ks_h = ub_ptr<half>(ks_f16_addr);
  auto t0 = ub_ptr<float>(scratch_addr);
  auto t1 = ub_ptr<float>(scratch_addr + 256);
  auto t2 = ub_ptr<float>(scratch_addr + 512);
  auto tmp = ub_ptr<float>(tmp_addr);
  mask_f32_full();
  if (visible < 384)
    fill_tail_cols<384>(ks, static_cast<int>(visible), 0.0f, 1);
  vmax(t0, ks, ks + 64, 1, 1, 1, 1, 8, 8, 8);
  vmax(t1, ks + 128, ks + 192, 1, 1, 1, 1, 8, 8, 8);
  vmax(t2, ks + 256, ks + 320, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t1, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t2, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vcmax(tmp, t0, 1, 1, 1, 8, Order_t::ONLY_VALUE);

  vadd(t0, ks, ks + 64, 1, 1, 1, 1, 8, 8, 8);
  vadd(t1, ks + 128, ks + 192, 1, 1, 1, 1, 8, 8, 8);
  vadd(t2, ks + 256, ks + 320, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vadd(t0, t0, t1, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vadd(t0, t0, t2, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vcadd(tmp + 8, t0, 1, 1, 1, 8, 0);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  float ks_max = tmp[0];
  float ks_sum = tmp[8];
  int32_t eff = static_cast<int32_t>(visible < 384 ? visible : 384);
  int32_t outlier = (ks_max * static_cast<float>(eff) > 4.0f * ks_sum) ? 1 : 0;
  *ub_ptr<int32_t>(outlier_addr) = outlier;
  *ub_ptr<float>(outlier_addr + 4) = ks_max;
  float inv_ks = ks_max > 1.0e-30f ? (1.0f / ks_max) : 0.0f;
  vmuls(t0, ks, inv_ks, 384 / 64, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322f16(ks_h, t0, 384 / 64, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);
}

template <int M>
__aiv__ __attribute__((always_inline)) void
prepare_window_scores(uint64_t c, uint64_t r, uint64_t qs, uint64_t ks,
                      uint64_t table, uint32_t visible, uint32_t first,
                      uint32_t head_stride) {
  using L = PagedInt8VecLayout<M>;
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  if (first) {
    copy_gm_to_ubuf(ub_ptr<float>(L::kQuery),
                    reinterpret_cast<__gm__ float *>(qs), 0, L::kTiles,
                    L::kRows / 8, (head_stride - L::kRows) / 8, 0);
  }
  auto ids = reinterpret_cast<__gm__ const int32_t *>(table);
  uint32_t pages = (visible + 63) / 64;
  for (int page = 0; page < 6; ++page) {
    uint64_t id = ids[static_cast<uint32_t>(page) < pages ? page : pages - 1];
    copy_gm_to_ubuf(ub_ptr<float>(L::kScales) + page * 64,
                    reinterpret_cast<__gm__ float *>(ks) + id * 64, 0, 1, 8, 0,
                    0);
  }
  for (int part = 0; part < L::kTiles; ++part) {
    int off = part * L::kRows * L::kWidth;
    copy_gm_to_ubuf(ub_ptr<int32_t>(L::kContent + part * L::kTileBytes),
                    reinterpret_cast<__gm__ int32_t *>(c) + off, 0, 1,
                    L::kRows * L::kWidth / 8, 0, 0);
    copy_gm_to_ubuf(
        ub_ptr<float>(L::kContent + part * L::kTileBytes + L::kTileBytes / 2),
        reinterpret_cast<__gm__ float *>(r) + off, 0, 1,
        L::kRows * L::kWidth / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, tile_event(part));
  }
}

template <int M, int tile>
__aiv__ __attribute__((always_inline)) void
encode_score_tile(uint64_t c, uint64_t r, uint64_t prob, uint64_t qs,
                  uint64_t ks, uint64_t table, uint64_t pv, uint64_t flag_gm,
                  uint32_t visible, uint32_t first, uint32_t slot,
                  float atten_scale, uint32_t head_stride,
                  uint32_t causal_pair) {
  using L = PagedInt8VecLayout<M>;
  mask_f32_full();
  if constexpr (tile == 0) {
    if constexpr (L::kTiles == 2)
      prepare_window_scores<M>(c, r, qs, ks, table, visible, first,
                               head_stride);
    else if (!first) {
      copy_gm_to_ubuf(ub_ptr<int32_t>(L::kPvStageM16),
                      reinterpret_cast<__gm__ int32_t *>(pv), 0, 1,
                      L::kRows * L::kDim / 8, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    }
  } else if (!first) {
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    if constexpr (M >= 64) {
      copy_gm_to_ubuf(ub_ptr<half>(L::kContent),
                      reinterpret_cast<__gm__ half *>(pv), 0, 1,
                      L::kRows * L::kDim / 16, 0, 0);
    } else {
      copy_gm_to_ubuf(ub_ptr<int32_t>(L::kContent),
                      reinterpret_cast<__gm__ int32_t *>(pv), 0, 1,
                      L::kRows * L::kDim / 8, 0, 0);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
  }

  int lane = tile * L::kRows;
  int off = lane * L::kWidth;
  int now = (slot * L::kHeads + lane) * 4;
  int prev = ((1 - slot) * L::kHeads + lane) * 4;
  int score_addr = L::kContent + tile * L::kTileBytes;
  int rope_addr = score_addr + L::kTileBytes / 2;
  auto score = ub_ptr<float>(score_addr);
  auto rope = ub_ptr<float>(rope_addr);
  wait_flag(PIPE_MTE2, PIPE_V, tile_event(tile));

  if constexpr (tile == 0) {
    if (flag_gm != 0) {
      compute_ks_outlier_and_norm_f16(L::kScales, L::kScalesF16, L::kScratch,
                                      L::kTmpRows, L::kKsOutlier, visible);
    } else if (visible < L::kWidth) {
      fill_tail_cols<L::kWidth>(ub_ptr<float>(L::kScales),
                                static_cast<int>(visible), 0.0f, 1);
    }
    if (first) {
      mask_short<L::kHeads>();
      vmuls(ub_ptr<float>(L::kQuery), ub_ptr<float>(L::kQuery), atten_scale, 1,
            1, 1, 8, 8);
      vector_dup(ub_ptr<float>(L::kSumBase + L::kHeads * 4), 0.0f, 1, 1, 1, 8,
                 0);
      vector_dup(ub_ptr<float>(L::kMaxBase + L::kHeads * 4), -__builtin_inff(),
                 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
      mask_f32_full();
      bcast_row_scalars<L::kRows>(ub_ptr<float>(L::kBcast),
                                  ub_ptr<float>(L::kQuery));
      if constexpr (L::kTiles == 2)
        bcast_row_scalars<L::kRows>(ub_ptr<float>(L::kBcast) + L::kRows * 8,
                                    ub_ptr<float>(L::kQuery) + L::kRows);
      pipe_barrier(PIPE_V);
    }
  }

  vconv_s322f32(score, reinterpret_cast<__ubuf__ int32_t *>(score),
                L::kRows * L::kWidth / 64, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vadd(score, score, rope, L::kRows * L::kWidth / 64, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  mul_row_bcast<L::kRows, L::kWidth>(score, score,
                                     ub_ptr<float>(L::kBcast) + lane * 8);
  pipe_barrier(PIPE_V);
  for (int col = 0; col < L::kWidth; col += 64)
    vmul(score + col, score + col, ub_ptr<float>(L::kScales) + col, L::kRows, 1,
         1, 1, L::kWidth / 8, L::kWidth / 8, 0);
  pipe_barrier(PIPE_V);

  uint32_t valid = visible - (tile == 0 ? causal_pair : 0);
  if (valid < L::kWidth) {
    fill_tail_cols<L::kWidth>(score, static_cast<int>(valid), -__builtin_inff(),
                              L::kRows);
    pipe_barrier(PIPE_V);
  }

  online_softmax_rows<L::kRows, L::kWidth, (M >= 64)>(
      score_addr, L::kScratch, L::kSumBase + now, L::kSumBase + prev,
      L::kMaxBase + now, L::kMaxBase + prev, L::kAlphaBase + now, L::kTmpRows,
      valid, first);

  uint64_t prob_hi_dst = prob + off;
  uint64_t prob_lo_dst = prob + 2 * L::kHeads * L::kWidth + off;
  bool tile_need_lo = false;
  if (flag_gm == 0) {
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    quantize_two_digit_fp32<L::kRows, L::kWidth, false>(
        score_addr, rope_addr, L::kScales, L::kScratch, L::kTmpRows,
        L::kStepBase + now, L::kPHiStage, L::kPLoStage, prob_hi_dst,
        prob_lo_dst);
    tile_need_lo = true;
  } else if (*ub_ptr<int32_t>(L::kKsOutlier) != 0) {
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    quantize_two_digit_fp32<L::kRows, L::kWidth, true>(
        score_addr, rope_addr, L::kScales, L::kScratch, L::kTmpRows,
        L::kStepBase + now, L::kPHiStage, L::kPLoStage, prob_hi_dst,
        prob_lo_dst);
    tile_need_lo = true;
  } else {
    float ks_max = *ub_ptr<float>(L::kKsOutlier + 4);
    quantize_p_hi_f16<L::kRows, L::kWidth>(
        score_addr, rope_addr, L::kScalesF16, L::kScratch, L::kTmpRows,
        L::kStepBase + now, L::kPHiStage, prob_hi_dst, ks_max, true);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float min_local_sum = *ub_ptr<float>(L::kScratch + 5120);
    if ((valid > 4) && (min_local_sum < 2.0f)) {
      quantize_two_digit_fp32<L::kRows, L::kWidth, true>(
          score_addr, rope_addr, L::kScales, L::kScratch, L::kTmpRows,
          L::kStepBase + now, L::kPHiStage, L::kPLoStage, prob_hi_dst,
          prob_lo_dst);
      tile_need_lo = true;
    }
  }

  if constexpr (L::kTiles == 2 && tile == 0) {
    if (!first)
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  }

  auto need_ub = ub_ptr<int32_t>(L::kNeedLoBase + slot * 32);
  int32_t cur_mask = tile_need_lo ? (1 << tile) : 0;
  if constexpr (tile == 1) {
    cur_mask |= *reinterpret_cast<volatile __ubuf__ int32_t *>(need_ub);
  }
  mask_short<8>();
  vector_dup(need_ub, cur_mask, 1, 1, 1, 8, 0);
  if constexpr (tile + 1 == L::kTiles) {
    if (flag_gm != 0) {
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ int32_t *>(flag_gm), need_ub, 0,
                      1, 1, 0, 0);
    }
  } else {
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  }
  mask_f32_full();
  pipe_barrier(PIPE_V);
}

// `M == 128` (`kRows = 16`, `kTiles = 4`, full 2-bank ping-pong `MTE2` + `V` +
// `MTE3` over FP32 `C`/`R` + FP16 `P_hi` quantization, plus FP16 `prior_pv[0]`
// prefetch during `tile == 3`).
__aiv__ __attribute__((always_inline)) void
encode_window_prob_128(uint32_t phase, uint64_t c, uint64_t r, uint64_t prob,
                       uint64_t qs, uint64_t ks, uint64_t table, uint64_t pv,
                       uint64_t flag_gm, uint32_t visible, uint32_t first,
                       uint32_t slot, float atten_scale, uint32_t head_stride,
                       uint32_t causal_pair) {
  using L = PagedInt8VecLayout<128>;
  mask_f32_full();
  if (phase == 0) {
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    if (first) {
      copy_gm_to_ubuf(ub_ptr<float>(L::kQuery),
                      reinterpret_cast<__gm__ float *>(qs), 0, 2, 32 / 8,
                      (head_stride - 32) / 8, 0);
    }
    auto ids = reinterpret_cast<__gm__ const int32_t *>(table);
    uint32_t pages = (visible + 63) / 64;
    for (int page = 0; page < 6; ++page) {
      uint64_t id = ids[static_cast<uint32_t>(page) < pages ? page : pages - 1];
      copy_gm_to_ubuf(ub_ptr<float>(L::kScales) + page * 64,
                      reinterpret_cast<__gm__ float *>(ks) + id * 64, 0, 1, 8,
                      0, 0);
    }
    copy_gm_to_ubuf(ub_ptr<int32_t>(L::kContent),
                    reinterpret_cast<__gm__ int32_t *>(c), 0, 1,
                    L::kRows * L::kWidth / 8, 0, 0);
    copy_gm_to_ubuf(ub_ptr<float>(L::kContent + L::kTileBytes / 2),
                    reinterpret_cast<__gm__ float *>(r), 0, 1,
                    L::kRows * L::kWidth / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  }

  int start_tile = phase == 0 ? 0 : 2;
  int end_tile = phase == 0 ? 2 : 4;
  int32_t acc_mask = 0;
  auto need_ub = ub_ptr<int32_t>(L::kNeedLoBase + slot * 32);
  if (phase == 1) {
    acc_mask = *reinterpret_cast<volatile __ubuf__ int32_t *>(need_ub);
  }

  for (int tile = start_tile; tile < end_tile; ++tile) {
    int bank = tile & 1;
    int next_tile = tile + 1;
    int next_bank = next_tile & 1;
    if (next_tile < 4) {
      if (tile > start_tile) {
        set_flag(PIPE_V, PIPE_MTE2, tile_event(next_bank));
        wait_flag(PIPE_V, PIPE_MTE2, tile_event(next_bank));
      }
      if (next_tile >= 2) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, tile_event(next_bank));
      }
      int next_off = next_tile * L::kRows * L::kWidth;
      copy_gm_to_ubuf(ub_ptr<int32_t>(L::kContent + next_bank * L::kTileBytes),
                      reinterpret_cast<__gm__ int32_t *>(c) + next_off, 0, 1,
                      L::kRows * L::kWidth / 8, 0, 0);
      copy_gm_to_ubuf(ub_ptr<float>(L::kContent + next_bank * L::kTileBytes +
                                    L::kTileBytes / 2),
                      reinterpret_cast<__gm__ float *>(r) + next_off, 0, 1,
                      L::kRows * L::kWidth / 8, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, tile_event(next_bank));
    } else if (!first) {
      // `tile == 3` (`bank == 1`): bank 0 (`65536..98304`, 32 KiB) is free and
      // disjoint from tile 2's `p_hi` (`65536 + 36864..43008`), so prefetch
      // FP16 `prior_pv[0]` (`32 x 512` FP16) now.
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      copy_gm_to_ubuf(ub_ptr<half>(L::kContent),
                      reinterpret_cast<__gm__ half *>(pv), 0, 1,
                      L::kUpdRows * L::kDim / 16, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    }

    wait_flag(PIPE_MTE2, PIPE_V, tile_event(bank));

    if (tile == 0) {
      compute_ks_outlier_and_norm_f16(L::kScales, L::kScalesF16, L::kScratch,
                                      L::kTmpRows, L::kKsOutlier, visible);
      if (first) {
        mask_short<L::kHeads>();
        vmuls(ub_ptr<float>(L::kQuery), ub_ptr<float>(L::kQuery), atten_scale,
              1, 1, 1, 8, 8);
        vector_dup(ub_ptr<float>(L::kSumBase + L::kHeads * 4), 0.0f, 1, 1, 1, 8,
                   0);
        vector_dup(ub_ptr<float>(L::kMaxBase + L::kHeads * 4),
                   -__builtin_inff(), 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        mask_f32_full();
        bcast_row_scalars<L::kHeads>(ub_ptr<float>(L::kBcast),
                                     ub_ptr<float>(L::kQuery));
        pipe_barrier(PIPE_V);
      }
    }

    int lane = tile * L::kRows;
    int off = lane * L::kWidth;
    int now = (slot * L::kHeads + lane) * 4;
    int prev = ((1 - slot) * L::kHeads + lane) * 4;
    int score_addr = L::kContent + bank * L::kTileBytes;
    int rope_addr = score_addr + L::kTileBytes / 2;
    auto score = ub_ptr<float>(score_addr);
    auto rope = ub_ptr<float>(rope_addr);

    mask_f32_full();
    vconv_s322f32(score, reinterpret_cast<__ubuf__ int32_t *>(score),
                  L::kRows * L::kWidth / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(score, score, rope, L::kRows * L::kWidth / 64, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    mul_row_bcast<L::kRows, L::kWidth>(score, score,
                                       ub_ptr<float>(L::kBcast) + lane * 8);
    pipe_barrier(PIPE_V);
    for (int col = 0; col < L::kWidth; col += 64)
      vmul(score + col, score + col, ub_ptr<float>(L::kScales) + col, L::kRows,
           1, 1, 1, L::kWidth / 8, L::kWidth / 8, 0);
    pipe_barrier(PIPE_V);

    uint32_t valid = visible - (tile < 2 ? causal_pair : 0);
    if (valid < L::kWidth) {
      fill_tail_cols<L::kWidth>(score, static_cast<int>(valid),
                                -__builtin_inff(), L::kRows);
      pipe_barrier(PIPE_V);
    }

    online_softmax_rows<L::kRows, L::kWidth, true>(
        score_addr, L::kScratch, L::kSumBase + now, L::kSumBase + prev,
        L::kMaxBase + now, L::kMaxBase + prev, L::kAlphaBase + now, L::kTmpRows,
        valid, first);

    uint64_t prob_hi_dst = prob + off;
    uint64_t prob_lo_dst = prob + 2 * L::kHeads * L::kWidth + off;
    int p_hi_addr = rope_addr + 12288;
    bool tile_need_lo = false;
    if (flag_gm == 0) {
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      quantize_two_digit_fp32<L::kRows, L::kWidth, false>(
          score_addr, rope_addr, L::kScales, L::kScratch, L::kTmpRows,
          L::kStepBase + now, L::kPHiStage, L::kPHiStage, prob_hi_dst,
          prob_lo_dst);
      tile_need_lo = true;
    } else if (*ub_ptr<int32_t>(L::kKsOutlier) != 0) {
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      quantize_two_digit_fp32<L::kRows, L::kWidth, true>(
          score_addr, rope_addr, L::kScales, L::kScratch, L::kTmpRows,
          L::kStepBase + now, L::kPHiStage, L::kPHiStage, prob_hi_dst,
          prob_lo_dst);
      tile_need_lo = true;
    } else {
      float ks_max = *ub_ptr<float>(L::kKsOutlier + 4);
      quantize_p_hi_f16<L::kRows, L::kWidth>(
          score_addr, rope_addr, L::kScalesF16, L::kScratch, L::kTmpRows,
          L::kStepBase + now, p_hi_addr, prob_hi_dst, ks_max, false);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float min_local_sum = *ub_ptr<float>(L::kScratch + 5120);
      if ((valid > 4) && (min_local_sum < 2.0f)) {
        quantize_two_digit_fp32<L::kRows, L::kWidth, true>(
            score_addr, rope_addr, L::kScales, L::kScratch, L::kTmpRows,
            L::kStepBase + now, L::kPHiStage, L::kPHiStage, prob_hi_dst,
            prob_lo_dst);
        tile_need_lo = true;
      }
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, tile_event(bank));
    if (tile_need_lo)
      acc_mask |= (1 << tile);
  }

  mask_short<8>();
  vector_dup(need_ub, acc_mask, 1, 1, 1, 8, 0);
  if (phase == 1) {
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    if (flag_gm != 0) {
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ int32_t *>(flag_gm), need_ub, 0,
                      1, 1, 0, 0);
    }
  } else {
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  }
  mask_f32_full();
  pipe_barrier(PIPE_V);
}
} // namespace

// Paged 384-token FP32 score scaling, online softmax, and adaptive FP16
// shifted-8-bit (`[-128, 127]`) / FP32 two-digit INT8 probability quantization
// on Ascend Vector cores (`M` in {16, 32, 64, 128}).
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_paged_int8_prob(uint32_t m, uint32_t phase, uint64_t c,
                                    uint64_t r, uint64_t prob, uint64_t qs,
                                    uint64_t ks, uint64_t table, uint64_t pv,
                                    uint64_t flag_gm, uint32_t visible,
                                    uint32_t first, uint32_t slot,
                                    float atten_scale, uint32_t head_stride,
                                    uint32_t causal_pair) {
  if (m == 16) {
    if (phase == 0) {
      prepare_window_scores<16>(c, r, qs, ks, table, visible, first,
                                head_stride);
    } else {
      encode_score_tile<16, 0>(c, r, prob, qs, ks, table, pv, flag_gm, visible,
                               first, slot, atten_scale, head_stride,
                               causal_pair);
    }
  } else if (m == 32) {
    if (phase == 0) {
      encode_score_tile<32, 0>(c, r, prob, qs, ks, table, pv, flag_gm, visible,
                               first, slot, atten_scale, head_stride,
                               causal_pair);
    } else {
      encode_score_tile<32, 1>(c, r, prob, qs, ks, table, pv, flag_gm, visible,
                               first, slot, atten_scale, head_stride,
                               causal_pair);
    }
  } else if (m == 64) {
    if (phase == 0) {
      encode_score_tile<64, 0>(c, r, prob, qs, ks, table, pv, flag_gm, visible,
                               first, slot, atten_scale, head_stride,
                               causal_pair);
    } else {
      encode_score_tile<64, 1>(c, r, prob, qs, ks, table, pv, flag_gm, visible,
                               first, slot, atten_scale, head_stride,
                               causal_pair);
    }
  } else if (m == 128) {
    encode_window_prob_128(phase, c, r, prob, qs, ks, table, pv, flag_gm,
                           visible, first, slot, atten_scale, head_stride,
                           causal_pair);
  }
}
