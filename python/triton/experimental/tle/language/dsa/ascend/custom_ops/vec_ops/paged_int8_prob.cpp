// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_int8_vec_common.h"

namespace {
template <int rows, int width>
__aiv__ __attribute__((always_inline)) void
online_softmax_rows(int score_addr, int work_addr, int sum_now, int sum_prev,
                    int max_now, int max_prev, int alpha_now, int tmp_addr,
                    uint32_t valid, uint32_t first) {
  auto score = ub_ptr<float>(score_addr);
  auto t0 = ub_ptr<float>(work_addr);
  auto t1 = ub_ptr<float>(work_addr + 4096);
  auto t2 = ub_ptr<float>(work_addr + 8192);
  mask_f32_full();
  vmax(t0, score, score + 64, rows, 1, 1, 1, 8, width / 8, width / 8);
  vmax(t1, score + 128, score + 192, rows, 1, 1, 1, 8, width / 8, width / 8);
  vmax(t2, score + 256, score + 320, rows, 1, 1, 1, 8, width / 8, width / 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vcmax(ub_ptr<float>(tmp_addr), t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);

  auto row_max = ub_ptr<float>(tmp_addr);
  auto prev_max = ub_ptr<float>(max_prev);
  auto now_max = ub_ptr<float>(max_now);
  auto safe_max = ub_ptr<float>(tmp_addr + 64);
  auto now_alpha = ub_ptr<float>(alpha_now);
  mask_short<rows>();
  if (first) {
    vadds(now_max, row_max, 0.0f, 1, 1, 1, 8, 8);
  } else {
    vmax(now_max, prev_max, row_max, 1, 1, 1, 1, 8, 8, 8);
  }
  pipe_barrier(PIPE_V);
  vmaxs(safe_max, now_max, -1.0e30f, 1, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  if (!first) {
    vsub(now_alpha, prev_max, safe_max, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vexp(now_alpha, now_alpha, 1, 1, 1, 8, 8);
  }

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
    vmul(now_sum, prev_sum, now_alpha, 1, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(now_sum, now_sum, row_max, 1, 1, 1, 1, 8, 8, 8);
  }
  pipe_barrier(PIPE_V);
  mask_f32_full();
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
                  uint64_t ks, uint64_t table, uint64_t pv, uint32_t visible,
                  uint32_t first, uint32_t slot, float atten_scale,
                  uint32_t head_stride, uint32_t causal_pair) {
  using L = PagedInt8VecLayout<M>;
  mask_f32_full();
  if constexpr (tile == 0) {
    if constexpr (L::kTiles == 2)
      prepare_window_scores<M>(c, r, qs, ks, table, visible, first,
                               head_stride);
  } else if (!first) {
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    copy_gm_to_ubuf(ub_ptr<int32_t>(L::kContent),
                    reinterpret_cast<__gm__ int32_t *>(pv), 0, 1,
                    L::kRows * L::kDim / 8, 0, 0);
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
    if (visible < L::kWidth)
      fill_tail_cols<L::kWidth>(ub_ptr<float>(L::kScales),
                                static_cast<int>(visible), 0.0f, 1);
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
  if (valid < L::kWidth)
    fill_tail_cols<L::kWidth>(score, static_cast<int>(valid), -__builtin_inff(),
                              L::kRows);
  pipe_barrier(PIPE_V);

  online_softmax_rows<L::kRows, L::kWidth>(
      score_addr, L::kScratch, L::kSumBase + now, L::kSumBase + prev,
      L::kMaxBase + now, L::kMaxBase + prev, L::kAlphaBase + now, L::kTmpRows,
      valid, first);

  for (int col = 0; col < L::kWidth; col += 64)
    vmul(score + col, score + col, ub_ptr<float>(L::kScales) + col, L::kRows, 1,
         1, 1, L::kWidth / 8, L::kWidth / 8, 0);
  pipe_barrier(PIPE_V);
  vabs(rope, score, L::kRows * L::kWidth / 64, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);

  auto t0 = ub_ptr<float>(L::kScratch);
  auto t1 = ub_ptr<float>(L::kScratch + 4096);
  auto t2 = ub_ptr<float>(L::kScratch + 8192);
  vmax(t0, rope, rope + 64, L::kRows, 1, 1, 1, 8, L::kWidth / 8, L::kWidth / 8);
  vmax(t1, rope + 128, rope + 192, L::kRows, 1, 1, 1, 8, L::kWidth / 8,
       L::kWidth / 8);
  vmax(t2, rope + 256, rope + 320, L::kRows, 1, 1, 1, 8, L::kWidth / 8,
       L::kWidth / 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t1, L::kRows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmax(t0, t0, t2, L::kRows, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  auto high = ub_ptr<float>(L::kTmpRows);
  vcmax(high, t0, L::kRows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);

  mask_short<L::kRows>();
  vmuls(ub_ptr<float>(L::kStepBase + now), high, 1.0f / 127.0f, 1, 1, 1, 8, 8);
  vmuls(ub_ptr<float>(L::kStepLoBase + now), high, 1.0f / 16129.0f, 1, 1, 1, 8,
        8);
  vmaxs(high, high, 1.0e-30f, 1, 1, 1, 8, 8);
  auto num = ub_ptr<float>(L::kTmpRows + 64);
  vector_dup(num, 127.0f, 1, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);
  vdiv(high, num, high, 1, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);

  mask_f32_full();
  bcast_row_scalars<L::kRows>(t0, high);
  pipe_barrier(PIPE_V);
  mul_row_bcast<L::kRows, L::kWidth>(score, score, t0);
  pipe_barrier(PIPE_V);
  auto halfs = ub_ptr<half>(L::kScratch);
  vconv_f322f16(halfs, score, L::kRows * L::kWidth / 64, 1, 1, 4, 8);
  pipe_barrier(PIPE_V);

  auto p_hi = ub_ptr<int8_t>(L::kPHiStage);
  auto p_lo = ub_ptr<int8_t>(L::kPLoStage);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
  mask_b16_full();
  vconv_f162s8r(p_hi, halfs, L::kRows * L::kWidth / 128, 1, 1, 4, 8);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
  copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(prob) + off, p_hi, 0, 1,
                  L::kRows * L::kWidth / 32, 0, 0);

  auto halfs_hi = reinterpret_cast<__ubuf__ half *>(rope);
  vconv_s82f16(halfs_hi, p_hi, L::kRows * L::kWidth / 128, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);
  vsub(halfs, halfs, halfs_hi, L::kRows * L::kWidth / 128, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmuls(halfs, halfs, static_cast<half>(127.0f), L::kRows * L::kWidth / 128, 1,
        1, 8, 8);
  pipe_barrier(PIPE_V);
  if constexpr (L::kTiles == 2 && tile == 0) {
    if (!first)
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  }
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
  vconv_f162s8r(p_lo, halfs, L::kRows * L::kWidth / 128, 1, 1, 4, 8);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
  copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(prob) +
                      2 * L::kHeads * L::kWidth + off,
                  p_lo, 0, 1, L::kRows * L::kWidth / 32, 0, 0);
  mask_f32_full();
  pipe_barrier(PIPE_V);
}

template <int M>
__aiv__ __attribute__((always_inline)) void
encode_window_prob(uint32_t phase, uint64_t c, uint64_t r, uint64_t prob,
                   uint64_t qs, uint64_t ks, uint64_t table, uint64_t pv,
                   uint32_t visible, uint32_t first, uint32_t slot,
                   float atten_scale, uint32_t head_stride,
                   uint32_t causal_pair) {
  using L = PagedInt8VecLayout<M>;
  if (phase == 0) {
    if constexpr (L::kTiles == 1)
      prepare_window_scores<M>(c, r, qs, ks, table, visible, first,
                               head_stride);
    else
      encode_score_tile<M, 0>(c, r, prob, qs, ks, table, pv, visible, first,
                              slot, atten_scale, head_stride, causal_pair);
    return;
  }
  if constexpr (L::kTiles == 1) {
    if (!first) {
      copy_gm_to_ubuf(ub_ptr<int32_t>(L::kPvStageM16),
                      reinterpret_cast<__gm__ int32_t *>(pv), 0, 1,
                      L::kRows * L::kDim / 8, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    }
    encode_score_tile<M, 0>(c, r, prob, qs, ks, table, pv, visible, first, slot,
                            atten_scale, head_stride, causal_pair);
  } else {
    encode_score_tile<M, 1>(c, r, prob, qs, ks, table, pv, visible, first, slot,
                            atten_scale, head_stride, causal_pair);
  }
}
} // namespace

// Online softmax + two-digit INT8 probability quantization (`P_hi`, `P_lo`)
// over a 384-token paged KV window on Ascend Vector cores (`M` in {16, 32,
// 64}).
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_paged_int8_prob(uint32_t m, uint32_t phase, uint64_t c,
                                    uint64_t r, uint64_t prob, uint64_t qs,
                                    uint64_t ks, uint64_t table, uint64_t pv,
                                    uint32_t visible, uint32_t first,
                                    uint32_t slot, float atten_scale,
                                    uint32_t head_stride,
                                    uint32_t causal_pair) {
  if (m == 16) {
    encode_window_prob<16>(phase, c, r, prob, qs, ks, table, pv, visible, first,
                           slot, atten_scale, head_stride, causal_pair);
  } else if (m == 32) {
    encode_window_prob<32>(phase, c, r, prob, qs, ks, table, pv, visible, first,
                           slot, atten_scale, head_stride, causal_pair);
  } else {
    encode_window_prob<64>(phase, c, r, prob, qs, ks, table, pv, visible, first,
                           slot, atten_scale, head_stride, causal_pair);
  }
}
