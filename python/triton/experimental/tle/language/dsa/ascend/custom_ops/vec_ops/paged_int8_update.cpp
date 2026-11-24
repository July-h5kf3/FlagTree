// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_int8_vec_common.h"

namespace {
template <int M>
__aiv__ __attribute__((always_inline)) void
update_window_accum(uint64_t pv, uint64_t out, uint64_t lse, uint32_t slot,
                    uint32_t first, uint32_t last, uint32_t empty,
                    uint32_t head_stride, uint32_t first_empty) {
  using L = PagedInt8VecLayout<M>;
  if (empty) {
    mask_short<L::kHeads>();
    vector_dup(ub_ptr<float>(L::kContent), __builtin_inff(), 1, 1, 1, 8, 0);
    mask_b16_full();
    vector_dup(ub_ptr<int16_t>(L::kRotary), static_cast<int16_t>(0),
               L::kHeads * L::kDim / 128, 1, 1, 8, 0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ float *>(lse),
                    ub_ptr<float>(L::kContent), 0, 1, L::kHeads / 8, 0, 0);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ int16_t *>(out),
                    ub_ptr<int16_t>(L::kRotary), 0, L::kTiles,
                    L::kRows * L::kDim / 16, 0,
                    (head_stride - L::kRows) * L::kDim / 16);
    mask_f32_full();
    pipe_barrier(PIPE_ALL);
    return;
  }

  mask_f32_full();
  constexpr int kPvBytes = L::kRows * L::kDim * 4;
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  copy_gm_to_ubuf(ub_ptr<int32_t>(L::kPvLoStage),
                  reinterpret_cast<__gm__ int32_t *>(pv) +
                      2 * L::kHeads * L::kDim,
                  0, 1, L::kRows * L::kDim / 8, 0, 0);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);

  for (int tile = 0; tile < L::kTiles; ++tile) {
    int off = tile * L::kRows * L::kDim;
    if (!last && tile == 0)
      continue;
    copy_gm_to_ubuf(ub_ptr<int32_t>(L::kTiles == 1
                                        ? L::kPvStageM16
                                        : L::kContent + tile * kPvBytes),
                    reinterpret_cast<__gm__ int32_t *>(pv) + off, 0, 1,
                    L::kRows * L::kDim / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, tile_event(tile));
  }

  for (int tile = 0; tile < L::kTiles; ++tile) {
    int lane = tile * L::kRows;
    int off = lane * L::kDim;
    int now = (slot * L::kHeads + lane) * 4;
    int stage_addr =
        L::kTiles == 1 ? L::kPvStageM16 : L::kContent + tile * kPvBytes;
    auto upd_hi = ub_ptr<float>(stage_addr);
    auto acc = ub_ptr<float>(L::kAccum) + off;
    auto factor = ub_ptr<float>(L::kScratch);

    if (tile == 0 && !last)
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    else
      wait_flag(PIPE_MTE2, PIPE_V, tile_event(tile));

    if (tile > 0) {
      copy_gm_to_ubuf(ub_ptr<int32_t>(L::kPvLoStage),
                      reinterpret_cast<__gm__ int32_t *>(pv) +
                          2 * L::kHeads * L::kDim + off,
                      0, 1, L::kRows * L::kDim / 8, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    }

    mask_f32_full();
    vconv_s322f32(upd_hi, reinterpret_cast<__ubuf__ int32_t *>(upd_hi),
                  L::kRows * L::kDim / 64, 1, 1, 8, 8);
    bcast_row_scalars<L::kRows>(factor, ub_ptr<float>(L::kStepBase + now));
    bcast_row_scalars<L::kRows>(factor + 2 * L::kRows * 8,
                                ub_ptr<float>(L::kStepLoBase + now));
    if (!first)
      bcast_row_scalars<L::kRows>(factor + L::kRows * 8,
                                  ub_ptr<float>(L::kAlphaBase + now));
    pipe_barrier(PIPE_V);

    if (first) {
      mul_row_bcast<L::kRows, L::kDim>(acc, upd_hi, factor);
    } else {
      mul_row_bcast<L::kRows, L::kDim>(upd_hi, upd_hi, factor);
      mul_row_bcast<L::kRows, L::kDim>(acc, acc, factor + L::kRows * 8);
      pipe_barrier(PIPE_V);
      vadd(acc, acc, upd_hi, L::kRows * L::kDim / 64, 1, 1, 1, 8, 8, 8);
    }

    auto upd_lo = ub_ptr<float>(L::kPvLoStage);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    vconv_s322f32(upd_lo, ub_ptr<int32_t>(L::kPvLoStage),
                  L::kRows * L::kDim / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    mul_row_bcast<L::kRows, L::kDim>(upd_lo, upd_lo, factor + 2 * L::kRows * 8);
    pipe_barrier(PIPE_V);
    vadd(acc, acc, upd_lo, L::kRows * L::kDim / 64, 1, 1, 1, 8, 8, 8);

    if (tile + 1 < L::kTiles) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    }

    if (last) {
      auto sums = ub_ptr<float>(L::kSumBase + now);
      auto logs = ub_ptr<float>(L::kTmpRows);
      auto inv = ub_ptr<float>(L::kTmpRows + 64);
      mask_short<L::kRows>();
      vln(logs, sums, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(logs, logs, ub_ptr<float>(L::kMaxBase + now), 1, 1, 1, 1, 8, 8, 8);
      vector_dup(inv, 1.0f, 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
      vdiv(inv, inv, sums, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);

      mask_f32_full();
      bcast_row_scalars<L::kRows>(factor, inv);
      pipe_barrier(PIPE_V);
      mul_row_bcast<L::kRows, L::kDim>(acc, acc, factor);
      pipe_barrier(PIPE_V);
      auto out_bf16 = ub_ptr<bfloat16_t>(L::kContent + tile * kPvBytes);
      vconv_f322bf16r(out_bf16, acc, L::kRows * L::kDim / 64, 1, 1, 4, 8);
      pipe_barrier(PIPE_V);

      if (tile == 0 && first_empty) {
        mask_short<L::kRows>();
        vector_dup(logs, __builtin_inff(), 1, 1, 1, 8, 0);
        mask_b16_full();
        vector_dup(ub_ptr<int16_t>(L::kContent + tile * kPvBytes),
                   static_cast<int16_t>(0), L::kRows * L::kDim / 128, 1, 1, 8,
                   0);
        mask_f32_full();
        pipe_barrier(PIPE_V);
      }

      auto ev = tile_event(tile);
      set_flag(PIPE_V, PIPE_MTE3, ev);
      wait_flag(PIPE_V, PIPE_MTE3, ev);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ float *>(lse) + lane, logs, 0, 1,
                      L::kRows / 8, 0, 0);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ int16_t *>(out) +
                          tile * head_stride * L::kDim,
                      ub_ptr<int16_t>(L::kContent + tile * kPvBytes), 0, 1,
                      L::kRows * L::kDim / 16, 0, 0);
      set_flag(PIPE_MTE3, PIPE_V, ev);
      wait_flag(PIPE_MTE3, PIPE_V, ev);
    }
    pipe_barrier(PIPE_V);
  }
  if (last)
    pipe_barrier(PIPE_ALL);
}
} // namespace

// Dual-digit INT32 PV rescaling, online accumulator update, and final BF16 /
// FP32 LSE writeback on Ascend Vector cores (`M` in {16, 32, 64}).
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_paged_int8_update(uint32_t m, uint64_t pv, uint64_t out,
                                      uint64_t lse, uint32_t slot,
                                      uint32_t first, uint32_t last,
                                      uint32_t empty, uint32_t head_stride,
                                      uint32_t first_empty) {
  if (m == 16) {
    update_window_accum<16>(pv, out, lse, slot, first, last, empty, head_stride,
                            first_empty);
  } else if (m == 32) {
    update_window_accum<32>(pv, out, lse, slot, first, last, empty, head_stride,
                            first_empty);
  } else {
    update_window_accum<64>(pv, out, lse, slot, first, last, empty, head_stride,
                            first_empty);
  }
}
