// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_int8_vec_common.h"

namespace {
// Full FP32 accumulator update for `M <= 32` (`int32` PV, 2-digit `[0, 127]`,
// zero `vsum` overhead).
template <int M>
__aiv__ __attribute__((always_inline)) void
update_window_accum(uint64_t pv, uint64_t vsum_gm, uint64_t out, uint64_t lse,
                    uint32_t slot, uint32_t first, uint32_t last,
                    uint32_t empty, uint32_t head_stride,
                    uint32_t first_empty) {
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

  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  uint32_t need_lo_mask =
      static_cast<uint32_t>(*reinterpret_cast<volatile __ubuf__ int32_t *>(
          ub_ptr<int32_t>(L::kNeedLoBase + slot * 32)));

  mask_f32_full();
  constexpr int kPvBytes = L::kRows * L::kDim * 4;
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  if (vsum_gm != 0) {
    copy_gm_to_ubuf(ub_ptr<int32_t>(L::kScratch),
                    reinterpret_cast<__gm__ int32_t *>(vsum_gm), 0, 1,
                    L::kDim / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  }
  if (need_lo_mask & 1u) {
    copy_gm_to_ubuf(ub_ptr<int32_t>(L::kPvLoStage),
                    reinterpret_cast<__gm__ int32_t *>(pv) +
                        2 * L::kHeads * L::kDim,
                    0, 1, L::kRows * L::kDim / 8, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
  }

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

  auto vsum_f32 = ub_ptr<float>(L::kScratch);
  if (vsum_gm != 0) {
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    vconv_s322f32(vsum_f32, ub_ptr<int32_t>(L::kScratch), L::kDim / 64, 1, 1, 8,
                  8);
    pipe_barrier(PIPE_V);
    vmuls(vsum_f32, vsum_f32, 128.0f, L::kDim / 64, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
  }

  for (int tile = 0; tile < L::kTiles; ++tile) {
    int lane = tile * L::kRows;
    int off = lane * L::kDim;
    int now = (slot * L::kHeads + lane) * 4;
    int stage_addr =
        L::kTiles == 1 ? L::kPvStageM16 : L::kContent + tile * kPvBytes;
    auto upd_hi = ub_ptr<float>(stage_addr);
    auto acc = ub_ptr<float>(L::kAccum) + off;
    auto factor = ub_ptr<float>(L::kScratch + 2048);
    bool tile_lo = (need_lo_mask & (1u << tile)) != 0;

    if (tile == 0 && !last)
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    else
      wait_flag(PIPE_MTE2, PIPE_V, tile_event(tile));

    if (tile > 0 && tile_lo) {
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
    if (!first)
      bcast_row_scalars<L::kRows>(factor + L::kRows * 8,
                                  ub_ptr<float>(L::kAlphaBase + now));
    pipe_barrier(PIPE_V);

    if (vsum_gm != 0) {
      for (int c = 0; c < L::kDim; c += 64)
        vadd(upd_hi + c, upd_hi + c, vsum_f32 + c, L::kRows, 1, 1, 1,
             L::kDim / 8, L::kDim / 8, 0);
    }
    if (!first) {
      mul_row_bcast<L::kRows, L::kDim>(acc, acc, factor + L::kRows * 8);
    }
    if (vsum_gm != 0 || !first)
      pipe_barrier(PIPE_V);

    if (tile_lo) {
      auto upd_lo = ub_ptr<float>(L::kPvLoStage);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      vconv_s322f32(upd_lo, ub_ptr<int32_t>(L::kPvLoStage),
                    L::kRows * L::kDim / 64, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vmuls(upd_lo, upd_lo, 1.0f / 127.0f, L::kRows * L::kDim / 64, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(upd_hi, upd_hi, upd_lo, L::kRows * L::kDim / 64, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }

    if (first) {
      mul_row_bcast<L::kRows, L::kDim>(acc, upd_hi, factor);
    } else {
      mla_row_bcast<L::kRows, L::kDim>(acc, upd_hi, factor);
    }
    pipe_barrier(PIPE_V);

    if (tile + 1 < L::kTiles && (need_lo_mask & 2u)) {
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

// Running-normalized FP16 accumulator update for `M >= 64` (`M == 64` and `M ==
// 128`) consuming Fixpipe `DEQF16` `PV_hi` (`/ 16384`), `vsum` (`/ 128`), and
// rare `PV_lo` (`/ (16384 * 127)`). Keeps `acc_h` normalized (`acc / sum`) so
// it never overflows or underflows for any sequence length while cutting
// `Update` vector repeats by 2.67x (`768` FP16 repeats vs `2048` FP32 repeats).
template <int M>
__aiv__ __attribute__((always_inline)) void
update_window_accum_fp16(uint64_t pv, uint64_t vsum_gm, uint64_t out,
                         uint64_t lse, uint32_t slot, uint32_t first,
                         uint32_t last, uint32_t empty, uint32_t head_stride,
                         uint32_t first_empty) {
  using L = PagedInt8VecLayout<M>;
  constexpr int kR = M == 128 ? 32 : 16;
  constexpr int kT = 2;
  constexpr int kPvBankBytes = kR * L::kDim * 2;
  if (empty) {
    mask_short<L::kHeads>();
    vector_dup(ub_ptr<float>(L::kScratch), __builtin_inff(), 1, 1, 1, 8, 0);
    mask_b16_full();
    vector_dup(ub_ptr<int16_t>(L::kContent), static_cast<int16_t>(0),
               kR * L::kDim / 128, 1, 1, 8, 0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ float *>(lse),
                    ub_ptr<float>(L::kScratch), 0, 1, L::kHeads / 8, 0, 0);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ int16_t *>(out),
                    ub_ptr<int16_t>(L::kContent), 0, 1, kR * L::kDim / 16, 0,
                    0);
    copy_ubuf_to_gm(
        reinterpret_cast<__gm__ int16_t *>(out) + head_stride * L::kDim,
        ub_ptr<int16_t>(L::kContent), 0, 1, kR * L::kDim / 16, 0, 0);
    mask_f32_full();
    pipe_barrier(PIPE_ALL);
    return;
  }

  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  uint32_t need_lo_mask =
      static_cast<uint32_t>(*reinterpret_cast<volatile __ubuf__ int32_t *>(
          ub_ptr<int32_t>(L::kNeedLoBase + slot * 32)));

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  auto vsum_h = ub_ptr<half>(L::kScratch); // 512 halfs = 1024 B
  if (vsum_gm != 0) {
    copy_gm_to_ubuf(vsum_h, reinterpret_cast<__gm__ half *>(vsum_gm), 0, 1,
                    L::kDim / 16, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  }
  for (int tile = 0; tile < kT; ++tile) {
    if (!last && tile == 0)
      continue;
    int off = tile * kR * L::kDim;
    copy_gm_to_ubuf(ub_ptr<half>(L::kContent + tile * kPvBankBytes),
                    reinterpret_cast<__gm__ half *>(pv) + off, 0, 1,
                    kR * L::kDim / 16, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, tile_event(tile));
  }

  bool lo0 = M == 128 ? ((need_lo_mask & 3u) != 0) : ((need_lo_mask & 1u) != 0);
  if (lo0) {
    copy_gm_to_ubuf(ub_ptr<half>(L::kPvLoStage),
                    reinterpret_cast<__gm__ half *>(pv) +
                        2 * L::kHeads * L::kDim,
                    0, 1, kR * L::kDim / 16, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
  }
  if (vsum_gm != 0) {
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    mask_b16_full();
    vmuls(vsum_h, vsum_h, static_cast<half>(128.0f), L::kDim / 128, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
  }

  // Scratch layout in `L::kScratch` (15 KiB):
  // - `0..1024`: `vsum_h` (512 halfs)
  // - `1024..2048`: `bcast_cur` (up to 32 x 16 halfs = 1024 B)
  // - `2048..3072`: `bcast_prev` (up to 32 x 16 halfs = 1024 B)
  // - `3072..3136`: `w_cur_h` (up to 32 halfs = 64 B)
  // - `3136..3200`: `w_prev_h` (up to 32 halfs = 64 B)
  auto bcast_cur = ub_ptr<half>(L::kScratch + 1024);
  auto bcast_prev = ub_ptr<half>(L::kScratch + 2048);
  auto w_cur_h = ub_ptr<half>(L::kScratch + 3072);
  auto w_prev_h = ub_ptr<half>(L::kScratch + 3136);

  for (int tile = 0; tile < kT; ++tile) {
    int lane = tile * kR;
    int off = lane * L::kDim;
    int now = (slot * L::kHeads + lane) * 4;
    int stage_addr = L::kContent + tile * kPvBankBytes;
    auto upd_hi_h = ub_ptr<half>(stage_addr);
    auto acc_h = ub_ptr<half>(L::kAccum) + off;
    bool tile_lo = M == 128 ? ((need_lo_mask & (3u << (tile * 2))) != 0)
                            : ((need_lo_mask & (1u << tile)) != 0);

    if (tile == 0 && !last)
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    else
      wait_flag(PIPE_MTE2, PIPE_V, tile_event(tile));

    if (tile > 0 && tile_lo) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      copy_gm_to_ubuf(ub_ptr<half>(L::kPvLoStage),
                      reinterpret_cast<__gm__ half *>(pv) +
                          2 * L::kHeads * L::kDim + off,
                      0, 1, kR * L::kDim / 16, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    }

    // `L::kStepBase + now` holds `16384 * step / now_sum` and
    // `L::kAlphaBase + now` holds `prev_sum * alpha / now_sum` (precomputed in
    // `Prob` before the next window overwrites `prev_sum`).
    mask_short<kR>();
    vconv_f322f16(w_cur_h, ub_ptr<float>(L::kStepBase + now), 1, 1, 1, 4, 8);
    if (!first)
      vconv_f322f16(w_prev_h, ub_ptr<float>(L::kAlphaBase + now), 1, 1, 1, 4,
                    8);
    pipe_barrier(PIPE_V);

    mask_b16_full();
    bcast_row_scalars_f16<kR>(bcast_cur, w_cur_h);
    if (!first)
      bcast_row_scalars_f16<kR>(bcast_prev, w_prev_h);
    if (vsum_gm != 0) {
      for (int c = 0; c < L::kDim; c += 128)
        vadd(upd_hi_h + c, upd_hi_h + c, vsum_h + c, kR, 1, 1, 1, L::kDim / 16,
             L::kDim / 16, 0);
    }
    pipe_barrier(PIPE_V);

    if (!first) {
      mul_row_bcast_f16<kR, L::kDim>(acc_h, acc_h, bcast_prev);
    }
    if (tile_lo) {
      auto upd_lo_h = ub_ptr<half>(L::kPvLoStage);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      vmuls(upd_lo_h, upd_lo_h, static_cast<half>(1.0f / 127.0f),
            kR * L::kDim / 128, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(upd_hi_h, upd_hi_h, upd_lo_h, kR * L::kDim / 128, 1, 1, 1, 8, 8, 8);
    }
    if (!first || tile_lo)
      pipe_barrier(PIPE_V);

    if (first) {
      mul_row_bcast_f16<kR, L::kDim>(acc_h, upd_hi_h, bcast_cur);
    } else {
      mla_row_bcast_f16<kR, L::kDim>(acc_h, upd_hi_h, bcast_cur);
    }
    pipe_barrier(PIPE_V);

    if (last) {
      auto sums = ub_ptr<float>(L::kSumBase + now);
      auto logs = ub_ptr<float>(L::kTmpRows);
      mask_short<kR>();
      vln(logs, sums, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vadd(logs, logs, ub_ptr<float>(L::kMaxBase + now), 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);

      // Convert normalized FP16 `acc_h` -> BF16 via 8-row FP32 chunks into
      // `stage_addr` (at least 16 KiB) -> `L::kPvLoStage` (32 KiB).
      auto tmp_f32 = ub_ptr<float>(stage_addr);
      auto out_bf16 = ub_ptr<bfloat16_t>(L::kPvLoStage);
      mask_f32_full();
      for (int sub = 0; sub < kR; sub += 8) {
        vconv_f162f32(tmp_f32, acc_h + sub * L::kDim, 8 * L::kDim / 64, 1, 1, 8,
                      4);
        pipe_barrier(PIPE_V);
        vconv_f322bf16r(out_bf16 + sub * L::kDim, tmp_f32, 8 * L::kDim / 64, 1,
                        1, 4, 8);
        pipe_barrier(PIPE_V);
      }

      if (tile == 0 && first_empty) {
        mask_short<kR>();
        vector_dup(logs, __builtin_inff(), 1, 1, 1, 8, 0);
        mask_b16_full();
        vector_dup(ub_ptr<int16_t>(L::kPvLoStage), static_cast<int16_t>(0),
                   kR * L::kDim / 128, 1, 1, 8, 0);
        mask_f32_full();
        pipe_barrier(PIPE_V);
      }

      auto ev = tile_event(tile);
      set_flag(PIPE_V, PIPE_MTE3, ev);
      wait_flag(PIPE_V, PIPE_MTE3, ev);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ float *>(lse) + lane, logs, 0, 1,
                      kR / 8, 0, 0);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ int16_t *>(out) +
                          tile * head_stride * L::kDim,
                      ub_ptr<int16_t>(L::kPvLoStage), 0, 1, kR * L::kDim / 16,
                      0, 0);
      set_flag(PIPE_MTE3, PIPE_V, ev);
      wait_flag(PIPE_MTE3, PIPE_V, ev);
    }
  }
  mask_f32_full();
  if (last)
    pipe_barrier(PIPE_ALL);
}
} // namespace

// Adaptive shifted-8-bit / residual INT8 PV rescaling, online accumulator
// update, and final BF16 / FP32 LSE writeback on Ascend Vector cores
// (`M` in {16, 32, 64, 128}).
extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_paged_int8_update(uint32_t m, uint64_t pv, uint64_t vsum_gm,
                                      uint64_t out, uint64_t lse, uint32_t slot,
                                      uint32_t first, uint32_t last,
                                      uint32_t empty, uint32_t head_stride,
                                      uint32_t first_empty) {
  if (m == 16) {
    update_window_accum<16>(pv, vsum_gm, out, lse, slot, first, last, empty,
                            head_stride, first_empty);
  } else if (m == 32) {
    update_window_accum<32>(pv, vsum_gm, out, lse, slot, first, last, empty,
                            head_stride, first_empty);
  } else if (m == 64) {
    update_window_accum_fp16<64>(pv, vsum_gm, out, lse, slot, first, last,
                                 empty, head_stride, first_empty);
  } else if (m == 128) {
    update_window_accum_fp16<128>(pv, vsum_gm, out, lse, slot, first, last,
                                  empty, head_stride, first_empty);
  }
}
