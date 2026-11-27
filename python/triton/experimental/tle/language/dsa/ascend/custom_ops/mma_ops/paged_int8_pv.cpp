// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_int8_common.h"

namespace {
// Single fixed FP32 bit pattern for Fixpipe `DEQF16` (`1 / 16384 = 2^-14`)
// shared across `vsum`, `PV_hi`, and `PV_lo` on `M >= 64` so the hardware
// `QUANT_PRE` SPR never changes mid-flight on `PIPE_FIX`.
constexpr uint64_t kDeqPvHi = 0x38800000ULL;

// Computes `vsum = (ones_{16 x 384} @ V_{384 x 512}) / 16384` in FP16 via
// Fixpipe `DEQF16` against the NZ value tiles resident in L1 slot `slot`.
__aicore__ __attribute__((always_inline)) void
compute_vsum(uint64_t ones_gm, uint64_t vsum_dst, uint32_t slot,
             uint32_t key_base, uint32_t ones_l1_base) {
  auto ones_l1 = l1_ptr<int8_t>(ones_l1_base);
  nd2nz_s8(ones_l1, ones_gm, 16, 128, 128, 16);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  for (uint32_t col = 0; col < 2; ++col) {
    wait_flag(PIPE_FIX, PIPE_M, bank_event(col));
    auto c_sum = l0c_ptr<int32_t>(col * 65536);
    for (uint32_t pair = 0; pair < 3; ++pair) {
      uint32_t bank = (col * 3 + pair) % 2;
      auto v = l1_ptr<int8_t>(key_base + (slot * 3 + pair) * 65536 +
                              col * 256 * 128);
      auto a_ones = l0a_ptr<int8_t>(bank * 32768);
      auto b = l0b_ptr<int8_t>(bank * 32768);
      wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      load_l0a(a_ones, ones_l1, 0, 128, 16, 128);
      load_l0b_value_s8(b, v, 256, 128);
      mma_step(c_sum, a_ones, b, 16, 256, 128, pair == 0);
      set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
    }
    store_l0c_deqf16(vsum_dst + col * 256 * 2, c_sum, 16, 256, 512, kDeqPvHi);
    set_flag(PIPE_FIX, PIPE_M, bank_event(col));
  }
}
} // namespace

// Adaptive shifted-8-bit (`P_hi @ V + 128 * sum(V)`, full [0, 255] dynamic
// range) or two-digit (`P_hi @ V + 128 * sum(V) + P_lo @ V / 127`) PV MMA
// against the 384x512 NZ value tiles left resident in L1 slot `slot` by
// `paged_int8_qk`. Supports `M` in {16, 32, 64, 128}.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_paged_int8_pv(uint32_t m, uint64_t p, uint64_t dst,
                                  uint64_t k, uint64_t kr, uint64_t table,
                                  uint64_t flag_gm, uint64_t ones_gm,
                                  uint32_t slot, uint32_t remaining,
                                  uint32_t last) {
  bool do_lo = true;
  if (flag_gm != 0) {
    auto f64 = reinterpret_cast<__gm__ int64_t *>(flag_gm);
    dcci(f64, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dcci(f64 + 4, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    auto f32 = reinterpret_cast<__gm__ const int32_t *>(flag_gm);
    do_lo = (f32[0] | f32[8]) != 0;
  }

  const bool m128 = (m == 128);
  const uint32_t key_base = m128 ? kKeyBase128 : kKeyBase;
  const uint32_t kr_base = m128 ? kKrBase128 : kKrBase;
  const uint32_t prob_base = m128 ? kProbBase128 : kProbBase;

  // Wait for `KR` (or previous `P`) to be free before writing to `prob_base`.
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);

  // When `ones_gm != 0`, emit `vsum` (`16 x 512` INT32, row 0 = sum_k V[k, :])
  // into `dst + m * 2048` (or `dst + 2 * m * 2048`? Wait: if `do_lo`, `dst +
  // m * 2048` holds `PV_lo`, so place `vsum` at `ones_gm + 2048`? No: `vsum`
  // is per-(core, slot), so store `vsum` at `dst + m * 2048` when `!do_lo`
  // and at `flag_gm + ...`? Even cleaner: `vsum` is ONLY 16x512 INT32 = 32 KiB
  // at `dst + m * 2048` when `!do_lo`, and when `do_lo` we STILL need `vsum`
  // because `P_hi` is shifted-8-bit in BOTH paths! Wait: `ones_gm` is 2 KiB
  // per core, and right after `ones_gm` in `window_pipeline` we can reserve
  // 2 x 32 KiB per core for `vsum` OR pass `vsum_gm`! Let's put `vsum` at
  // `ones_gm + (1 + slot * 16) * 2048` = 32 KiB per slot right after `ones`!).
  if (ones_gm != 0) {
    uint64_t vsum_dst = ones_gm + (1 + slot * 16) * 2048;
    compute_vsum(ones_gm, vsum_dst, slot, key_base, prob_base);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  }

  if (m128) {
    auto p_hi = l1_ptr<int8_t>(kProbBase128);
    nd2nz_s8(p_hi, p, 128, 384, 384, 128);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    for (uint32_t col = 0; col < 4; ++col) {
      uint32_t c_bank = col % 2;
      wait_flag(PIPE_FIX, PIPE_M, bank_event(c_bank));
      auto c_hi = l0c_ptr<int32_t>(c_bank * 65536);
      for (uint32_t pair = 0; pair < 3; ++pair) {
        uint32_t bank = (col * 3 + pair) % 2;
        auto v = l1_ptr<int8_t>(kKeyBase128 + (slot * 3 + pair) * 65536 +
                                col * 128 * 128);
        auto a_hi = l0a_ptr<int8_t>(bank * 32768);
        auto b = l0b_ptr<int8_t>(bank * 32768);
        wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
        load_l0a(a_hi, p_hi, pair, 128, 128, 128);
        load_l0b_value_s8(b, v, 128, 128);
        if (!do_lo && col == 3 && remaining)
          refill_page_pair(k, kr, table, remaining, slot, pair, kKeyBase128,
                           kKrBase128);
        if (!do_lo && col == 3 && pair == 2 && !remaining)
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        mma_step(c_hi, a_hi, b, 128, 128, 128, pair == 0);
        set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      }
      store_l0c_deqf16(dst + col * 128 * 2, c_hi, 128, 128, 512, kDeqPvHi);
      set_flag(PIPE_FIX, PIPE_M, bank_event(c_bank));
    }
    if (do_lo) {
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      nd2nz_s8(p_hi, p + 128 * 384, 128, 384, 384, 128);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
      for (uint32_t col = 0; col < 4; ++col) {
        uint32_t c_bank = col % 2;
        wait_flag(PIPE_FIX, PIPE_M, bank_event(c_bank));
        auto c_lo = l0c_ptr<int32_t>(c_bank * 65536);
        for (uint32_t pair = 0; pair < 3; ++pair) {
          uint32_t bank = (col * 3 + pair) % 2;
          auto v = l1_ptr<int8_t>(kKeyBase128 + (slot * 3 + pair) * 65536 +
                                  col * 128 * 128);
          auto a_lo = l0a_ptr<int8_t>(bank * 32768);
          auto b = l0b_ptr<int8_t>(bank * 32768);
          wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
          load_l0a(a_lo, p_hi, pair, 128, 128, 128);
          load_l0b_value_s8(b, v, 128, 128);
          if (col == 3 && remaining)
            refill_page_pair(k, kr, table, remaining, slot, pair, kKeyBase128,
                             kKrBase128);
          if (col == 3 && pair == 2 && !remaining)
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          mma_step(c_lo, a_lo, b, 128, 128, 128, pair == 0);
          set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
        }
        store_l0c_deqf16(dst + 128 * 1024 + col * 128 * 2, c_lo, 128, 128, 512,
                         kDeqPvHi);
        set_flag(PIPE_FIX, PIPE_M, bank_event(c_bank));
      }
    }
  } else if (!do_lo) {
    auto p_hi = l1_ptr<int8_t>(kProbBase);
    nd2nz_s8(p_hi, p, m, 384, 384, m);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    for (uint32_t col = 0; col < 2; ++col) {
      wait_flag(PIPE_FIX, PIPE_M, bank_event(col));
      auto c_hi = l0c_ptr<int32_t>(col * 65536);
      for (uint32_t pair = 0; pair < 3; ++pair) {
        uint32_t instruction = col * 3 + pair;
        uint32_t bank = instruction % 2;
        auto v = l1_ptr<int8_t>(kKeyBase + (slot * 3 + pair) * 65536 +
                                col * 256 * 128);
        auto a_hi = l0a_ptr<int8_t>(bank * 32768);
        auto b = l0b_ptr<int8_t>(bank * 32768);
        wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
        load_l0a(a_hi, p_hi, pair, 128, m, 128);
        load_l0b_value_s8(b, v, 256, 128);
        if (col == 1 && remaining)
          refill_page_pair(k, kr, table, remaining, slot, pair);
        if (instruction == 5 && (m <= 32 || !remaining))
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        mma_step(c_hi, a_hi, b, m, 256, 128, pair == 0);
        set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      }
      if (m >= 64)
        store_l0c_deqf16(dst + col * 256 * 2, c_hi, m, 256, 512, kDeqPvHi);
      else
        store_l0c(dst + col * 256 * 4, c_hi, m, 256, 512);
      set_flag(PIPE_FIX, PIPE_M, bank_event(col));
    }
  } else if (m <= 32) {
    auto p_hi = l1_ptr<int8_t>(kProbBase);
    auto p_lo = l1_ptr<int8_t>(kProbBase + 16384);
    nd2nz_s8(p_hi, p, m, 384, 384, m);
    nd2nz_s8(p_lo, p + m * 384, m, 384, 384, m);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    for (uint32_t col = 0; col < 2; ++col) {
      wait_flag(PIPE_FIX, PIPE_M, bank_event(col));
      auto c_hi = l0c_ptr<int32_t>(col * 65536);
      auto c_lo = l0c_ptr<int32_t>(col * 65536 + 32768);
      for (uint32_t pair = 0; pair < 3; ++pair) {
        uint32_t instruction = col * 3 + pair;
        uint32_t bank = instruction % 2;
        auto v = l1_ptr<int8_t>(kKeyBase + (slot * 3 + pair) * 65536 +
                                col * 256 * 128);
        auto a_hi = l0a_ptr<int8_t>(bank * 32768);
        auto a_lo = l0a_ptr<int8_t>(bank * 32768 + 16384);
        auto b = l0b_ptr<int8_t>(bank * 32768);
        wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
        load_l0a(a_hi, p_hi, pair, 128, m, 128);
        load_l0a(a_lo, p_lo, pair, 128, m, 128);
        load_l0b_value_s8(b, v, 256, 128);
        if (col == 1 && remaining)
          refill_page_pair(k, kr, table, remaining, slot, pair);
        if (instruction == 5)
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        mma_pair_s8(c_hi, a_hi, c_lo, a_lo, b, m, 256, 128, pair == 0);
        set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      }
      store_l0c(dst + col * 256 * 4, c_hi, m, 256, 512);
      store_l0c(dst + m * 2048 + col * 256 * 4, c_lo, m, 256, 512);
      set_flag(PIPE_FIX, PIPE_M, bank_event(col));
    }
  } else {
    auto p_hi = l1_ptr<int8_t>(kProbBase);
    auto p_lo_01 = l1_ptr<int8_t>(kProbBase + 24576);
    auto p_lo_2 = l1_ptr<int8_t>(kKrBase + 2 * 16384);
    nd2nz_s8(p_hi, p, 64, 384, 384, 64);
    uint64_t p_lo_gm = p + 64 * 384;
    nd2nz_s8(p_lo_01, p_lo_gm, 64, 256, 384, 64);
    nd2nz_s8(p_lo_2, p_lo_gm + 256, 64, 128, 384, 64);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
    for (uint32_t col = 0; col < 4; ++col) {
      uint32_t c_bank = col % 2;
      wait_flag(PIPE_FIX, PIPE_M, bank_event(c_bank));
      auto c_hi = l0c_ptr<int32_t>(c_bank * 65536);
      auto c_lo = l0c_ptr<int32_t>(c_bank * 65536 + 32768);
      for (uint32_t pair = 0; pair < 3; ++pair) {
        uint32_t bank = (col * 3 + pair) % 2;
        auto v = l1_ptr<int8_t>(kKeyBase + (slot * 3 + pair) * 65536 +
                                col * 128 * 128);
        auto a_hi = l0a_ptr<int8_t>(bank * 32768);
        auto a_lo = l0a_ptr<int8_t>(bank * 32768 + 16384);
        auto b = l0b_ptr<int8_t>(bank * 32768);
        wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
        load_l0a(a_hi, p_hi, pair, 128, 64, 128);
        if (pair < 2)
          load_l0a(a_lo, p_lo_01, pair, 128, 64, 128);
        else
          load_l0a(a_lo, p_lo_2, 0, 128, 64, 128);
        load_l0b_value_s8(b, v, 128, 128);
        if (col == 3 && remaining)
          refill_page_pair(k, kr, table, remaining, slot, pair);
        if (col == 3 && pair == 2 && !remaining)
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        mma_pair_s8(c_hi, a_hi, c_lo, a_lo, b, 64, 128, 128, pair == 0);
        set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      }
      store_l0c_deqf16(dst + col * 128 * 2, c_hi, 64, 128, 512, kDeqPvHi);
      store_l0c_deqf16(dst + 64 * 1024 + col * 128 * 2, c_lo, 64, 128, 512,
                       kDeqPvHi);
      set_flag(PIPE_FIX, PIPE_M, bank_event(c_bank));
    }
  }
  if (last) {
    for (uint32_t bank = 0; bank < 2; ++bank) {
      wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      wait_flag(PIPE_FIX, PIPE_M, bank_event(bank));
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  }
}
