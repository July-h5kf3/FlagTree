// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_int8_common.h"

// Dual-digit INT8 PV MMA (`P_hi @ V` and `P_lo @ V` -> INT32) against the
// 384x512 NZ value tiles left resident in L1 slot `slot` by `paged_int8_qk`.
// Reuses the fractal-transposed L0B value slice across both probability digits
// and prefetches the next window's page pairs as each L1 slot retires.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_paged_int8_pv(uint32_t m, uint64_t p, uint64_t dst,
                                  uint64_t k, uint64_t kr, uint64_t table,
                                  uint32_t slot, uint32_t remaining,
                                  uint32_t last) {
  if (m <= 32) {
    auto p_hi = l1_ptr<int8_t>(kProbBase);
    auto p_lo = l1_ptr<int8_t>(kProbBase + 16384);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
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
        mma_step(c_hi, a_hi, b, m, 256, 128, pair == 0);
        mma_step(c_lo, a_lo, b, m, 256, 128, pair == 0);
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
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
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
        mma_step(c_hi, a_hi, b, 64, 128, 128, pair == 0);
        mma_step(c_lo, a_lo, b, 64, 128, 128, pair == 0);
        set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      }
      store_l0c(dst + col * 128 * 4, c_hi, 64, 128, 512);
      store_l0c(dst + 64 * 2048 + col * 128 * 4, c_lo, 64, 128, 512);
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
