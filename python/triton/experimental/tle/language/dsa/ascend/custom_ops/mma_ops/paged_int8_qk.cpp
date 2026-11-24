// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_int8_common.h"

// Paged 6-page (384-token) QK MMA over INT8 content (D=512 -> INT32) and BF16
// RoPE (D=64 -> FP32). Keeps the three 128x512 NZ content tiles resident in L1
// slot `slot` so the paired `paged_int8_pv` call consumes them without
// re-reading global memory.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_paged_int8_qk(uint32_t m, uint64_t q, uint64_t qr,
                                  uint64_t k, uint64_t kr, uint64_t table,
                                  uint64_t c, uint64_t r, uint32_t remaining,
                                  uint32_t first, uint32_t slot,
                                  uint32_t head_stride) {
  constexpr uint32_t width = 384;
  auto q1 = l1_ptr<int8_t>(kQueryBase);
  auto qr1 = l1_ptr<bfloat16_t>(kQrBase);
  if (first) {
    if (m == 16 || head_stride == m / 4) {
      nd2nz_s8(q1, q, m, 512, 512, m);
      nd2nz_b16(qr1, qr, m, 64, 64, m);
    } else {
      for (uint32_t tile = 0; tile < 4; ++tile) {
        uint32_t source = (tile % 2) * head_stride + (tile / 2) * (m / 4);
        nd2nz_s8(q1 + tile * (m / 4) * 32, q + source * 512, m / 4, 512, 512,
                 m);
        nd2nz_b16(qr1 + tile * (m / 4) * 16, qr + source * 128, m / 4, 64, 64,
                  m);
      }
    }
    for (uint32_t bank = 0; bank < 2; ++bank) {
      set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      set_flag(PIPE_FIX, PIPE_M, bank_event(bank));
    }
    if (m <= 32)
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    for (uint32_t pair = 0; pair < 3; ++pair)
      prefetch_page_pair(k, kr, table, remaining, slot, pair);
  }
  for (uint32_t pair = 0; pair < 3; ++pair) {
    auto kv1 = l1_ptr<int8_t>(kKeyBase + (slot * 3 + pair) * 65536);
    auto rope1 = l1_ptr<bfloat16_t>(kKrBase + pair * 16384);
    auto ic = l0c_ptr<int32_t>((pair % 2) * 65536);
    auto rc = l0c_ptr<float>((pair % 2) * 65536 + 32768);
    wait_flag(PIPE_MTE2, PIPE_MTE1, pair_event(pair));
    wait_flag(PIPE_FIX, PIPE_M, bank_event(pair % 2));
    for (uint32_t part = 0; part < 2; ++part) {
      uint32_t bank = (3 * pair + part) % 2;
      wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
      auto a = l0a_ptr<int8_t>(bank * 32768);
      auto b = l0b_ptr<int8_t>(bank * 32768);
      load_l0a(a, q1, part, 256, m, 256);
      load_l0b_transposed(b, kv1, part, 256, 256, 128);
      mma_step(ic, a, b, m, 128, 256, part == 0);
      set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
    }
    store_l0c(c + pair * 512, ic, m, 128, width);
    uint32_t bank = pair % 2;
    wait_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
    auto a = l0a_ptr<bfloat16_t>(bank * 32768);
    auto b = l0b_ptr<bfloat16_t>(bank * 32768);
    load_l0a(a, qr1, 0, 0, m, 64);
    load_l0b_transposed(b, rope1, 0, 0, 64, 128);
    if (first && remaining > 6)
      refill_page_pair(k, kr, table + 24, remaining - 6, 1, pair);
    if (m == 64 && pair == 2 && !(first && remaining > 6))
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    mma_step(rc, a, b, m, 128, 64, true);
    set_flag(PIPE_M, PIPE_MTE1, bank_event(bank));
    store_l0c(r + pair * 512, rc, m, 128, width);
    set_flag(PIPE_FIX, PIPE_M, bank_event(pair % 2));
  }
}
