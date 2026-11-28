// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIC/Cube) into custom_ops.bc.
// Stateless chunk-pipelined GM->L1 ND2NZ + INT8 L1-ZN x L1-ZN -> FP16 GM
// Fixpipe DEQF16 matmul on Ascend910B (dav-c220-cube), with optional
// next-iteration chunk-0 block-gather prefetch into `rhs_l1` overlapped behind
// the final tile's MMA and Fixpipe stores (zero get_imm() L1 addressing, zero
// cross-call event state).
#include "Macro/common.h"
#include "Macro/coord.h"
#include "Macro/operators/iterator.h"
#include "Macro/operators/iterators/gm_to_l1_iterator.h"
#include "Macro/operators/iterators/l1_to_l0_iterator.h"
#include "Macro/operators/mma.h"
#include "Macro/params.h"
#include "Utils.h"

namespace {
constexpr ArchType kArch = ArchType::ASCEND_V220;

template <typename T>
__aicore__ __attribute__((always_inline)) __ca__ T *l0a_bank(uint32_t bank) {
  return reinterpret_cast<__ca__ T *>(
      reinterpret_cast<__ca__ uint8_t *>(get_imm(0)) + bank * 32768u);
}

template <typename T>
__aicore__ __attribute__((always_inline)) __cb__ T *l0b_bank(uint32_t bank) {
  return reinterpret_cast<__cb__ T *>(
      reinterpret_cast<__cb__ uint8_t *>(get_imm(0)) + bank * 32768u);
}

template <typename T>
__aicore__ __attribute__((always_inline)) __cc__ T *l0c_bank(uint32_t bank) {
  return reinterpret_cast<__cc__ T *>(
      reinterpret_cast<__cc__ uint8_t *>(get_imm(0)) + bank * 65536u);
}

__aicore__ __attribute__((always_inline)) event_t bank_ev(uint32_t bank) {
  return bank ? EVENT_ID7 : EVENT_ID6;
}

__aicore__ __attribute__((always_inline)) event_t mte1_m_ev(uint32_t bank) {
  return bank ? EVENT_ID5 : EVENT_ID4;
}

__aicore__ __attribute__((always_inline)) event_t m_fix_ev(uint32_t bank) {
  return bank ? EVENT_ID5 : EVENT_ID4;
}

__aicore__ __attribute__((always_inline)) event_t kchunk_ev(uint32_t idx) {
  return idx == 0u ? EVENT_ID0 : (idx == 1u ? EVENT_ID1 : EVENT_ID2);
}
} // namespace

extern "C" {

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_nd2nz_matmul_l1_to_gm_s8_deqf16(
    uint64_t lhs_gm, uint64_t out_gm, uint32_t m, uint32_t n, uint32_t k,
    uint32_t dst_stride, float deq_scale, uint64_t prefetch_gm,
    uint64_t prefetch_ids_gm, uint32_t prefetch_blocks,
    uint32_t prefetch_valid_blocks, uint32_t prefetch_block_rows,
    uint32_t prefetch_block_stride, memref_t<__cbuf__ int8_t, 2> *scratch_l1,
    memref_t<__cbuf__ int8_t, 2> *rhs_l1) {
  constexpr uint32_t kC0 = 32u;
  constexpr uint32_t kFrac = 512u;

  auto b1 = rhs_l1->aligned + rhs_l1->offset;
  auto a_l1 = scratch_l1->aligned + scratch_l1->offset;
  auto gm_lhs = reinterpret_cast<__gm__ int8_t *>(lhs_gm);
  auto gm_out = reinterpret_cast<__gm__ half *>(out_gm);

  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID7);

  // Determine first m-tile and whether k-chunked ND2NZ pipelining applies.
  const uint32_t m_first = (m > 128u) ? 128u : m;
  const bool k_chunked = (k % 128u == 0u) && (k >= 256u) && (k <= 384u);
  const uint32_t num_kchunks = k_chunked ? (k / 128u) : 1u;

  if (k_chunked) {
    for (uint32_t kc = 0u; kc < num_kchunks; ++kc) {
      uint32_t k_off = kc * 128u;
      gm_to_l1<kArch, int8_t, CubeFormat::ND, CubeFormat::ZN> load_kc(
          a_l1 + static_cast<uint64_t>(k_off) * m, gm_lhs + k_off, m_first, m,
          m_first, 128u, 128u, k);
      set_flag(PIPE_MTE2, PIPE_MTE1, kchunk_ev(kc));
    }
    if (m > 128u) {
      gm_to_l1<kArch, int8_t, CubeFormat::ND, CubeFormat::ZN> load_rest(
          a_l1 + static_cast<uint64_t>(128u) * kC0,
          gm_lhs + static_cast<uint64_t>(128u) * k, m - 128u, m, m - 128u, k, k,
          k);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    }
  } else {
    gm_to_l1<kArch, int8_t, CubeFormat::ND, CubeFormat::ZN> load_all(
        a_l1, gm_lhs, m, m, m, k, k, k);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  }

  uint32_t deq_bits = 0;
  __builtin_memcpy(&deq_bits, &deq_scale, sizeof(deq_bits));
  bool quant_set = false;
  bool prefetch_issued = false;

  uint32_t ab_step = 0u;
  uint32_t c_step = 0u;

  for (uint32_t m0 = 0u; m0 < m;) {
    uint32_t m_tile = (m - m0 > 128u) ? 128u : (m - m0);
    uint32_t n_cap = (m_tile <= 64u && (n % 256u == 0u)) ? 256u : 128u;
    bool is_last_m = (m0 + m_tile >= m);

    if (k_chunked && m0 == 128u) {
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    }

    for (uint32_t n0 = 0u; n0 < n;) {
      uint32_t n_tile = (n - n0 > n_cap) ? n_cap : (n - n0);
      bool is_last_mn = is_last_m && (n0 + n_tile >= n);
      uint32_t k_lim_a = (32768u / m_tile) & ~(kC0 - 1u);
      uint32_t k_lim_b = (32768u / n_tile) & ~(kC0 - 1u);
      uint32_t k_cap = (k_lim_a < k_lim_b) ? k_lim_a : k_lim_b;
      if (k_cap > 128u && (k % 128u == 0u)) {
        k_cap = 128u;
      }

      uint32_t cb = c_step & 1u;
      wait_flag(PIPE_FIX, PIPE_M, bank_ev(cb));
      auto l0c = l0c_bank<int32_t>(cb);

      for (uint32_t k0 = 0u; k0 < k;) {
        uint32_t k_tile = (k - k0 > k_cap) ? k_cap : (k - k0);
        bool is_last_k = (k0 + k_tile >= k);
        if (m0 == 0u && n0 == 0u) {
          if (k_chunked) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, kchunk_ev(k0 / 128u));
          } else if (k0 == 0u) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          }
        }

        uint32_t abb = ab_step & 1u;
        wait_flag(PIPE_M, PIPE_MTE1, bank_ev(abb));
        auto l0a = l0a_bank<int8_t>(abb);
        auto l0b = l0b_bank<int8_t>(abb);

        auto a_src =
            a_l1 +
            (static_cast<uint64_t>(k0 / kC0) * (m / 16u) + (m0 / 16u)) * kFrac;
        l1_to_l0_a<kArch, int8_t, false, CubeFormat::ZN, CubeFormat::ZZ> load_a(
            l0a, a_src, m_tile, k_tile, 1u, m / 16u, k_tile / kC0, 1u);
        auto b_src =
            b1 +
            (static_cast<uint64_t>(n0 / kC0) * (k / 16u) + (k0 / 16u)) * kFrac;
        l1_to_l0_b<kArch, int8_t, false, CubeFormat::ZN, CubeFormat::NZ> load_b(
            l0b, b_src, n_tile, k_tile, k / 16u, 1u, 1u, n_tile / kC0);

        auto mte1_ev = mte1_m_ev(abb);
        set_flag(PIPE_MTE1, PIPE_M, mte1_ev);

        // Once the final L1->L0B load from `rhs_l1` is issued on PIPE_MTE1,
        // overlap prefetching the next group's leading blocks into canonical ZN
        // `rhs_l1` behind this final MMA and the last two Fixpipe stores.
        if (is_last_mn && is_last_k && prefetch_gm != 0u &&
            prefetch_blocks > 0u) {
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
          auto gm_pf = reinterpret_cast<__gm__ int8_t *>(prefetch_gm);
          auto pf_ids =
              reinterpret_cast<__gm__ const int32_t *>(prefetch_ids_gm);
          const uint32_t last_valid =
              prefetch_valid_blocks > 0u ? (prefetch_valid_blocks - 1u) : 0u;
          for (uint32_t pb = 0u; pb < prefetch_blocks; ++pb) {
            uint32_t idx = (pb < prefetch_valid_blocks) ? pb : last_valid;
            int64_t bid = static_cast<int64_t>(pf_ids[idx]);
            gm_to_l1<kArch, int8_t, CubeFormat::ND, CubeFormat::ZN> pf_copy(
                b1 + static_cast<uint64_t>(pb) * prefetch_block_rows * kC0,
                gm_pf + static_cast<uint64_t>(bid) * prefetch_block_stride,
                prefetch_block_rows, k, prefetch_block_rows, n, n, n);
          }
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          prefetch_issued = true;
        }

        wait_flag(PIPE_MTE1, PIPE_M, mte1_ev);
        if ((m_tile / 16u) * (n_tile / 16u) < 10u) {
          pipe_barrier(PIPE_M);
        }
        mmad<kArch, int8_t, int8_t, int32_t, false> run(
            l0c, l0a, l0b, m_tile, n_tile, k_tile, k0 == 0u);
        set_flag(PIPE_M, PIPE_MTE1, bank_ev(abb));
        ++ab_step;
        k0 += k_tile;
      }

      if (!quant_set) {
        set_flag(PIPE_FIX, PIPE_S, EVENT_ID6);
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID6);
        set_quant_pre(static_cast<uint64_t>(deq_bits));
        quant_set = true;
      }

      auto fix_ev = m_fix_ev(cb);
      set_flag(PIPE_M, PIPE_FIX, fix_ev);
      wait_flag(PIPE_M, PIPE_FIX, fix_ev);
      auto dst_ptr = gm_out + static_cast<uint64_t>(m0) * dst_stride + n0;
      copy_matrix_cc_to_gm(dst_ptr, l0c, 0, n_tile, m_tile, dst_stride, m_tile,
                           0, DEQF16, 0, false, true);
      set_flag(PIPE_FIX, PIPE_M, bank_ev(cb));
      ++c_step;
      n0 += n_tile;
    }
    m0 += m_tile;
  }

  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID7);
  if (prefetch_issued) {
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  }
  pipe_barrier(PIPE_MTE1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
}

} // extern "C"
