// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIC/Cube) into custom_ops.bc.
// Stateless general L1-ZN x L1-ZN -> GM matrix multiplication on Ascend910B
// (dav-c220-cube) with double-buffered L0A/L0B/L0C ping-pong, optional B
// transpose (`trans_b`), arbitrary `dst_stride`, and optional hardware Fixpipe
// `DEQF16` (`int32 -> fp16` scaled dequantization).
//
// Both input matrices live in compiler-allocated L1/CBUF buffers in canonical
// ZN fractal layout (as produced by `nd2nz_gm_to_l1` or
// `gather_blocks_gm_to_l1`). L0A/L0B/L0C scratch is strictly ephemeral within a
// single invocation and fully drained before return (zero cross-call state).
#include "Macro/common.h"
#include "Macro/coord.h"
#include "Macro/operators/iterator.h"
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

template <typename InT, typename AccT, typename OutT, bool kUseDeqF16>
__aicore__ __attribute__((always_inline)) void
matmul_l1_to_gm_impl(memref_t<__cbuf__ InT, 2> *lhs,
                     memref_t<__cbuf__ InT, 2> *rhs, uint64_t out_gm,
                     uint32_t m, uint32_t n, uint32_t k, uint32_t dst_stride,
                     uint32_t trans_b, float deq_scale) {
  constexpr uint32_t kC0 = 32 / sizeof(InT);
  constexpr uint32_t kFrac = 512 / sizeof(InT);

  auto a1 = lhs->aligned + lhs->offset;
  auto b1 = rhs->aligned + rhs->offset;
  auto gm_out = reinterpret_cast<__gm__ OutT *>(out_gm);

  if constexpr (kUseDeqF16) {
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID6);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID6);
    uint32_t deq_bits = 0;
    __builtin_memcpy(&deq_bits, &deq_scale, sizeof(deq_bits));
    set_quant_pre(static_cast<uint64_t>(deq_bits));
  }

  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID7);

  uint32_t ab_step = 0;
  uint32_t c_step = 0;

  for (uint32_t m0 = 0; m0 < m;) {
    uint32_t m_tile = (m - m0 > 128u) ? 128u : (m - m0);
    uint32_t n_cap = (m_tile <= 64u) ? 256u : 128u;
    if (n_cap == 256u && (n % 256u != 0u) && (n % 128u == 0u)) {
      n_cap = 128u;
    }

    for (uint32_t n0 = 0; n0 < n;) {
      uint32_t n_tile = (n - n0 > n_cap) ? n_cap : (n - n0);
      uint32_t k_lim_a = (32768u / (m_tile * sizeof(InT))) & ~(kC0 - 1u);
      uint32_t k_lim_b = (32768u / (n_tile * sizeof(InT))) & ~(kC0 - 1u);
      uint32_t k_cap = (k_lim_a < k_lim_b) ? k_lim_a : k_lim_b;

      uint32_t cb = c_step & 1u;
      wait_flag(PIPE_FIX, PIPE_M, bank_ev(cb));
      auto l0c = l0c_bank<AccT>(cb);

      for (uint32_t k0 = 0; k0 < k;) {
        uint32_t k_tile = (k - k0 > k_cap) ? k_cap : (k - k0);
        uint32_t abb = ab_step & 1u;
        wait_flag(PIPE_M, PIPE_MTE1, bank_ev(abb));
        auto l0a = l0a_bank<InT>(abb);
        auto l0b = l0b_bank<InT>(abb);

        auto a_src =
            a1 +
            (static_cast<uint64_t>(k0 / kC0) * (m / 16u) + (m0 / 16u)) * kFrac;
        l1_to_l0_a<kArch, InT, false, CubeFormat::ZN, CubeFormat::ZZ> load_a(
            l0a, a_src, m_tile, k_tile, 1u, m / 16u, k_tile / kC0, 1u);

        if (trans_b) {
          auto b_src =
              b1 + (static_cast<uint64_t>(k0 / kC0) * (n / 16u) + (n0 / 16u)) *
                       kFrac;
          l1_to_l0_b<kArch, InT, true, CubeFormat::ZN, CubeFormat::NZ> load_b(
              l0b, b_src, n_tile, k_tile, 1u, n / 16u, 1u, n_tile / 16u);
        } else {
          auto b_src =
              b1 + (static_cast<uint64_t>(n0 / kC0) * (k / 16u) + (k0 / 16u)) *
                       kFrac;
          l1_to_l0_b<kArch, InT, false, CubeFormat::ZN, CubeFormat::NZ> load_b(
              l0b, b_src, n_tile, k_tile, k / 16u, 1u, 1u, n_tile / kC0);
        }

        auto mte1_ev = mte1_m_ev(abb);
        set_flag(PIPE_MTE1, PIPE_M, mte1_ev);
        wait_flag(PIPE_MTE1, PIPE_M, mte1_ev);
        if ((m_tile / 16u) * (n_tile / 16u) < 10u) {
          pipe_barrier(PIPE_M);
        }
        mmad<kArch, InT, InT, AccT, false> run(l0c, l0a, l0b, m_tile, n_tile,
                                               k_tile, k0 == 0u);
        set_flag(PIPE_M, PIPE_MTE1, bank_ev(abb));

        ++ab_step;
        k0 += k_tile;
      }

      auto fix_ev = m_fix_ev(cb);
      set_flag(PIPE_M, PIPE_FIX, fix_ev);
      wait_flag(PIPE_M, PIPE_FIX, fix_ev);
      auto dst_ptr = gm_out + static_cast<uint64_t>(m0) * dst_stride + n0;
      if constexpr (kUseDeqF16) {
        copy_matrix_cc_to_gm(dst_ptr, l0c, 0, n_tile, m_tile, dst_stride,
                             m_tile, 0, DEQF16, 0, false, true);
      } else {
        copy_matrix_cc_to_gm(dst_ptr, l0c, 0, n_tile, m_tile, dst_stride,
                             m_tile, 0, NoQuant, 0, false, true);
      }
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
  pipe_barrier(PIPE_MTE1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
}
} // namespace

extern "C" {

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_matmul_l1_to_gm_s8_s32(memref_t<__cbuf__ int8_t, 2> *lhs,
                                           memref_t<__cbuf__ int8_t, 2> *rhs,
                                           uint64_t out_gm, uint32_t m,
                                           uint32_t n, uint32_t k,
                                           uint32_t dst_stride,
                                           uint32_t trans_b) {
  matmul_l1_to_gm_impl<int8_t, int32_t, int32_t, false>(
      lhs, rhs, out_gm, m, n, k, dst_stride, trans_b, 0.0f);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_matmul_l1_to_gm_s8_deqf16(
    memref_t<__cbuf__ int8_t, 2> *lhs, memref_t<__cbuf__ int8_t, 2> *rhs,
    uint64_t out_gm, uint32_t m, uint32_t n, uint32_t k, uint32_t dst_stride,
    uint32_t trans_b, float deq_scale) {
  matmul_l1_to_gm_impl<int8_t, int32_t, half, true>(
      lhs, rhs, out_gm, m, n, k, dst_stride, trans_b, deq_scale);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_matmul_l1_to_gm_bf16_f32(
    memref_t<__cbuf__ bfloat16_t, 2> *lhs,
    memref_t<__cbuf__ bfloat16_t, 2> *rhs, uint64_t out_gm, uint32_t m,
    uint32_t n, uint32_t k, uint32_t dst_stride, uint32_t trans_b) {
  matmul_l1_to_gm_impl<bfloat16_t, float, float, false>(
      lhs, rhs, out_gm, m, n, k, dst_stride, trans_b, 0.0f);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_matmul_l1_to_gm_f16_f32(memref_t<__cbuf__ half, 2> *lhs,
                                            memref_t<__cbuf__ half, 2> *rhs,
                                            uint64_t out_gm, uint32_t m,
                                            uint32_t n, uint32_t k,
                                            uint32_t dst_stride,
                                            uint32_t trans_b) {
  matmul_l1_to_gm_impl<half, float, float, false>(lhs, rhs, out_gm, m, n, k,
                                                  dst_stride, trans_b, 0.0f);
}

} // extern "C"
