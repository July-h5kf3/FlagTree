// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIC/Cube) into custom_ops.bc.
// Stateless chunk-pipelined paged block-gather + L1-ZN x L1-ZN^T -> GM matmul
// on Ascend910B (dav-c220-cube).
//
// Streams `num_blocks` indexed `[block_rows, k]` row-blocks from GM into
// compiler-allocated L1/CBUF ZN buffers on PIPE_MTE2 while overlapping
// `lhs_l1 @ rhs_l1^T -> out_gm` on PIPE_MTE1/M/FIX per chunk (2 blocks = 128
// rows when divisible). When `preloaded_rhs0_chunks != 0`, chunk 0 of `rhs0`
// is already resident in `rhs0_l1` (e.g. prefetched during the preceding
// matmul's final column), allowing the first `lhs0 @ rhs0^T` MMA to launch
// immediately at t=0 while `rhs1` and remaining chunks stream concurrently.
// Zero get_imm() L1 addressing, zero cross-call state.
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

__aicore__ __attribute__((always_inline)) event_t chunk_ev(uint32_t slot) {
  return slot == 0u ? EVENT_ID0 : (slot == 1u ? EVENT_ID1 : EVENT_ID2);
}

template <typename CopyT>
__aicore__ __attribute__((always_inline)) void
issue_gather_chunk(__cbuf__ CopyT *l1_dst, __gm__ CopyT *gm_src,
                   __gm__ const int32_t *ids, uint32_t start_block,
                   uint32_t chunk_blocks, uint32_t valid_blocks,
                   uint32_t block_rows, uint32_t total_rows, uint32_t cols,
                   uint32_t src_block_stride) {
  constexpr uint32_t kC0 = 32u / sizeof(CopyT);
  const uint32_t last_valid = valid_blocks > 0u ? (valid_blocks - 1u) : 0u;
  for (uint32_t i = 0u; i < chunk_blocks; ++i) {
    uint32_t b = start_block + i;
    uint32_t idx = (b < valid_blocks) ? b : last_valid;
    int64_t bid = static_cast<int64_t>(ids[idx]);
    gm_to_l1<kArch, CopyT, CubeFormat::ND, CubeFormat::ZN> copy(
        l1_dst + static_cast<uint64_t>(b) * block_rows * kC0,
        gm_src + static_cast<uint64_t>(bid) * src_block_stride, block_rows,
        total_rows, block_rows, cols, cols, cols);
  }
}

template <typename InT, typename AccT, typename OutT>
__aicore__ __attribute__((always_inline)) void
matmul_chunk_trans_b(__cbuf__ InT *a1, __cbuf__ InT *b1, __gm__ OutT *gm_out,
                     uint32_t m, uint32_t total_n, uint32_t n_start,
                     uint32_t chunk_n, uint32_t k, uint32_t dst_stride,
                     uint32_t &ab_step, uint32_t &c_step) {
  constexpr uint32_t kC0 = 32u / sizeof(InT);
  constexpr uint32_t kFrac = 512u / sizeof(InT);

  for (uint32_t m0 = 0u; m0 < m;) {
    uint32_t m_tile = (m - m0 > 128u) ? 128u : (m - m0);
    for (uint32_t n_off = 0u; n_off < chunk_n;) {
      uint32_t n_tile = (chunk_n - n_off > 128u) ? 128u : (chunk_n - n_off);
      uint32_t n0 = n_start + n_off;
      uint32_t k_lim_a = (32768u / (m_tile * sizeof(InT))) & ~(kC0 - 1u);
      uint32_t k_lim_b = (32768u / (n_tile * sizeof(InT))) & ~(kC0 - 1u);
      uint32_t k_cap = (k_lim_a < k_lim_b) ? k_lim_a : k_lim_b;

      uint32_t cb = c_step & 1u;
      wait_flag(PIPE_FIX, PIPE_M, bank_ev(cb));
      auto l0c = l0c_bank<AccT>(cb);

      for (uint32_t k0 = 0u; k0 < k;) {
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

        auto b_src = b1 + (static_cast<uint64_t>(k0 / kC0) * (total_n / 16u) +
                           (n0 / 16u)) *
                              kFrac;
        l1_to_l0_b<kArch, InT, true, CubeFormat::ZN, CubeFormat::NZ> load_b(
            l0b, b_src, n_tile, k_tile, 1u, total_n / 16u, 1u, n_tile / 16u);

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
      copy_matrix_cc_to_gm(dst_ptr, l0c, 0, n_tile, m_tile, dst_stride, m_tile,
                           0, NoQuant, 0, false, true);
      set_flag(PIPE_FIX, PIPE_M, bank_ev(cb));

      ++c_step;
      n_off += n_tile;
    }
    m0 += m_tile;
  }
}
} // namespace

extern "C" {

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_blocks_matmul_pair_s8s32_bf16f32(
    memref_t<__cbuf__ int8_t, 2> *lhs0, uint64_t rhs0_gm,
    memref_t<__cbuf__ bfloat16_t, 2> *lhs1, uint64_t rhs1_gm,
    uint64_t block_ids_gm, uint64_t out0_gm, uint64_t out1_gm,
    uint32_t num_blocks, uint32_t valid_blocks, uint32_t block_rows, uint32_t m,
    uint32_t k0, uint32_t k1, uint32_t src0_block_stride,
    uint32_t src1_block_stride, uint32_t dst_stride,
    uint32_t preloaded_rhs0_chunks, memref_t<__cbuf__ int8_t, 2> *rhs0_l1,
    memref_t<__cbuf__ bfloat16_t, 2> *rhs1_l1) {
  auto a0 = lhs0->aligned + lhs0->offset;
  auto a1 = lhs1->aligned + lhs1->offset;
  auto b0 = rhs0_l1->aligned + rhs0_l1->offset;
  auto b1 = rhs1_l1->aligned + rhs1_l1->offset;
  auto gm_rhs0 = reinterpret_cast<__gm__ int8_t *>(rhs0_gm);
  auto gm_rhs1 = reinterpret_cast<__gm__ half *>(rhs1_gm);
  auto l1_b1_h = reinterpret_cast<__cbuf__ half *>(b1);
  auto ids = reinterpret_cast<__gm__ const int32_t *>(block_ids_gm);
  auto gm_out0 = reinterpret_cast<__gm__ int32_t *>(out0_gm);
  auto gm_out1 = reinterpret_cast<__gm__ float *>(out1_gm);

  const uint32_t total_n = num_blocks * block_rows;
  const uint32_t chunk_blocks =
      ((num_blocks & 1u) == 0u && (2u * block_rows <= 128u)) ? 2u : 1u;
  const uint32_t num_chunks = num_blocks / chunk_blocks;
  const uint32_t chunk_n = chunk_blocks * block_rows;
  const bool chunk0_preloaded = (preloaded_rhs0_chunks != 0u);

  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID7);

  // Issue up to 3 chunks ahead on PIPE_MTE2. When chunk 0's rhs0 is already
  // preloaded in rhs0_l1, skip its DMA so lhs0 @ rhs0^T launches immediately.
  const uint32_t init_chunks = (num_chunks < 3u) ? num_chunks : 3u;
  for (uint32_t c = 0u; c < init_chunks; ++c) {
    uint32_t sb = c * chunk_blocks;
    if (!(c == 0u && chunk0_preloaded)) {
      issue_gather_chunk<int8_t>(b0, gm_rhs0, ids, sb, chunk_blocks,
                                 valid_blocks, block_rows, total_n, k0,
                                 src0_block_stride);
      if (c == 0u) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      }
    }
    issue_gather_chunk<half>(l1_b1_h, gm_rhs1, ids, sb, chunk_blocks,
                             valid_blocks, block_rows, total_n, k1,
                             src1_block_stride);
    set_flag(PIPE_MTE2, PIPE_MTE1, chunk_ev(c));
  }

  uint32_t ab_step = 0u;
  uint32_t c_step = 0u;
  for (uint32_t c = 0u; c < num_chunks; ++c) {
    const uint32_t slot = c % 3u;
    const uint32_t n_start = c * chunk_n;

    if (c == 0u) {
      if (!chunk0_preloaded) {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      }
      matmul_chunk_trans_b<int8_t, int32_t, int32_t>(
          a0, b0, gm_out0, m, total_n, n_start, chunk_n, k0, dst_stride,
          ab_step, c_step);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      if (c + 3u < num_chunks) {
        uint32_t nb = (c + 3u) * chunk_blocks;
        issue_gather_chunk<int8_t>(b0, gm_rhs0, ids, nb, chunk_blocks,
                                   valid_blocks, block_rows, total_n, k0,
                                   src0_block_stride);
        issue_gather_chunk<half>(l1_b1_h, gm_rhs1, ids, nb, chunk_blocks,
                                 valid_blocks, block_rows, total_n, k1,
                                 src1_block_stride);
        set_flag(PIPE_MTE2, PIPE_MTE1, chunk_ev(slot));
      }
      matmul_chunk_trans_b<bfloat16_t, float, float>(
          a1, b1, gm_out1, m, total_n, n_start, chunk_n, k1, dst_stride,
          ab_step, c_step);
    } else {
      wait_flag(PIPE_MTE2, PIPE_MTE1, chunk_ev(slot));
      if (c + 3u < num_chunks) {
        uint32_t nb = (c + 3u) * chunk_blocks;
        issue_gather_chunk<int8_t>(b0, gm_rhs0, ids, nb, chunk_blocks,
                                   valid_blocks, block_rows, total_n, k0,
                                   src0_block_stride);
        issue_gather_chunk<half>(l1_b1_h, gm_rhs1, ids, nb, chunk_blocks,
                                 valid_blocks, block_rows, total_n, k1,
                                 src1_block_stride);
        set_flag(PIPE_MTE2, PIPE_MTE1, chunk_ev(slot));
      }
      matmul_chunk_trans_b<int8_t, int32_t, int32_t>(
          a0, b0, gm_out0, m, total_n, n_start, chunk_n, k0, dst_stride,
          ab_step, c_step);
      matmul_chunk_trans_b<bfloat16_t, float, float>(
          a1, b1, gm_out1, m, total_n, n_start, chunk_n, k1, dst_stride,
          ab_step, c_step);
    }
  }

  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID7);
  pipe_barrier(PIPE_MTE1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
}

} // extern "C"
