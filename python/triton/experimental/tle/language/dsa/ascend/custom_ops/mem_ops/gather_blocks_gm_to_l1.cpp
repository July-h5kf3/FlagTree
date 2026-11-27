// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIC/Cube) into custom_ops.bc.
// Stateless multi-block GM -> L1/CBUF ND-to-ZN gather: copies `num_blocks`
// contiguous `[block_rows, cols]` row-blocks indexed by GM `int32` block IDs
// (clamped to `[0, valid_blocks - 1]`) into a canonical `[num_blocks *
// block_rows, cols]` ZN layout in a compiler-allocated L1 buffer, issuing one
// ND2NZ DMA per block.
#include "Macro/common.h"
#include "Macro/coord.h"
#include "Macro/operators/iterator.h"
#include "Macro/operators/iterators/gm_to_l1_iterator.h"
#include "Macro/params.h"
#include "Utils.h"

namespace {
constexpr ArchType kArch = ArchType::ASCEND_V220;

template <typename T, typename CopyT = T>
__aicore__ __attribute__((always_inline)) void gather_blocks_gm_to_l1_impl(
    uint64_t src_gm, uint64_t block_ids_gm, uint32_t num_blocks,
    uint32_t valid_blocks, uint32_t block_rows, uint32_t cols,
    uint32_t src_block_stride, memref_t<__cbuf__ T, 2> *dst) {
  constexpr uint32_t kC0 = 32 / sizeof(CopyT);
  auto gm_src = reinterpret_cast<__gm__ CopyT *>(src_gm);
  auto ids = reinterpret_cast<__gm__ const int32_t *>(block_ids_gm);
  auto l1_dst = reinterpret_cast<__cbuf__ CopyT *>(dst->aligned + dst->offset);
  const uint32_t total_rows = num_blocks * block_rows;
  const uint32_t last_valid = valid_blocks > 0 ? (valid_blocks - 1) : 0;

  for (uint32_t b = 0; b < num_blocks; ++b) {
    uint32_t idx = (b < valid_blocks) ? b : last_valid;
    int64_t bid = static_cast<int64_t>(ids[idx]);
    gm_to_l1<kArch, CopyT, CubeFormat::ND, CubeFormat::ZN> copy(
        l1_dst + static_cast<uint64_t>(b) * block_rows * kC0,
        gm_src + bid * static_cast<int64_t>(src_block_stride), block_rows,
        total_rows, block_rows, cols, cols, cols);
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
}
} // namespace

extern "C" {

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_blocks_gm_to_l1_s8(
    uint64_t src_gm, uint64_t block_ids_gm, uint32_t num_blocks,
    uint32_t valid_blocks, uint32_t block_rows, uint32_t cols,
    uint32_t src_block_stride, memref_t<__cbuf__ int8_t, 2> *dst) {
  gather_blocks_gm_to_l1_impl<int8_t>(src_gm, block_ids_gm, num_blocks,
                                      valid_blocks, block_rows, cols,
                                      src_block_stride, dst);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_blocks_gm_to_l1_half(
    uint64_t src_gm, uint64_t block_ids_gm, uint32_t num_blocks,
    uint32_t valid_blocks, uint32_t block_rows, uint32_t cols,
    uint32_t src_block_stride, memref_t<__cbuf__ half, 2> *dst) {
  gather_blocks_gm_to_l1_impl<half>(src_gm, block_ids_gm, num_blocks,
                                    valid_blocks, block_rows, cols,
                                    src_block_stride, dst);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_gather_blocks_gm_to_l1_bf16(
    uint64_t src_gm, uint64_t block_ids_gm, uint32_t num_blocks,
    uint32_t valid_blocks, uint32_t block_rows, uint32_t cols,
    uint32_t src_block_stride, memref_t<__cbuf__ bfloat16_t, 2> *dst) {
  gather_blocks_gm_to_l1_impl<bfloat16_t, half>(
      src_gm, block_ids_gm, num_blocks, valid_blocks, block_rows, cols,
      src_block_stride, dst);
}

} // extern "C"
