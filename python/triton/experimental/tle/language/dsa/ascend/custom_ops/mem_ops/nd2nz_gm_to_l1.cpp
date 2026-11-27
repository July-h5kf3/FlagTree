// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIC/Cube) into custom_ops.bc.
// Stateless GM -> L1/CBUF ND-to-ZN fractal copy for 2D matrices
// (int8/fp16/bf16) into a compiler-allocated L1 buffer.
#include "Macro/common.h"
#include "Macro/coord.h"
#include "Macro/operators/iterator.h"
#include "Macro/operators/iterators/gm_to_l1_iterator.h"
#include "Macro/params.h"
#include "Utils.h"

namespace {
constexpr ArchType kArch = ArchType::ASCEND_V220;

template <typename T, typename CopyT = T>
__aicore__ __attribute__((always_inline)) void
nd2nz_gm_to_l1_impl(uint64_t src_gm, uint32_t rows, uint32_t cols,
                    uint32_t src_stride, memref_t<__cbuf__ T, 2> *dst) {
  auto gm_src = reinterpret_cast<__gm__ CopyT *>(src_gm);
  auto l1_dst = reinterpret_cast<__cbuf__ CopyT *>(dst->aligned + dst->offset);
  gm_to_l1<kArch, CopyT, CubeFormat::ND, CubeFormat::ZN> copy(
      l1_dst, gm_src, rows, rows, rows, cols, cols, src_stride);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
}
} // namespace

extern "C" {

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_nd2nz_gm_to_l1_s8(uint64_t src_gm, uint32_t rows,
                                      uint32_t cols, uint32_t src_stride,
                                      memref_t<__cbuf__ int8_t, 2> *dst) {
  nd2nz_gm_to_l1_impl<int8_t>(src_gm, rows, cols, src_stride, dst);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_nd2nz_gm_to_l1_half(uint64_t src_gm, uint32_t rows,
                                        uint32_t cols, uint32_t src_stride,
                                        memref_t<__cbuf__ half, 2> *dst) {
  nd2nz_gm_to_l1_impl<half>(src_gm, rows, cols, src_stride, dst);
}

__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_nd2nz_gm_to_l1_bf16(uint64_t src_gm, uint32_t rows,
                                        uint32_t cols, uint32_t src_stride,
                                        memref_t<__cbuf__ bfloat16_t, 2> *dst) {
  nd2nz_gm_to_l1_impl<bfloat16_t, half>(src_gm, rows, cols, src_stride, dst);
}

} // extern "C"
