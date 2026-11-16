// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// CANN 9.1 dav_c220/kernel_operator_fixpipe_impl.h:
// SetFixpipePreQuantFlagImpl and the scalar DEQF16 writeback intrinsic.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_copy_l0c2gm_deqf16(uint64_t dst_address, uint32_t src_address,
                                int32_t n_size, int32_t m_size,
                                int32_t dst_stride, int32_t src_stride,
                                int32_t unit_flag, uint64_t scale_bits) {
  set_quant_pre(scale_bits);
  copy_matrix_cc_to_gm(
      reinterpret_cast<__gm__ half *>(dst_address),
      reinterpret_cast<__cc__ int32_t *>(static_cast<uint64_t>(src_address)), 0,
      static_cast<uint16_t>(n_size), static_cast<uint16_t>(m_size),
      static_cast<uint32_t>(dst_stride), static_cast<uint16_t>(src_stride),
      static_cast<uint8_t>(unit_flag), QuantMode_t::DEQF16, 0, false, true);
}
