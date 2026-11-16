// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "Utils.h"

// Test-only producer of a caller-owned 128x128 INT32 L0C tile.
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_fixpipe_test_setup(uint64_t a_address, uint64_t b_address,
                                uint32_t l0c_address) {
  auto *a = reinterpret_cast<__cbuf__ int8_t *>(0);
  auto *b = reinterpret_cast<__cbuf__ int8_t *>(16384);
  copy_gm_to_cbuf_multi_nd2nz_b8(a,
                                 reinterpret_cast<__gm__ int8_t *>(a_address),
                                 0, 1, 128, 128, 0, 128, 128, 1, 0);
  copy_gm_to_cbuf_multi_nd2nz_b8(b,
                                 reinterpret_cast<__gm__ int8_t *>(b_address),
                                 0, 1, 128, 128, 0, 128, 128, 1, 0);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  set_fmatrix(uint64_t(128) | (uint64_t(1) << 16) | (uint64_t(255) << 56));
  img2colv2_cbuf_to_ca(reinterpret_cast<__ca__ int8_t *>(0), a, 128, 128, 0, 0,
                       1, 1, 1, 1, 1, 1, false, false, false, false, 128);
  load_cbuf_to_cb(reinterpret_cast<__cb__ int8_t *>(0), b, 0, 32, 1, 0, 0, 0,
                  inc);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  mad(reinterpret_cast<__cc__ int32_t *>(uint64_t(l0c_address)),
      reinterpret_cast<__ca__ int8_t *>(0),
      reinterpret_cast<__cb__ int8_t *>(0), 128, 128, 128, 0, false, false,
      true);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  set_nd_para(1);
}
