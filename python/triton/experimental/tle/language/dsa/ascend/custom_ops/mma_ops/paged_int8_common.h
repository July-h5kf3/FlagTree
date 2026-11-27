// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef TLE_ASCEND_CUSTOM_OPS_PAGED_INT8_COMMON_H
#define TLE_ASCEND_CUSTOM_OPS_PAGED_INT8_COMMON_H

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

// Double-buffered 6-page (384-token) KV layout in 512 KiB L1 (`M` in {16, 32,
// 64}).
constexpr uint32_t kQueryBase = 0;
constexpr uint32_t kQrBase = 32768;
constexpr uint32_t kKeyBase = 40960;
constexpr uint32_t kKrBase = kKeyBase + 6 * 65536;
constexpr uint32_t kProbBase = kKrBase + 3 * 16384;
static_assert(kQueryBase + 64 * 512 == kQrBase);
static_assert(kQrBase + 64 * 64 * 2 == kKeyBase);
static_assert(kProbBase + 64 * 384 <= 512 * 1024);

// Double-buffered 6-page (384-token) KV layout in 512 KiB L1 for `M == 128`.
// `kProbBase128` aliases `kKrBase128` (48 KiB) once `paged_int8_qk` finishes
// consuming `KR`; `refill_page_pair` reclaims each 16 KiB pair slot in lockstep
// with the final column of `paged_int8_pv`.
constexpr uint32_t kQueryBase128 = 0;
constexpr uint32_t kQrBase128 = 65536;
constexpr uint32_t kKeyBase128 = 81920;
constexpr uint32_t kKrBase128 = kKeyBase128 + 6 * 65536;
constexpr uint32_t kProbBase128 = kKrBase128;
static_assert(kQueryBase128 + 128 * 512 == kQrBase128);
static_assert(kQrBase128 + 128 * 64 * 2 == kKeyBase128);
static_assert(kProbBase128 + 128 * 384 <= 512 * 1024);

template <typename T>
__aicore__ __attribute__((always_inline)) __cbuf__ T *l1_ptr(uint32_t bytes) {
  return reinterpret_cast<__cbuf__ T *>(
      reinterpret_cast<__cbuf__ uint8_t *>(get_imm(0)) + bytes);
}

template <typename T>
__aicore__ __attribute__((always_inline)) __ca__ T *l0a_ptr(uint32_t bytes) {
  return reinterpret_cast<__ca__ T *>(
      reinterpret_cast<__ca__ uint8_t *>(get_imm(0)) + bytes);
}

template <typename T>
__aicore__ __attribute__((always_inline)) __cb__ T *l0b_ptr(uint32_t bytes) {
  return reinterpret_cast<__cb__ T *>(
      reinterpret_cast<__cb__ uint8_t *>(get_imm(0)) + bytes);
}

template <typename T>
__aicore__ __attribute__((always_inline)) __cc__ T *l0c_ptr(uint32_t bytes) {
  return reinterpret_cast<__cc__ T *>(
      reinterpret_cast<__cc__ uint8_t *>(get_imm(0)) + bytes);
}

__aicore__ __attribute__((always_inline)) event_t bank_event(uint32_t bank) {
  return bank ? EVENT_ID1 : EVENT_ID0;
}

__aicore__ __attribute__((always_inline)) event_t pair_event(uint32_t pair) {
  return pair == 0 ? EVENT_ID3 : (pair == 1 ? EVENT_ID4 : EVENT_ID5);
}

__aicore__ __attribute__((always_inline)) void
nd2nz_s8(__cbuf__ int8_t *dst, uint64_t src, uint32_t rows, uint32_t width,
         uint32_t src_stride, uint32_t tile_rows) {
  gm_to_l1<kArch, int8_t, CubeFormat::ND, CubeFormat::ZN> copy(
      dst, reinterpret_cast<__gm__ int8_t *>(src), rows, tile_rows, rows, width,
      width, src_stride);
}

__aicore__ __attribute__((always_inline)) void
nd2nz_b16(__cbuf__ bfloat16_t *dst, uint64_t src, uint32_t rows, uint32_t width,
          uint32_t src_stride, uint32_t tile_rows) {
  gm_to_l1<kArch, half, CubeFormat::ND, CubeFormat::ZN> copy(
      reinterpret_cast<__cbuf__ half *>(dst),
      reinterpret_cast<__gm__ half *>(src), rows, tile_rows, rows, width, width,
      src_stride);
}

template <typename DT>
__aicore__ __attribute__((always_inline)) void
load_l0a(__ca__ DT *dst, __cbuf__ DT *src, uint32_t part, uint32_t k_split,
         uint32_t m, uint32_t sub_k) {
  constexpr uint32_t block = 32 / sizeof(DT);
  l1_to_l0_a<kArch, DT, false, CubeFormat::ZN, CubeFormat::ZZ> copy(
      dst, src + part * k_split * m, m, sub_k, 1, m / 16, sub_k / block, 1);
}

template <typename DT>
__aicore__ __attribute__((always_inline)) void
load_l0b_transposed(__cb__ DT *dst, __cbuf__ DT *src, uint32_t part,
                    uint32_t k_split, uint32_t sub_k, uint32_t n) {
  l1_to_l0_b<kArch, DT, true, CubeFormat::ZN, CubeFormat::NZ> copy(
      dst, src + part * k_split * n, n, sub_k, 1, n / 16, 1, n / 16);
}

__aicore__ __attribute__((always_inline)) void
load_l0b_value_s8(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint32_t n,
                  uint32_t sub_k) {
  l1_to_l0_b<kArch, int8_t, false, CubeFormat::ZN, CubeFormat::NZ> copy(
      dst, src, n, sub_k, sub_k / 16, 1, 1, n / 32);
}

template <typename A, typename B, typename C>
__aicore__ __attribute__((always_inline)) void
mma_step(__cc__ C *dst, __ca__ A *a, __cb__ B *b, uint32_t m, uint32_t n,
         uint32_t k, bool init) {
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
  if ((m / 16) * (n / 16) < 10)
    pipe_barrier(PIPE_M);
  mmad<kArch, A, B, C, false> run(dst, a, b, m, n, k, init);
}

__aicore__ __attribute__((always_inline)) void
mma_pair_s8(__cc__ int32_t *dst_hi, __ca__ int8_t *a_hi, __cc__ int32_t *dst_lo,
            __ca__ int8_t *a_lo, __cb__ int8_t *b, uint32_t m, uint32_t n,
            uint32_t k, bool init) {
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
  if ((m / 16) * (n / 16) < 10)
    pipe_barrier(PIPE_M);
  mmad<kArch, int8_t, int8_t, int32_t, false> run0(dst_hi, a_hi, b, m, n, k,
                                                   init);
  if ((m / 16) * (n / 16) < 10)
    pipe_barrier(PIPE_M);
  mmad<kArch, int8_t, int8_t, int32_t, false> run1(dst_lo, a_lo, b, m, n, k,
                                                   init);
}

template <typename T>
__aicore__ __attribute__((always_inline)) void
store_l0c(uint64_t dst, __cc__ T *src, uint32_t rows, uint32_t cols,
          uint32_t stride) {
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
  copy_matrix_cc_to_gm(reinterpret_cast<__gm__ T *>(dst), src, 0, cols, rows,
                       stride, rows, 0, NoQuant, 0, false, true);
}

// Dequantizes INT32 L0C accumulators by a fixed FP32 scale (raw IEEE-754 bits
// in `deq_scale`) to FP16 in Fixpipe hardware.
__aicore__ __attribute__((always_inline)) void
store_l0c_deqf16(uint64_t dst, __cc__ int32_t *src, uint32_t rows,
                 uint32_t cols, uint32_t stride, uint64_t deq_scale) {
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
  set_quant_pre(deq_scale);
  copy_matrix_cc_to_gm(reinterpret_cast<__gm__ half *>(dst), src, 0, cols, rows,
                       stride, rows, 0, DEQF16, 0, false, true);
}

__aicore__ __attribute__((always_inline)) void
prefetch_page_pair(uint64_t k, uint64_t kr, uint64_t table, uint32_t remaining,
                   uint32_t slot, uint32_t pair, uint32_t key_base = kKeyBase,
                   uint32_t kr_base = kKrBase) {
  auto ids = reinterpret_cast<__gm__ const int32_t *>(table);
  auto kv1 = l1_ptr<int8_t>(key_base + (slot * 3 + pair) * 65536);
  auto rope1 = l1_ptr<bfloat16_t>(kr_base + pair * 16384);
  for (uint32_t page = 0; page < 2; ++page) {
    uint32_t p = pair * 2 + page;
    uint64_t id = ids[p < remaining ? p : remaining - 1];
    nd2nz_s8(kv1 + page * 64 * 32, k + id * 32768, 64, 512, 512, 128);
    nd2nz_b16(rope1 + page * 64 * 16, kr + id * 8192, 64, 64, 64, 128);
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, pair_event(pair));
}

__aicore__ __attribute__((always_inline)) void
refill_page_pair(uint64_t k, uint64_t kr, uint64_t table, uint32_t remaining,
                 uint32_t slot, uint32_t pair, uint32_t key_base = kKeyBase,
                 uint32_t kr_base = kKrBase) {
  auto ev = pair_event(pair);
  set_flag(PIPE_MTE1, PIPE_MTE2, ev);
  wait_flag(PIPE_MTE1, PIPE_MTE2, ev);
  prefetch_page_pair(k, kr, table, remaining, slot, pair, key_base, kr_base);
}
} // namespace

#endif // TLE_ASCEND_CUSTOM_OPS_PAGED_INT8_COMMON_H
