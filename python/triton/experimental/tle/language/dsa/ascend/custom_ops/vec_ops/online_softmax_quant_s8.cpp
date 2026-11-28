// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIV/Vector) into custom_ops.bc.
// Stateless fused score scaling, exact FP32 online softmax, and dynamic per-row
// INT8 quantization on Ascend910B (dav-c220-vec).
//
// Operates on arbitrary `[total_rows, cols]` tiles (`total_rows % 8 == 0`,
// `cols % 64 == 0`) with 2-bank MTE2/V/MTE3 ping-pong over 8- or 16-row tiles,
// using strictly compiler-allocated UB buffers (zero `get_imm()` UB addressing
// and zero cross-call state).
//
// Optional GM arguments:
//   - `col_scale_gm` + `col_block_ids_gm` (when both != 0): gathers `cols/64`
//     64-element blocks of FP32 column scales from GM overlapped with Tile 0
//     DMA; otherwise consumes `col_scale_ub` directly.
//   - `flag_gm` (when != 0): writes a 32-byte int32 indicator (`1` if `p_lo_gm`
//     was written, else `0`) directly to GM.
//
// Flags bitmask:
//   bit 0 (1):  is_first         - initialize online softmax state
//   bit 1 (2):  use_shifted      - shifted 8-bit [-128, 127] (0 -> unshifted
//                                  [0, 127] unsigned 7-bit)
//   bit 2 (4):  force_two_digit  - always emit 2-digit FP32 quantization
//   bit 3 (8):  running_norm     - normalize step weights by running l_now
//   bit 4 (16): group_scale_fast - allow shared group scale (ks_max / 127) on
//                                  tight-range unshifted 384-col tiles
#include "Utils.h"

namespace {

__aicore__ __attribute__((always_inline)) void mask_f32_full() {
  set_mask_norm();
  set_vector_mask(0, ~uint64_t(0));
}

__aicore__ __attribute__((always_inline)) void mask_b16_full() {
  set_mask_norm();
  set_vector_mask(~uint64_t(0), ~uint64_t(0));
}

__aicore__ __attribute__((always_inline)) void mask_short_dyn(uint32_t count) {
  set_mask_norm();
  uint64_t lo = (count >= 64u) ? ~uint64_t(0) : ((uint64_t(1) << count) - 1u);
  set_vector_mask(0, lo);
}

__aicore__ __attribute__((always_inline)) event_t bank_ev(uint32_t bank) {
  return bank ? EVENT_ID4 : EVENT_ID3;
}

__aicore__ __attribute__((always_inline)) void
bcast_row_scalars_f32(__ubuf__ float *dst, __ubuf__ float *src, uint32_t rows) {
  vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(dst),
        reinterpret_cast<__ubuf__ uint32_t *>(src), 1, 8, rows / 8u);
}

__aicore__ __attribute__((always_inline)) void
bcast_row_scalars_f16(__ubuf__ half *dst, __ubuf__ half *src, uint32_t rows) {
  vbrcb(reinterpret_cast<__ubuf__ uint16_t *>(dst),
        reinterpret_cast<__ubuf__ uint16_t *>(src), 1, 8, rows / 8u);
}

__aicore__ __attribute__((always_inline)) void
fill_tail_cols_f32(__ubuf__ float *dst, uint32_t valid, float fill,
                   uint32_t rows, uint32_t cols) {
  uint32_t row_blk = cols / 8u;
  for (uint32_t c = 0u; c < cols; c += 64u) {
    if (valid >= c + 64u)
      continue;
    uint64_t m = ~uint64_t(0);
    if (valid > c)
      m <<= (valid - c);
    set_vector_mask(0, m);
    vector_dup(dst + c, fill, rows, 1, 1, row_blk, 0);
  }
  mask_f32_full();
}

__aicore__ __attribute__((always_inline)) void
reduce_row_max_f32(__ubuf__ float *out, __ubuf__ float *src, __ubuf__ float *t0,
                   uint32_t rows, uint32_t cols) {
  mask_f32_full();
  if (cols == 64u) {
    vcmax(out, src, rows, 1, 1, 8, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);
    return;
  }
  uint32_t row_blk = cols / 8u;
  uint32_t stride_t = rows * 64u;
  if (cols == 256u) {
    __ubuf__ float *t1 = t0 + stride_t;
    vmax(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vmax(t1, src + 128u, src + 192u, rows, 1, 1, 1, 8, row_blk, row_blk);
    pipe_barrier(PIPE_V);
    vmax(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcmax(out, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);
    return;
  }
  if (cols == 384u) {
    __ubuf__ float *t1 = t0 + stride_t;
    __ubuf__ float *t2 = t1 + stride_t;
    vmax(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vmax(t1, src + 128u, src + 192u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vmax(t2, src + 256u, src + 320u, rows, 1, 1, 1, 8, row_blk, row_blk);
    pipe_barrier(PIPE_V);
    vmax(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmax(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcmax(out, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);
    return;
  }
  vmax(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
  for (uint32_t c = 128u; c < cols; c += 64u) {
    pipe_barrier(PIPE_V);
    vmax(t0, t0, src + c, rows, 1, 1, 1, 8, 8, row_blk);
  }
  pipe_barrier(PIPE_V);
  vcmax(out, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) void
reduce_row_min_f32(__ubuf__ float *out, __ubuf__ float *src, __ubuf__ float *t0,
                   uint32_t rows, uint32_t cols) {
  mask_f32_full();
  if (cols == 64u) {
    vcmin(out, src, rows, 1, 1, 8, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);
    return;
  }
  uint32_t row_blk = cols / 8u;
  uint32_t stride_t = rows * 64u;
  if (cols == 384u) {
    __ubuf__ float *t1 = t0 + stride_t;
    __ubuf__ float *t2 = t1 + stride_t;
    vmin(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vmin(t1, src + 128u, src + 192u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vmin(t2, src + 256u, src + 320u, rows, 1, 1, 1, 8, row_blk, row_blk);
    pipe_barrier(PIPE_V);
    vmin(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmin(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcmin(out, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);
    return;
  }
  vmin(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
  for (uint32_t c = 128u; c < cols; c += 64u) {
    pipe_barrier(PIPE_V);
    vmin(t0, t0, src + c, rows, 1, 1, 1, 8, 8, row_blk);
  }
  pipe_barrier(PIPE_V);
  vcmin(out, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) void
reduce_row_sum_f32(__ubuf__ float *out, __ubuf__ float *src, __ubuf__ float *t0,
                   uint32_t rows, uint32_t cols) {
  mask_f32_full();
  if (cols == 64u) {
    vcadd(out, src, rows, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    return;
  }
  uint32_t row_blk = cols / 8u;
  uint32_t stride_t = rows * 64u;
  if (cols == 256u) {
    __ubuf__ float *t1 = t0 + stride_t;
    vadd(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vadd(t1, src + 128u, src + 192u, rows, 1, 1, 1, 8, row_blk, row_blk);
    pipe_barrier(PIPE_V);
    vadd(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcadd(out, t0, rows, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    return;
  }
  if (cols == 384u) {
    __ubuf__ float *t1 = t0 + stride_t;
    __ubuf__ float *t2 = t1 + stride_t;
    vadd(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vadd(t1, src + 128u, src + 192u, rows, 1, 1, 1, 8, row_blk, row_blk);
    vadd(t2, src + 256u, src + 320u, rows, 1, 1, 1, 8, row_blk, row_blk);
    pipe_barrier(PIPE_V);
    vadd(t0, t0, t1, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(t0, t0, t2, rows, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vcadd(out, t0, rows, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    return;
  }
  vadd(t0, src, src + 64u, rows, 1, 1, 1, 8, row_blk, row_blk);
  for (uint32_t c = 128u; c < cols; c += 64u) {
    pipe_barrier(PIPE_V);
    vadd(t0, t0, src + c, rows, 1, 1, 1, 8, 8, row_blk);
  }
  pipe_barrier(PIPE_V);
  vcadd(out, t0, rows, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);
}

__aicore__ __attribute__((always_inline)) void
reduce_row_max_f16(__ubuf__ half *out, __ubuf__ half *src, __ubuf__ half *t0,
                   uint32_t rows, uint32_t cols) {
  mask_b16_full();
  if (cols == 128u) {
    vcmax(out, src, rows, 1, 1, 8, Order_t::ONLY_VALUE);
    pipe_barrier(PIPE_V);
    return;
  }
  uint32_t row_blk = cols / 16u;
  vmax(t0, src, src + 128u, rows, 1, 1, 1, 8, row_blk, row_blk);
  for (uint32_t c = 256u; c < cols; c += 128u) {
    pipe_barrier(PIPE_V);
    vmax(t0, t0, src + c, rows, 1, 1, 1, 8, 8, row_blk);
  }
  pipe_barrier(PIPE_V);
  vcmax(out, t0, rows, 1, 1, 8, Order_t::ONLY_VALUE);
  pipe_barrier(PIPE_V);
}

#define FOR_CHUNKS_F32(N_ELEMS)                                                \
  for (uint32_t off = 0u, reps = 0u;                                           \
       off < (N_ELEMS) && ((reps = ((N_ELEMS) - off) / 64u),                   \
                           (reps = reps > 255u ? 255u : reps), true);          \
       off += reps * 64u)

#define FOR_CHUNKS_F16(N_ELEMS)                                                \
  for (uint32_t off = 0u, reps = 0u;                                           \
       off < (N_ELEMS) && ((reps = ((N_ELEMS) - off) / 128u),                  \
                           (reps = reps > 255u ? 255u : reps), true);          \
       off += reps * 128u)

__aicore__ __attribute__((always_inline)) void
issue_tile_load(__ubuf__ float *bank_base, uint64_t c_gm, uint64_t r_gm,
                uint32_t tile_off_elems, uint32_t tile_elems, uint32_t bank) {
  uint32_t burst = tile_elems / 8u;
  copy_gm_to_ubuf(reinterpret_cast<__ubuf__ int32_t *>(bank_base),
                  reinterpret_cast<__gm__ int32_t *>(c_gm) + tile_off_elems, 0,
                  1, burst, 0, 0);
  if (r_gm != 0u) {
    copy_gm_to_ubuf(bank_base + tile_elems,
                    reinterpret_cast<__gm__ float *>(r_gm) + tile_off_elems, 0,
                    1, burst, 0, 0);
  }
  set_flag(PIPE_MTE2, PIPE_V, bank_ev(bank));
}

} // namespace

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_online_softmax_quant_s8(
    uint64_t c_gm, uint64_t r_gm, uint64_t row_scale_gm, uint64_t col_scale_gm,
    uint64_t col_block_ids_gm, uint64_t flag_gm,
    memref_t<__ubuf__ float, 2> *col_scale_ub,
    memref_t<__ubuf__ float, 2> *state_in, uint64_t p_hi_gm, uint64_t p_lo_gm,
    uint32_t total_rows, uint32_t cols, uint32_t valid_cols, float scale,
    float out_scale, uint32_t flags, memref_t<__ubuf__ float, 2> *state_out,
    memref_t<__ubuf__ float, 1> *work_ub) {
  const bool is_first = (flags & 1u) != 0u;
  const bool use_shifted = (flags & 2u) != 0u;
  const bool force_two_digit = (flags & 4u) != 0u;
  const bool running_norm = (flags & 8u) != 0u;
  const bool allow_group_fast = (flags & 16u) != 0u;

  const uint32_t tile_rows = (total_rows >= 128u || cols > 384u)
                                 ? 8u
                                 : ((total_rows >= 16u) ? 16u : 8u);
  const uint32_t num_tiles = total_rows / tile_rows;
  const uint32_t tile_elems = tile_rows * cols;
  const uint32_t bank_floats = 2u * tile_elems;
  const uint32_t num_banks = (num_tiles >= 2u) ? 2u : 1u;

  auto st_in = state_in->aligned + state_in->offset;
  const int64_t st_in_stride = state_in->strides[0];
  auto st_out = state_out->aligned + state_out->offset;
  const int64_t st_out_stride = state_out->strides[0];
  auto work = work_ub->aligned + work_ub->offset;

  __ubuf__ float *banks[2] = {work,
                              work + (num_banks == 2u ? bank_floats : 0u)};
  __ubuf__ float *aux = work + num_banks * bank_floats;

  // Partition ephemeral aux region (all 32-byte aligned, multiples of 64 f32).
  uint32_t qs_pad = (total_rows + 63u) & ~63u;
  uint32_t ks_pad = (cols + 63u) & ~63u;
  __ubuf__ float *qs_ub = aux;
  __ubuf__ float *ks_loaded = qs_ub + qs_pad;
  __ubuf__ float *ks_tmp = ks_loaded + ks_pad;
  __ubuf__ half *ks_f16 = reinterpret_cast<__ubuf__ half *>(ks_tmp + ks_pad);
  __ubuf__ float *t0_buf =
      reinterpret_cast<__ubuf__ float *>(ks_f16 + ((cols + 127u) & ~127u));
  __ubuf__ float *tmp = t0_buf + tile_rows * 64u;
  __ubuf__ float *row_max = tmp + 0u;
  __ubuf__ float *beta_f32 = tmp + 64u;
  __ubuf__ float *safe_max = tmp + 128u;
  __ubuf__ float *inv_sum = tmp + 192u;
  __ubuf__ float *high_f32 = tmp + 256u;
  __ubuf__ float *bcast_buf = tmp + 320u;  // tile_rows * 8 floats (<= 128 f32)
  __ubuf__ float *scalar_out = tmp + 448u; // ks_max, ks_sum, ks_min, min_l

  const bool gather_ks = (col_scale_gm != 0u && col_block_ids_gm != 0u);
  __ubuf__ float *ks =
      gather_ks ? ks_loaded : (col_scale_ub->aligned + col_scale_ub->offset);

  mask_f32_full();

  // Ensure any prior PIPE_V consumer of shared work_ub has completed before
  // PIPE_MTE2 overwrites work_ub.
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);

  // Issue initial row_scale, optional paged col_scale, and tile-0 DMAs upfront.
  if (row_scale_gm != 0u) {
    copy_gm_to_ubuf(qs_ub, reinterpret_cast<__gm__ float *>(row_scale_gm), 0, 1,
                    total_rows / 8u, 0, 0);
  }
  if (gather_ks) {
    auto ids = reinterpret_cast<__gm__ const int32_t *>(col_block_ids_gm);
    auto gm_ks = reinterpret_cast<__gm__ float *>(col_scale_gm);
    uint32_t num_b = cols / 64u;
    uint32_t valid_b = (valid_cols + 63u) / 64u;
    uint32_t last_valid = (valid_b > 0u) ? (valid_b - 1u) : 0u;
    for (uint32_t b = 0u; b < num_b; ++b) {
      uint32_t idx = (b < valid_b) ? b : last_valid;
      int64_t bid = static_cast<int64_t>(ids[idx]);
      copy_gm_to_ubuf(ks_loaded + b * 64u, gm_ks + bid * 64u, 0, 1, 8, 0, 0);
    }
  }
  if (row_scale_gm != 0u || gather_ks) {
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
  }
  issue_tile_load(banks[0], c_gm, r_gm, 0u, tile_elems, 0u);

  if (row_scale_gm != 0u || gather_ks) {
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
  }
  if (gather_ks && valid_cols < cols) {
    fill_tail_cols_f32(ks_loaded, valid_cols, 0.0f, 1u, cols);
    pipe_barrier(PIPE_V);
  }

  // Prescale row_scale once across all rows while tile 0 DMA is in flight.
  if (row_scale_gm != 0u) {
    for (uint32_t r0 = 0u; r0 < total_rows; r0 += 64u) {
      uint32_t rem = total_rows - r0;
      uint32_t cnt = (rem > 64u) ? 64u : rem;
      mask_short_dyn(cnt);
      vmuls(qs_ub + r0, qs_ub + r0, scale, 1, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    mask_f32_full();
  }

  // Detect outlier column scale & tight-range ks, then precompute FP16 ks.
  bool use_f16_fast = false;
  bool use_group_fast = false;
  float ks_max_s = 0.0f;
  float max_q = use_shifted ? 255.0f : 127.0f;
  const bool check_group_fast =
      allow_group_fast && !use_shifted && valid_cols >= cols && cols == 384u;
  if (!force_two_digit && (cols % 128u == 0u)) {
    FOR_CHUNKS_F32(cols) { vabs(ks_tmp + off, ks + off, reps, 1, 1, 8, 8); }
    pipe_barrier(PIPE_V);
    if (valid_cols < cols) {
      fill_tail_cols_f32(ks_tmp, valid_cols, 0.0f, 1u, cols);
      pipe_barrier(PIPE_V);
    }
    reduce_row_max_f32(scalar_out, ks_tmp, t0_buf, 1u, cols);
    reduce_row_sum_f32(scalar_out + 8u, ks_tmp, t0_buf, 1u, cols);
    if (check_group_fast) {
      reduce_row_min_f32(scalar_out + 16u, ks, t0_buf, 1u, cols);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    ks_max_s = *reinterpret_cast<volatile __ubuf__ float *>(scalar_out);
    float ks_sum_s =
        *reinterpret_cast<volatile __ubuf__ float *>(scalar_out + 8u);
    float eff_vis = static_cast<float>(
        static_cast<int32_t>((valid_cols < cols) ? valid_cols : cols));
    bool ks_outlier = (ks_max_s * eff_vis) > (4.0f * ks_sum_s);
    if (!ks_outlier && ks_max_s > 0.0f) {
      use_f16_fast = true;
      if (check_group_fast) {
        float ks_min_s =
            *reinterpret_cast<volatile __ubuf__ float *>(scalar_out + 16u);
        if (ks_min_s >= 0.40f * ks_max_s) {
          use_group_fast = true;
        }
      }
      float ks_mul = use_group_fast ? (127.0f / ks_max_s) : (1.0f / ks_max_s);
      mask_f32_full();
      FOR_CHUNKS_F32(cols) {
        vmuls(ks_tmp + off, ks + off, ks_mul, reps, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
      FOR_CHUNKS_F32(cols) {
        vconv_f322f16(ks_f16 + off, ks_tmp + off, reps, 1, 1, 4, 8);
      }
      pipe_barrier(PIPE_V);
    }
  }

  bool any_need_lo = false;
  const uint32_t row_blk_f32 = cols / 8u;
  const uint32_t row_blk_f16 = cols / 16u;
  const uint32_t half_elems = tile_elems / 2u;

  // Main exact-FP32 softmax loop with ZERO per-tile PIPE_V->S stalls.
exact_pass:
  for (uint32_t tile = 0u; tile < num_tiles; ++tile) {
    uint32_t bank = (num_banks == 2u) ? (tile & 1u) : 0u;
    uint32_t next_tile = tile + 1u;

    if (num_banks == 2u && next_tile < num_tiles) {
      uint32_t next_bank = next_tile & 1u;
      if (next_tile >= 2u) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, bank_ev(next_bank));
        set_flag(PIPE_V, PIPE_MTE2, bank_ev(next_bank));
        wait_flag(PIPE_V, PIPE_MTE2, bank_ev(next_bank));
      }
      issue_tile_load(banks[next_bank], c_gm, r_gm, next_tile * tile_elems,
                      tile_elems, next_bank);
    }

    wait_flag(PIPE_MTE2, PIPE_V, bank_ev(bank));

    __ubuf__ float *score = banks[bank];
    __ubuf__ float *rope = score + tile_elems;
    uint32_t row_off = tile * tile_rows;
    uint32_t elem_off = tile * tile_elems;

    // 1. Exact FP32 score assembly: (float(C) + R) * (qs * scale) * ks.
    mask_f32_full();
    FOR_CHUNKS_F32(tile_elems) {
      vconv_s322f32(score + off,
                    reinterpret_cast<__ubuf__ int32_t *>(score) + off, reps, 1,
                    1, 8, 8);
    }
    if (row_scale_gm != 0u) {
      bcast_row_scalars_f32(bcast_buf, qs_ub + row_off, tile_rows);
    }
    pipe_barrier(PIPE_V);

    if (r_gm != 0u) {
      FOR_CHUNKS_F32(tile_elems) {
        vadd(score + off, score + off, rope + off, reps, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);
    }

    if (row_scale_gm != 0u) {
      for (uint32_t c = 0u; c < cols; c += 64u) {
        vmul(score + c, score + c, bcast_buf, tile_rows, 1, 1, 0, row_blk_f32,
             row_blk_f32, 1);
      }
    } else {
      FOR_CHUNKS_F32(tile_elems) {
        vmuls(score + off, score + off, scale, reps, 1, 1, 8, 8);
      }
    }
    pipe_barrier(PIPE_V);

    for (uint32_t c = 0u; c < cols; c += 64u) {
      vmul(score + c, score + c, ks + c, tile_rows, 1, 1, 1, row_blk_f32,
           row_blk_f32, 0);
    }
    pipe_barrier(PIPE_V);

    if (valid_cols < cols) {
      fill_tail_cols_f32(score, valid_cols, -__builtin_inff(), tile_rows, cols);
      pipe_barrier(PIPE_V);
    }

    // 2. Exact FP32 row max, online max stats, shift, vexp, and row sum.
    reduce_row_max_f32(row_max, score, (cols >= 128u) ? rope : t0_buf,
                       tile_rows, cols);

    __ubuf__ float *m_prev = st_in + 0 * st_in_stride + row_off;
    __ubuf__ float *l_prev = st_in + 1 * st_in_stride + row_off;
    __ubuf__ float *m_now = st_out + 0 * st_out_stride + row_off;
    __ubuf__ float *l_now = st_out + 1 * st_out_stride + row_off;
    __ubuf__ float *w_prev = st_out + 2 * st_out_stride + row_off;
    __ubuf__ float *w_cur = st_out + 3 * st_out_stride + row_off;

    if (valid_cols >= cols) {
      // Full valid group: row_max and m_now are always finite; broadcast
      // row_max directly alongside m_now update to eliminate 1 pipe_barrier.
      mask_f32_full();
      bcast_row_scalars_f32(bcast_buf, row_max, tile_rows);
      mask_short_dyn(tile_rows);
      if (is_first) {
        vadds(m_now, row_max, 0.0f, 1, 1, 1, 8, 8);
        vector_dup(w_prev, 0.0f, 1, 1, 1, 8, 0);
        vector_dup(beta_f32, 1.0f, 1, 1, 1, 8, 0);
      } else {
        vmax(m_now, m_prev, row_max, 1, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);

      mask_f32_full();
      for (uint32_t c = 0u; c < cols; c += 64u) {
        vsub(score + c, score + c, bcast_buf, tile_rows, 1, 1, 0, row_blk_f32,
             row_blk_f32, 1);
      }
      if (!is_first) {
        mask_short_dyn(tile_rows);
        vsub(w_prev, m_prev, m_now, 1, 1, 1, 1, 8, 8, 8);
        vsub(beta_f32, row_max, m_now, 1, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);

      if (!is_first) {
        mask_short_dyn(tile_rows);
        vexp(w_prev, w_prev, 1, 1, 1, 8, 8);
        vexp(beta_f32, beta_f32, 1, 1, 1, 8, 8);
      }
    } else {
      mask_short_dyn(tile_rows);
      vmaxs(safe_max, row_max, -1.0e30f, 1, 1, 1, 8, 8);
      if (is_first) {
        vadds(m_now, row_max, 0.0f, 1, 1, 1, 8, 8);
        vector_dup(w_prev, 0.0f, 1, 1, 1, 8, 0);
        vector_dup(beta_f32, 1.0f, 1, 1, 1, 8, 0);
      } else {
        vmax(m_now, m_prev, row_max, 1, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);

      mask_f32_full();
      bcast_row_scalars_f32(bcast_buf, safe_max, tile_rows);
      if (!is_first) {
        mask_short_dyn(tile_rows);
        vmaxs(safe_max, m_now, -1.0e30f, 1, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);

      mask_f32_full();
      for (uint32_t c = 0u; c < cols; c += 64u) {
        vsub(score + c, score + c, bcast_buf, tile_rows, 1, 1, 0, row_blk_f32,
             row_blk_f32, 1);
      }
      if (!is_first) {
        mask_short_dyn(tile_rows);
        vsub(w_prev, m_prev, safe_max, 1, 1, 1, 1, 8, 8, 8);
        vsub(beta_f32, row_max, safe_max, 1, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);

      mask_f32_full();
      FOR_CHUNKS_F32(tile_elems) {
        vmaxs(score + off, score + off, -90.0f, reps, 1, 1, 8, 8);
      }
      if (!is_first) {
        mask_short_dyn(tile_rows);
        vexp(w_prev, w_prev, 1, 1, 1, 8, 8);
        vexp(beta_f32, beta_f32, 1, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
    }

    auto rope_u8 = reinterpret_cast<__ubuf__ uint8_t *>(rope);
    auto p_ub = reinterpret_cast<__ubuf__ int8_t *>(rope_u8 + 3u * tile_elems);

    if (use_f16_fast && valid_cols >= cols && cols == 384u) {
      // Hybrid fast path: FP16 vexp on shifted scores (<= 0) + exact FP32 row
      // sum (`lse_err ~ 4e-5`), followed by either tight-range group-scale or
      // per-row FP16 dynamic INT8 quantization.
      auto score_h = reinterpret_cast<__ubuf__ half *>(rope_u8);
      auto high_h = reinterpret_cast<__ubuf__ half *>(high_f32 + 32u);
      auto bcast_h = reinterpret_cast<__ubuf__ half *>(bcast_buf);
      auto sum_h = reinterpret_cast<__ubuf__ half *>(t0_buf);

      mask_f32_full();
      FOR_CHUNKS_F32(tile_elems) {
        vconv_f322f16(score_h + off, score + off, reps, 1, 1, 4, 8);
      }
      pipe_barrier(PIPE_V);
      mask_b16_full();
      FOR_CHUNKS_F16(tile_elems) {
        vexp(score_h + off, score_h + off, reps, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);

      vadd(sum_h, score_h, score_h + 128u, tile_rows, 1, 1, 1, 8, row_blk_f16,
           row_blk_f16);
      pipe_barrier(PIPE_V);
      vadd(sum_h, sum_h, score_h + 256u, tile_rows, 1, 1, 1, 8, 8, row_blk_f16);
      pipe_barrier(PIPE_V);

      mask_f32_full();
      FOR_CHUNKS_F32(tile_rows * 128u) {
        vconv_f162f32(score + off, sum_h + off, reps, 1, 1, 8, 4);
      }
      pipe_barrier(PIPE_V);

      vadd(score, score, score + 64u, tile_rows, 1, 1, 1, 8, 16, 16);
      mask_b16_full();
      for (uint32_t c = 0u; c < cols; c += 128u) {
        vmul(score_h + c, score_h + c, ks_f16 + c, tile_rows, 1, 1, 1,
             row_blk_f16, row_blk_f16, 0);
      }
      pipe_barrier(PIPE_V);

      if (use_group_fast) {
        mask_f32_full();
        vcadd(row_max, score, tile_rows, 1, 1, 8, 0);
        mask_b16_full();
        FOR_CHUNKS_F16(tile_elems) {
          vconv_f162s8r(p_ub + off, score_h + off, reps, 1, 1, 4, 8);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(p_hi_gm) + elem_off,
                        p_ub, 0, 1, tile_elems / 32u, 0, 0);
        pipe_barrier(PIPE_V);

        mask_short_dyn(tile_rows);
        if (is_first) {
          vadds(l_now, row_max, 0.0f, 1, 1, 1, 8, 8);
        } else {
          vmul(safe_max, row_max, beta_f32, 1, 1, 1, 1, 8, 8, 8);
          vmul(l_now, l_prev, w_prev, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          if (running_norm) {
            vadds(w_prev, l_now, 0.0f, 1, 1, 1, 8, 8);
          }
          vadd(l_now, l_now, safe_max, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);
        float w_scale = (ks_max_s / 127.0f) * out_scale;
        if (running_norm) {
          vmaxs(safe_max, l_now, 1.0e-30f, 1, 1, 1, 8, 8);
          vector_dup(inv_sum, 1.0f, 1, 1, 1, 8, 0);
          pipe_barrier(PIPE_V);
          vdiv(inv_sum, inv_sum, safe_max, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          if (!is_first) {
            vmul(w_prev, w_prev, inv_sum, 1, 1, 1, 1, 8, 8, 8);
          }
          vmul(beta_f32, beta_f32, inv_sum, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
        }
        vmuls(w_cur, beta_f32, w_scale, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
      } else {
        mask_f32_full();
        vcadd(row_max, score, tile_rows, 1, 1, 8, 0);
        reduce_row_max_f16(high_h, score_h,
                           reinterpret_cast<__ubuf__ half *>(t0_buf), tile_rows,
                           cols);

        mask_short_dyn(tile_rows);
        if (is_first) {
          vadds(l_now, row_max, 0.0f, 1, 1, 1, 8, 8);
        } else {
          vmul(safe_max, row_max, beta_f32, 1, 1, 1, 1, 8, 8, 8);
          vmul(l_now, l_prev, w_prev, 1, 1, 1, 1, 8, 8, 8);
        }
        vconv_f162f32(high_f32, high_h, 1, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);

        if (!is_first) {
          if (running_norm) {
            vadds(w_prev, l_now, 0.0f, 1, 1, 1, 8, 8);
          }
          vadd(l_now, l_now, safe_max, 1, 1, 1, 1, 8, 8, 8);
        }
        float w_scale = (ks_max_s / max_q) * out_scale;
        vmuls(w_cur, high_f32, w_scale, 1, 1, 1, 8, 8);
        vmaxs(high_f32, high_f32, 1.0e-30f, 1, 1, 1, 8, 8);
        vector_dup(row_max, max_q, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);

        if (running_norm) {
          vmaxs(safe_max, l_now, 1.0e-30f, 1, 1, 1, 8, 8);
          vector_dup(inv_sum, 1.0f, 1, 1, 1, 8, 0);
        }
        vdiv(high_f32, row_max, high_f32, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        if (running_norm) {
          vdiv(inv_sum, inv_sum, safe_max, 1, 1, 1, 1, 8, 8, 8);
        }
        vconv_f322f16(high_h, high_f32, 1, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        if (running_norm) {
          if (!is_first) {
            vmul(w_prev, w_prev, inv_sum, 1, 1, 1, 1, 8, 8, 8);
          }
          vmul(beta_f32, beta_f32, inv_sum, 1, 1, 1, 1, 8, 8, 8);
        }
        mask_b16_full();
        bcast_row_scalars_f16(bcast_h, high_h, tile_rows);
        pipe_barrier(PIPE_V);

        for (uint32_t c = 0u; c < cols; c += 128u) {
          vmul(score_h + c, score_h + c, bcast_h, tile_rows, 1, 1, 0,
               row_blk_f16, row_blk_f16, 1);
        }
        mask_short_dyn(tile_rows);
        vmul(w_cur, w_cur, beta_f32, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        mask_b16_full();
        if (use_shifted) {
          FOR_CHUNKS_F16(tile_elems) {
            vadds(score_h + off, score_h + off, static_cast<half>(-128.0f),
                  reps, 1, 1, 8, 8);
          }
          pipe_barrier(PIPE_V);
        }
        FOR_CHUNKS_F16(tile_elems) {
          vconv_f162s8r(p_ub + off, score_h + off, reps, 1, 1, 4, 8);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(p_hi_gm) + elem_off,
                        p_ub, 0, 1, tile_elems / 32u, 0, 0);
      }
    } else {
      mask_f32_full();
      FOR_CHUNKS_F32(tile_elems) {
        vexp(score + off, score + off, reps, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
      if (valid_cols < cols) {
        fill_tail_cols_f32(score, valid_cols, 0.0f, tile_rows, cols);
        pipe_barrier(PIPE_V);
      }
      reduce_row_sum_f32(row_max, score, (cols >= 128u) ? rope : t0_buf,
                         tile_rows, cols);

      if (use_f16_fast) {
        auto score_h = reinterpret_cast<__ubuf__ half *>(rope_u8);
        auto high_h = reinterpret_cast<__ubuf__ half *>(high_f32 + 32u);
        auto bcast_h = reinterpret_cast<__ubuf__ half *>(bcast_buf);

        mask_f32_full();
        FOR_CHUNKS_F32(tile_elems) {
          vconv_f322f16(score_h + off, score + off, reps, 1, 1, 4, 8);
        }
        mask_short_dyn(tile_rows);
        if (is_first) {
          vadds(l_now, row_max, 0.0f, 1, 1, 1, 8, 8);
        } else {
          vmul(safe_max, row_max, beta_f32, 1, 1, 1, 1, 8, 8, 8);
          vmul(l_now, l_prev, w_prev, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);

        mask_b16_full();
        for (uint32_t c = 0u; c < cols; c += 128u) {
          vmul(score_h + c, score_h + c, ks_f16 + c, tile_rows, 1, 1, 1,
               row_blk_f16, row_blk_f16, 0);
        }
        if (!is_first) {
          mask_short_dyn(tile_rows);
          if (running_norm) {
            vadds(w_prev, l_now, 0.0f, 1, 1, 1, 8, 8);
          }
          vadd(l_now, l_now, safe_max, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);

        reduce_row_max_f16(high_h, score_h,
                           reinterpret_cast<__ubuf__ half *>(t0_buf), tile_rows,
                           cols);

        mask_short_dyn(tile_rows);
        vconv_f162f32(high_f32, high_h, 1, 1, 1, 8, 4);
        if (running_norm) {
          vmaxs(safe_max, l_now, 1.0e-30f, 1, 1, 1, 8, 8);
          vector_dup(inv_sum, 1.0f, 1, 1, 1, 8, 0);
        }
        pipe_barrier(PIPE_V);

        float w_scale = (ks_max_s / max_q) * out_scale;
        vmuls(w_cur, high_f32, w_scale, 1, 1, 1, 8, 8);
        vmaxs(high_f32, high_f32, 1.0e-30f, 1, 1, 1, 8, 8);
        vector_dup(row_max, max_q, 1, 1, 1, 8, 0);
        if (running_norm) {
          vdiv(inv_sum, inv_sum, safe_max, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);

        vdiv(high_f32, row_max, high_f32, 1, 1, 1, 1, 8, 8, 8);
        if (running_norm) {
          if (!is_first) {
            vmul(w_prev, w_prev, inv_sum, 1, 1, 1, 1, 8, 8, 8);
          }
          vmul(beta_f32, beta_f32, inv_sum, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);

        vconv_f322f16(high_h, high_f32, 1, 1, 1, 4, 8);
        vmul(w_cur, w_cur, beta_f32, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        mask_b16_full();
        bcast_row_scalars_f16(bcast_h, high_h, tile_rows);
        pipe_barrier(PIPE_V);
        for (uint32_t c = 0u; c < cols; c += 128u) {
          vmul(score_h + c, score_h + c, bcast_h, tile_rows, 1, 1, 0,
               row_blk_f16, row_blk_f16, 1);
        }
        pipe_barrier(PIPE_V);
        if (use_shifted) {
          FOR_CHUNKS_F16(tile_elems) {
            vadds(score_h + off, score_h + off, static_cast<half>(-128.0f),
                  reps, 1, 1, 8, 8);
          }
          pipe_barrier(PIPE_V);
        }
        FOR_CHUNKS_F16(tile_elems) {
          vconv_f162s8r(p_ub + off, score_h + off, reps, 1, 1, 4, 8);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(p_hi_gm) + elem_off,
                        p_ub, 0, 1, tile_elems / 32u, 0, 0);
      } else {
        mask_short_dyn(tile_rows);
        if (is_first) {
          vadds(l_now, row_max, 0.0f, 1, 1, 1, 8, 8);
        } else {
          vmul(safe_max, row_max, beta_f32, 1, 1, 1, 1, 8, 8, 8);
          vmul(l_now, l_prev, w_prev, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          if (running_norm) {
            vadds(w_prev, l_now, 0.0f, 1, 1, 1, 8, 8);
          }
          vadd(l_now, l_now, safe_max, 1, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);

        if (running_norm) {
          vmaxs(safe_max, l_now, 1.0e-30f, 1, 1, 1, 8, 8);
          vector_dup(inv_sum, 1.0f, 1, 1, 1, 8, 0);
          pipe_barrier(PIPE_V);
          vdiv(inv_sum, inv_sum, safe_max, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          if (!is_first) {
            vmul(w_prev, w_prev, inv_sum, 1, 1, 1, 1, 8, 8, 8);
          }
          vmul(beta_f32, beta_f32, inv_sum, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
        }
        if (out_scale != 1.0f) {
          vmuls(beta_f32, beta_f32, out_scale, 1, 1, 1, 8, 8);
          pipe_barrier(PIPE_V);
        }

        auto tmp_f32 = reinterpret_cast<__ubuf__ float *>(rope_u8);
        auto h_tmp =
            reinterpret_cast<__ubuf__ half *>(rope_u8 + 2u * tile_elems);
        mask_f32_full();
        for (uint32_t c = 0u; c < cols; c += 64u) {
          vmul(score + c, score + c, ks + c, tile_rows, 1, 1, 1, row_blk_f32,
               row_blk_f32, 0);
        }
        pipe_barrier(PIPE_V);

        reduce_row_max_f32(high_f32, score, (cols >= 128u) ? rope : t0_buf,
                           tile_rows, cols);

        mask_short_dyn(tile_rows);
        vmuls(w_cur, high_f32, 1.0f / max_q, 1, 1, 1, 8, 8);
        vmaxs(high_f32, high_f32, 1.0e-30f, 1, 1, 1, 8, 8);
        vector_dup(inv_sum, max_q, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        vmul(w_cur, w_cur, beta_f32, 1, 1, 1, 1, 8, 8, 8);
        vdiv(high_f32, inv_sum, high_f32, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        mask_f32_full();
        bcast_row_scalars_f32(bcast_buf, high_f32, tile_rows);
        pipe_barrier(PIPE_V);
        for (uint32_t c = 0u; c < cols; c += 64u) {
          vmul(score + c, score + c, bcast_buf, tile_rows, 1, 1, 0, row_blk_f32,
               row_blk_f32, 1);
        }
        pipe_barrier(PIPE_V);
        if (use_shifted) {
          FOR_CHUNKS_F32(tile_elems) {
            vadds(score + off, score + off, -128.0f, reps, 1, 1, 8, 8);
          }
          pipe_barrier(PIPE_V);
        }

        for (uint32_t h = 0u; h < 2u; ++h) {
          uint32_t h_off = h * half_elems;
          mask_f32_full();
          FOR_CHUNKS_F32(half_elems) {
            vconv_f322f16(h_tmp + off, score + h_off + off, reps, 1, 1, 4, 8);
          }
          pipe_barrier(PIPE_V);
          mask_b16_full();
          FOR_CHUNKS_F16(half_elems) {
            vconv_f162s8r(p_ub + h_off + off, h_tmp + off, reps, 1, 1, 4, 8);
          }
          pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(p_hi_gm) + elem_off,
                        p_ub, 0, 1, tile_elems / 32u, 0, 0);

        if (p_lo_gm != 0u) {
          any_need_lo = true;
          for (uint32_t h = 0u; h < 2u; ++h) {
            uint32_t h_off = h * half_elems;
            mask_b16_full();
            FOR_CHUNKS_F16(half_elems) {
              vconv_s82f16(h_tmp + off, p_ub + h_off + off, reps, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            mask_f32_full();
            FOR_CHUNKS_F32(half_elems) {
              vconv_f162f32(tmp_f32 + off, h_tmp + off, reps, 1, 1, 8, 4);
            }
            pipe_barrier(PIPE_V);
            FOR_CHUNKS_F32(half_elems) {
              vsub(score + h_off + off, score + h_off + off, tmp_f32 + off,
                   reps, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);
            FOR_CHUNKS_F32(half_elems) {
              vmuls(score + h_off + off, score + h_off + off, 127.0f, reps, 1,
                    1, 8, 8);
            }
            pipe_barrier(PIPE_V);
          }
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
          for (uint32_t h = 0u; h < 2u; ++h) {
            uint32_t h_off = h * half_elems;
            mask_f32_full();
            FOR_CHUNKS_F32(half_elems) {
              vconv_f322f16(h_tmp + off, score + h_off + off, reps, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);
            mask_b16_full();
            FOR_CHUNKS_F16(half_elems) {
              vconv_f162s8r(p_ub + h_off + off, h_tmp + off, reps, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);
          }
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
          copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(p_lo_gm) + elem_off,
                          p_ub, 0, 1, tile_elems / 32u, 0, 0);
        }
      }
    }

    if (num_banks == 2u) {
      set_flag(PIPE_MTE3, PIPE_MTE2, bank_ev(bank));
    }
  }

  // Single end-of-call peaked-row check across all total_rows in st_out[1],
  // overlapped with the final tile's PIPE_MTE3 store.
  bool trigger_fallback = false;
  if (use_f16_fast && p_lo_gm != 0u && valid_cols > 4u) {
    __ubuf__ float *l_all = st_out + 1 * st_out_stride;
    if (total_rows <= 64u) {
      mask_short_dyn(total_rows);
      vcmin(scalar_out, l_all, 1, 1, 1, 8, Order_t::ONLY_VALUE);
    } else {
      mask_f32_full();
      vmin(t0_buf, l_all, l_all + 64u, 1, 1, 1, 1, 8, 8, 8);
      for (uint32_t r0 = 128u; r0 < total_rows; r0 += 64u) {
        pipe_barrier(PIPE_V);
        vmin(t0_buf, t0_buf, l_all + r0, 1, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);
      vcmin(scalar_out, t0_buf, 1, 1, 1, 8, Order_t::ONLY_VALUE);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float min_l_all = *reinterpret_cast<volatile __ubuf__ float *>(scalar_out);
    if (min_l_all < 2.0f) {
      trigger_fallback = true;
    }
  }

  if (num_banks == 2u) {
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  }

  if (trigger_fallback) {
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
    use_f16_fast = false;
    use_group_fast = false;
    issue_tile_load(banks[0], c_gm, r_gm, 0u, tile_elems, 0u);
    goto exact_pass;
  }

  // Store aggregated `need_lo` flag into state_out row 4 (and optional
  // flag_gm).
  float need_lo_val = any_need_lo ? 1.0f : 0.0f;
  __ubuf__ float *need_row = st_out + 4 * st_out_stride;
  for (uint32_t r0 = 0u; r0 < total_rows; r0 += 64u) {
    uint32_t rem = total_rows - r0;
    uint32_t cnt = (rem > 64u) ? 64u : rem;
    mask_short_dyn(cnt);
    vector_dup(need_row + r0, need_lo_val, 1, 1, 1, 8, 0);
  }
  if (flag_gm != 0u) {
    int32_t flag_i32 = any_need_lo ? 1 : 0;
    mask_short_dyn(8u);
    vector_dup(reinterpret_cast<__ubuf__ int32_t *>(scalar_out), flag_i32, 1, 1,
               1, 8, 0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
    copy_ubuf_to_gm(reinterpret_cast<__gm__ int32_t *>(flag_gm),
                    reinterpret_cast<__ubuf__ int32_t *>(scalar_out), 0, 1, 1,
                    0, 0);
  }
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
  mask_f32_full();
  pipe_barrier(PIPE_V);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
}
