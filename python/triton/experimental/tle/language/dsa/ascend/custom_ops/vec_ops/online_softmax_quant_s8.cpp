// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIV/Vector) into custom_ops.bc.
// Stateless fused score scaling, FP32 online softmax, and dynamic per-row
// INT8 quantization on Ascend910B (dav-c220-vec).
//
// Operates on arbitrary `[total_rows, cols]` tiles (`total_rows % 8 == 0`,
// `cols % 64 == 0`) with 2-bank MTE2/V/MTE3 ping-pong over 16-row (or 8-row)
// tiles, using strictly compiler-allocated UB buffers (zero `get_imm()` UB
// addressing and zero cross-call state).
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
  for (uint32_t c = 0; c < cols; c += 64u) {
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
    uint64_t c_gm, uint64_t r_gm, uint64_t row_scale_gm,
    memref_t<__ubuf__ float, 2> *col_scale_ub,
    memref_t<__ubuf__ float, 2> *state_in, uint64_t p_hi_gm, uint64_t p_lo_gm,
    uint32_t total_rows, uint32_t cols, uint32_t valid_cols, float scale,
    float out_scale, uint32_t flags, memref_t<__ubuf__ float, 2> *state_out,
    memref_t<__ubuf__ float, 1> *work_ub) {
  const bool is_first = (flags & 1u) != 0u;
  const bool use_shifted = (flags & 2u) != 0u;
  const bool force_two_digit = (flags & 4u) != 0u;
  const bool running_norm = (flags & 8u) != 0u;

  const uint32_t tile_rows = (total_rows >= 16u) ? 16u : 8u;
  const uint32_t num_tiles = total_rows / tile_rows;
  const uint32_t tile_elems = tile_rows * cols;
  const uint32_t bank_floats = 2u * tile_elems;
  const uint32_t num_banks = (num_tiles >= 2u) ? 2u : 1u;

  auto ks = col_scale_ub->aligned + col_scale_ub->offset;
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
  __ubuf__ float *ks_tmp = qs_ub + qs_pad;
  __ubuf__ half *ks_f16 = reinterpret_cast<__ubuf__ half *>(ks_tmp + ks_pad);
  __ubuf__ float *t0_buf =
      reinterpret_cast<__ubuf__ float *>(ks_f16 + ((cols + 127u) & ~127u));
  // t0_buf holds tile_rows * 64 floats (>= 512 f32, enough for 1-row 3-way tree
  // & f16 max).
  __ubuf__ float *tmp = t0_buf + tile_rows * 64u;
  __ubuf__ float *row_max = tmp + 0u;
  __ubuf__ float *beta_f32 = tmp + 64u;
  __ubuf__ float *safe_max = tmp + 128u;
  __ubuf__ float *inv_sum = tmp + 192u;
  __ubuf__ float *high_f32 = tmp + 256u;
  __ubuf__ float *bcast_buf = tmp + 320u;  // tile_rows * 8 floats (<= 128 f32)
  __ubuf__ float *scalar_out = tmp + 448u; // min_sum, ks_max, ks_sum

  mask_f32_full();

  // Issue initial loads: row_scale (if provided) + tile 0 C/R.
  if (row_scale_gm != 0u) {
    copy_gm_to_ubuf(qs_ub, reinterpret_cast<__gm__ float *>(row_scale_gm), 0, 1,
                    total_rows / 8u, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
  }
  issue_tile_load(banks[0], c_gm, r_gm, 0u, tile_elems, 0u);

  // Detect outlier column scale & prepare normalized FP16 ks_f16 when eligible.
  bool use_f16_fast = false;
  float ks_max_s = 0.0f;
  if (use_shifted && !force_two_digit && (cols % 128u == 0u)) {
    FOR_CHUNKS_F32(cols) { vabs(ks_tmp + off, ks + off, reps, 1, 1, 8, 8); }
    pipe_barrier(PIPE_V);
    if (valid_cols < cols) {
      fill_tail_cols_f32(ks_tmp, valid_cols, 0.0f, 1u, cols);
      pipe_barrier(PIPE_V);
    }
    reduce_row_max_f32(scalar_out, ks_tmp, t0_buf, 1u, cols);
    reduce_row_sum_f32(scalar_out + 8u, ks_tmp, t0_buf, 1u, cols);
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
      float inv_ks_max = 1.0f / ks_max_s;
      mask_f32_full();
      FOR_CHUNKS_F32(cols) {
        vmuls(ks_tmp + off, ks + off, inv_ks_max, reps, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
      FOR_CHUNKS_F32(cols) {
        vconv_f322f16(ks_f16 + off, ks_tmp + off, reps, 1, 1, 4, 8);
      }
      pipe_barrier(PIPE_V);
    }
  }

  // Wait and prescale row_scale once across all rows.
  if (row_scale_gm != 0u) {
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
    for (uint32_t r0 = 0; r0 < total_rows; r0 += 64u) {
      uint32_t rem = total_rows - r0;
      uint32_t cnt = (rem > 64u) ? 64u : rem;
      mask_short_dyn(cnt);
      vmuls(qs_ub + r0, qs_ub + r0, scale, 1, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    mask_f32_full();
  }

  bool any_need_lo = false;
  const uint32_t row_blk_f32 = cols / 8u;
  const uint32_t row_blk_f16 = cols / 16u;
  const uint32_t half_elems = tile_elems / 2u;

  for (uint32_t tile = 0; tile < num_tiles; ++tile) {
    uint32_t bank = (num_banks == 2u) ? (tile & 1u) : 0u;
    uint32_t next_tile = tile + 1u;

    // Prefetch next tile into the alternate bank (if 2-bank ping-pong active).
    if (num_banks == 2u && next_tile < num_tiles) {
      uint32_t next_bank = next_tile & 1u;
      if (next_tile >= 2u) {
        // Wait for previous vector + MTE3 uses of next_bank before DMA refill.
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

    // 1. Convert int32 C -> fp32, add optional R, scale by row_scale &
    // col_scale.
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
      for (uint32_t c = 0; c < cols; c += 64u) {
        vmul(score + c, score + c, bcast_buf, tile_rows, 1, 1, 0, row_blk_f32,
             row_blk_f32, 1);
      }
    } else {
      FOR_CHUNKS_F32(tile_elems) {
        vmuls(score + off, score + off, scale, reps, 1, 1, 8, 8);
      }
    }
    pipe_barrier(PIPE_V);

    for (uint32_t c = 0; c < cols; c += 64u) {
      vmul(score + c, score + c, ks + c, tile_rows, 1, 1, 1, row_blk_f32,
           row_blk_f32, 0);
    }
    pipe_barrier(PIPE_V);

    if (valid_cols < cols) {
      fill_tail_cols_f32(score, valid_cols, -__builtin_inff(), tile_rows, cols);
      pipe_barrier(PIPE_V);
    }

    // 2. Online softmax statistics update (FP32).
    reduce_row_max_f32(row_max, score, (cols >= 192u) ? rope : t0_buf,
                       tile_rows, cols);

    __ubuf__ float *m_prev = st_in + 0 * st_in_stride + row_off;
    __ubuf__ float *l_prev = st_in + 1 * st_in_stride + row_off;
    __ubuf__ float *m_now = st_out + 0 * st_out_stride + row_off;
    __ubuf__ float *l_now = st_out + 1 * st_out_stride + row_off;
    __ubuf__ float *w_prev = st_out + 2 * st_out_stride + row_off;
    __ubuf__ float *w_cur = st_out + 3 * st_out_stride + row_off;

    mask_short_dyn(tile_rows);
    if (is_first) {
      vadds(m_now, row_max, 0.0f, 1, 1, 1, 8, 8);
      vector_dup(w_prev, 0.0f, 1, 1, 1, 8, 0);
      vector_dup(beta_f32, 1.0f, 1, 1, 1, 8, 0);
    } else {
      vmax(m_now, m_prev, row_max, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vmaxs(safe_max, m_now, -1.0e30f, 1, 1, 1, 8, 8);
      pipe_barrier(PIPE_V);
      vsub(w_prev, m_prev, safe_max, 1, 1, 1, 1, 8, 8, 8);
      vsub(beta_f32, row_max, safe_max, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vexp(w_prev, w_prev, 1, 1, 1, 8, 8);
      vexp(beta_f32, beta_f32, 1, 1, 1, 8, 8);
    }
    vmaxs(safe_max, row_max, -1.0e30f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    mask_f32_full();
    bcast_row_scalars_f32(bcast_buf, safe_max, tile_rows);
    pipe_barrier(PIPE_V);
    for (uint32_t c = 0; c < cols; c += 64u) {
      vsub(score + c, score + c, bcast_buf, tile_rows, 1, 1, 0, row_blk_f32,
           row_blk_f32, 1);
    }
    pipe_barrier(PIPE_V);
    if (valid_cols < cols) {
      FOR_CHUNKS_F32(tile_elems) {
        vmaxs(score + off, score + off, -90.0f, reps, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
    }
    FOR_CHUNKS_F32(tile_elems) {
      vexp(score + off, score + off, reps, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    if (valid_cols < cols) {
      fill_tail_cols_f32(score, valid_cols, 0.0f, tile_rows, cols);
      pipe_barrier(PIPE_V);
    }

    reduce_row_sum_f32(row_max, score, (cols >= 192u) ? rope : t0_buf,
                       tile_rows, cols);

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

    vcmin(scalar_out, l_now, 1, 1, 1, 8, Order_t::ONLY_VALUE);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);

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

    // 3. Per-row dynamic INT8 quantization (disjoint buffers inside `rope`).
    //    Layout inside `rope` (4 * tile_elems bytes):
    //      Fast FP16 path:
    //        score_h [0 .. 2*tile_elems), p_ub [3*tile_elems .. 4*tile_elems)
    //      Exact FP32 2-digit path (chunked by half_elems = tile_elems/2):
    //        tmp_f32 [0 .. 2*tile_elems), h_tmp [2*tile_elems .. 3*tile_elems),
    //        p_ub    [3*tile_elems .. 4*tile_elems)
    auto rope_u8 = reinterpret_cast<__ubuf__ uint8_t *>(rope);
    auto p_ub = reinterpret_cast<__ubuf__ int8_t *>(rope_u8 + 3u * tile_elems);
    bool do_fp32_two_digit = !use_f16_fast;

    if (use_f16_fast) {
      auto score_h = reinterpret_cast<__ubuf__ half *>(rope_u8);
      auto high_h = reinterpret_cast<__ubuf__ half *>(high_f32 + 32u);
      auto bcast_h = reinterpret_cast<__ubuf__ half *>(bcast_buf);
      mask_f32_full();
      FOR_CHUNKS_F32(tile_elems) {
        vconv_f322f16(score_h + off, score + off, reps, 1, 1, 4, 8);
      }
      pipe_barrier(PIPE_V);

      mask_b16_full();
      for (uint32_t c = 0; c < cols; c += 128u) {
        vmul(score_h + c, score_h + c, ks_f16 + c, tile_rows, 1, 1, 1,
             row_blk_f16, row_blk_f16, 0);
      }
      pipe_barrier(PIPE_V);

      reduce_row_max_f16(high_h, score_h,
                         reinterpret_cast<__ubuf__ half *>(t0_buf), tile_rows,
                         cols);

      mask_short_dyn(tile_rows);
      vconv_f162f32(high_f32, high_h, 1, 1, 1, 8, 4);
      pipe_barrier(PIPE_V);
      vmuls(w_cur, high_f32, ks_max_s / 255.0f, 1, 1, 1, 8, 8);
      vmaxs(high_f32, high_f32, 1.0e-30f, 1, 1, 1, 8, 8);
      vector_dup(inv_sum, 255.0f, 1, 1, 1, 8, 0);
      pipe_barrier(PIPE_V);
      vmul(w_cur, w_cur, beta_f32, 1, 1, 1, 1, 8, 8, 8);
      vdiv(high_f32, inv_sum, high_f32, 1, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      vconv_f322f16(high_h, high_f32, 1, 1, 1, 4, 8);
      pipe_barrier(PIPE_V);

      mask_b16_full();
      bcast_row_scalars_f16(bcast_h, high_h, tile_rows);
      pipe_barrier(PIPE_V);
      for (uint32_t c = 0; c < cols; c += 128u) {
        vmul(score_h + c, score_h + c, bcast_h, tile_rows, 1, 1, 0, row_blk_f16,
             row_blk_f16, 1);
      }
      pipe_barrier(PIPE_V);
      FOR_CHUNKS_F16(tile_elems) {
        vadds(score_h + off, score_h + off, static_cast<half>(-128.0f), reps, 1,
              1, 8, 8);
      }
      pipe_barrier(PIPE_V);
      FOR_CHUNKS_F16(tile_elems) {
        vconv_f162s8r(p_ub + off, score_h + off, reps, 1, 1, 4, 8);
      }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ int8_t *>(p_hi_gm) + elem_off,
                      p_ub, 0, 1, tile_elems / 32u, 0, 0);

      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float min_l_s = *reinterpret_cast<volatile __ubuf__ float *>(scalar_out);
      if (p_lo_gm != 0u && valid_cols > 4u && min_l_s < 2.0f) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        do_fp32_two_digit = true;
      }
    } else {
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    }

    if (do_fp32_two_digit) {
      auto tmp_f32 = reinterpret_cast<__ubuf__ float *>(rope_u8);
      auto h_tmp = reinterpret_cast<__ubuf__ half *>(rope_u8 + 2u * tile_elems);

      mask_f32_full();
      for (uint32_t c = 0; c < cols; c += 64u) {
        vmul(score + c, score + c, ks + c, tile_rows, 1, 1, 1, row_blk_f32,
             row_blk_f32, 0);
      }
      pipe_barrier(PIPE_V);

      reduce_row_max_f32(high_f32, score, (cols >= 192u) ? rope : t0_buf,
                         tile_rows, cols);

      float max_q = use_shifted ? 255.0f : 127.0f;
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
      for (uint32_t c = 0; c < cols; c += 64u) {
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

      for (uint32_t h = 0; h < 2u; ++h) {
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
        for (uint32_t h = 0; h < 2u; ++h) {
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
            vsub(score + h_off + off, score + h_off + off, tmp_f32 + off, reps,
                 1, 1, 1, 8, 8, 8);
          }
          pipe_barrier(PIPE_V);
          FOR_CHUNKS_F32(half_elems) {
            vmuls(score + h_off + off, score + h_off + off, 127.0f, reps, 1, 1,
                  8, 8);
          }
          pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
        for (uint32_t h = 0; h < 2u; ++h) {
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

    if (num_banks == 2u) {
      set_flag(PIPE_MTE3, PIPE_MTE2, bank_ev(bank));
    }
  }

  // Drain all internal ping-pong MTE3->MTE2 events and ensure final MTE3 store
  // from work_ub completes before compiler-managed work_ub can be reused.
  if (num_banks == 2u) {
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  }
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);

  // Store aggregated `need_lo` flag (0.0 or 1.0) into state_out row 4.
  float need_lo_val = any_need_lo ? 1.0f : 0.0f;
  __ubuf__ float *need_row = st_out + 4 * st_out_stride;
  for (uint32_t r0 = 0; r0 < total_rows; r0 += 64u) {
    uint32_t rem = total_rows - r0;
    uint32_t cnt = (rem > 64u) ? 64u : rem;
    mask_short_dyn(cnt);
    vector_dup(need_row + r0, need_lo_val, 1, 1, 1, 8, 0);
  }
  mask_f32_full();
  pipe_barrier(PIPE_V);
}
