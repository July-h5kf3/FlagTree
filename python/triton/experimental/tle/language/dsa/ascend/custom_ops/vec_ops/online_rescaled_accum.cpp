// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled standalone (AIV/Vector) into custom_ops.bc.
// Stateless online rescaled accumulator update (`acc = alpha * acc + step *
// (src + bias_scale * bias + lo_scale * src_lo)`) with 2-bank contiguous
// GM->UB DMA ping-pong and optional final BF16 GM writeback on Ascend910B
// (dav-c220-vec), using strictly compiler-allocated UB buffers (zero get_imm()
// UB addressing, zero cross-call state).
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

} // namespace

extern "C" {

// FP16 running-normalized accumulator update:
//   src_gm, src_lo_gm (optional), bias_gm (optional): GM `half*`
//   state_ub: [>=4, total_rows] FP32 (row 2 = w_prev, row 3 = w_cur)
//   out_gm: optional BF16 GM output written when (flags & 2) != 0
//   acc_ub: [total_rows, dim] FP16 UB accumulator (updated in place)
//   work_ub: 1D FP32 UB scratch (>= (nb*tile_rows + lo_rows)*dim/2 + dim/2 +
//            tile_rows*16 + 128 floats, where tile_rows = 16 for rows>=128,
//            32 for 32<=rows<128, else rows; lo_rows = min(tile_rows, 8))
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_online_rescaled_accum_f16(
    uint64_t src_gm, uint64_t src_lo_gm, uint64_t bias_gm,
    memref_t<__ubuf__ float, 2> *state_ub, uint64_t out_gm, uint32_t total_rows,
    uint32_t dim, float bias_scale, float lo_scale, uint32_t flags,
    memref_t<__ubuf__ half, 2> *acc_ub, memref_t<__ubuf__ float, 1> *work_ub) {
  const bool is_first = (flags & 1u) != 0u;
  const bool is_last = (flags & 2u) != 0u;
  const bool first_empty = (flags & 4u) != 0u;
  const bool check_state_need_lo = (flags & 16u) != 0u;

  const uint32_t tile_rows =
      (total_rows >= 128u) ? 16u : ((total_rows >= 32u) ? 32u : total_rows);
  const uint32_t num_tiles = total_rows / tile_rows;
  const uint32_t tile_elems = tile_rows * dim;
  const uint32_t num_banks = (num_tiles >= 2u) ? 2u : 1u;
  const uint32_t row_blk_f16 = dim / 16u;
  const uint32_t lo_rows = (tile_rows >= 8u) ? 8u : tile_rows;
  const uint32_t lo_elems = lo_rows * dim;

  auto st = state_ub->aligned + state_ub->offset;
  const int64_t st_stride = state_ub->strides[0];
  auto acc_base = acc_ub->aligned + acc_ub->offset;
  auto work_u8 =
      reinterpret_cast<__ubuf__ uint8_t *>(work_ub->aligned + work_ub->offset);

  const uint32_t bank_bytes = tile_elems * 2u;
  const uint32_t lo_bytes = lo_elems * 2u;
  __ubuf__ half *banks[2] = {
      reinterpret_cast<__ubuf__ half *>(work_u8),
      reinterpret_cast<__ubuf__ half *>(work_u8 +
                                        (num_banks == 2u ? bank_bytes : 0u))};
  // Compact lo_stage (`lo_rows x dim` half) doubles as BF16 writeback buffer.
  __ubuf__ half *lo_stage =
      reinterpret_cast<__ubuf__ half *>(work_u8 + num_banks * bank_bytes);
  __ubuf__ uint8_t *aux_u8 = work_u8 + num_banks * bank_bytes + lo_bytes;
  __ubuf__ half *bias_h = reinterpret_cast<__ubuf__ half *>(aux_u8);
  __ubuf__ half *bcast_cur =
      reinterpret_cast<__ubuf__ half *>(aux_u8 + dim * 2u);
  __ubuf__ half *bcast_prev =
      reinterpret_cast<__ubuf__ half *>(aux_u8 + dim * 2u + tile_rows * 32u);
  __ubuf__ half *w_cur_h =
      reinterpret_cast<__ubuf__ half *>(aux_u8 + dim * 2u + tile_rows * 64u);
  __ubuf__ half *w_prev_h = w_cur_h + 64u;

  // Ensure any prior PIPE_V consumer of shared work_ub has completed before
  // PIPE_MTE2 overwrites work_ub.
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);

  if (bias_gm != 0u) {
    copy_gm_to_ubuf(bias_h, reinterpret_cast<__gm__ half *>(bias_gm), 0, 1,
                    dim / 16u, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  }
  copy_gm_to_ubuf(banks[0], reinterpret_cast<__gm__ half *>(src_gm), 0, 1,
                  tile_elems / 16u, 0, 0);
  set_flag(PIPE_MTE2, PIPE_V, bank_ev(0u));
  if (check_state_need_lo && src_lo_gm != 0u) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float need_lo_s =
        *reinterpret_cast<volatile __ubuf__ float *>(st + 4 * st_stride);
    if (need_lo_s <= 0.5f) {
      src_lo_gm = 0u;
    }
  }
  if (bias_gm != 0u) {
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    mask_b16_full();
    FOR_CHUNKS_F16(dim) {
      vmuls(bias_h + off, bias_h + off, static_cast<half>(bias_scale), reps, 1,
            1, 8, 8);
    }
    pipe_barrier(PIPE_V);
  }

  for (uint32_t tile = 0u; tile < num_tiles; ++tile) {
    uint32_t bank = (num_banks == 2u) ? (tile & 1u) : 0u;
    uint32_t next_tile = tile + 1u;
    if (num_banks == 2u && next_tile < num_tiles) {
      uint32_t next_bank = next_tile & 1u;
      if (next_tile >= 2u) {
        set_flag(PIPE_V, PIPE_MTE2, bank_ev(next_bank));
        wait_flag(PIPE_V, PIPE_MTE2, bank_ev(next_bank));
      }
      copy_gm_to_ubuf(banks[next_bank],
                      reinterpret_cast<__gm__ half *>(src_gm) +
                          next_tile * tile_elems,
                      0, 1, tile_elems / 16u, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, bank_ev(next_bank));
    }

    wait_flag(PIPE_MTE2, PIPE_V, bank_ev(bank));

    uint32_t row_off = tile * tile_rows;
    __ubuf__ half *upd_h = banks[bank];
    __ubuf__ half *acc_h = acc_base + row_off * dim;

    mask_short_dyn(tile_rows);
    vconv_f322f16(w_cur_h, st + 3 * st_stride + row_off, 1, 1, 1, 4, 8);
    if (!is_first) {
      vconv_f322f16(w_prev_h, st + 2 * st_stride + row_off, 1, 1, 1, 4, 8);
    }
    pipe_barrier(PIPE_V);

    mask_b16_full();
    bcast_row_scalars_f16(bcast_cur, w_cur_h, tile_rows);
    if (!is_first) {
      bcast_row_scalars_f16(bcast_prev, w_prev_h, tile_rows);
    }
    if (bias_gm != 0u) {
      for (uint32_t c = 0u; c < dim; c += 128u) {
        vadd(upd_h + c, upd_h + c, bias_h + c, tile_rows, 1, 1, 1, row_blk_f16,
             row_blk_f16, 0);
      }
    }
    pipe_barrier(PIPE_V);

    if (!is_first) {
      for (uint32_t c = 0u; c < dim; c += 128u) {
        vmul(acc_h + c, acc_h + c, bcast_prev, tile_rows, 1, 1, 0, row_blk_f16,
             row_blk_f16, 1);
      }
    }
    if (src_lo_gm != 0u) {
      for (uint32_t sub = 0u; sub < tile_rows; sub += lo_rows) {
        if (tile > 0u || sub > 0u) {
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        }
        copy_gm_to_ubuf(lo_stage,
                        reinterpret_cast<__gm__ half *>(src_lo_gm) +
                            (row_off + sub) * dim,
                        0, 1, lo_elems / 16u, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        FOR_CHUNKS_F16(lo_elems) {
          vmuls(lo_stage + off, lo_stage + off, static_cast<half>(lo_scale),
                reps, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        FOR_CHUNKS_F16(lo_elems) {
          vadd(upd_h + sub * dim + off, upd_h + sub * dim + off, lo_stage + off,
               reps, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);
      }
    } else if (!is_first) {
      pipe_barrier(PIPE_V);
    }

    if (is_first) {
      for (uint32_t c = 0u; c < dim; c += 128u) {
        vmul(acc_h + c, upd_h + c, bcast_cur, tile_rows, 1, 1, 0, row_blk_f16,
             row_blk_f16, 1);
      }
    } else {
      for (uint32_t c = 0u; c < dim; c += 128u) {
        vmla(acc_h + c, upd_h + c, bcast_cur, tile_rows, 1, 1, 0, row_blk_f16,
             row_blk_f16, 1);
      }
    }
    pipe_barrier(PIPE_V);

    if (is_last && out_gm != 0u) {
      auto out_bf16 = reinterpret_cast<__ubuf__ bfloat16_t *>(lo_stage);
      const uint32_t chunk_r = (tile_rows >= 16u) ? 8u : (tile_rows / 2u);
      const uint32_t chunk_elems = chunk_r * dim;
      for (uint32_t sub = 0u; sub < tile_rows; sub += chunk_r) {
        if (first_empty) {
          mask_b16_full();
          FOR_CHUNKS_F16(chunk_elems) {
            vector_dup(reinterpret_cast<__ubuf__ int16_t *>(out_bf16) + off,
                       static_cast<int16_t>(0), reps, 1, 1, 8, 0);
          }
          pipe_barrier(PIPE_V);
        } else {
          auto tmp_f32 = reinterpret_cast<__ubuf__ float *>(upd_h);
          mask_f32_full();
          FOR_CHUNKS_F32(chunk_elems) {
            vconv_f162f32(tmp_f32 + off, acc_h + sub * dim + off, reps, 1, 1, 8,
                          4);
          }
          pipe_barrier(PIPE_V);
          FOR_CHUNKS_F32(chunk_elems) {
            vconv_f322bf16r(out_bf16 + off, tmp_f32 + off, reps, 1, 1, 4, 8);
          }
          pipe_barrier(PIPE_V);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ bfloat16_t *>(out_gm) +
                            (row_off + sub) * dim,
                        out_bf16, 0, 1, chunk_elems / 16u, 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      }
    }
  }
  mask_f32_full();
  pipe_barrier(PIPE_V);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
}

// FP32 accumulator update (consumes int32 GM src / src_lo):
//   state_ub: [>=4, total_rows] FP32 (row 1 = l, row 2 = alpha, row 3 = step)
//   acc_ub: [total_rows, dim] FP32 UB accumulator (updated in place)
__aicore__ __attribute__((always_inline)) void
_mlir_ciface_custom_online_rescaled_accum_f32(
    uint64_t src_gm, uint64_t src_lo_gm, uint64_t bias_gm,
    memref_t<__ubuf__ float, 2> *state_ub, uint64_t out_gm, uint32_t total_rows,
    uint32_t dim, float bias_scale, float lo_scale, uint32_t flags,
    memref_t<__ubuf__ float, 2> *acc_ub, memref_t<__ubuf__ float, 1> *work_ub) {
  const bool is_first = (flags & 1u) != 0u;
  const bool is_last = (flags & 2u) != 0u;
  const bool first_empty = (flags & 4u) != 0u;
  const bool check_state_need_lo = (flags & 16u) != 0u;
  const bool normalize_out = (flags & 8u) != 0u;

  const uint32_t tile_rows = (total_rows >= 8u) ? 8u : total_rows;
  const uint32_t num_tiles = total_rows / tile_rows;
  const uint32_t tile_elems = tile_rows * dim;
  const uint32_t num_banks = (num_tiles >= 2u) ? 2u : 1u;
  const uint32_t row_blk_f32 = dim / 8u;

  auto st = state_ub->aligned + state_ub->offset;
  const int64_t st_stride = state_ub->strides[0];
  auto acc_base = acc_ub->aligned + acc_ub->offset;
  auto work = work_ub->aligned + work_ub->offset;

  __ubuf__ float *banks[2] = {work, work + (num_banks == 2u ? tile_elems : 0u)};
  __ubuf__ float *lo_stage = work + num_banks * tile_elems;
  __ubuf__ float *aux = work + (num_banks + 1u) * tile_elems;
  __ubuf__ float *bias_f = aux;
  __ubuf__ float *bcast_cur = aux + dim;
  __ubuf__ float *bcast_prev = bcast_cur + tile_rows * 8u;
  __ubuf__ float *inv_l = bcast_prev + tile_rows * 8u;
  __ubuf__ float *safe_l = inv_l + 64u;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);

  if (bias_gm != 0u) {
    copy_gm_to_ubuf(reinterpret_cast<__ubuf__ int32_t *>(bias_f),
                    reinterpret_cast<__gm__ int32_t *>(bias_gm), 0, 1, dim / 8u,
                    0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  }
  copy_gm_to_ubuf(reinterpret_cast<__ubuf__ int32_t *>(banks[0]),
                  reinterpret_cast<__gm__ int32_t *>(src_gm), 0, 1,
                  tile_elems / 8u, 0, 0);
  set_flag(PIPE_MTE2, PIPE_V, bank_ev(0u));
  if (check_state_need_lo && src_lo_gm != 0u) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    float need_lo_s =
        *reinterpret_cast<volatile __ubuf__ float *>(st + 4 * st_stride);
    if (need_lo_s <= 0.5f) {
      src_lo_gm = 0u;
    }
  }
  if (src_lo_gm != 0u) {
    copy_gm_to_ubuf(reinterpret_cast<__ubuf__ int32_t *>(lo_stage),
                    reinterpret_cast<__gm__ int32_t *>(src_lo_gm), 0, 1,
                    tile_elems / 8u, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
  }
  if (bias_gm != 0u) {
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    mask_f32_full();
    FOR_CHUNKS_F32(dim) {
      vconv_s322f32(bias_f + off,
                    reinterpret_cast<__ubuf__ int32_t *>(bias_f) + off, reps, 1,
                    1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    FOR_CHUNKS_F32(dim) {
      vmuls(bias_f + off, bias_f + off, bias_scale, reps, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
  }

  for (uint32_t tile = 0u; tile < num_tiles; ++tile) {
    uint32_t bank = (num_banks == 2u) ? (tile & 1u) : 0u;
    uint32_t next_tile = tile + 1u;
    if (num_banks == 2u && next_tile < num_tiles) {
      uint32_t next_bank = next_tile & 1u;
      if (next_tile >= 2u) {
        set_flag(PIPE_V, PIPE_MTE2, bank_ev(next_bank));
        wait_flag(PIPE_V, PIPE_MTE2, bank_ev(next_bank));
      }
      copy_gm_to_ubuf(reinterpret_cast<__ubuf__ int32_t *>(banks[next_bank]),
                      reinterpret_cast<__gm__ int32_t *>(src_gm) +
                          next_tile * tile_elems,
                      0, 1, tile_elems / 8u, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, bank_ev(next_bank));
    }

    wait_flag(PIPE_MTE2, PIPE_V, bank_ev(bank));
    if (tile > 0u && src_lo_gm != 0u) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      copy_gm_to_ubuf(reinterpret_cast<__ubuf__ int32_t *>(lo_stage),
                      reinterpret_cast<__gm__ int32_t *>(src_lo_gm) +
                          tile * tile_elems,
                      0, 1, tile_elems / 8u, 0, 0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
    }

    uint32_t row_off = tile * tile_rows;
    __ubuf__ float *upd = banks[bank];
    __ubuf__ float *acc = acc_base + row_off * dim;

    mask_f32_full();
    FOR_CHUNKS_F32(tile_elems) {
      vconv_s322f32(upd + off, reinterpret_cast<__ubuf__ int32_t *>(upd) + off,
                    reps, 1, 1, 8, 8);
    }
    bcast_row_scalars_f32(bcast_cur, st + 3 * st_stride + row_off, tile_rows);
    if (!is_first) {
      bcast_row_scalars_f32(bcast_prev, st + 2 * st_stride + row_off,
                            tile_rows);
    }
    pipe_barrier(PIPE_V);

    if (bias_gm != 0u) {
      for (uint32_t c = 0u; c < dim; c += 64u) {
        vadd(upd + c, upd + c, bias_f + c, tile_rows, 1, 1, 1, row_blk_f32,
             row_blk_f32, 0);
      }
    }
    if (!is_first) {
      for (uint32_t c = 0u; c < dim; c += 64u) {
        vmul(acc + c, acc + c, bcast_prev, tile_rows, 1, 1, 0, row_blk_f32,
             row_blk_f32, 1);
      }
    }
    if (bias_gm != 0u || !is_first) {
      pipe_barrier(PIPE_V);
    }

    if (src_lo_gm != 0u) {
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      FOR_CHUNKS_F32(tile_elems) {
        vconv_s322f32(lo_stage + off,
                      reinterpret_cast<__ubuf__ int32_t *>(lo_stage) + off,
                      reps, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
      FOR_CHUNKS_F32(tile_elems) {
        vmuls(lo_stage + off, lo_stage + off, lo_scale, reps, 1, 1, 8, 8);
      }
      pipe_barrier(PIPE_V);
      FOR_CHUNKS_F32(tile_elems) {
        vadd(upd + off, upd + off, lo_stage + off, reps, 1, 1, 1, 8, 8, 8);
      }
      pipe_barrier(PIPE_V);
    }

    if (is_first) {
      for (uint32_t c = 0u; c < dim; c += 64u) {
        vmul(acc + c, upd + c, bcast_cur, tile_rows, 1, 1, 0, row_blk_f32,
             row_blk_f32, 1);
      }
    } else {
      for (uint32_t c = 0u; c < dim; c += 64u) {
        vmla(acc + c, upd + c, bcast_cur, tile_rows, 1, 1, 0, row_blk_f32,
             row_blk_f32, 1);
      }
    }
    pipe_barrier(PIPE_V);

    if (is_last && out_gm != 0u) {
      auto out_bf16 = reinterpret_cast<__ubuf__ bfloat16_t *>(upd);
      if (first_empty) {
        mask_b16_full();
        FOR_CHUNKS_F16(tile_elems) {
          vector_dup(reinterpret_cast<__ubuf__ int16_t *>(out_bf16) + off,
                     static_cast<int16_t>(0), reps, 1, 1, 8, 0);
        }
        pipe_barrier(PIPE_V);
      } else {
        if (normalize_out) {
          mask_short_dyn(tile_rows);
          vmaxs(safe_l, st + 1 * st_stride + row_off, 1.0e-30f, 1, 1, 1, 8, 8);
          vector_dup(inv_l, 1.0f, 1, 1, 1, 8, 0);
          pipe_barrier(PIPE_V);
          vdiv(inv_l, inv_l, safe_l, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          mask_f32_full();
          bcast_row_scalars_f32(bcast_cur, inv_l, tile_rows);
          pipe_barrier(PIPE_V);
          for (uint32_t c = 0u; c < dim; c += 64u) {
            vmul(acc + c, acc + c, bcast_cur, tile_rows, 1, 1, 0, row_blk_f32,
                 row_blk_f32, 1);
          }
          pipe_barrier(PIPE_V);
        }
        mask_f32_full();
        FOR_CHUNKS_F32(tile_elems) {
          vconv_f322bf16r(out_bf16 + off, acc + off, reps, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_V);
      }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      copy_ubuf_to_gm(reinterpret_cast<__gm__ bfloat16_t *>(out_gm) +
                          row_off * dim,
                      out_bf16, 0, 1, tile_elems / 16u, 0, 0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    }
  }
  mask_f32_full();
  pipe_barrier(PIPE_V);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
}

} // extern "C"
