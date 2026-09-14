// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "../ascendc_common.h"

template <typename T, typename MaskWord>
__aiv__ __attribute__((always_inline)) void
gather_impl(memref_t<__ubuf__ T, 1> *src, memref_t<__ubuf__ uint16_t, 1> *mask,
            memref_t<__ubuf__ T, 1> *dst,
            memref_t<__ubuf__ int32_t, 1> *count) {
  AscendC::PipeBarrier<PIPE_V>();
  auto input = ascendc_local_tensor(src);
  auto bits = ascendc_local_tensor(mask).template ReinterpretCast<MaskWord>();
  auto output = ascendc_local_tensor(dst);
  auto number = ascendc_local_tensor(count);
  uint64_t found = 0;
  AscendC::GatherMaskParams params{1, 1, 8, 1};
  AscendC::GatherMask(output, input, bits, true, src->sizes[0], params, found);
  AscendC::PipeBarrier<PIPE_V>();
  // Consume the reserved register before any other compaction can overwrite it.
  found = get_rsvd_cnt();
  AscendC::Duplicate(number, static_cast<int32_t>(found), count->sizes[0]);
  AscendC::PipeBarrier<PIPE_V>();
}
#define GATHER_ENTRY(TYPE, SUFFIX, WORD)                                       \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
  _mlir_ciface_custom_gather_mask_##SUFFIX(                                    \
      memref_t<__ubuf__ TYPE, 1> *src, memref_t<__ubuf__ uint16_t, 1> *mask,   \
      memref_t<__ubuf__ TYPE, 1> *dst, memref_t<__ubuf__ int32_t, 1> *count) { \
    gather_impl<TYPE, WORD>(src, mask, dst, count);                            \
  }
GATHER_ENTRY(float, float, uint32_t)
GATHER_ENTRY(half, half, uint16_t)
