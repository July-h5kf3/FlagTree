// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "../ascendc_common.h"

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_merge_sort4_float(memref_t<__ubuf__ float, 1> *src,
                                      int32_t length, int32_t ways,
                                      memref_t<__ubuf__ float, 1> *dst) {
  auto input = ascendc_local_tensor(src);
  auto output = ascendc_local_tensor(dst);
  AscendC::MrgSortSrcList<float> lists(input, input[2 * length],
                                       input[ways == 4 ? 4 * length : 0],
                                       input[ways == 4 ? 6 * length : 0]);
  const uint16_t lengths[4] = {static_cast<uint16_t>(length),
                               static_cast<uint16_t>(length),
                               static_cast<uint16_t>(ways == 4 ? length : 0),
                               static_cast<uint16_t>(ways == 4 ? length : 0)};
  const auto groups = src->sizes[0] / (2 * ways * length);
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::MrgSort(
      output, lists,
      AscendC::MrgSort4Info(lengths, false, ways == 4 ? 15 : 3, groups));
  AscendC::PipeBarrier<PIPE_ALL>();
}
