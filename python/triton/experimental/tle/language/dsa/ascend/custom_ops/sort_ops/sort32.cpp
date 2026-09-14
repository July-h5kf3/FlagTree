// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "../ascendc_common.h"

extern "C" __aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_sort32_float(memref_t<__ubuf__ float, 1> *src,
                                 memref_t<__ubuf__ uint32_t, 1> *indices,
                                 memref_t<__ubuf__ float, 1> *dst) {
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::Sort32(ascendc_local_tensor(dst), ascendc_local_tensor(src),
                  ascendc_local_tensor(indices), src->sizes[0] / 32);
  AscendC::PipeBarrier<PIPE_ALL>();
}
