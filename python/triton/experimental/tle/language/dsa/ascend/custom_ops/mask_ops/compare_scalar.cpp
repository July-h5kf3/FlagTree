// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "../ascendc_common.h"

template <typename T>
__aiv__ __attribute__((always_inline)) void
compare_impl(memref_t<__ubuf__ T, 1> *src, float scalar, int32_t comparison,
             memref_t<__ubuf__ uint16_t, 1> *dst) {
  AscendC::PipeBarrier<PIPE_V>();
  auto input = ascendc_local_tensor(src);
  auto mask = ascendc_local_tensor(dst).template ReinterpretCast<uint8_t>();
  if (comparison == 0)
    AscendC::CompareScalar(mask, input, static_cast<T>(scalar),
                           AscendC::CMPMODE::EQ, src->sizes[0]);
  else if (comparison == 1)
    AscendC::CompareScalar(mask, input, static_cast<T>(scalar),
                           AscendC::CMPMODE::GT, src->sizes[0]);
  else
    AscendC::CompareScalar(mask, input, static_cast<T>(scalar),
                           AscendC::CMPMODE::GE, src->sizes[0]);
  AscendC::PipeBarrier<PIPE_V>();
}

#define COMPARE_ENTRY(TYPE, SUFFIX)                                            \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_compare_scalar_##SUFFIX(                             \
          memref_t<__ubuf__ TYPE, 1> *src, float scalar,                       \
          memref_t<__ubuf__ uint16_t, 1> *dst) {                               \
    compare_impl(src, scalar, 0, dst);                                         \
  }                                                                            \
  extern "C" __aiv__ __attribute__((always_inline)) void                       \
      _mlir_ciface_custom_compare_scalar_##SUFFIX##_mode(                      \
          memref_t<__ubuf__ TYPE, 1> *src, float scalar, int32_t comparison,   \
          memref_t<__ubuf__ uint16_t, 1> *dst) {                               \
    compare_impl(src, scalar, comparison, dst);                                \
  }
COMPARE_ENTRY(float, float)
COMPARE_ENTRY(half, half)
