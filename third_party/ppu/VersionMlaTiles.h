// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_VERSION_MLA_TILES_H
#define TRITON_PPU_VERSION_MLA_TILES_H

namespace llvm {
class Function;
}
namespace triton::ppu::mla_full_tiles {
unsigned run(llvm::Function &function, bool q2Tail = false);
}
#endif
