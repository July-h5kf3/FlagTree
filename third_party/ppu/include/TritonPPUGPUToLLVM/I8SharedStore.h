// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_I8_SHARED_STORE_H
#define TRITON_PPU_I8_SHARED_STORE_H

#include <string>

namespace mlir::triton::ppu {

inline std::string groupedI8SharedStoreAsm() {
  std::string code = "{ .reg .u32 aligned, bit; .reg .pred swap; "
                     "ppu.and.b32 aligned, $64, 0xfffffff0; "
                     "ppu.and.b32 bit, $64, 8; ppu.setp.ne.u32 swap, bit, 0;";
  for (unsigned rotation : {2u, 0u}) {
    for (unsigned packet = 0; packet < 16; ++packet) {
      code += rotation ? " @swap" : " @!swap";
      code += " ppu.st.shared.v4.b32 [aligned + " +
              std::to_string(packet * 8192) + "], {";
      for (unsigned word = 0; word < 4; ++word) {
        if (word)
          code += ',';
        code += '$' + std::to_string(packet * 4 + (word + rotation) % 4);
      }
      code += "};";
    }
  }
  return code + " }";
}

} // namespace mlir::triton::ppu
#endif
