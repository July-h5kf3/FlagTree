// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_ASYNC_MLA_KV_COPY_H
#define TRITON_PPU_ASYNC_MLA_KV_COPY_H

#include "mlir/IR/Matchers.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::ppu {

inline unsigned promoteMlaKvCopy(ModuleOp module) {
  using namespace mlir::triton::gpu;
  ModuleAxisInfoAnalysis axis(module);
  SmallVector<LocalAllocOp> roots;
  module.walk([&](LocalAllocOp root) { roots.push_back(root); });
  unsigned changed = 0;
  for (auto root : roots) {
    if (!root.getSrc())
      continue;
    auto type = root.getType();
    auto encoding = dyn_cast<SwizzledSharedEncodingAttr>(type.getEncoding());
    auto load = root.getSrc().getDefiningOp<LoadOp>();
    if (!load || !load.getResult().hasOneUse() ||
        load->getBlock() != root->getBlock() || !root.isSharedMemoryAlloc() ||
        type.getMutableMemory() || !type.getElementType().isInteger(8) ||
        type.getShape() != ArrayRef<int64_t>({256, 512}) || !encoding ||
        encoding.getVec() != 16 || encoding.getPerPhase() != 1 ||
        encoding.getMaxPhase() != 16 ||
        encoding.getOrder() != ArrayRef<unsigned>({1, 0}) ||
        load.getIsVolatile() ||
        (load.getOther() && !matchPattern(load.getOther(), m_Zero())) ||
        axis.getContiguity(load.getPtr()) < 16 ||
        (load.getMask() && axis.getMaskAlignment(load.getMask()) < 16))
      continue;
    SmallVector<Value> descriptors{root.getResult()};
    SmallVector<LocalLoadOp> readers;
    bool supported = true;
    for (unsigned index = 0; index < descriptors.size(); ++index) {
      for (auto *user : descriptors[index].getUsers()) {
        if (auto view = dyn_cast<MemDescTransOp>(user))
          descriptors.push_back(view.getResult());
        else if (auto reader = dyn_cast<LocalLoadOp>(user))
          readers.push_back(reader);
        else
          supported = false;
      }
    }
    if (!supported || readers.empty())
      continue;
    OpBuilder builder(load);
    auto mutableType =
        MemDescType::get(type.getShape(), type.getElementType(),
                         type.getEncoding(), type.getMemorySpace(), true);
    auto allocation = builder.create<LocalAllocOp>(load.getLoc(), mutableType);
    auto copy = builder.create<AsyncCopyGlobalToLocalOp>(
        load.getLoc(), load.getPtr(), allocation.getResult(), load.getMask(),
        load.getOther(), load.getCache(), load.getEvict(), false);
    auto commit =
        builder.create<AsyncCommitGroupOp>(load.getLoc(), copy.getToken());
    auto wait =
        builder.create<AsyncWaitOp>(load.getLoc(), commit.getResult(), 0);
    for (Value descriptor : descriptors) {
      auto previous = cast<MemDescType>(descriptor.getType());
      descriptor.setType(MemDescType::get(
          previous.getShape(), previous.getElementType(),
          previous.getEncoding(), previous.getMemorySpace(), true));
    }
    root.replaceAllUsesWith(allocation.getResult());
    for (auto reader : readers)
      reader.getTokenMutable().assign(wait.getResult());
    root.erase();
    load.erase();
    ++changed;
  }
  return changed;
}

} // namespace mlir::triton::ppu
#endif
