// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_REUSE_MLA_KV_VIEWS_H
#define TRITON_PPU_REUSE_MLA_KV_VIEWS_H

#include "Dialect/TritonPPUGPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::ppu {

inline unsigned reuseMlaKvViews(ModuleOp module) {
  using namespace mlir::triton::gpu;
  SmallVector<TransOp> transposes;
  module.walk([&](TransOp transpose) { transposes.push_back(transpose); });
  unsigned changed = 0;
  for (auto transpose : transposes) {
    auto resultType = dyn_cast<RankedTensorType>(transpose.getType());
    if (!resultType || !resultType.getElementType().isInteger(8) ||
        resultType.getShape() != ArrayRef<int64_t>({512, 256}) ||
        transpose.getOrder() != ArrayRef<int32_t>({1, 0}))
      continue;
    auto dot = dyn_cast<DotOperandEncodingAttr>(resultType.getEncoding());
    auto mma = dot ? dyn_cast<PPUMmaEncodingAttr>(dot.getParent())
                   : PPUMmaEncodingAttr();
    if (!dot || dot.getOpIdx() != 1 || dot.getKWidth() != 4 || !mma ||
        !mma.isPPU0010() ||
        (mma.getWarpsPerCTA() != ArrayRef<unsigned>({2, 8}) &&
         mma.getWarpsPerCTA() != ArrayRef<unsigned>({1, 16})))
      continue;
    auto convert = transpose.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert || !convert.getResult().hasOneUse())
      continue;
    Value key = convert.getSrc();
    LocalAllocOp root;
    bool ambiguous = false;
    for (auto *user : key.getUsers()) {
      if (auto allocation = dyn_cast<LocalAllocOp>(user)) {
        if (allocation.getType().getShape() != ArrayRef<int64_t>({256, 512}))
          continue;
        if (root)
          ambiguous = true;
        root = allocation;
      }
    }
    if (!root || ambiguous || root->getBlock() != convert->getBlock() ||
        !root->isBeforeInBlock(convert) || !root.isSharedMemoryAlloc())
      continue;
    auto originalType = root.getType();
    auto encoding =
        dyn_cast<SwizzledSharedEncodingAttr>(originalType.getEncoding());
    if (!originalType.getElementType().isInteger(8) ||
        originalType.getMutableMemory() ||
        originalType.getAllocShape() != originalType.getShape() || !encoding ||
        encoding.getVec() != 8 || encoding.getPerPhase() != 1 ||
        encoding.getMaxPhase() != 16 ||
        encoding.getOrder() != ArrayRef<unsigned>({1, 0}))
      continue;
    auto alignment = root.getAlignmentAttr();
    if (alignment && alignment.getInt() < 16)
      continue;
    bool readersOnly = true;
    for (auto *user : root.getResult().getUsers())
      readersOnly &= isa<LocalLoadOp>(user);
    if (!readersOnly)
      continue;
    auto *context = module.getContext();
    auto cta = encoding.getCTALayout();
    auto counts = cta.getCTAsPerCGA();
    auto splits = cta.getCTASplitNum();
    if (llvm::any_of(counts, [](unsigned count) { return count != 1; }) ||
        llvm::any_of(splits, [](unsigned count) { return count != 1; }))
      continue;
    auto physical = SwizzledSharedEncodingAttr::get(
        context, 16, 1, 16, ArrayRef<unsigned>({1, 0}), cta);
    auto logical = SwizzledSharedEncodingAttr::get(
        context, 16, 1, 16, ArrayRef<unsigned>({0, 1}), cta);
    auto rootType =
        MemDescType::get(originalType.getShape(), originalType.getElementType(),
                         physical, originalType.getMemorySpace());
    auto viewType =
        MemDescType::get(resultType.getShape(), resultType.getElementType(),
                         logical, originalType.getMemorySpace());
    OpBuilder builder(convert);
    root.getResult().setType(rootType);
    auto view = builder.create<MemDescTransOp>(
        convert.getLoc(), viewType, root.getResult(),
        builder.getDenseI32ArrayAttr({1, 0}));
    builder.setInsertionPoint(transpose);
    auto loaded = builder.create<LocalLoadOp>(transpose.getLoc(), resultType,
                                              view.getResult());
    transpose.getResult().replaceAllUsesWith(loaded.getResult());
    transpose.erase();
    convert.erase();
    ++changed;
  }
  if (changed)
    module->setAttr(
        "mla.group_vec16_kv",
        IntegerAttr::get(IntegerType::get(module.getContext(), 32), 1));
  return changed;
}

} // namespace mlir::triton::ppu
#endif
