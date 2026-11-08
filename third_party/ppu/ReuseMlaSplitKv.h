// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_REUSE_MLA_SPLIT_KV_H
#define TRITON_PPU_REUSE_MLA_SPLIT_KV_H

#include "Dialect/TritonPPUGPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::ppu {
namespace mla_split_kv {
using namespace mlir::triton::gpu;

template <typename Op> Op onlyUser(Value value, Block *block) {
  if (!value.hasOneUse())
    return {};
  Operation *user = *value.getUsers().begin();
  return user->getBlock() == block ? dyn_cast<Op>(user) : Op();
}

inline bool isInt8Tile(Value value, ArrayRef<int64_t> shape) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getElementType().isInteger(8) && type.getShape() == shape;
}

// The reshape splits the feature dimension into two contiguous halves.
inline SplitOp matchHalves(Value tile, int64_t halfWidth,
                           SmallVectorImpl<Operation *> &chain, Block *block) {
  auto reshape = onlyUser<ReshapeOp>(tile, block);
  if (!reshape || reshape.getAllowReorder() ||
      !isInt8Tile(reshape.getResult(), {128, 2, halfWidth}))
    return {};
  auto trans = onlyUser<TransOp>(reshape.getResult(), block);
  if (!trans || trans.getOrder() != ArrayRef<int32_t>({0, 2, 1}))
    return {};
  auto split = onlyUser<SplitOp>(trans.getResult(), block);
  if (!split || !isInt8Tile(split.getOutLHS(), {128, halfWidth}))
    return {};
  chain.append({reshape, trans, split});
  return split;
}

inline DotOperandEncodingAttr matchRhs(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto dot = type ? dyn_cast<DotOperandEncodingAttr>(type.getEncoding())
                  : DotOperandEncodingAttr();
  auto mma = dot ? dyn_cast<PPUMmaEncodingAttr>(dot.getParent())
                 : PPUMmaEncodingAttr();
  if (!dot || dot.getOpIdx() != 1 || dot.getKWidth() != 4 || !mma ||
      !mma.isPPU0010() || mma.getWarpsPerCTA() != ArrayRef<unsigned>({1, 8}))
    return {};
  return dot;
}

inline bool reuse(LocalAllocOp root) {
  Value source = root.getSrc();
  Block *block = root->getBlock();
  if (!source || !isInt8Tile(source, {128, 512}) ||
      !root.isSharedMemoryAlloc() || !source.getDefiningOp<LoadOp>() ||
      source.getDefiningOp()->getBlock() != block || !source.hasNUses(2))
    return false;
  MemDescType oldType = root.getType();
  auto shared = dyn_cast<SwizzledSharedEncodingAttr>(oldType.getEncoding());
  if (!shared || oldType.getMutableMemory() ||
      oldType.getAllocShape() != oldType.getShape() || shared.getVec() != 8 ||
      shared.getPerPhase() != 1 || shared.getMaxPhase() != 16 ||
      shared.getOrder() != ArrayRef<unsigned>({1, 0}))
    return false;
  auto alignment = root.getAlignmentAttr();
  if (alignment && alignment.getInt() < 16)
    return false;
  auto cta = shared.getCTALayout();
  if (cta.getCTAsPerCGA() != SmallVector<unsigned>({1, 1}) ||
      cta.getCTASplitNum() != SmallVector<unsigned>({1, 1}))
    return false;
  auto pv = onlyUser<LocalLoadOp>(root.getResult(), block);
  if (!pv || pv.getToken() || !matchRhs(pv.getResult()) ||
      !isInt8Tile(pv.getResult(), {128, 512}))
    return false;
  ConvertLayoutOp convert;
  for (Operation *user : source.getUsers())
    if (user != root.getOperation())
      convert = dyn_cast<ConvertLayoutOp>(user);
  if (!convert || convert->getBlock() != block ||
      !root->isBeforeInBlock(convert))
    return false;

  SmallVector<Operation *> chain{convert};
  auto halves = matchHalves(convert.getResult(), 256, chain, block);
  if (!halves)
    return false;
  SmallVector<TransOp> reads;
  Value accumulator;
  for (Value half : {halves.getOutLHS(), halves.getOutRHS()}) {
    auto quarters = matchHalves(half, 128, chain, block);
    if (!quarters)
      return false;
    for (Value quarter : {quarters.getOutLHS(), quarters.getOutRHS()}) {
      auto trans = onlyUser<TransOp>(quarter, block);
      if (!trans || trans.getOrder() != ArrayRef<int32_t>({1, 0}) ||
          !matchRhs(trans.getResult()))
        return false;
      auto dot = onlyUser<DotOp>(trans.getResult(), block);
      if (!dot || dot.getB() != trans.getResult() ||
          !isInt8Tile(dot.getA(), {16, 128}) ||
          (accumulator && dot.getC() != accumulator))
        return false;
      accumulator = dot.getResult();
      reads.push_back(trans);
      chain.push_back(trans);
    }
  }

  auto context = root.getContext();
  auto physical =
      SwizzledSharedEncodingAttr::get(context, 16, 1, 16, {1, 0}, cta);
  auto logical =
      SwizzledSharedEncodingAttr::get(context, 16, 1, 16, {0, 1}, cta);
  auto element = oldType.getElementType();
  auto space = oldType.getMemorySpace();
  auto rootType = MemDescType::get({128, 512}, element, physical, space);
  auto transType = MemDescType::get({512, 128}, element, logical, space);
  auto sliceType = MemDescType::get({128, 128}, element, logical, space, false,
                                    transType.getShape());
  root.getResult().setType(rootType);
  OpBuilder builder(convert);
  auto trans = builder.create<MemDescTransOp>(
      reads.front().getLoc(), transType, root.getResult(),
      builder.getDenseI32ArrayAttr({1, 0}));
  for (auto [index, read] : llvm::enumerate(reads)) {
    builder.setInsertionPoint(read);
    auto slice = builder.create<MemDescSubsliceOp>(
        read.getLoc(), sliceType, trans.getResult(),
        builder.getDenseI32ArrayAttr({static_cast<int32_t>(128 * index), 0}));
    auto loaded = builder.create<LocalLoadOp>(read.getLoc(), read.getType(),
                                              slice.getResult());
    read.getResult().replaceAllUsesWith(loaded.getResult());
  }
  for (Operation *op : llvm::reverse(chain))
    op->erase();
  return true;
}
} // namespace mla_split_kv

inline unsigned reuseMlaSplitKv(ModuleOp module) {
  SmallVector<gpu::LocalAllocOp> roots;
  module.walk([&](gpu::LocalAllocOp root) { roots.push_back(root); });
  unsigned changed = 0;
  for (auto root : roots)
    changed += mla_split_kv::reuse(root);
  if (changed)
    module->setAttr(
        "ppu.mla_h16_split_kv",
        IntegerAttr::get(IntegerType::get(module.getContext(), 32), 1));
  return changed;
}
} // namespace mlir::triton::ppu

#endif // TRITON_PPU_REUSE_MLA_SPLIT_KV_H
