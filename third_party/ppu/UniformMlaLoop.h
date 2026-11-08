// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_UNIFORM_MLA_LOOP_H
#define TRITON_PPU_UNIFORM_MLA_LOOP_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include <functional>

namespace mlir::triton::ppu {
// Triton scalars are per-program values. Reject opaque scalar assembly and
// unknown producers rather than assuming that their result is CTA-uniform.
inline unsigned proveUniformMlaLoop(ModuleOp module, bool allowUniformIf) {
  SmallVector<scf::ForOp> loops;
  module.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  if (loops.size() != 1)
    return 0;
  auto loop = loops.front();
  auto marker =
      module->getAttrOfType<IntegerAttr>("ppu.mla_h64_k128_pv_layout");
  auto h32Marker =
      module->getAttrOfType<IntegerAttr>("ppu.mla_h32_k128_pv_layout");
  bool h32 = h32Marker && h32Marker.getInt() == 1;
  bool small = (marker && marker.getInt() == 1) || h32;
  llvm::APInt step;
  if (!loop.getLowerBound().getType().isInteger(32) ||
      !loop.getUpperBound().getType().isInteger(32) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
      (step != 4 && !(small && step == 2)))
    return 0;
  llvm::DenseSet<Value> proved;
  std::function<bool(Value)> uniform = [&](Value value) {
    if (isa<RankedTensorType>(value.getType()))
      return false;
    if (proved.contains(value))
      return true;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto function =
          dyn_cast<triton::FuncOp>(argument.getOwner()->getParentOp());
      if (!function || !function.isPublic() ||
          argument.getOwner() != &function.getBody().front())
        return false;
      proved.insert(value);
      return true;
    }
    auto *op = value.getDefiningOp();
    if (!op)
      return false;
    if (auto load = dyn_cast<triton::LoadOp>(op))
      if (load.getIsVolatile())
        return false;
    if (auto reduction = dyn_cast<triton::ReduceOp>(op)) {
      if (reduction.getAxis() != 0 ||
          llvm::any_of(reduction.getOperands(), [](Value input) {
            auto type = dyn_cast<RankedTensorType>(input.getType());
            return !type || type.getRank() != 1;
          }))
        return false;
      for (Operation &nested : reduction.getCombineOp().front())
        if (!isa<triton::ReduceReturnOp>(nested) &&
            (nested.getName().getDialectNamespace() != "arith" ||
             nested.getNumRegions()))
          return false;
      proved.insert(value);
      return true;
    }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (!allowUniformIf || !uniform(branch.getCondition()) ||
          !llvm::hasSingleElement(branch.getThenRegion()) ||
          !llvm::hasSingleElement(branch.getElseRegion()))
        return false;
      unsigned result = cast<OpResult>(value).getResultNumber();
      auto thenYield = dyn_cast<scf::YieldOp>(
          branch.getThenRegion().front().getTerminator());
      auto elseYield = dyn_cast<scf::YieldOp>(
          branch.getElseRegion().front().getTerminator());
      if (!thenYield || !elseYield || result >= thenYield.getNumOperands() ||
          result >= elseYield.getNumOperands() ||
          !uniform(thenYield.getOperand(result)) ||
          !uniform(elseYield.getOperand(result)))
        return false;
      proved.insert(value);
      return true;
    }
    bool known =
        op->getName().getDialectNamespace() == "arith" ||
        isa<triton::GetProgramIdOp, triton::GetNumProgramsOp, triton::LoadOp,
            triton::AddPtrOp, triton::IntToPtrOp, triton::PtrToIntOp>(op);
    if (!known || op->getNumRegions() ||
        !llvm::all_of(op->getOperands(), uniform))
      return false;
    proved.insert(value);
    return true;
  };
  return uniform(loop.getLowerBound()) && uniform(loop.getUpperBound());
}
} // namespace mlir::triton::ppu
#endif
