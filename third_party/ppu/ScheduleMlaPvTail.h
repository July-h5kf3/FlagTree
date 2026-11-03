// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_SCHEDULE_MLA_PV_TAIL_H
#define TRITON_PPU_SCHEDULE_MLA_PV_TAIL_H

#include "ScalarizeMlaPv.h"

namespace triton::ppu {
template <unsigned Width>
inline unsigned scheduleMlaPvTailImpl(llvm::Module &module, unsigned threads,
                                      unsigned shared) {
  using namespace llvm;
  if (threads != 512 || shared != (Width == 64 ? 123392 : 168192))
    return 0;
  unsigned changed = 0;
  for (Function &function : module) {
    if (function.isDeclaration() ||
        function.getCallingConv() != CallingConv::PTX_Kernel ||
        !function.hasFnAttribute("ppu.mla.kv.pipeline") ||
        function.hasFnAttribute("ppu.mla.pv.tail_schedule"))
      continue;
    BasicBlock *tail = nullptr;
    for (BasicBlock &block : function)
      if (block.getName() == "mla.pv.cont")
        tail = &block;
    if (!tail)
      continue;
    SmallVector<Instruction *> region;
    InsertElementInst *root = nullptr;
    unsigned products = 0;
    bool accepted = true;
    for (Instruction &instruction : *tail) {
      if (instruction.isTerminator() || isa<PHINode>(instruction))
        continue;
      accepted &= mla_pv_scalarization::pure(instruction);
      region.push_back(&instruction);
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
          products += assembly->getAsmString() == mla_nope_prefetch::ProductAsm;
      auto *insert = dyn_cast<InsertElementInst>(&instruction);
      auto *type =
          insert ? dyn_cast<FixedVectorType>(insert->getType()) : nullptr;
      auto *index =
          insert ? dyn_cast<ConstantInt>(insert->getOperand(2)) : nullptr;
      if (type && type->getNumElements() == Width &&
          type->getElementType()->isFloatTy() && index &&
          index->equalsInt(Width - 1) &&
          insert->getName().starts_with("pv_stream_pack")) {
        if (root)
          accepted = false;
        root = insert;
      }
    }
    if (!accepted || products != (Width == 64 ? 39 : 35) || !root)
      continue;
    std::array<Value *, Width> lanes;
    Value *cursor = root;
    for (int index = Width - 1; index >= 0; --index) {
      auto *insert = dyn_cast<InsertElementInst>(cursor);
      auto *slot =
          insert ? dyn_cast<ConstantInt>(insert->getOperand(2)) : nullptr;
      if (!insert || insert->getParent() != tail || !slot ||
          !slot->equalsInt(index)) {
        accepted = false;
        break;
      }
      lanes[index] = insert->getOperand(1);
      auto *scalar = dyn_cast<Instruction>(lanes[index]);
      if (!scalar || scalar->getParent() != tail ||
          !scalar->getType()->isFloatTy() ||
          !scalar->getName().starts_with("pv_stream_scalar"))
        accepted = false;
      cursor = insert->getOperand(0);
    }
    if (!accepted || !isa<PoisonValue>(cursor))
      continue;
    // The continuation has no writers or synchronization. SSA dependencies
    // retain each arithmetic expression and every MMA operand unchanged.
    SmallPtrSet<Instruction *, 32> members(region.begin(), region.end()),
        emitted;
    SmallVector<Instruction *> ordered;
    std::function<void(Value *)> emit = [&](Value *value) {
      auto *instruction = dyn_cast<Instruction>(value);
      if (!instruction || !members.contains(instruction) ||
          !emitted.insert(instruction).second)
        return;
      for (Value *operand : instruction->operands())
        emit(operand);
      ordered.push_back(instruction);
    };
    for (Value *lane : lanes)
      emit(lane);
    for (Instruction *instruction : region)
      emit(instruction);
    if (ordered == region)
      continue;
    for (Instruction *instruction : ordered)
      instruction->moveBefore(*tail, tail->getTerminator()->getIterator());
    function.addFnAttr("ppu.mla.pv.tail_schedule", "1");
    ++changed;
  }
  return changed;
}
inline unsigned scheduleMlaPvTail(llvm::Module &module, unsigned threads,
                                  unsigned shared) {
  return shared == 123392 ? scheduleMlaPvTailImpl<64>(module, threads, shared)
                          : scheduleMlaPvTailImpl<32>(module, threads, shared);
}

} // namespace triton::ppu
#endif
