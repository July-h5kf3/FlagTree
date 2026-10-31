// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_EARLY_SHARED_FLOAT_SCALES_H
#define TRITON_PPU_EARLY_SHARED_FLOAT_SCALES_H

#include "ReadonlySharedMatrices.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IRBuilder.h"
#include <algorithm>
#include <functional>
#include <utility>

namespace triton::ppu {
namespace early_shared_float_scales {

inline bool barrier(llvm::Instruction *instruction) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(instruction);
  auto *callee = call ? call->getCalledFunction() : nullptr;
  auto *argument =
      call && call->arg_size() == 1
          ? llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0))
          : nullptr;
  return callee && argument && argument->isZero() &&
         callee->getName() == "llvm.nvvm.barrier.cta.sync.aligned.all";
}

inline bool vector2(llvm::Type *type) {
  auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type);
  return vector && vector->getNumElements() == 2 &&
         vector->getElementType()->isFloatTy();
}

inline bool floatLoad(llvm::LoadInst *load) {
  if (load->isVolatile() || load->isAtomic() ||
      load->getPointerAddressSpace() != 3)
    return false;
  unsigned alignment = load->getAlign().value();
  if (alignment != 4 && alignment != 8 && alignment != 16)
    return false;
  if (load->getType()->isFloatTy())
    return true;
  auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(load->getType());
  if (!vector || !vector->getElementType()->isFloatTy())
    return false;
  unsigned count = vector->getNumElements();
  return count == 1 || count == 2 || count == 4 || count == 8;
}

inline unsigned rewrite(llvm::Function &function, unsigned shared,
                        unsigned threads, llvm::MDNode *annotation) {
  using namespace llvm;
  SmallVector<CallInst *> products;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
          if (assembly->getAsmString().contains(".s32.s8.s8.s32"))
            products.push_back(call);
  if (products.size() != 96)
    return shared;
  DominatorTree dominators(function);
  LoopInfo loops(dominators);
  auto *header = products.front()->getParent();
  auto *loop = loops.getLoopFor(header);
  auto *pv = products[32]->getParent();
  if (!loop || loop->getParentLoop() || loop->getHeader() != header ||
      loop->getLoopLatch() != pv || !dominators.dominates(header, pv))
    return shared;
  for (auto *product : ArrayRef(products).drop_front(32))
    if (product->getParent() != pv)
      return shared;
  bool headerBarrier = false, headerWrite = false;
  for (Instruction &instruction : *header) {
    if (&instruction == products.front())
      break;
    if (barrier(&instruction))
      headerBarrier = true;
    bool writes = false;
    if (auto *store = dyn_cast<StoreInst>(&instruction))
      writes = store->getPointerAddressSpace() == 3;
    if (auto *call = dyn_cast<CallInst>(&instruction))
      if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
        writes |=
            assembly->getAsmString().contains("ppu.st.shared.") ||
            assembly->getAsmString() ==
                "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;";
    if (writes) {
      if (!headerBarrier)
        return shared;
      headerWrite = true;
    }
  }
  if (!headerBarrier || !headerWrite)
    return shared;
  Instruction *anchor = nullptr;
  for (Instruction &instruction : *pv) {
    if (&instruction == products[32])
      break;
    if (barrier(&instruction))
      anchor = &instruction;
  }
  if (!anchor)
    return shared;
  // The retained publication barrier must cover every original loop write.
  for (Instruction *instruction = anchor->getNextNode();
       instruction != products.back()->getNextNode();
       instruction = instruction->getNextNode())
    if (isa<StoreInst, AtomicRMWInst, AtomicCmpXchgInst, FenceInst>(
            instruction))
      return shared;
  SmallVector<Instruction *> effects;
  for (Instruction *instruction = products.back()->getNextNode();
       instruction != pv->getTerminator();
       instruction = instruction->getNextNode())
    if (isa<CallBase>(instruction) || instruction->mayReadOrWriteMemory() ||
        isa<FenceInst>(instruction))
      effects.push_back(instruction);
  if (effects.size() < 8 || !barrier(effects[0]) || !barrier(effects[2]))
    return shared;
  auto *store0 = dyn_cast<StoreInst>(effects[1]);
  if (!store0 || store0->isVolatile() || store0->isAtomic() ||
      store0->getPointerAddressSpace() != 3 || store0->getAlign() != Align(8) ||
      !vector2(store0->getValueOperand()->getType()))
    return shared;
  SmallVector<LoadInst *> reads;
  SmallVector<unsigned> phases;
  unsigned index = 3;
  for (; index < effects.size(); ++index) {
    auto *load = dyn_cast<LoadInst>(effects[index]);
    if (!load)
      break;
    if (!floatLoad(load))
      return shared;
    reads.push_back(load);
    phases.push_back(0);
  }
  if (reads.empty() || reads.size() > 8 || index + 3 >= effects.size() ||
      !barrier(effects[index]) || !barrier(effects[index + 2]))
    return shared;
  auto *store1 = dyn_cast<StoreInst>(effects[index + 1]);
  if (!store1 || store1->isVolatile() || store1->isAtomic() ||
      store1->getPointerAddressSpace() != 3 || store1->getAlign() != Align(8) ||
      store1->getValueOperand()->getType() !=
          store0->getValueOperand()->getType() ||
      store1->getPointerOperand() != store0->getPointerOperand())
    return shared;
  SmallVector<Instruction *> barriers = {effects[0], effects[2], effects[index],
                                         effects[index + 2]};
  unsigned firstPhase = reads.size();
  for (index += 3; index < effects.size(); ++index) {
    auto *load = dyn_cast<LoadInst>(effects[index]);
    if (!load || !floatLoad(load))
      return shared;
    reads.push_back(load);
    phases.push_back(1);
  }
  if (reads.size() == firstPhase || reads.size() - firstPhase > 8)
    return shared;
  auto &layout = function.getParent()->getDataLayout();
  unsigned extent = 0;
  for (unsigned tid = 0; tid < threads; ++tid) {
    DenseMap<Value *, APInt> cache;
    auto check = [&](Value *pointer, unsigned width, unsigned alignment) {
      auto *instruction = dyn_cast<Instruction>(pointer);
      if (instruction && !dominators.dominates(instruction, anchor))
        return false;
      auto offset =
          readonly_shared_matrices::sharedOffset(pointer, tid, layout, cache);
      if (!offset || *offset < 0 || *offset % alignment ||
          *offset + width > 4096)
        return false;
      extent = std::max(extent, static_cast<unsigned>(*offset + width));
      return true;
    };
    if (!check(store0->getPointerOperand(), 8, 8))
      return shared;
    for (auto *load : reads)
      if (!check(load->getPointerOperand(),
                 layout.getTypeStoreSize(load->getType()).getFixedValue(),
                 load->getAlign().value()))
        return shared;
  }
  extent = (extent + 31) / 32 * 32;
  unsigned total = shared + 2 * extent;
  if (total > 262144)
    return shared;
  BitVector scratch(total);
  for (unsigned byte = shared; byte < total; ++byte)
    scratch.set(byte);
  if (!readonly_shared_matrices::proveNoQueryWrites(*loop, scratch, layout,
                                                    threads))
    return shared;
  SmallPtrSet<Instruction *, 8> moving;
  std::function<bool(Value *)> gather = [&](Value *value) {
    auto *instruction = dyn_cast<Instruction>(value);
    if (!instruction || dominators.dominates(instruction, anchor))
      return true;
    if (instruction->getParent() != pv ||
        !products.back()->comesBefore(instruction))
      return false;
    bool insert =
        isa<InsertElementInst>(instruction) && vector2(instruction->getType());
    bool multiply = instruction->getOpcode() == Instruction::FMul &&
                    instruction->getType()->isFloatTy();
    if (!insert && !multiply)
      return false;
    if (moving.contains(instruction))
      return true;
    for (Value *operand : instruction->operands())
      if (!gather(operand))
        return false;
    moving.insert(instruction);
    return true;
  };
  if (!gather(store0->getValueOperand()) ||
      !gather(store1->getValueOperand()) || moving.size() != 6)
    return shared;
  SmallVector<Instruction *> ordered;
  for (Instruction &instruction : *pv)
    if (moving.contains(&instruction))
      ordered.push_back(&instruction);
  for (auto *instruction : ordered)
    instruction->moveBefore(anchor->getIterator());
  IRBuilder<> builder(anchor);
  Value *write =
      builder.CreateGEP(builder.getInt8Ty(), store0->getPointerOperand(),
                        builder.getInt32(shared), "early_write");
  Value *scaleWrite =
      builder.CreateGEP(builder.getInt8Ty(), write, builder.getInt32(extent),
                        "early_write_scale");
  for (unsigned i = 0; i < reads.size(); ++i) {
    Value *pointer = builder.CreateGEP(
        builder.getInt8Ty(), reads[i]->getPointerOperand(),
        builder.getInt32(shared + phases[i] * extent), "early_read");
    reads[i]->setOperand(0, pointer);
  }
  store0->setOperand(1, write);
  store1->setOperand(1, scaleWrite);
  store0->moveBefore(anchor->getIterator());
  store1->moveBefore(anchor->getIterator());
  for (unsigned i = 1; i < reads.size(); ++i)
    if (reads[i]->getPrevNode() != reads[i - 1])
      reads[i]->moveAfter(reads[i - 1]);
  for (auto *instruction : barriers)
    instruction->eraseFromParent();
  annotation->replaceOperandWith(
      2, ConstantAsMetadata::get(builder.getInt32(total)));
  return total;
}

} // namespace early_shared_float_scales

inline std::pair<unsigned, unsigned>
publishSharedFloatScalesEarly(llvm::Module &module, unsigned threads,
                              unsigned shared) {
  using namespace llvm;
  if (threads != 512 || (shared != 217088 && shared != 167936))
    return {shared, 0};
  auto *annotations = module.getNamedMetadata("nvvm.annotations");
  if (!annotations)
    return {shared, 0};
  for (Function &function : module) {
    if (function.isDeclaration())
      continue;
    MDNode *annotation = nullptr;
    for (MDNode *entry : annotations->operands()) {
      if (entry->getNumOperands() != 3)
        continue;
      auto *owner = dyn_cast_or_null<ValueAsMetadata>(entry->getOperand(0));
      auto *name = dyn_cast_or_null<MDString>(entry->getOperand(1));
      auto *size =
          mdconst::dyn_extract_or_null<ConstantInt>(entry->getOperand(2));
      if (owner && owner->getValue() == &function && name &&
          name->getString() == "smemsize" && size && size->equalsInt(shared)) {
        if (annotation)
          return {shared, 0};
        annotation = entry;
      }
    }
    if (!annotation)
      continue;
    unsigned result = early_shared_float_scales::rewrite(function, shared,
                                                         threads, annotation);
    if (result != shared)
      return {result, 1};
  }
  return {shared, 0};
}

} // namespace triton::ppu
#endif
