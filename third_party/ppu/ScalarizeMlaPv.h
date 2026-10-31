// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_SCALARIZE_MLA_PV_H
#define TRITON_PPU_SCALARIZE_MLA_PV_H

#include "EarlySharedFloatScales.h"
#include "PrefetchMlaNope.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NoFolder.h"
#include <map>

namespace triton::ppu {
namespace mla_pv_scalarization {

struct Lane {
  unsigned opcode = 0;
  llvm::Value *value = nullptr;
  unsigned first = 0, second = 0, index = 0;
};

struct Tail {
  llvm::Instruction *publication;
  llvm::SmallVector<Lane> lanes;
  std::map<std::pair<llvm::Value *, unsigned>, unsigned> memo;
  llvm::SmallPtrSet<llvm::Instruction *, 32> removable;

  unsigned append(Lane lane) {
    lanes.push_back(lane);
    return lanes.size() - 1;
  }

  unsigned scalar(llvm::Value *vector, unsigned index) {
    using namespace llvm;
    auto key = std::make_pair(vector, index);
    if (auto found = memo.find(key); found != memo.end())
      return found->second;
    auto *type = cast<FixedVectorType>(vector->getType());
    unsigned result;
    if (isa<PoisonValue>(vector)) {
      result = append({0, PoisonValue::get(type->getElementType())});
    } else if (auto *constant = dyn_cast<Constant>(vector)) {
      result = append({0, constant->getAggregateElement(index)});
    } else if (auto *insert = dyn_cast<InsertElementInst>(vector);
               insert && isa<ConstantInt>(insert->getOperand(2))) {
      removable.insert(insert);
      auto slot = cast<ConstantInt>(insert->getOperand(2))->getZExtValue();
      result = slot == index ? append({0, insert->getOperand(1)})
                             : scalar(insert->getOperand(0), index);
    } else if (auto *shuffle = dyn_cast<ShuffleVectorInst>(vector)) {
      removable.insert(shuffle);
      int slot = shuffle->getMaskValue(index);
      unsigned width = cast<FixedVectorType>(shuffle->getOperand(0)->getType())
                           ->getNumElements();
      result = slot < 0 ? append({0, PoisonValue::get(type->getElementType())})
                        : scalar(shuffle->getOperand(slot >= int(width)),
                                 slot % width);
    } else {
      auto *instruction = dyn_cast<Instruction>(vector);
      bool eligible = type->getNumElements() == 32 ||
                      (instruction &&
                       instruction->getParent() == publication->getParent() &&
                       publication->comesBefore(instruction));
      if (eligible && isa_and_nonnull<SIToFPInst>(instruction)) {
        removable.insert(instruction);
        unsigned input = scalar(instruction->getOperand(0), index);
        result = append({Instruction::SIToFP, instruction, input});
      } else if (eligible && type->getElementType()->isFloatTy() &&
                 instruction &&
                 (instruction->getOpcode() == Instruction::FMul ||
                  instruction->getOpcode() == Instruction::FAdd ||
                  instruction->getOpcode() == Instruction::FSub)) {
        removable.insert(instruction);
        unsigned first = scalar(instruction->getOperand(0), index);
        unsigned second = scalar(instruction->getOperand(1), index);
        result = append({instruction->getOpcode(), instruction, first, second});
      } else {
        result = append({Instruction::ExtractElement, vector, 0, 0, index});
      }
    }
    memo.emplace(key, result);
    return result;
  }
};

inline bool pure(llvm::Instruction &instruction) {
  using namespace llvm;
  if (auto *load = dyn_cast<LoadInst>(&instruction))
    return load->getPointerAddressSpace() == 3 && !load->isVolatile() &&
           !load->isAtomic();
  if (auto *call = dyn_cast<CallInst>(&instruction)) {
    if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
      return assembly->getAsmString() == mla_nope_prefetch::ProductAsm ||
             assembly->getAsmString() == mla_nope_prefetch::MatrixAsm;
    auto *function = call->getCalledFunction();
    return function &&
           (function->getName() == "__flagtree_ppu_intrinsic__mat.trans.b8" ||
            function->getIntrinsicID() == Intrinsic::vector_reduce_or);
  }
  return instruction.isBinaryOp() ||
         isa<CastInst, GetElementPtrInst, ExtractValueInst, ExtractElementInst,
             InsertElementInst, ShuffleVectorInst, ICmpInst, SelectInst>(
             instruction);
}

inline unsigned rewrite(llvm::Function &function) {
  using namespace llvm;
  SmallVector<CallInst *> products;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
          if (assembly->getAsmString() == mla_nope_prefetch::ProductAsm) {
            if (call->arg_size() != 16)
              return 0;
            products.push_back(call);
          }
  if (products.size() != 96)
    return 0;
  auto *block = products[32]->getParent();
  Instruction *publication = nullptr;
  for (Instruction &instruction : *block) {
    if (&instruction == products[32])
      break;
    if (early_shared_float_scales::barrier(&instruction))
      publication = &instruction;
  }
  if (!publication)
    return 0;
  for (Instruction *instruction = publication->getNextNode();
       instruction != block->getTerminator();
       instruction = instruction->getNextNode())
    if (!pure(*instruction))
      return 0;
  std::array<std::array<ExtractValueInst *, 8>, 64> outputs{};
  SmallPtrSet<Value *, 32> highWords, terminals;
  SmallVector<ExtractValueInst *> terminalOrder;
  for (unsigned index = 0; index < 64; ++index) {
    auto *product = products[index + 32];
    if (product->getParent() != block ||
        !mla_nope_prefetch::extracts(product, outputs[index]))
      return 0;
    bool consumed = false;
    for (auto *word : outputs[index])
      for (User *user : word->users())
        if (auto *call = dyn_cast<CallInst>(user))
          consumed |=
              llvm::is_contained(ArrayRef(products).drop_front(64), call);
    if (index >= 32 && !consumed)
      for (auto *word : outputs[index]) {
        terminals.insert(word);
        terminalOrder.push_back(word);
      }
    if (index < 32)
      for (auto *word : outputs[index])
        highWords.insert(word);
  }
  if (terminals.size() != 32)
    return 0;
  Instruction *root = nullptr;
  for (Instruction *instruction = products.back()->getNextNode();
       instruction != block->getTerminator();
       instruction = instruction->getNextNode()) {
    auto *type = dyn_cast<FixedVectorType>(instruction->getType());
    if (instruction->getOpcode() == Instruction::FAdd && type &&
        type->getNumElements() == 32 && type->getElementType()->isFloatTy()) {
      if (root)
        return 0;
      root = instruction;
    }
  }
  if (!root)
    return 0;
  Tail tail{publication};
  std::array<unsigned, 32> roots{};
  for (unsigned index = 0; index < 32; ++index)
    roots[index] = tail.scalar(root, index);
  std::function<bool(unsigned, SmallVectorImpl<Value *> &)> words =
      [&](unsigned index, SmallVectorImpl<Value *> &found) {
        const Lane &lane = tail.lanes[index];
        if (!lane.value)
          return false;
        if (!lane.opcode) {
          if (terminals.contains(lane.value))
            found.push_back(lane.value);
        } else if (lane.opcode != Instruction::ExtractElement) {
          if (!words(lane.first, found))
            return false;
          if (lane.opcode != Instruction::SIToFP && !words(lane.second, found))
            return false;
        }
        return true;
      };
  std::array<Value *, 32> laneWord{};
  unsigned covered = 0;
  for (unsigned index = 0; index < 32; ++index) {
    SmallVector<Value *> found;
    if (!words(roots[index], found) || found.size() > 1)
      return 0;
    if (!found.empty()) {
      laneWord[index] = found.front();
      ++covered;
    }
  }
  if (!covered)
    return 0;
  SmallVector<std::pair<ExtractElementInst *, Value *>> seeds;
  SmallPtrSet<Instruction *, 32> seedNodes;
  SmallPtrSet<Value *, 32> seenSeeds;
  unsigned scalarSeeds = 0;
  for (auto *product : ArrayRef(products).drop_front(64))
    for (unsigned index = 0; index < 8; ++index) {
      auto *argument = product->getArgOperand(index);
      auto *existing = dyn_cast<BinaryOperator>(argument);
      auto *factor =
          existing ? dyn_cast<ConstantInt>(existing->getOperand(1)) : nullptr;
      if (existing && existing->getOpcode() == Instruction::Mul &&
          existing->getType()->isIntegerTy(32) && factor &&
          factor->equalsInt(127) &&
          highWords.contains(existing->getOperand(0))) {
        if (seenSeeds.insert(existing).second)
          ++scalarSeeds;
        continue;
      }
      auto *seed = dyn_cast<ExtractElementInst>(argument);
      if (!seed || !seenSeeds.insert(seed).second)
        continue;
      auto *slot = dyn_cast<ConstantInt>(seed->getIndexOperand());
      auto *multiply = dyn_cast<BinaryOperator>(seed->getVectorOperand());
      auto *type =
          multiply ? dyn_cast<FixedVectorType>(multiply->getType()) : nullptr;
      auto *vectorFactor =
          multiply ? dyn_cast<Constant>(multiply->getOperand(1)) : nullptr;
      auto *splat =
          vectorFactor
              ? dyn_cast_or_null<ConstantInt>(vectorFactor->getSplatValue())
              : nullptr;
      if (!slot || slot->getZExtValue() >= 32 || !multiply || !type ||
          type->getNumElements() != 32 ||
          !type->getElementType()->isIntegerTy(32) ||
          multiply->getOpcode() != Instruction::Mul || !splat ||
          !splat->equalsInt(127))
        return 0;
      Value *cursor = multiply->getOperand(0), *high = nullptr;
      while (auto *insert = dyn_cast<InsertElementInst>(cursor)) {
        auto *position = dyn_cast<ConstantInt>(insert->getOperand(2));
        if (!position)
          return 0;
        seedNodes.insert(insert);
        if (!high && position->getValue() == slot->getValue())
          high = insert->getOperand(1);
        cursor = insert->getOperand(0);
      }
      if (!isa<PoisonValue>(cursor) || !high || !highWords.contains(high))
        return 0;
      seeds.emplace_back(seed, high);
      seedNodes.insert(multiply);
      seedNodes.insert(seed);
    }
  if (seeds.size() + scalarSeeds != 32)
    return 0;
  // All effect, lane and seed checks precede any changes to the SSA graph.
  IRBuilder<NoFolder> builder(root);
  SmallVector<Value *> scalars;
  for (const Lane &lane : tail.lanes) {
    Value *value = lane.value;
    if (lane.opcode == Instruction::ExtractElement)
      value = builder.CreateExtractElement(value, builder.getInt64(lane.index),
                                           "pv_stream_scalar");
    else if (lane.opcode == Instruction::SIToFP)
      value = builder.CreateSIToFP(scalars[lane.first], builder.getFloatTy(),
                                   "pv_stream_scalar");
    else if (lane.opcode) {
      value = builder.CreateBinOp(
          static_cast<Instruction::BinaryOps>(lane.opcode), scalars[lane.first],
          scalars[lane.second], "pv_stream_scalar");
      cast<Instruction>(value)->copyIRFlags(lane.value);
    }
    scalars.push_back(value);
  }
  Value *packed = PoisonValue::get(root->getType());
  for (unsigned index = 0; index < 32; ++index)
    packed =
        builder.CreateInsertElement(packed, scalars[roots[index]],
                                    builder.getInt64(index), "pv_stream_pack");
  root->replaceAllUsesWith(packed);
  tail.removable.insert(root);
  for (auto [seed, high] : seeds) {
    builder.SetInsertPoint(seed);
    Value *value =
        builder.CreateMul(high, builder.getInt32(127), "pv_stream_seed");
    seed->replaceAllUsesWith(value);
  }
  for (auto *instruction : seedNodes)
    tail.removable.insert(instruction);
  bool changed;
  do {
    SmallVector<Instruction *> dead;
    for (auto *instruction : tail.removable)
      if (instruction->use_empty())
        dead.push_back(instruction);
    for (auto *instruction : dead) {
      tail.removable.erase(instruction);
      instruction->eraseFromParent();
    }
    changed = !dead.empty();
  } while (changed);
  SmallVector<Instruction *> region, ordered;
  SmallPtrSet<Instruction *, 32> members, emitted;
  for (Instruction *instruction = publication->getNextNode();
       instruction != block->getTerminator();
       instruction = instruction->getNextNode()) {
    region.push_back(instruction);
    members.insert(instruction);
  }
  std::function<void(Value *)> emit = [&](Value *value) {
    auto *instruction = dyn_cast<Instruction>(value);
    if (!instruction || !members.contains(instruction) ||
        !emitted.insert(instruction).second)
      return;
    for (Value *operand : instruction->operands())
      emit(operand);
    ordered.push_back(instruction);
  };
  for (unsigned index = 0; index < 64; ++index) {
    emit(products[index + 32]);
    for (auto *word : outputs[index])
      emit(word);
  }
  for (Value *word : terminalOrder)
    for (unsigned index = 0; index < 32; ++index)
      if (laneWord[index] == word)
        emit(scalars[roots[index]]);
  for (unsigned index = 0; index < 32; ++index)
    if (!laneWord[index])
      emit(scalars[roots[index]]);
  for (auto *instruction : region)
    emit(instruction);
  for (auto *instruction : ordered)
    instruction->moveBefore(*block, block->getTerminator()->getIterator());
  return 1;
}

} // namespace mla_pv_scalarization

inline unsigned scalarizeMlaPv(llvm::Module &module, unsigned threads,
                               unsigned shared) {
  if (threads != 512 || shared != 168192)
    return 0;
  unsigned changed = 0;
  for (llvm::Function &function : module)
    if (!function.isDeclaration() &&
        function.getCallingConv() == llvm::CallingConv::PTX_Kernel)
      changed += mla_pv_scalarization::rewrite(function);
  return changed;
}

} // namespace triton::ppu
#endif
