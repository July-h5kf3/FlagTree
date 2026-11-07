// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_MLA_ROW_REDUCTION_H
#define TRITON_PPU_MLA_ROW_REDUCTION_H

#include "PrefetchMlaRope.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NoFolder.h"

namespace triton::ppu {
namespace mla_row_groups {

struct Access {
  llvm::Instruction *instruction;
  llvm::Value *pointer;
  llvm::Value *predicate;
  bool write;
};

inline std::optional<Access> access(llvm::Instruction &instruction) {
  using namespace llvm;
  if (auto *load = dyn_cast<LoadInst>(&instruction)) {
    if (load->getPointerAddressSpace() == 3 && load->getType()->isFloatTy() &&
        !load->isVolatile() && !load->isAtomic())
      return Access{load, load->getPointerOperand(), nullptr, false};
    return std::nullopt;
  }
  auto *call = dyn_cast<CallInst>(&instruction);
  auto *code = call ? dyn_cast<InlineAsm>(call->getCalledOperand()) : nullptr;
  if (!code || !code->hasSideEffects() || code->isAlignStack() ||
      code->canThrow() || code->getDialect() != InlineAsm::AD_ATT ||
      call->hasOperandBundles())
    return std::nullopt;
  bool write =
      code->getAsmString() == "@$2 ppu.st.shared.b32 [ $0 + 0 ], $1;" &&
      code->getConstraintString() == "r,r,b" && call->arg_size() == 3 &&
      call->getType()->isVoidTy() &&
      call->getArgOperand(1)->getType() ==
          FixedVectorType::get(Type::getInt32Ty(call->getContext()), 1);
  bool read = code->getAsmString() == "@$2 ppu.ld.shared.b32 $0, [ $1 + 0 ];" &&
              code->getConstraintString() == "=r,r,b" &&
              call->arg_size() == 2 && call->getType()->isIntegerTy(32);
  if ((!write && !read) || !call->getArgOperand(0)->getType()->isPointerTy() ||
      call->getArgOperand(0)->getType()->getPointerAddressSpace() != 3 ||
      !call->getArgOperand(write ? 2 : 1)->getType()->isIntegerTy(1))
    return std::nullopt;
  return Access{call, call->getArgOperand(0),
                call->getArgOperand(write ? 2 : 1), write};
}

inline bool pure(llvm::Instruction &instruction) {
  using namespace llvm;
  if (auto *call = dyn_cast<CallInst>(&instruction)) {
    if (call->hasOperandBundles())
      return false;
    if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand()))
      return !code->hasSideEffects() && !code->canThrow() &&
             code->getConstraintString().find("memory") == std::string::npos &&
             !code->getAsmString().contains("ppu.ld") &&
             !code->getAsmString().contains("ppu.mma") &&
             readonly_shared_matrices::readOnlyAsm(code->getAsmString());
    auto *callee = call->getCalledFunction();
    return callee && callee->isDeclaration() &&
           StringSwitch<bool>(callee->getName())
               .Cases("llvm.maxnum.f32", "llvm.maxnum.v2f32", true)
               .Cases("llvm.nvvm.ex2.approx.ftz.f", "llvm.nvvm.ex2.approx.f",
                      true)
               .Cases("__nvvm_reflect", "__builtin_ppu_rcpf", true)
               .Case("llvm.nvvm.shfl.sync.bfly.i32", true)
               .Default(false);
  }
  return instruction.isBinaryOp() ||
         isa<CastInst, GetElementPtrInst, ExtractElementInst, ExtractValueInst,
             InsertElementInst, ShuffleVectorInst, ICmpInst, FCmpInst,
             SelectInst, PHINode>(instruction);
}

struct Region {
  llvm::CallInst *begin = nullptr, *end = nullptr;
  llvm::SmallVector<llvm::CallInst *> barriers;
  llvm::SmallVector<Access> partial, middle, final;
  unsigned fields = 0;
};

inline bool collect(Region &region, llvm::Loop *loop) {
  using namespace llvm;
  auto *beginBlock = region.begin->getParent();
  unsigned phase = 0;
  SmallPtrSet<BasicBlock *, 32> visited, active;
  std::function<bool(BasicBlock *, Instruction *)> walk =
      [&](BasicBlock *block, Instruction *first) {
        if (!loop->contains(block) || active.contains(block) ||
            visited.size() >= 128)
          return false;
        if (!visited.insert(block).second)
          return true;
        active.insert(block);
        for (auto it = first->getIterator(); it != block->end(); ++it) {
          Instruction &instruction = *it;
          if (readonly_shared_matrices::ctaBarrier(&instruction, 256)) {
            auto *call = cast<CallInst>(&instruction);
            if (call->getCalledFunction()->getName() ==
                "__ppukernel_barrier_sync_cnt_void") {
              if (phase != 2 || (region.end && region.end != call))
                return false;
              region.end = call;
              active.erase(block);
              return true;
            }
            if (block != beginBlock || phase >= 2)
              return false;
            region.barriers.push_back(call);
            ++phase;
          } else if (auto memory = access(instruction)) {
            (phase == 0   ? region.partial
             : phase == 1 ? region.middle
                          : region.final)
                .push_back(*memory);
          } else if (auto *branch = dyn_cast<BranchInst>(&instruction)) {
            if (phase != 2)
              return false;
            for (unsigned edge = 0; edge < branch->getNumSuccessors(); ++edge) {
              auto *successor = branch->getSuccessor(edge);
              if (!walk(successor, &*successor->begin()))
                return false;
            }
          } else if (!pure(instruction)) {
            return false;
          }
        }
        active.erase(block);
        return true;
      };
  return walk(beginBlock, region.begin->getNextNode()) && region.end &&
         region.barriers.size() == 2;
}

inline std::optional<int64_t> offset(llvm::Value *pointer, unsigned tid,
                                     const llvm::DataLayout &layout) {
  llvm::DenseMap<llvm::Value *, llvm::Constant *> constants;
  llvm::DenseMap<llvm::Value *, llvm::APInt> integers;
  if (!mla_nope_prefetch::preparePointer(pointer, tid, layout, constants,
                                         integers))
    return std::nullopt;
  return readonly_shared_matrices::sharedOffset(pointer, tid, layout, integers);
}

inline std::optional<bool> predicate(llvm::Value *value, unsigned tid,
                                     const llvm::DataLayout &layout) {
  if (!value)
    return true;
  llvm::DenseMap<llvm::Value *, llvm::Constant *> constants;
  auto *constant = llvm::dyn_cast_or_null<llvm::ConstantInt>(
      mla_nope_prefetch::constantValue(value, tid, layout, constants));
  if (!constant || !constant->getType()->isIntegerTy(1))
    return std::nullopt;
  return !constant->isZero();
}

inline bool prove(Region &region, const llvm::DataLayout &layout) {
  using namespace llvm;
  unsigned fields = region.partial.size() / 2;
  if ((fields != 1 && fields != 2) || region.partial.size() != fields * 2 ||
      region.middle.size() != fields * 2 || region.final.size() != fields * 2)
    return false;
  region.fields = fields;
  std::array<BitVector, 2> partial = {BitVector(1024), BitVector(1024)};
  std::array<BitVector, 2> published = {BitVector(1024), BitVector(1024)};
  SmallVector<Access *> reads, writes;
  for (auto &memory : region.middle)
    (memory.write ? writes : reads).push_back(&memory);
  if (reads.size() != fields || writes.size() != fields)
    return false;
  for (auto &memory : region.partial) {
    if (!memory.write)
      return false;
    for (unsigned tid = 0; tid < 256; ++tid) {
      auto enabled = predicate(memory.predicate, tid, layout);
      auto address = offset(memory.pointer, tid, layout);
      if (!enabled || !address || *enabled != (tid % 4 == 0))
        return false;
      if (!*enabled)
        continue;
      int64_t byte = *address - 81920;
      unsigned group = tid / 128;
      if (byte < 0 || byte + 4 > fields * 512 || byte % 4 ||
          unsigned(byte % 512) / 256 != group)
        return false;
      for (unsigned i = 0; i < 4; ++i) {
        if (partial[group].test(byte + i))
          return false;
        partial[group].set(byte + i);
      }
    }
  }
  for (unsigned field = 0; field < fields; ++field) {
    auto *read = reads[field];
    auto found = llvm::find_if(
        writes, [&](Access *write) { return write->pointer == read->pointer; });
    auto *gep = dyn_cast<GetElementPtrInst>(read->pointer);
    if (found == writes.end() || !gep || gep->getNumIndices() != 1 ||
        !gep->getSourceElementType()->isFloatTy())
      return false;
    auto base = offset(gep->getPointerOperand(), 0, layout);
    if (!base || *base != 81920 + field * 512)
      return false;
    for (unsigned tid = 0; tid < 256; ++tid) {
      auto address = offset(read->pointer, tid, layout);
      auto loadMask = predicate(read->predicate, tid, layout);
      auto storeMask = predicate((*found)->predicate, tid, layout);
      if (!address || !loadMask || !storeMask || *address != *base + tid * 4 ||
          *loadMask != (tid < 128) || *storeMask != (tid < 128 && tid % 4 == 0))
        return false;
      if ((tid & 64) != 0)
        continue;
      unsigned logical = (tid & 63) | ((tid >> 1) & 64);
      unsigned byte = field * 512 + logical * 4, group = tid / 128;
      for (unsigned i = 0; i < 4; ++i) {
        if (!partial[group].test(byte + i))
          return false;
        if (tid % 4 == 0)
          published[group].set(byte + i);
      }
    }
  }
  for (auto &memory : region.final) {
    if (memory.write || memory.predicate || !isa<LoadInst>(memory.instruction))
      return false;
    for (unsigned tid = 0; tid < 256; ++tid) {
      auto address = offset(memory.pointer, tid, layout);
      if (!address || *address < 81920 || *address + 4 > 81920 + fields * 512 ||
          *address % 4)
        return false;
      for (unsigned i = 0; i < 4; ++i)
        if (!published[tid / 128].test(*address - 81920 + i))
          return false;
    }
  }
  // Middle values may escape only through their matched masked table stores.
  SmallPtrSet<Value *, 32> data;
  SmallPtrSet<Value *, 8> roots;
  for (auto *read : reads)
    roots.insert(read->instruction);
  std::function<bool(Value *)> visit = [&](Value *value) {
    if (roots.contains(value) || isa<Constant>(value))
      return true;
    if (!data.insert(value).second)
      return true;
    auto *instruction = dyn_cast<Instruction>(value);
    if (!instruction || instruction->getParent() != region.begin->getParent())
      return false;
    if (auto *call = dyn_cast<CallInst>(instruction)) {
      auto *callee = call->getCalledFunction();
      if (!callee || call->hasOperandBundles())
        return false;
      if (callee->getName() == "llvm.nvvm.shfl.sync.bfly.i32") {
        if (call->arg_size() != 4)
          return false;
        auto *mask = dyn_cast<ConstantInt>(call->getArgOperand(0));
        auto *delta = dyn_cast<ConstantInt>(call->getArgOperand(2));
        auto *width = dyn_cast<ConstantInt>(call->getArgOperand(3));
        return mask && mask->isMinusOne() && delta &&
               (delta->equalsInt(1) || delta->equalsInt(2)) && width &&
               width->equalsInt(31) && visit(call->getArgOperand(1));
      }
      if (callee->getName() != "llvm.maxnum.f32" || call->arg_size() != 2)
        return false;
    } else if (!isa<BitCastInst>(instruction) &&
               instruction->getOpcode() != Instruction::FAdd) {
      return false;
    }
    return llvm::all_of(instruction->operands(), [&](const Use &operand) {
      return isa<Function>(operand) || visit(operand);
    });
  };
  for (auto *write : writes)
    if (!visit(cast<CallInst>(write->instruction)->getArgOperand(1)))
      return false;
  for (Value *root : roots)
    data.insert(root);
  for (Value *value : data)
    for (User *user : value->users()) {
      bool store = llvm::any_of(writes, [&](Access *write) {
        return write->instruction == user &&
               cast<CallInst>(user)->getArgOperand(1) == value;
      });
      if (!data.contains(user) && !store)
        return false;
    }
  return true;
}

} // namespace mla_row_groups
} // namespace triton::ppu
#endif
