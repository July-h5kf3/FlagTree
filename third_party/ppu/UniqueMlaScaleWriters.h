// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_UNIQUE_MLA_SCALE_WRITERS_H
#define TRITON_PPU_UNIQUE_MLA_SCALE_WRITERS_H

#include "EarlySharedFloatScales.h"
#include "GatherMlaRowReductions.h"

namespace triton::ppu {
namespace mla_scale_writers {

struct Equivalent {
  unsigned first, second;
  const llvm::DataLayout &layout;
  llvm::LoopInfo &loops;
  llvm::DenseMap<llvm::Value *, bool> known;
  llvm::SmallPtrSet<llvm::Value *, 8> active;
  llvm::DenseMap<llvm::Value *, llvm::Constant *> constants[2];

  Equivalent(unsigned first, unsigned second, const llvm::DataLayout &layout,
             llvm::LoopInfo &loops)
      : first(first), second(second), layout(layout), loops(loops) {}

  bool value(llvm::Value *input, unsigned depth = 0) {
    using namespace llvm;
    if (depth > 256)
      return false;
    if (isa<Constant, Argument>(input))
      return true;
    if (auto it = known.find(input); it != known.end())
      return it->second;
    if (active.contains(input))
      return isa<PHINode>(input);
    auto *instruction = dyn_cast<Instruction>(input);
    if (!instruction)
      return false;
    if (auto *a = mla_nope_prefetch::constantValue(input, first, layout,
                                                   constants[0]))
      if (auto *b = mla_nope_prefetch::constantValue(input, second, layout,
                                                     constants[1]))
        return a == b;
    active.insert(input);
    bool equal = false;
    if (auto *load = dyn_cast<LoadInst>(instruction)) {
      auto a = mla_row_groups::offset(load->getPointerOperand(), first, layout);
      auto b =
          mla_row_groups::offset(load->getPointerOperand(), second, layout);
      unsigned width = layout.getTypeStoreSize(load->getType()).getFixedValue();
      if (load->getPointerAddressSpace() == 3 && !load->isVolatile() &&
          !load->isAtomic() && a && b && *a == *b && *a >= 81920 &&
          *a + width <= 82944) {
        for (Instruction *previous = load->getPrevNode(); previous;
             previous = previous->getPrevNode()) {
          if (readonly_shared_matrices::ctaBarrier(previous, 256)) {
            equal = true;
            break;
          }
          if (previous->mayWriteToMemory())
            break;
        }
      }
    } else if (auto *phi = dyn_cast<PHINode>(instruction)) {
      auto *loop = loops.getLoopFor(phi->getParent());
      bool control = false;
      if (loop && loop->getHeader() == phi->getParent()) {
        control = early_shared_float_scales::vector2(phi->getType()) &&
                  phi->getNumIncomingValues() == 2 &&
                  loop->getLoopPreheader() && loop->getLoopLatch() &&
                  phi->getBasicBlockIndex(loop->getLoopPreheader()) >= 0 &&
                  phi->getBasicBlockIndex(loop->getLoopLatch()) >= 0;
      } else if (phi->getNumIncomingValues() == 2) {
        auto *left = phi->getIncomingBlock(0);
        auto *right = phi->getIncomingBlock(1);
        auto *leftJump = dyn_cast<BranchInst>(left->getTerminator());
        auto *rightJump = dyn_cast<BranchInst>(right->getTerminator());
        auto *branchBlock = left->getSinglePredecessor();
        auto *branch = branchBlock
                           ? dyn_cast<BranchInst>(branchBlock->getTerminator())
                           : nullptr;
        if (leftJump && rightJump && leftJump->isUnconditional() &&
            rightJump->isUnconditional() && branch &&
            branchBlock == right->getSinglePredecessor() &&
            branch->isConditional() &&
            ((branch->getSuccessor(0) == left &&
              branch->getSuccessor(1) == right) ||
             (branch->getSuccessor(0) == right &&
              branch->getSuccessor(1) == left)))
          control = value(branch->getCondition(), depth + 1);
      }
      equal = control;
      for (Value *incoming : phi->incoming_values())
        equal &= value(incoming, depth + 1);
    } else {
      bool pure =
          instruction->isBinaryOp() ||
          isa<CastInst, ExtractElementInst, InsertElementInst,
              ShuffleVectorInst, FCmpInst, ICmpInst, SelectInst>(instruction);
      if (auto *call = dyn_cast<CallInst>(instruction)) {
        if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand()))
          pure = !call->hasOperandBundles() && !code->hasSideEffects() &&
                 !code->canThrow() && !code->isAlignStack() &&
                 code->getAsmString() == "ppu.div.full.f32 $0, $1, $2;" &&
                 code->getConstraintString() == "=f,f,f" &&
                 call->arg_size() == 2 && call->getType()->isFloatTy();
        else if (auto *callee = call->getCalledFunction())
          pure = !call->hasOperandBundles() && callee->isDeclaration() &&
                 StringSwitch<bool>(callee->getName())
                     .Cases("llvm.maxnum.f32", "llvm.maxnum.v2f32", true)
                     .Cases("llvm.nvvm.ex2.approx.ftz.f",
                            "llvm.nvvm.ex2.approx.f", true)
                     .Case("__nvvm_reflect", true)
                     .Default(false);
        else
          pure = false;
      }
      if (pure) {
        equal = true;
        if (auto *call = dyn_cast<CallInst>(instruction)) {
          for (Value *argument : call->args())
            equal &= value(argument, depth + 1);
        } else {
          for (Value *operand : instruction->operands())
            equal &= value(operand, depth + 1);
        }
      }
    }
    active.erase(input);
    known[input] = equal;
    return equal;
  }
};

inline unsigned run(llvm::Function &function) {
  using namespace llvm;
  if (function.getCallingConv() != CallingConv::PTX_Kernel ||
      function.hasFnAttribute("ppu.mla.scale.unique"))
    return 0;
  for (auto [name, expected] : {std::pair{"nvvm.reqntid", "256"},
                                {"ppu.mla.kv.pipeline", "1"},
                                {"ppu.mla.pv.batch", "4"},
                                {"ppu.mla.rope.prefetch", "1"}})
    if (!function.hasFnAttribute(name) ||
        function.getFnAttribute(name).getValueAsString() != expected)
      return 0;
  const auto &layout = function.getParent()->getDataLayout();
  DominatorTree dominance(function);
  LoopInfo loops(dominance);
  SmallVector<StoreInst *> stores;
  CallInst *thread = nullptr;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block) {
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *callee = call->getCalledFunction();
            callee && callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
          if (thread || !call->arg_empty())
            return 0;
          thread = call;
        }
      auto *store = dyn_cast<StoreInst>(&instruction);
      auto *loop = loops.getLoopFor(&block);
      if (!store || !loop || loop->getParentLoop() ||
          !early_shared_float_scales::vector2(
              store->getValueOperand()->getType()))
        continue;
      auto base = mla_row_groups::offset(store->getPointerOperand(), 0, layout);
      if (!base || (*base != 94208 && *base != 94336))
        continue;
      if (store->isVolatile() || store->isAtomic() ||
          store->getPointerAddressSpace() != 3 || store->getAlign() != Align(8))
        return 0;
      stores.push_back(store);
    }
  if (!thread || stores.size() != 2 || stores[0]->getNextNode() != stores[1] ||
      !dominance.dominates(thread, stores[0]))
    return 0;
  for (unsigned phase = 0; phase < 2; ++phase)
    for (unsigned tid = 0; tid < 256; ++tid) {
      auto address = mla_row_groups::offset(stores[phase]->getPointerOperand(),
                                            tid, layout);
      unsigned expected =
          94208 + phase * 128 + 4 * (tid & 28) + ((tid & 128) >> 4);
      if (!address || *address != expected)
        return 0;
      unsigned owner = tid & ~99u;
      Equivalent proof(tid, owner, layout, loops);
      if (!proof.value(stores[phase]->getValueOperand()))
        return 0;
    }
  IRBuilder<NoFolder> builder(stores[0]);
  Value *writer = builder.CreateICmpEQ(
      builder.CreateAnd(thread, builder.getInt32(99)), builder.getInt32(0));
  auto *assembly = InlineAsm::get(
      FunctionType::get(builder.getVoidTy(),
                        {stores[0]->getPointerOperand()->getType(),
                         builder.getInt32Ty(), builder.getInt32Ty(),
                         builder.getInt1Ty()},
                        false),
      "@$3 ppu.st.shared.v2.b32 [ $0 + 0 ], { $1, $2 };", "r,r,r,b", true);
  for (auto *store : stores) {
    builder.SetInsertPoint(store);
    Value *words[2];
    for (unsigned index = 0; index < 2; ++index)
      words[index] = builder.CreateBitCast(
          builder.CreateExtractElement(store->getValueOperand(),
                                       builder.getInt64(index)),
          builder.getInt32Ty());
    builder.CreateCall(
        assembly, {store->getPointerOperand(), words[0], words[1], writer});
    store->eraseFromParent();
  }
  function.addFnAttr("ppu.mla.scale.unique", "1");
  return 2;
}
} // namespace mla_scale_writers

inline unsigned uniqueMlaScaleWriters(llvm::Module &module, unsigned threads,
                                      unsigned shared) {
  if (threads != 256 || shared != 94464)
    return 0;
  unsigned changed = 0;
  for (llvm::Function &function : module)
    if (!function.isDeclaration())
      changed += mla_scale_writers::run(function);
  return changed;
}
} // namespace triton::ppu
#endif
