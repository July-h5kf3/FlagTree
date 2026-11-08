// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_FOLD_MLA_LOGIT_SCALE_H
#define TRITON_PPU_FOLD_MLA_LOGIT_SCALE_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"

namespace triton::ppu {
namespace mla_logit_scale {
using namespace llvm;

inline bool scaleLoad(Value *value, unsigned argument) {
  auto *bits = dyn_cast<BitCastInst>(value);
  auto *load = bits ? dyn_cast<CallInst>(bits->getOperand(0)) : nullptr;
  auto *assembly =
      load ? dyn_cast<InlineAsm>(load->getCalledOperand()) : nullptr;
  if (!assembly || load->arg_size() != 3 || !load->getType()->isIntegerTy(32) ||
      !load->getArgOperand(2)->getType()->isIntegerTy(1) ||
      !isa<ConstantInt>(load->getArgOperand(0)) ||
      !cast<ConstantInt>(load->getArgOperand(0))->isZero() ||
      assembly->getAsmString() !=
          "ppu.mov.u32 $0, $1;\n\t@$3 ppu.ld.global.b32 { $0 }, [ $2 + 0 ];" ||
      assembly->getConstraintString() != "=r,r,l,b" ||
      !assembly->hasSideEffects() || assembly->canThrow() ||
      load->hasOperandBundles() ||
      !load->getArgOperand(1)->getType()->isPointerTy() ||
      load->getArgOperand(1)->getType()->getPointerAddressSpace() != 1)
    return false;
  auto *pointer = load->getArgOperand(1);
  while (auto *gep = dyn_cast<GEPOperator>(pointer))
    pointer = gep->getPointerOperand();
  auto *arg = dyn_cast<Argument>(pointer);
  return arg && arg->getArgNo() == argument;
}

struct Chain {
  BinaryOperator *last, *keyProduct;
  Instruction *key;
};

inline bool fold(Function &function) {
  SmallVector<Chain> chains;
  SmallPtrSet<Instruction *, 8> keys, queries;
  ConstantFP *factor = nullptr;
  BasicBlock *body = nullptr;
  for (auto &block : function)
    for (auto &instruction : block) {
      auto *last = dyn_cast<BinaryOperator>(&instruction);
      auto *constant = last && last->getOpcode() == Instruction::FMul
                           ? dyn_cast<ConstantFP>(last->getOperand(1))
                           : nullptr;
      if (!constant || !constant->getType()->isFloatTy() ||
          constant->getValueAPF().bitcastToAPInt() != APInt(32, 0x3d76384f))
        continue;
      auto *key = dyn_cast<BinaryOperator>(last->getOperand(0));
      auto *query =
          key ? dyn_cast<BinaryOperator>(key->getOperand(0)) : nullptr;
      auto *score =
          query ? dyn_cast<ExtractValueInst>(query->getOperand(0)) : nullptr;
      if (!key || !query || !score || key->getOpcode() != Instruction::FMul ||
          query->getOpcode() != Instruction::FMul || !key->hasOneUse() ||
          !query->hasOneUse() || last->getFastMathFlags().any() ||
          key->getFastMathFlags().any() || query->getFastMathFlags().any() ||
          key->getParent() != &block || query->getParent() != &block ||
          (body && body != &block) || !scaleLoad(key->getOperand(1), 5) ||
          !scaleLoad(query->getOperand(1), 4))
        return false;
      factor = constant;
      body = &block;
      auto *keyScale = cast<Instruction>(key->getOperand(1));
      keys.insert(keyScale);
      queries.insert(cast<Instruction>(query->getOperand(1)));
      chains.push_back({last, key, keyScale});
    }
  if (chains.size() != 16 || keys.size() != 8 || queries.size() != 2)
    return false;
  DenseMap<Instruction *, Value *> combined;
  for (const Chain &chain : chains) {
    auto *&scaled = combined[chain.key];
    if (!scaled) {
      IRBuilder<> builder(chain.key->getNextNode());
      builder.SetCurrentDebugLocation(DebugLoc());
      scaled = builder.CreateFMul(chain.key, factor, "log2.key.scale");
    }
    chain.keyProduct->setOperand(1, scaled);
    chain.last->replaceAllUsesWith(chain.keyProduct);
  }
  // The post-LLVM SDK pipeline removes the now dead constant multiplies.
  return true;
}
} // namespace mla_logit_scale

inline unsigned foldMlaLogitScale(llvm::Module &module, unsigned threads,
                                  unsigned shared) {
  if (threads != 256 || shared != 106496)
    return 0;
  unsigned changed = 0;
  for (auto &function : module)
    if (!function.isDeclaration() &&
        function.getCallingConv() == llvm::CallingConv::PTX_Kernel &&
        function.getFnAttribute("ppu.mla.kv.pipeline").getValueAsString() ==
            "1" &&
        function.getFnAttribute("ppu.mla.pv.batch").getValueAsString() == "4")
      changed += mla_logit_scale::fold(function);
  return changed;
}
} // namespace triton::ppu

#endif // TRITON_PPU_FOLD_MLA_LOGIT_SCALE_H
