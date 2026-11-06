// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_PREFETCH_MLA_ROPE_H
#define TRITON_PPU_PREFETCH_MLA_ROPE_H

#include "PrefetchMlaNope.h"
#include "llvm/ADT/StringSwitch.h"

namespace triton::ppu {
namespace mla_rope_prefetch {

constexpr llvm::StringLiteral LoadAsm =
    "ppu.mov.u32 $0, $4;\n\tppu.mov.u32 $1, $5;\n\t"
    "ppu.mov.u32 $2, $6;\n\tppu.mov.u32 $3, $7;\n\t"
    "@$9 ppu.ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];";

inline bool globalLoad(llvm::CallInst *call) {
  using namespace llvm;
  auto *code = dyn_cast<InlineAsm>(call->getCalledOperand());
  auto *word = Type::getInt32Ty(call->getContext());
  auto *result =
      StructType::get(call->getContext(), SmallVector<Type *>(4, word));
  if (!code || code->getAsmString() != LoadAsm ||
      code->getConstraintString() != "=r,=r,=r,=r,r,r,r,r,l,b" ||
      !code->hasSideEffects() || code->isAlignStack() || code->canThrow() ||
      code->getDialect() != InlineAsm::AD_ATT || call->arg_size() != 6 ||
      call->getType() != result || call->hasOperandBundles())
    return false;
  for (unsigned index = 0; index < 4; ++index) {
    auto *initial = dyn_cast<ConstantInt>(call->getArgOperand(index));
    if (!initial || !initial->getType()->isIntegerTy(32) || !initial->isZero())
      return false;
  }
  Value *pointer = call->getArgOperand(4);
  if (!pointer->getType()->isPointerTy() ||
      pointer->getType()->getPointerAddressSpace() != 1 ||
      !call->getArgOperand(5)->getType()->isIntegerTy(1))
    return false;
  while (auto *gep = dyn_cast<GEPOperator>(pointer))
    pointer = gep->getPointerOperand();
  auto *base = dyn_cast<Argument>(pointer);
  return base && base->getArgNo() == 3;
}

inline bool addressOperation(llvm::Instruction *instruction) {
  using namespace llvm;
  if (auto *gep = dyn_cast<GetElementPtrInst>(instruction))
    return gep->getPointerAddressSpace() == 1;
  auto integer = [](Type *type) {
    return type->isIntegerTy() && type->getIntegerBitWidth() <= 64;
  };
  if (!integer(instruction->getType()))
    return false;
  if (auto *cast = dyn_cast<CastInst>(instruction))
    return integer(cast->getSrcTy());
  if (isa<ICmpInst, SelectInst>(instruction))
    return llvm::all_of(instruction->operands(), [&](const Use &operand) {
      return integer(operand->getType());
    });
  return StringSwitch<bool>(instruction->getOpcodeName())
      .Cases("add", "sub", "mul", true)
      .Cases("shl", "lshr", "ashr", true)
      .Cases("and", "or", "xor", true)
      .Default(false);
}

inline unsigned run(llvm::Function &function) {
  using namespace llvm;
  if (function.getCallingConv() != CallingConv::PTX_Kernel ||
      !function.hasFnAttribute("ppu.mla.kv.pipeline") ||
      function.getFnAttribute("ppu.mla.pv.batch").getValueAsString() != "4" ||
      function.hasFnAttribute("ppu.mla.rope.prefetch"))
    return 0;
  SmallVector<CallInst *> products;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand()))
          if (code->getAsmString() == mla_nope_prefetch::ProductAsm)
            products.push_back(call);
  if (products.size() != 96)
    return 0;
  auto *anchor = products.front();
  auto *block = anchor->getParent();
  if (llvm::any_of(ArrayRef(products).take_front(32), [&](auto *product) {
        return product->getParent() != block;
      }))
    return 0;
  SmallVector<CallInst *> loads;
  for (Instruction &instruction : *block)
    if (auto *call = dyn_cast<CallInst>(&instruction))
      if (globalLoad(call))
        loads.push_back(call);
  if (loads.size() != 4 || !products[31]->comesBefore(loads.front()))
    return 0;
  // No writer, barrier, unknown call or volatile access may be crossed.
  // In particular the input has the same value at both read positions even
  // when a caller aliases kernel arguments.
  for (Instruction *instruction = anchor; instruction != loads.back();
       instruction = instruction->getNextNode()) {
    if (auto *load = dyn_cast<LoadInst>(instruction)) {
      if (load->isVolatile() || load->isAtomic())
        return 0;
    } else if (auto *call = dyn_cast<CallInst>(instruction)) {
      if (llvm::is_contained(loads, call))
        continue;
      auto *code = dyn_cast<InlineAsm>(call->getCalledOperand());
      if (!code ||
          code->getConstraintString().find("memory") != std::string::npos ||
          code->canThrow() || call->hasOperandBundles() ||
          (code->getAsmString() != mla_nope_prefetch::ProductAsm &&
           code->getAsmString() != mla_nope_prefetch::MatrixAsm))
        return 0;
    } else if (instruction->mayReadOrWriteMemory())
      return 0;
  }
  DominatorTree dominance(function);
  SmallPtrSet<Instruction *, 32> seen;
  SmallVector<Instruction *> ordered;
  std::function<bool(Value *)> collect = [&](Value *value) {
    auto *instruction = dyn_cast<Instruction>(value);
    if (!instruction)
      return isa<Constant, Argument>(value);
    // Loop PHIs, existing page reads and scalar bases are leaves whenever they
    // already dominate the new issue position; never recurse around backedges.
    if (dominance.dominates(instruction, anchor))
      return true;
    if (instruction->getParent() != block || !addressOperation(instruction))
      return false;
    if (seen.size() >= 64)
      return false;
    if (!seen.insert(instruction).second)
      return true;
    for (Value *operand : instruction->operands())
      if (!collect(operand))
        return false;
    ordered.push_back(instruction);
    return true;
  };
  for (auto *load : loads) {
    for (Value *argument : load->args())
      if (!collect(argument))
        return 0;
    ordered.push_back(load);
  }
  // All effect and dominance checks complete before any existing node moves.
  for (Instruction *instruction : ordered)
    instruction->moveBefore(anchor->getIterator());
  function.addFnAttr("ppu.mla.rope.prefetch", "1");
  return loads.size();
}
} // namespace mla_rope_prefetch

inline unsigned prefetchMlaRope(llvm::Module &module, unsigned threads,
                                unsigned shared) {
  if (threads != 256 || shared != 94464)
    return 0;
  unsigned changed = 0;
  for (llvm::Function &function : module)
    if (!function.isDeclaration())
      changed += mla_rope_prefetch::run(function);
  return changed;
}
} // namespace triton::ppu
#endif
