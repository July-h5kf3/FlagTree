// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_JOINT_MLA_KR_COPY_H
#define TRITON_PPU_JOINT_MLA_KR_COPY_H

#include "PrefetchMlaRope.h"
#include "ReadonlySharedMatrices.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/KnownBits.h"

namespace triton::ppu {
namespace mla_kr_copy {
using namespace llvm;

constexpr unsigned OldShared = 94464;
constexpr unsigned OldBase = 81920;
constexpr unsigned PlaneBytes = 8192;
constexpr unsigned Delta = 16384;
constexpr unsigned NewShared = OldBase + Delta + PlaneBytes;

inline bool matrix(CallInst *call) {
  auto *callee = call ? call->getCalledFunction() : nullptr;
  return callee &&
         callee->getName() ==
             "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3" &&
         call->arg_size() == 1 &&
         call->getArgOperand(0)->getType()->isPointerTy() &&
         call->getArgOperand(0)->getType()->getPointerAddressSpace() == 3 &&
         !call->hasOperandBundles();
}

inline bool covered(ArrayRef<Value *> pointers, unsigned expected,
                    const DataLayout &layout) {
  std::vector<unsigned> bytes(PlaneBytes);
  for (unsigned tid = 0; tid < 256; ++tid) {
    DenseMap<Value *, APInt> cache;
    for (Value *ptr : pointers) {
      auto offset =
          readonly_shared_matrices::sharedOffset(ptr, tid, layout, cache);
      if (!offset || *offset < OldBase || *offset + 16 > OldBase + PlaneBytes ||
          *offset % 16)
        return false;
      for (unsigned i = 0; i < 16; ++i)
        ++bytes[*offset - OldBase + i];
    }
  }
  return llvm::all_of(bytes,
                      [expected](unsigned count) { return count == expected; });
}

inline bool collectAddressChoices(Value *value,
                                  SmallVectorImpl<SelectInst *> &choices,
                                  SmallPtrSetImpl<Value *> &seen,
                                  unsigned depth = 0) {
  if (depth > 32)
    return false;
  if (!seen.insert(value).second)
    return true;
  if (auto *choice = dyn_cast<SelectInst>(value)) {
    if (choices.size() == 2 || !isa<ConstantInt>(choice->getTrueValue()) ||
        !isa<ConstantInt>(choice->getFalseValue()))
      return false;
    choices.push_back(choice);
    return true;
  }
  if (auto *op = dyn_cast<Operator>(value))
    for (Value *operand : op->operands())
      if (!collectAddressChoices(operand, choices, seen, depth + 1))
        return false;
  return true;
}

inline bool disjoint(Value *pointer, const DataLayout &layout) {
  SmallVector<SelectInst *, 2> choices;
  SmallPtrSet<Value *, 32> seen;
  if (!collectAddressChoices(pointer, choices, seen))
    return false;
  for (unsigned variant = 0; variant < (1u << choices.size()); ++variant)
    for (unsigned tid = 0; tid < 256; ++tid) {
      DenseMap<Value *, APInt> cache;
      for (auto [i, choice] : llvm::enumerate(choices)) {
        Value *arm = (variant >> i) & 1 ? choice->getTrueValue()
                                        : choice->getFalseValue();
        cache.try_emplace(choice, cast<ConstantInt>(arm)->getValue());
      }
      auto offset =
          readonly_shared_matrices::sharedOffset(pointer, tid, layout, cache);
      if (!offset || *offset < 0 || *offset + 16 > OldShared ||
          (*offset < OldBase + PlaneBytes && *offset + 16 > OldBase))
        return false;
    }
  return true;
}

// Lowering has already selected a 16-byte Arg3 global load. Keep its base
// alignment contract and check that each intervening offset preserves it.
inline bool alignedRopeOffset(Value *value, const DataLayout &layout) {
  while (auto *gep = dyn_cast<GEPOperator>(value)) {
    if (gep->getNumIndices() != 1)
      return false;
    Type *element = gep->getSourceElementType();
    unsigned lowBits = element->isIntegerTy(8) ? 4
                       : element->isBFloatTy() ? 3
                                               : 0;
    if (!lowBits || computeKnownBits(gep->idx_begin()->get(), layout)
                            .countMinTrailingZeros() < lowBits)
      return false;
    value = gep->getPointerOperand();
  }
  auto *arg = dyn_cast<Argument>(value);
  return arg && arg->getArgNo() == 3;
}

inline bool matchStore(StoreInst *store, CallInst *load,
                       SmallVectorImpl<Instruction *> &dead) {
  if (!store || store->getPointerAddressSpace() != 3 || store->isVolatile() ||
      store->isAtomic() || store->getAlign() != Align(16) || !load->hasNUses(4))
    return false;
  Value *packed = store->getValueOperand();
  SmallVector<Instruction *> extracts;
  for (int slot = 3; slot >= 0; --slot) {
    auto *insert = dyn_cast<InsertElementInst>(packed);
    auto *index =
        insert ? dyn_cast<ConstantInt>(insert->getOperand(2)) : nullptr;
    auto *word =
        insert ? dyn_cast<ExtractValueInst>(insert->getOperand(1)) : nullptr;
    if (!index || !index->equalsInt(slot) || !insert->hasOneUse() || !word ||
        !word->hasOneUse() || word->getAggregateOperand() != load ||
        word->getIndices() != ArrayRef<unsigned>({static_cast<unsigned>(slot)}))
      return false;
    dead.push_back(insert);
    extracts.push_back(word);
    packed = insert->getOperand(0);
  }
  if (!isa<UndefValue>(packed))
    return false;
  dead.append(extracts);
  return true;
}

inline bool rewrite(Function &kernel, MDNode &annotation) {
  if (kernel.getFnAttribute("nvvm.reqntid").getValueAsString() != "256" ||
      kernel.getFnAttribute("ppu.mla.kv.pipeline").getValueAsString() != "1" ||
      kernel.getFnAttribute("ppu.mla.pv.batch").getValueAsString() != "4" ||
      kernel.getFnAttribute("ppu.mla.rope.prefetch").getValueAsString() != "1")
    return false;
  SmallVector<CallInst *> reads;
  for (auto &block : kernel)
    for (auto &inst : block)
      if (auto *call = dyn_cast<CallInst>(&inst); matrix(call))
        reads.push_back(call);
  if (reads.size() != 8)
    return false;
  auto *header = reads.front()->getParent();
  DominatorTree dominance(kernel);
  LoopInfo loops(dominance);
  auto *loop = loops.getLoopFor(header);
  if (!loop || loop->getHeader() != header || loop->getParentLoop() ||
      !loop->getLoopLatch())
    return false;
  SmallVector<Instruction *> effects;
  SmallVector<CallInst *> loads;
  for (auto &inst : *header) {
    if (isa<CallBase, FenceInst>(inst) || inst.mayReadOrWriteMemory())
      effects.push_back(&inst);
    if (auto *call = dyn_cast<CallInst>(&inst);
        call && mla_rope_prefetch::globalLoad(call))
      loads.push_back(call);
  }
  auto firstRead = llvm::find(effects, reads.front());
  if (loads.size() != 4 || firstRead == effects.end() ||
      std::distance(effects.begin(), firstRead) < 3 ||
      std::distance(firstRead, effects.end()) < 12)
    return false;
  auto phase = ArrayRef(effects).slice(
      std::distance(effects.begin(), firstRead) - 3, 15);
  for (unsigned i = 0; i < 8; ++i)
    if (phase[i < 4 ? i + 3 : i + 7] != reads[i] ||
        (i >= 4 &&
         reads[i]->getArgOperand(0) != reads[i - 4]->getArgOperand(0)))
      return false;
  for (unsigned index : {2u, 7u, 10u})
    if (!readonly_shared_matrices::ctaBarrier(phase[index], 256))
      return false;
  SmallVector<Value *> storePointers, readPointers;
  SmallVector<Instruction *> dead;
  const auto &layout = kernel.getParent()->getDataLayout();
  for (auto [i, load] : llvm::enumerate(loads)) {
    auto *store = dyn_cast<StoreInst>(phase[i < 2 ? i : i + 6]);
    if (!load->comesBefore(phase[0]) || !matchStore(store, load, dead) ||
        !dominance.dominates(store->getPointerOperand(), load) ||
        !alignedRopeOffset(load->getArgOperand(4), layout))
      return false;
    storePointers.push_back(store->getPointerOperand());
  }
  if (storePointers[0] != storePointers[2] ||
      storePointers[1] != storePointers[3])
    return false;
  for (auto *read : ArrayRef(reads).take_front(4))
    readPointers.push_back(read->getArgOperand(0));
  if (!covered(ArrayRef(storePointers).take_front(2), 1, layout) ||
      !covered(readPointers, 2, layout))
    return false;
  // Reuse the original loop-header CTA publication as the prior-iteration
  // scratch lifetime boundary. Between it and the exchange, the only memory
  // readers other than the replaced loads are disjoint NoPE matrix reads.
  auto begin = llvm::find(effects, loads.front());
  while (begin != effects.begin() &&
         !readonly_shared_matrices::ctaBarrier(*begin, 256))
    --begin;
  if (!readonly_shared_matrices::ctaBarrier(*begin, 256))
    return false;
  for (auto it = std::next(begin); *it != phase[0]; ++it) {
    auto *call = dyn_cast<CallInst>(*it);
    auto *code = call ? dyn_cast<InlineAsm>(call->getCalledOperand()) : nullptr;
    if (!code || call->hasOperandBundles() || code->canThrow() ||
        code->getConstraintString().contains("memory"))
      return false;
    if (llvm::is_contained(loads, call) ||
        code->getAsmString() == mla_nope_prefetch::ProductAsm)
      continue;
    auto *address = call->arg_size() == 1
                        ? dyn_cast<PtrToIntInst>(call->getArgOperand(0))
                        : nullptr;
    if (code->getAsmString() != mla_nope_prefetch::MatrixAsm || !address ||
        !disjoint(address->getPointerOperand(), layout))
      return false;
  }

  auto &module = *kernel.getParent();
  auto *commit = module.getFunction("llvm.nvvm.cp.async.commit.group");
  auto *wait = module.getFunction("llvm.nvvm.cp.async.wait.group");
  if (!commit || !wait || !commit->isIntrinsic() || !wait->isIntrinsic())
    return false;
  IRBuilder<> builder(loads.front());
  auto *copyType = FunctionType::get(builder.getVoidTy(),
                                     {storePointers[0]->getType(),
                                      loads[0]->getArgOperand(4)->getType(),
                                      builder.getInt32Ty()},
                                     false);
  auto *copy = InlineAsm::get(
      copyType, "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;",
      "r,l,r", true);
  for (auto [i, load] : llvm::enumerate(loads)) {
    builder.SetInsertPoint(load);
    builder.SetCurrentDebugLocation(phase[i < 2 ? i : i + 6]->getDebugLoc());
    auto *size =
        builder.CreateSelect(load->getArgOperand(5), builder.getInt32(16),
                             builder.getInt32(0), "kr.copy.bytes");
    Value *destination = storePointers[i];
    if (i < 2)
      destination = builder.CreateGEP(builder.getInt8Ty(), destination,
                                      builder.getInt32(Delta), "kr.copy.dst");
    auto *call = builder.CreateCall(
        copyType, copy, {destination, load->getArgOperand(4), size});
    call->setTailCallKind(CallInst::TCK_Tail);
  }
  builder.CreateCall(commit)->setTailCallKind(CallInst::TCK_Tail);
  builder.SetInsertPoint(phase[2]);
  builder.SetCurrentDebugLocation(DebugLoc());
  builder.CreateCall(wait, {builder.getInt32(0)})
      ->setTailCallKind(CallInst::TCK_Tail);
  for (auto *read : ArrayRef(reads).take_front(4)) {
    builder.SetInsertPoint(read);
    read->setArgOperand(
        0, builder.CreateGEP(builder.getInt8Ty(), read->getArgOperand(0),
                             builder.getInt32(Delta), "kr.copy.reader"));
  }
  for (unsigned index : {0u, 1u, 7u, 8u, 9u, 10u})
    phase[index]->eraseFromParent();
  for (auto *inst : dead)
    inst->eraseFromParent();
  for (auto *load : loads)
    load->eraseFromParent();
  annotation.replaceOperandWith(
      2, ConstantAsMetadata::get(builder.getInt32(NewShared)));
  return true;
}
} // namespace mla_kr_copy

inline std::pair<unsigned, unsigned>
jointMlaKrCopy(llvm::Module &module, unsigned threads, unsigned shared) {
  using namespace llvm;
  if (threads != 256 || shared != mla_kr_copy::OldShared)
    return {shared, 0};
  Function *kernel = nullptr;
  for (auto &function : module)
    if (!function.isDeclaration() &&
        function.getCallingConv() == CallingConv::PTX_Kernel) {
      if (kernel)
        return {shared, 0};
      kernel = &function;
    }
  auto *annotations = module.getNamedMetadata("nvvm.annotations");
  if (!kernel || !annotations)
    return {shared, 0};
  MDNode *annotation = nullptr;
  for (auto *entry : annotations->operands()) {
    if (entry->getNumOperands() < 2)
      continue;
    auto *owner = dyn_cast_or_null<ValueAsMetadata>(entry->getOperand(0));
    auto *key = dyn_cast_or_null<MDString>(entry->getOperand(1));
    if (!owner || owner->getValue() != kernel || !key ||
        key->getString() != "smemsize")
      continue;
    if (annotation || entry->getNumOperands() != 3)
      return {shared, 0};
    auto *value =
        mdconst::dyn_extract_or_null<ConstantInt>(entry->getOperand(2));
    if (!value || !value->getType()->isIntegerTy(32) ||
        !value->equalsInt(shared))
      return {shared, 0};
    annotation = entry;
  }
  if (!annotation || !mla_kr_copy::rewrite(*kernel, *annotation))
    return {shared, 0};
  return {mla_kr_copy::NewShared, 1};
}
} // namespace triton::ppu

#endif // TRITON_PPU_JOINT_MLA_KR_COPY_H
