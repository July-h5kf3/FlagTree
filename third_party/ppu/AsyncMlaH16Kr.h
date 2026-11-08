// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_ASYNC_MLA_H16_KR_H
#define TRITON_PPU_ASYNC_MLA_H16_KR_H

#include "FoldMlaLogitScale.h"
#include "JointMlaKrCopy.h"

namespace triton::ppu {
namespace mla_h16_kr {
using namespace llvm;

// All possible per-thread addresses must remain inside the specified plane.
inline bool inside(Value *pointer, const DataLayout &layout, unsigned begin,
                   unsigned end) {
  SmallVector<SelectInst *, 2> choices;
  SmallPtrSet<Value *, 32> seen;
  if (!mla_kr_copy::collectAddressChoices(pointer, choices, seen))
    return false;
  for (unsigned variant = 0; variant < (1u << choices.size()); ++variant)
    for (unsigned tid = 0; tid < 256; ++tid) {
      DenseMap<Value *, APInt> cache;
      for (auto [i, choice] : llvm::enumerate(choices))
        cache.try_emplace(choice,
                          cast<ConstantInt>((variant >> i) & 1
                                                ? choice->getTrueValue()
                                                : choice->getFalseValue())
                              ->getValue());
      auto offset =
          readonly_shared_matrices::sharedOffset(pointer, tid, layout, cache);
      if (!offset || *offset < begin || *offset + 16 > end || *offset % 16)
        return false;
    }
  return true;
}

inline bool run(Function &kernel) {
  if (kernel.getCallingConv() != CallingConv::PTX_Kernel ||
      kernel.getFnAttribute("nvvm.reqntid").getValueAsString() != "256" ||
      kernel.getFnAttribute("ppu.mla.kv.pipeline").getValueAsString() != "1" ||
      kernel.hasFnAttribute("ppu.mla.h16.kr.copy"))
    return false;
  SmallVector<CallInst *> products, loads;
  for (auto &block : kernel)
    for (auto &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction)) {
        if (mla_rope_prefetch::globalLoad(call))
          loads.push_back(call);
        if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand());
            code && code->getAsmString() == mla_nope_prefetch::ProductAsm)
          products.push_back(call);
      }
  if (products.size() != 48 || loads.size() != 4)
    return false;
  auto *header = products.front()->getParent();
  auto *anchor = products[15]->getNextNode();
  DominatorTree dominance(kernel);
  LoopInfo loops(dominance);
  auto *loop = loops.getLoopFor(header);
  if (!anchor || !loop || loop->getHeader() != header ||
      loop->getParentLoop() || !loop->getLoopLatch() ||
      llvm::any_of(ArrayRef(products).take_front(16),
                   [&](auto *call) { return call->getParent() != header; }) ||
      llvm::any_of(loads, [&](auto *call) {
        return call->getParent() != header ||
               !call->comesBefore(products.front());
      }))
    return false;

  const auto &layout = kernel.getParent()->getDataLayout();
  SmallVector<StoreInst *> stores;
  SmallVector<Instruction *> dead;
  SmallVector<Value *> destinations;
  for (auto *load : loads) {
    StoreInst *matched = nullptr;
    for (auto &instruction : *header)
      if (auto *store = dyn_cast<StoreInst>(&instruction)) {
        SmallVector<Instruction *> chain;
        if (mla_kr_copy::matchStore(store, load, chain)) {
          if (matched || !load->comesBefore(store) ||
              !dominance.dominates(store->getPointerOperand(), load))
            return false;
          matched = store;
          dead.append(chain);
        }
      }
    if (!matched ||
        !mla_kr_copy::alignedRopeOffset(load->getArgOperand(4), layout) ||
        !inside(matched->getPointerOperand(), layout, 65536, 81920))
      return false;
    stores.push_back(matched);
    destinations.push_back(matched->getPointerOperand());
  }
  std::vector<unsigned> coverage(16384);
  for (unsigned tid = 0; tid < 256; ++tid) {
    DenseMap<Value *, APInt> cache;
    for (auto *pointer : destinations) {
      auto offset =
          readonly_shared_matrices::sharedOffset(pointer, tid, layout, cache);
      if (!offset || *offset < 65536 || *offset + 16 > 81920 || *offset % 16)
        return false;
      for (unsigned byte = 0; byte < 16; ++byte)
        ++coverage[*offset - 65536 + byte];
    }
  }
  if (llvm::any_of(coverage, [](unsigned count) { return count != 1; }))
    return false;
  auto *lastStore = *llvm::max_element(
      stores, [](auto *a, auto *b) { return a->comesBefore(b); });
  auto *publish = lastStore->getNextNode();
  if (!publish || !readonly_shared_matrices::ctaBarrier(publish, 256) ||
      !loads.back()->comesBefore(stores.front()) ||
      !publish->comesBefore(products.front()))
    return false;

  // The previous KV wait and publication also end the prior KR scratch use.
  Instruction *prior = loads.front()->getPrevNode();
  while (prior && !prior->mayReadOrWriteMemory() && !isa<CallBase>(prior))
    prior = prior->getPrevNode();
  auto *oldWait =
      prior ? dyn_cast_or_null<CallInst>(prior->getPrevNode()) : nullptr;
  auto *waitFunction = oldWait ? oldWait->getCalledFunction() : nullptr;
  if (!prior || !readonly_shared_matrices::ctaBarrier(prior, 256) ||
      !waitFunction || !waitFunction->isIntrinsic() ||
      waitFunction->getName() != "llvm.nvvm.cp.async.wait.group" ||
      oldWait->arg_size() != 1 ||
      !isa<ConstantInt>(oldWait->getArgOperand(0)) ||
      !cast<ConstantInt>(oldWait->getArgOperand(0))->isZero())
    return false;

  SmallVector<Instruction *> readers;
  unsigned matrices = 0;
  for (auto *instruction = prior->getNextNode(); instruction != anchor;
       instruction = instruction->getNextNode()) {
    if (!instruction || instruction->getParent() != header)
      return false;
    if (instruction == publish || llvm::is_contained(stores, instruction))
      continue;
    if (auto *call = dyn_cast<CallInst>(instruction)) {
      if (call->hasOperandBundles())
        return false;
      if (llvm::is_contained(loads, call) ||
          llvm::is_contained(ArrayRef(products).take_front(16), call))
        continue;
      if (mla_kr_copy::matrix(call)) {
        Value *pointer = call->getArgOperand(0);
        if (inside(pointer, layout, 0, 65536))
          continue;
        if (!inside(pointer, layout, 65536, 81920) ||
            !publish->comesBefore(call) ||
            !call->comesBefore(products.front()) || !call->hasNUses(4))
          return false;
        SmallVector<ExtractValueInst *> words(4, nullptr);
        for (User *user : call->users()) {
          auto *word = dyn_cast<ExtractValueInst>(user);
          if (!word || word->getIndices().size() != 1 ||
              word->getIndices()[0] >= 4 || words[word->getIndices()[0]] ||
              word->getParent() != header ||
              !word->getType()->isIntegerTy(32) ||
              llvm::any_of(word->users(), [&](User *consumer) {
                auto *use = dyn_cast<Instruction>(consumer);
                return !use || !dominance.dominates(anchor, use);
              }))
            return false;
          words[word->getIndices()[0]] = word;
        }
        readers.push_back(call);
        readers.append(words.begin(), words.end());
        ++matrices;
        continue;
      }
      // The four scalar K-scale reads are read-only and remain in place.
      if (call->hasOneUse() &&
          mla_logit_scale::scaleLoad(*call->user_begin(), 5))
        continue;
      return false;
    }
    if (instruction->mayReadOrWriteMemory() || isa<FenceInst>(instruction))
      return false;
  }
  if (matrices != 4)
    return false;

  auto &module = *kernel.getParent();
  auto *commit = module.getFunction("llvm.nvvm.cp.async.commit.group");
  if (!commit || !commit->isIntrinsic())
    return false;
  // Every guard is complete before changing existing IR.
  IRBuilder<> builder(loads.front());
  auto *copyType = FunctionType::get(builder.getVoidTy(),
                                     {destinations[0]->getType(),
                                      loads[0]->getArgOperand(4)->getType(),
                                      builder.getInt32Ty()},
                                     false);
  auto *copy = InlineAsm::get(
      copyType, "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;",
      "r,l,r", true);
  for (auto [index, load] : llvm::enumerate(loads)) {
    builder.SetInsertPoint(load);
    builder.SetCurrentDebugLocation(DebugLoc());
    auto *size =
        builder.CreateSelect(load->getArgOperand(5), builder.getInt32(16),
                             builder.getInt32(0), "h16.kr.bytes");
    builder
        .CreateCall(copyType, copy,
                    {destinations[index], load->getArgOperand(4), size})
        ->setTailCallKind(CallInst::TCK_Tail);
  }
  builder.SetInsertPoint(loads.back()->getNextNode());
  builder.CreateCall(commit)->setTailCallKind(CallInst::TCK_Tail);
  builder.SetInsertPoint(anchor);
  builder.CreateCall(waitFunction, {builder.getInt32(0)})
      ->setTailCallKind(CallInst::TCK_Tail);
  publish->moveBefore(anchor->getIterator());
  for (auto *reader : readers)
    reader->moveBefore(anchor->getIterator());
  for (auto *store : stores)
    store->eraseFromParent();
  for (auto *instruction : dead)
    instruction->eraseFromParent();
  for (auto *load : loads)
    load->eraseFromParent();
  kernel.addFnAttr("ppu.mla.h16.kr.copy", "1");
  return true;
}
} // namespace mla_h16_kr

inline unsigned asyncMlaH16Kr(llvm::Module &module, unsigned threads,
                              unsigned shared) {
  if (threads != 256 || shared != 83968)
    return 0;
  unsigned changed = 0;
  for (auto &function : module)
    if (!function.isDeclaration())
      changed += mla_h16_kr::run(function);
  return changed;
}
} // namespace triton::ppu
#endif
