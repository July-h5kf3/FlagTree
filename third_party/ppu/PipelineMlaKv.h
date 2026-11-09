// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_PIPELINE_MLA_KV_H
#define TRITON_PPU_PIPELINE_MLA_KV_H
#include "EarlySharedFloatScales.h"
#include "PrefetchMlaNope.h"
#include "ScheduleMlaPvTail.h"

namespace triton::ppu {
namespace mla_kv_pipeline {
inline bool pageRead(llvm::CallInst *call) {
  using namespace llvm;
  auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand());
  if (!assembly ||
      assembly->getAsmString() !=
          "ppu.mov.u32 $0, 0x0;\n\t@$2 ppu.ld.global.b32 { $0 }, [ $1 + 0 ];" ||
      assembly->getConstraintString() != "=r,l,b" || call->arg_size() != 2 ||
      !call->getType()->isIntegerTy(32) ||
      !call->getArgOperand(0)->getType()->isPointerTy() ||
      call->getArgOperand(0)->getType()->getPointerAddressSpace() != 1 ||
      !call->getArgOperand(1)->getType()->isIntegerTy(1))
    return false;
  Value *base = call->getArgOperand(0);
  while (auto *gep = dyn_cast<GEPOperator>(base))
    base = gep->getPointerOperand();
  auto *argument = dyn_cast<Argument>(base);
  return argument && argument->getArgNo() == 6;
}

inline unsigned run(llvm::Function &function, unsigned shared, bool reusePages,
                    bool small = false, unsigned threads = 512) {
  using namespace llvm;
  if (function.getCallingConv() != CallingConv::PTX_Kernel ||
      function.hasFnAttribute("ppu.mla.kv.pipeline"))
    return 0;
  bool h32 = threads == 256 && shared == 94464;
  bool h16 = threads == 256 && shared == 83968;
  unsigned qkProducts = h16 ? 16 : 32;
  unsigned kvBytes = small || h32 || h16 ? 65536 : 131072;
  unsigned pageStep = small || h32 || h16 ? 2 : 4;
  bool pvBatch =
      ((threads == 512 && shared == 123392) || h32) &&
      function.getFnAttribute("ppu.mla.pv.batch").getValueAsString() == "4";
  unsigned remainingProducts = h16       ? 32
                               : pvBatch ? (h32 ? 49 : 51)
                                         : (h32 ? 37 : (small ? 39 : 35));
  SmallVector<CallInst *> products, copies;
  DenseMap<Instruction *, unsigned> position;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block) {
      position[&instruction] = position.size();
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand())) {
          if (assembly->getAsmString().contains(".s32.s8.s8.s32"))
            products.push_back(call);
          if (assembly->getAsmString() ==
              "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;")
            copies.push_back(call);
        }
    }
  if (products.size() != (h16 ? 48 : 96) || copies.size() != (small ? 8 : 16))
    return 0;
  auto *header = products.front()->getParent();
  DominatorTree dominance(function);
  LoopInfo loops(dominance);
  auto *loop = loops.getLoopFor(header);
  if (!loop || loop->getHeader() != header || loop->getParentLoop())
    return 0;
  auto *preheader = loop->getLoopPreheader();
  auto *latch = loop->getLoopLatch();
  if (!preheader || !latch || latch == header ||
      llvm::any_of(copies,
                   [&](auto *call) { return call->getParent() != header; }) ||
      llvm::any_of(ArrayRef(products).drop_front(qkProducts),
                   [&](auto *call) { return call->getParent() != latch; }))
    return 0;
  auto *backedge = dyn_cast<BranchInst>(latch->getTerminator());
  if (!backedge || !backedge->isConditional() ||
      backedge->getSuccessor(0) != header ||
      loop->contains(backedge->getSuccessor(1)))
    return 0;
  auto *condition = dyn_cast<ICmpInst>(backedge->getCondition());
  if (!condition || condition->getPredicate() != CmpInst::ICMP_SLT ||
      (condition->getParent() != latch && condition->getParent() != header) ||
      !loop->isLoopInvariant(condition->getOperand(1)))
    return 0;
  auto *next = dyn_cast<BinaryOperator>(condition->getOperand(0));
  auto *step = next ? dyn_cast<ConstantInt>(next->getOperand(1)) : nullptr;
  auto *iv = next ? dyn_cast<PHINode>(next->getOperand(0)) : nullptr;
  if (!next || next->getOpcode() != Instruction::Add ||
      next->getParent() != condition->getParent() || !step ||
      !step->equalsInt(pageStep) || !iv || iv->getParent() != header ||
      (!iv->getType()->isIntegerTy(32) &&
       !((h32 || h16) && iv->getType()->isIntegerTy(64))) ||
      iv->getNumIncomingValues() != 2 || iv->getBasicBlockIndex(latch) < 0 ||
      iv->getIncomingValueForBlock(latch) != next ||
      iv->getBasicBlockIndex(preheader) < 0)
    return 0;
  CallInst *protect = nullptr, *commit = nullptr, *wait = nullptr;
  unsigned commits = 0, waits = 0;
  for (Instruction &instruction : *header) {
    if (position[&instruction] < position[copies.front()] &&
        early_shared_float_scales::barrier(&instruction, threads))
      protect = cast<CallInst>(&instruction);
    if (auto *call = dyn_cast<CallInst>(&instruction))
      if (auto *callee = call->getCalledFunction()) {
        if (callee->getName() == "llvm.nvvm.cp.async.commit.group") {
          commit = call;
          ++commits;
        }
        if (callee->getName() == "llvm.nvvm.cp.async.wait.group") {
          wait = call;
          ++waits;
        }
      }
  }
  auto *groups = wait && wait->arg_size() == 1
                     ? dyn_cast<ConstantInt>(wait->getArgOperand(0))
                     : nullptr;
  if (!protect || !commit || commits != 1 || waits != 1 || !groups ||
      !groups->isZero() || !commit->arg_empty() ||
      !commit->getType()->isVoidTy() || !wait->getType()->isVoidTy() ||
      !copies.back()->comesBefore(commit) || !commit->comesBefore(wait) ||
      !wait->comesBefore(products.front()))
    return 0;
  for (auto *copy : copies) {
    auto *assembly = cast<InlineAsm>(copy->getCalledOperand());
    if (copy->arg_size() != 3 || !copy->getType()->isVoidTy() ||
        assembly->getConstraintString() != "r,l,r" ||
        !copy->getArgOperand(0)->getType()->isPointerTy() ||
        copy->getArgOperand(0)->getType()->getPointerAddressSpace() != 3 ||
        !copy->getArgOperand(1)->getType()->isPointerTy() ||
        copy->getArgOperand(1)->getType()->getPointerAddressSpace() != 1 ||
        !copy->getArgOperand(2)->getType()->isIntegerTy(32))
      return 0;
  }
  // All loop effects are known shared writes or reads. No input is modified.
  for (auto *block : loop->blocks())
    for (Instruction &instruction : *block)
      if (auto *store = dyn_cast<StoreInst>(&instruction))
        if (store->getPointerAddressSpace() == 1)
          return 0;
  const auto &layout = function.getParent()->getDataLayout();
  BitVector empty(shared);
  if (!readonly_shared_matrices::proveNoQueryWrites(*loop, empty, layout,
                                                    threads))
    return 0;
  auto footprint = [&](Value *pointer,
                       unsigned width) -> std::optional<BitVector> {
    BitVector bytes(shared);
    for (unsigned tid = 0; tid < threads; ++tid) {
      DenseMap<Value *, Constant *> constants;
      DenseMap<Value *, APInt> integers;
      if (!mla_nope_prefetch::preparePointer(pointer, tid, layout, constants,
                                             integers))
        return std::nullopt;
      auto offset = readonly_shared_matrices::sharedOffset(pointer, tid, layout,
                                                           integers);
      if (!offset || *offset < 0 || *offset + width > shared)
        return std::nullopt;
      bytes.set(*offset, *offset + width);
    }
    return bytes;
  };
  BitVector kv(shared), destinations(shared), expected(shared);
  expected.set(0, kvBytes);
  SmallPtrSet<LoadInst *, 32> readers;
  LoadInst *last = nullptr;
  for (auto *product : ArrayRef(products).drop_front(qkProducts))
    for (unsigned index = 12; index < 16; ++index) {
      if (product->arg_size() != 16)
        return 0;
      auto *transpose = dyn_cast<CallInst>(product->getArgOperand(index));
      auto *callee = transpose ? transpose->getCalledFunction() : nullptr;
      if (!callee ||
          callee->getName() != "__flagtree_ppu_intrinsic__mat.trans.b8" ||
          transpose->arg_size() != 1)
        return 0;
      auto *load = dyn_cast<LoadInst>(transpose->getArgOperand(0));
      if (!load || load->getParent() != latch ||
          load->getPointerAddressSpace() != 3 ||
          !load->getType()->isIntegerTy(32) || load->isVolatile() ||
          load->isAtomic())
        return 0;
      if (readers.insert(load).second) {
        auto bytes = footprint(load->getPointerOperand(), 4);
        if (!bytes)
          return 0;
        kv |= *bytes;
      }
      if (!last || position[last] < position[load])
        last = load;
    }
  for (auto *copy : copies) {
    auto bytes = footprint(copy->getArgOperand(0), 16);
    if (!bytes)
      return 0;
    destinations |= *bytes;
  }
  if (readers.size() != (small ? 32 : 64) || kv != expected ||
      destinations != expected || !last)
    return 0;
  unsigned remaining = 0;
  for (Instruction *instruction = last->getNextNode(); instruction;
       instruction = instruction->getNextNode()) {
    remaining += llvm::is_contained(products, instruction);
    Value *pointer = nullptr;
    unsigned width = 0;
    if (auto *load = dyn_cast<LoadInst>(instruction)) {
      if (load->getPointerAddressSpace() != 3)
        continue;
      if (load->isVolatile() || load->isAtomic())
        return 0;
      pointer = load->getPointerOperand();
      width = layout.getTypeStoreSize(load->getType()).getFixedValue();
    } else if (auto *store = dyn_cast<StoreInst>(instruction)) {
      if (store->getPointerAddressSpace() != 3)
        return 0;
      pointer = store->getPointerOperand();
      width = layout.getTypeStoreSize(store->getValueOperand()->getType())
                  .getFixedValue();
    } else if (auto *call = dyn_cast<CallInst>(instruction)) {
      if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand())) {
        if (assembly->getAsmString() == mla_nope_prefetch::MatrixAsm) {
          auto *address = call->arg_size() == 1
                              ? dyn_cast<PtrToIntInst>(call->getArgOperand(0))
                              : nullptr;
          if (!address || address->getPointerAddressSpace() != 3)
            return 0;
          pointer = address->getPointerOperand();
          width = 16;
        } else if (assembly->getAsmString().contains("shared") ||
                   !readonly_shared_matrices::readOnlyAsm(
                       assembly->getAsmString()))
          return 0;
      } else if (!call->doesNotAccessMemory())
        return 0;
    } else if (instruction->mayReadOrWriteMemory())
      return 0;
    if (pointer) {
      auto bytes = footprint(pointer, width);
      if (!bytes)
        return 0;
      *bytes &= kv;
      if (bytes->any())
        return 0;
    }
  }
  // The batch scheduler may make the final V read earlier, leaving more
  // independent PV work after it.  That is safe (and gives the asynchronous
  // copy more cover); only the original minimum product tail is required.
  if (remaining < remainingProducts ||
      (condition->getParent() == latch &&
       (position[next] <= position[last] ||
        position[condition] <= position[next])))
    return 0;
  // Only the IV and a bounded pure integer/page-read DAG may be remapped.
  SmallPtrSet<Value *, 32> checked;
  std::function<bool(Value *)> prove = [&](Value *value) {
    if (value == iv)
      return true;
    auto *instruction = dyn_cast<Instruction>(value);
    if (!instruction)
      return isa<Constant, Argument>(value);
    if (!loop->contains(instruction))
      return dominance.dominates(instruction, preheader->getTerminator());
    if (instruction->getParent() != header || isa<PHINode>(instruction))
      return false;
    if (!checked.insert(value).second)
      return true;
    bool accepted = false;
    if (auto *call = dyn_cast<CallInst>(instruction)) {
      accepted = pageRead(call);
      if (auto *intrinsic = dyn_cast<IntrinsicInst>(call))
        accepted = intrinsic->getIntrinsicID() == Intrinsic::vector_reduce_or;
    } else if (isa<GetElementPtrInst>(instruction))
      accepted = true;
    else if (auto *cast = dyn_cast<CastInst>(instruction))
      accepted = mla_nope_prefetch::integerType(cast->getType()) &&
                 mla_nope_prefetch::integerType(cast->getSrcTy());
    else
      accepted = mla_nope_prefetch::integerType(instruction->getType()) &&
                 (instruction->isBinaryOp() ||
                  isa<ICmpInst, SelectInst, InsertElementInst,
                      ExtractElementInst, ShuffleVectorInst>(instruction));
    if (!accepted)
      return false;
    if (auto *call = dyn_cast<CallInst>(instruction)) {
      for (Value *argument : call->args())
        if (!prove(argument))
          return false;
    } else {
      for (Value *operand : instruction->operands())
        if (!prove(operand))
          return false;
    }
    return true;
  };
  for (auto *copy : copies)
    for (Value *operand : copy->args())
      if (!prove(operand))
        return 0;
  SmallVector<CallInst *> pages;
  SmallVector<SmallVector<CallInst *>> aliases;
  if (reusePages) {
    for (Instruction &instruction : *header)
      if (checked.contains(&instruction))
        if (auto *call = dyn_cast<CallInst>(&instruction))
          if (pageRead(call))
            pages.push_back(call);
    if (pages.size() != pageStep)
      return 0;
    for (unsigned i = 0; i < pages.size(); ++i) {
      SmallVector<CallInst *> matching;
      for (Instruction &instruction : *header)
        if (auto *call = dyn_cast<CallInst>(&instruction))
          if (pageRead(call) &&
              call->getCalledOperand() == pages[i]->getCalledOperand() &&
              call->getArgOperand(0) == pages[i]->getArgOperand(0) &&
              call->getArgOperand(1) == pages[i]->getArgOperand(1))
            matching.push_back(call);
      if (matching.empty())
        return 0;
      for (unsigned j = 0; j < i; ++j)
        if (pages[i]->getArgOperand(0) == pages[j]->getArgOperand(0) &&
            pages[i]->getArgOperand(1) == pages[j]->getArgOperand(1))
          return 0;
      aliases.push_back(std::move(matching));
    }
  }
  // No modification precedes the complete range/effect/SSA proofs above.
  Instruction *anchor = last->getNextNode();
  // A mask using the loop continuation may already compute both values in
  // the header. They dominate the prefetch there; do not sink those
  // definitions.
  if (condition->getParent() == latch) {
    next->moveBefore(anchor->getIterator());
    condition->moveBefore(anchor->getIterator());
  }
  auto *continuation =
      latch->splitBasicBlock(anchor->getIterator(), "mla.pv.cont");
  auto *prefetch = BasicBlock::Create(function.getContext(), "mla.kv.prefetch",
                                      &function, continuation);
  BranchInst::Create(continuation, prefetch);
  auto *oldBranch = latch->getTerminator();
  BranchInst::Create(prefetch, continuation, condition,
                     oldBranch->getIterator());
  oldBranch->eraseFromParent();
  auto emit = [&](BasicBlock *block, Value *iteration) {
    auto *destination = block->getTerminator();
    DenseMap<Value *, Value *> mapped;
    mapped[iv] = iteration;
    std::function<Value *(Value *)> clone = [&](Value *value) -> Value * {
      if (auto found = mapped.find(value); found != mapped.end())
        return found->second;
      auto *instruction = dyn_cast<Instruction>(value);
      if (!instruction || instruction->getParent() != header)
        return value;
      auto *copy = instruction->clone();
      for (unsigned index = 0; index < copy->getNumOperands(); ++index)
        copy->setOperand(index, clone(instruction->getOperand(index)));
      copy->setName("mla.kv.source");
      copy->insertBefore(destination->getIterator());
      mapped[value] = copy;
      return copy;
    };
    protect->clone()->insertBefore(destination->getIterator());
    for (auto *original : copies) {
      auto *copy = cast<CallInst>(original->clone());
      for (unsigned index = 0; index < copy->arg_size(); ++index)
        copy->setArgOperand(index, clone(original->getArgOperand(index)));
      copy->insertBefore(destination->getIterator());
    }
    commit->clone()->insertBefore(destination->getIterator());
    SmallVector<Value *> results;
    for (auto *page : pages)
      results.push_back(mapped.lookup(page));
    return results;
  };
  auto initialPages = emit(preheader, iv->getIncomingValueForBlock(preheader));
  auto followingPages = emit(prefetch, next);
  for (unsigned i = 0; i < pages.size(); ++i) {
    auto *joined = PHINode::Create(Type::getInt32Ty(function.getContext()), 2,
                                   "mla.kv.page.next",
                                   continuation->getFirstInsertionPt());
    joined->addIncoming(followingPages[i], prefetch);
    joined->addIncoming(PoisonValue::get(joined->getType()), latch);
    auto *current = PHINode::Create(joined->getType(), 2, "mla.kv.page.current",
                                    header->getFirstInsertionPt());
    current->addIncoming(initialPages[i], preheader);
    current->addIncoming(joined, continuation);
    for (auto *alias : aliases[i])
      alias->replaceAllUsesWith(current);
  }
  for (auto *copy : copies)
    copy->eraseFromParent();
  commit->eraseFromParent();
  protect->eraseFromParent();
  function.addFnAttr("ppu.mla.kv.pipeline", "1");
  return reusePages ? pageStep + 1 : 1;
}
} // namespace mla_kv_pipeline
inline unsigned pipelineMlaKv(llvm::Module &module, unsigned threads,
                              unsigned shared, bool reusePages = false) {
  if (!((threads == 512 && (shared == 168192 || shared == 123392)) ||
        (threads == 256 && (shared == 94464 || shared == 83968))))
    return 0;
  unsigned changed = 0;
  for (auto &function : module)
    if (!function.isDeclaration())
      changed += mla_kv_pipeline::run(function, shared, reusePages,
                                      shared == 123392, threads);
  if (changed && shared == 168192)
    scheduleMlaPvTail(module, threads, shared);
  return changed;
}
} // namespace triton::ppu
#endif
