// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_UNIQUE_MLA_FINAL_ROWS_H
#define TRITON_PPU_UNIQUE_MLA_FINAL_ROWS_H
#include "UniqueMlaScaleWriters.h"
#include "llvm/Analysis/PostDominators.h"
namespace triton::ppu::mla_final_rows {
inline llvm::Constant *
foldIndex(llvm::Value *value, unsigned tid, const llvm::DataLayout &layout,
          llvm::DenseMap<llvm::Value *, llvm::Constant *> &cache,
          llvm::SmallPtrSetImpl<llvm::Value *> &active, unsigned depth = 0) {
  using namespace llvm;
  if (depth > 64 || !mla_nope_prefetch::integerType(value->getType()))
    return nullptr;
  if (auto *constant =
          mla_nope_prefetch::constantValue(value, tid, layout, cache))
    return constant;
  auto *instruction = dyn_cast<Instruction>(value);
  if (!instruction || !active.insert(value).second)
    return nullptr;
  Constant *result = nullptr;
  if (auto *phi = dyn_cast<PHINode>(instruction)) {
    for (Value *incoming : phi->incoming_values()) {
      auto *constant = dyn_cast_or_null<ConstantInt>(
          foldIndex(incoming, tid, layout, cache, active, depth + 1));
      if (!constant || (result && result != constant)) {
        result = nullptr;
        break;
      }
      result = constant;
    }
  } else if (instruction->isBinaryOp() ||
             isa<CastInst, ICmpInst, SelectInst, InsertElementInst,
                 ExtractElementInst, ShuffleVectorInst>(instruction)) {
    for (Value *operand : instruction->operands())
      foldIndex(operand, tid, layout, cache, active, depth + 1);
    result = mla_nope_prefetch::constantValue(value, tid, layout, cache);
  }
  active.erase(value);
  if (result)
    cache.try_emplace(value, result);
  return result;
}

inline std::optional<int64_t> offset(llvm::Value *pointer, unsigned tid,
                                     const llvm::DataLayout &layout) {
  using namespace llvm;
  DenseMap<Value *, Constant *> constants;
  DenseMap<Value *, APInt> integers;
  SmallPtrSet<Value *, 32> active;
  std::function<bool(Value *)> prepare = [&](Value *value) {
    if (isa<GlobalVariable>(value))
      return true;
    auto *gep = dyn_cast<GEPOperator>(value);
    if (!gep || !prepare(gep->getPointerOperand()))
      return false;
    for (Value *index : gep->indices())
      if (!foldIndex(index, tid, layout, constants, active))
        return false;
    return true;
  };
  if (!prepare(pointer) || !mla_nope_prefetch::preparePointer(
                               pointer, tid, layout, constants, integers))
    return std::nullopt;
  return readonly_shared_matrices::sharedOffset(pointer, tid, layout, integers);
}
inline bool pure(llvm::Instruction &instruction) {
  using namespace llvm;
  if (mla_row_groups::pure(instruction))
    return true;
  auto *call = dyn_cast<CallInst>(&instruction);
  auto *callee = call ? call->getCalledFunction() : nullptr;
  return callee && callee->isDeclaration() && !call->hasOperandBundles() &&
         call->arg_size() == 3 && call->getType()->isFloatTy() &&
         (callee->getName() == "llvm.nvvm.fma.rn.f" ||
          callee->getName() == "llvm.nvvm.fma.rn.ftz.f");
}

struct ExitEqual {
  llvm::BasicBlock *entry;
  llvm::DominatorTree &dominance;
  llvm::LoopInfo &loops;
  llvm::DenseMap<llvm::Value *, bool> known;
  llvm::SmallPtrSet<llvm::Value *, 32> active;
  ExitEqual(llvm::BasicBlock *entry, llvm::DominatorTree &dominance,
            llvm::LoopInfo &loops)
      : entry(entry), dominance(dominance), loops(loops) {}
  bool value(llvm::Value *input, unsigned depth = 0) {
    using namespace llvm;
    if (depth > 256)
      return false;
    if (isa<Constant, Argument>(input))
      return true;
    if (auto it = known.find(input); it != known.end())
      return it->second;
    auto *instruction = dyn_cast<Instruction>(input);
    if (!instruction || !active.insert(input).second)
      return false;
    bool equal = false;
    if (auto *phi = dyn_cast<PHINode>(instruction)) {
      // The caller proves the exit-region branch selectors separately.
      equal = !loops.getLoopFor(phi->getParent()) &&
              dominance.dominates(entry, phi->getParent());
      for (Value *incoming : phi->incoming_values())
        equal &= value(incoming, depth + 1);
    } else if (pure(*instruction) && !isa<LoadInst>(instruction)) {
      equal = true;
      if (auto *call = dyn_cast<CallInst>(instruction)) {
        for (Value *argument : call->args())
          equal &= value(argument, depth + 1);
      } else {
        for (Value *operand : instruction->operands())
          equal &= value(operand, depth + 1);
      }
    }
    active.erase(input);
    known[input] = equal;
    return equal;
  }
};

inline bool pureControl(llvm::BranchInst *branch,
                        llvm::PostDominatorTree &post) {
  using namespace llvm;
  auto *node = post.getNode(branch->getParent());
  auto *join = node && node->getIDom() ? node->getIDom()->getBlock() : nullptr;
  if (!join)
    return false;
  SmallVector<BasicBlock *> pending;
  for (BasicBlock *block : successors(branch->getParent()))
    pending.push_back(block);
  SmallPtrSet<BasicBlock *, 32> seen;
  while (!pending.empty()) {
    auto *block = pending.pop_back_val();
    if (block == join || !seen.insert(block).second)
      continue;
    if (seen.size() > 64)
      return false;
    for (Instruction &instruction : *block)
      if (!isa<BranchInst>(instruction) && !pure(instruction))
        return false;
    auto *jump = dyn_cast<BranchInst>(block->getTerminator());
    if (!jump)
      return false;
    for (BasicBlock *successor : successors(block))
      pending.push_back(successor);
  }
  return true;
}

struct Match {
  llvm::PHINode *state, *exit;
  llvm::Instruction *update;
  llvm::CallInst *thread;
  llvm::StoreInst *final[2];
};
inline std::optional<Match> match(llvm::Function &function, bool uniformLoop) {
  using namespace llvm;
  if (!uniformLoop || function.getCallingConv() != CallingConv::PTX_Kernel ||
      function.hasFnAttribute("ppu.mla.final.rows.unique"))
    return std::nullopt;
  for (auto [name, expected] : {std::pair{"nvvm.reqntid", "256"},
                                {"ppu.mla.kv.pipeline", "1"},
                                {"ppu.mla.pv.batch", "4"},
                                {"ppu.mla.rope.prefetch", "1"}})
    if (!function.hasFnAttribute(name) ||
        function.getFnAttribute(name).getValueAsString() != expected)
      return std::nullopt;
  const auto &layout = function.getParent()->getDataLayout();
  DominatorTree dominance(function);
  PostDominatorTree post(function);
  LoopInfo loops(dominance);
  Match result{};
  Loop *matchedLoop = nullptr;
  for (Loop *loop : loops) {
    if (!loop->getLoopPreheader() || !loop->getLoopLatch())
      continue;
    for (PHINode &phi : loop->getHeader()->phis()) {
      auto *initial = dyn_cast<Constant>(
          phi.getIncomingValueForBlock(loop->getLoopPreheader()));
      if (early_shared_float_scales::vector2(phi.getType()) &&
          phi.getNumIncomingValues() == 2 && initial &&
          initial->isNullValue()) {
        if (result.state)
          return std::nullopt;
        result.state = &phi;
        matchedLoop = loop;
      }
    }
  }
  if (!result.state)
    return std::nullopt;
  auto *update = dyn_cast<BinaryOperator>(
      result.state->getIncomingValueForBlock(matchedLoop->getLoopLatch()));
  auto *multiply =
      update ? dyn_cast<BinaryOperator>(update->getOperand(0)) : nullptr;
  if (!update || update->getOpcode() != Instruction::FAdd || !multiply ||
      multiply->getOpcode() != Instruction::FMul ||
      multiply->getOperand(0) != result.state || !result.state->hasOneUse() ||
      *result.state->user_begin() != multiply)
    return std::nullopt;
  result.update = update;
  for (User *user : update->users()) {
    if (user == result.state)
      continue;
    auto *phi = dyn_cast<PHINode>(user);
    if (!phi || result.exit || loops.getLoopFor(phi->getParent()) ||
        phi->getNumIncomingValues() != 2)
      return std::nullopt;
    result.exit = phi;
    for (unsigned index = 0; index < 2; ++index)
      if (phi->getIncomingValue(index) == update) {
        if (phi->getIncomingBlock(index)->getSinglePredecessor() !=
            matchedLoop->getLoopLatch())
          return std::nullopt;
      } else {
        auto *initial = dyn_cast<Constant>(phi->getIncomingValue(index));
        if (!initial || !initial->isNullValue())
          return std::nullopt;
      }
  }
  if (!result.exit)
    return std::nullopt;
  SmallVector<Value *> pending{result.state};
  SmallPtrSet<Instruction *, 32> seen;
  SmallVector<StoreInst *> stores;
  unsigned controls = 0;
  while (!pending.empty()) {
    auto *instruction = dyn_cast<Instruction>(pending.pop_back_val());
    if (!instruction || !seen.insert(instruction).second)
      continue;
    if (auto *store = dyn_cast<StoreInst>(instruction)) {
      if (store->getPointerAddressSpace() != 3 || store->isVolatile() ||
          store->isAtomic() || store->getAlign() != Align(8) ||
          !early_shared_float_scales::vector2(
              store->getValueOperand()->getType()) ||
          loops.getLoopFor(store->getParent()) ||
          !dominance.dominates(result.exit->getParent(), store->getParent()))
        return std::nullopt;
      stores.push_back(store);
      continue;
    }
    if (auto *branch = dyn_cast<BranchInst>(instruction)) {
      if (!branch->isConditional() || loops.getLoopFor(branch->getParent()) ||
          !dominance.dominates(result.exit->getParent(), branch->getParent()) ||
          !pureControl(branch, post))
        return std::nullopt;
      ++controls;
      continue;
    }
    if (!pure(*instruction))
      return std::nullopt;
    for (User *user : instruction->users()) {
      auto *next = dyn_cast<Instruction>(user);
      if (!next)
        return std::nullopt;
      pending.push_back(next);
    }
  }
  if (stores.size() != 2 || controls != 2)
    return std::nullopt;
  if (!dominance.dominates(stores[0], stores[1]))
    std::swap(stores[0], stores[1]);
  if (!dominance.dominates(stores[0], stores[1]))
    return std::nullopt;
  result.final[0] = stores[0];
  result.final[1] = stores[1];
  for (StoreInst *store : stores) {
    // Every address class must participate in the original publication.
    auto *protect = store->getPrevNode();
    while (protect && pure(*protect))
      protect = protect->getPrevNode();
    if (!protect || !readonly_shared_matrices::ctaBarrier(protect, 256) ||
        !readonly_shared_matrices::ctaBarrier(store->getNextNode(), 256))
      return std::nullopt;
  }
  for (BasicBlock &block : function)
    for (Instruction &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *callee = call->getCalledFunction();
            callee && callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
          if (result.thread || !call->arg_empty())
            return std::nullopt;
          result.thread = call;
        }
  if (!result.thread || !dominance.dominates(result.thread, stores[0]))
    return std::nullopt;
  SmallVector<PHINode *> seeds;
  for (PHINode &phi : result.exit->getParent()->phis())
    if (early_shared_float_scales::vector2(phi.getType())) {
      if (phi.getNumIncomingValues() != 2)
        return std::nullopt;
      seeds.push_back(&phi);
    }
  if (seeds.size() != 2)
    return std::nullopt;
  for (unsigned tid = 0; tid < 256; ++tid) {
    mla_scale_writers::Equivalent loopProof(tid, tid & ~99u, layout, loops);
    if (!loopProof.value(result.state))
      return std::nullopt;
    ExitEqual proof(result.exit->getParent(), dominance, loops);
    for (auto *phi : seeds) {
      for (Value *incoming : phi->incoming_values())
        if (!loopProof.value(incoming))
          return std::nullopt;
      proof.known[phi] = true;
    }
    for (BasicBlock &block : function)
      if (dominance.dominates(result.exit->getParent(), &block))
        if (auto *branch = dyn_cast<BranchInst>(block.getTerminator());
            branch && branch->isConditional())
          if (!proof.value(branch->getCondition()))
            return std::nullopt;
    for (unsigned phase = 0; phase < 2; ++phase) {
      auto address = offset(stores[phase]->getPointerOperand(), tid, layout);
      unsigned expected = phase == 0 ? 4 * (tid & 28) + ((tid & 128) >> 4)
                                     : 2 * (tid & 28) + ((tid & 128) >> 1);
      if (!address || *address != expected ||
          !proof.value(stores[phase]->getValueOperand()))
        return std::nullopt;
    }
  }
  return result;
}

inline unsigned finalRows(llvm::Function &function, bool uniformLoop) {
  using namespace llvm;
  auto found = match(function, uniformLoop);
  if (!found)
    return 0;
  Match &item = *found;
  IRBuilder<NoFolder> entry(item.final[0]);
  Value *writer = entry.CreateICmpEQ(
      entry.CreateAnd(item.thread, entry.getInt32(99)), entry.getInt32(0));
  auto *assembly = InlineAsm::get(
      FunctionType::get(entry.getVoidTy(),
                        {item.final[0]->getPointerOperand()->getType(),
                         entry.getInt32Ty(), entry.getInt32Ty(),
                         entry.getInt1Ty()},
                        false),
      "@$3 ppu.st.shared.v2.b32 [ $0 + 0 ], { $1, $2 };", "r,r,r,b", true);
  for (StoreInst *store : item.final) {
    IRBuilder<NoFolder> builder(store);
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
  function.addFnAttr("ppu.mla.final.rows.unique", "1");
  return 2;
}

} // namespace triton::ppu::mla_final_rows
namespace triton::ppu {
inline unsigned uniqueMlaFinalRows(llvm::Module &module, unsigned threads,
                                   unsigned shared, bool uniformLoop) {
  if (threads != 256 || shared != 94464 || !uniformLoop)
    return 0;
  unsigned changed = 0;
  for (llvm::Function &function : module)
    if (!function.isDeclaration())
      changed += mla_final_rows::finalRows(function, uniformLoop);
  return changed;
}
} // namespace triton::ppu
#endif
