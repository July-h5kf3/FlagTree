// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_GATHER_MLA_ROW_REDUCTIONS_H
#define TRITON_PPU_GATHER_MLA_ROW_REDUCTIONS_H
#include "MlaRowReduction.h"

namespace triton::ppu {
namespace mla_row_gather {

inline llvm::Value *unpack(llvm::Value *value) {
  if (auto *cast = llvm::dyn_cast<llvm::BitCastInst>(value))
    return cast->getOperand(0);
  return value;
}

inline std::optional<std::pair<llvm::Value *, llvm::Value *>>
operands(llvm::Value *value, bool sum) {
  using namespace llvm;
  if (sum) {
    auto *add = dyn_cast<BinaryOperator>(value);
    if (add && add->getOpcode() == Instruction::FAdd &&
        add->getType()->isFloatTy())
      return std::pair{add->getOperand(0), add->getOperand(1)};
  } else if (auto *call = dyn_cast<CallInst>(value)) {
    auto *callee = call->getCalledFunction();
    if (callee && callee->getName() == "llvm.maxnum.f32" &&
        call->arg_size() == 2 && !call->hasOperandBundles())
      return std::pair{call->getArgOperand(0), call->getArgOperand(1)};
  }
  return std::nullopt;
}

inline bool matchesShuffle(llvm::Value *value, llvm::Value *source,
                           unsigned delta) {
  using namespace llvm;
  auto *call = dyn_cast<CallInst>(unpack(value));
  auto *callee = call ? call->getCalledFunction() : nullptr;
  auto constant = [&](unsigned index, uint64_t expected) {
    auto *arg = dyn_cast<ConstantInt>(call->getArgOperand(index));
    return arg && arg->getZExtValue() == expected;
  };
  return callee && callee->getName() == "llvm.nvvm.shfl.sync.bfly.i32" &&
         call->arg_size() == 4 && constant(0, 0xffffffff) &&
         constant(2, delta) && constant(3, 31) &&
         unpack(call->getArgOperand(1)) == source;
}

struct Tree {
  llvm::Instruction *first, *last;
  bool sum;
};

inline std::optional<Tree> tree(mla_row_groups::Access &read,
                                mla_row_groups::Access &write) {
  using namespace llvm;
  auto *root = dyn_cast<Instruction>(
      unpack(cast<CallInst>(write.instruction)->getArgOperand(1)));
  if (!root)
    return std::nullopt;
  bool sum = root->getOpcode() == Instruction::FAdd;
  auto last = operands(root, sum);
  auto first = last ? operands(last->first, sum) : std::nullopt;
  if (!last || !first || unpack(first->first) != read.instruction ||
      !matchesShuffle(first->second, read.instruction, 2) ||
      !matchesShuffle(last->second, last->first, 1))
    return std::nullopt;
  return Tree{cast<Instruction>(last->first), root, sum};
}

inline bool killed(mla_row_groups::Region &region,
                   const llvm::DataLayout &layout) {
  using namespace llvm;
  BitVector bytes(region.fields * 512);
  for (Instruction *instruction = region.end->getNextNode(); instruction;
       instruction = instruction->getNextNode()) {
    if (readonly_shared_matrices::ctaBarrier(instruction, 256))
      return bytes.all();
    if (auto *store = dyn_cast<StoreInst>(instruction)) {
      unsigned width =
          layout.getTypeStoreSize(store->getValueOperand()->getType())
              .getFixedValue();
      if (store->getPointerAddressSpace() != 3 || store->isVolatile() ||
          store->isAtomic() || width > 16)
        return false;
      for (unsigned tid = 0; tid < 256; ++tid) {
        auto address =
            mla_row_groups::offset(store->getPointerOperand(), tid, layout);
        if (!address || *address < 0 || *address + width > 94464)
          return false;
        for (unsigned byte = 0; byte < width; ++byte)
          if (*address + byte >= 81920 &&
              *address + byte < 81920 + bytes.size())
            bytes.set(*address + byte - 81920);
      }
    } else if (auto *call = dyn_cast<CallInst>(instruction);
               call && call->getCalledFunction() &&
               (call->getCalledFunction()->getName() ==
                    "__flagtree_ppu_intrinsic__mat.trans.b8" ||
                call->getCalledFunction()->getName() ==
                    "llvm.nvvm.shfl.sync.idx.i32")) {
      if (call->hasOperandBundles())
        return false;
    } else if (!mla_row_groups::pure(*instruction)) {
      return false;
    }
  }
  return false;
}

inline llvm::Value *combine(llvm::IRBuilder<llvm::NoFolder> &builder,
                            llvm::Instruction *original, llvm::Value *a,
                            llvm::Value *b) {
  llvm::Instruction *value;
  if (original->getOpcode() == llvm::Instruction::FAdd)
    value = llvm::cast<llvm::Instruction>(builder.CreateFAdd(a, b));
  else
    value = builder.CreateCall(
        llvm::cast<llvm::CallInst>(original)->getCalledFunction(), {a, b});
  value->copyIRFlags(original);
  return value;
}

// Admit two independent physical groups after gather removes
// the cross-group intermediate exchange. Initial and final CTA guards remain.
inline llvm::CallInst *
groupThread(llvm::Function &function,
            llvm::ArrayRef<mla_row_groups::Region> regions,
            const llvm::DataLayout &layout) {
  using namespace llvm;
  CallInst *tid = nullptr;
  for (auto &block : function)
    for (auto &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction)) {
        if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand())) {
          if (code->getAsmString().contains("bar.") ||
              code->getAsmString().contains("barrier"))
            return nullptr;
        }
        auto *callee = call->getCalledFunction();
        if (!callee)
          continue;
        if (callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
          if (tid || call->arg_size() || !call->getType()->isIntegerTy(32))
            return nullptr;
          tid = call;
        }
        if ((callee->getName().starts_with("llvm.nvvm.bar") ||
             callee->getName().contains("barrier")) &&
            !readonly_shared_matrices::ctaBarrier(call, 256))
          return nullptr;
      }
  if (!tid || tid->getParent() != &function.getEntryBlock())
    return nullptr;
  DominatorTree dominance(function);
  for (auto &region : regions) {
    if (!dominance.dominates(tid, region.begin) ||
        !dominance.dominates(tid, region.end))
      return nullptr;
    std::array<std::vector<unsigned>, 2> written = {
        std::vector<unsigned>(region.fields * 512),
        std::vector<unsigned>(region.fields * 512)};
    for (auto &memory : region.partial)
      for (unsigned thread = 0; thread < 256; ++thread) {
        auto active =
            mla_row_groups::predicate(memory.predicate, thread, layout);
        auto address = mla_row_groups::offset(memory.pointer, thread, layout);
        if (!active || !address)
          return nullptr;
        if (!*active)
          continue;
        auto byte = *address - 81920;
        if (byte < 0 || byte + 4 > region.fields * 512 || byte % 4 ||
            unsigned(byte % 512) / 256 != thread / 128)
          return nullptr;
        for (unsigned i = 0; i < 4; ++i)
          if (++written[thread / 128][byte + i] != 1)
            return nullptr;
      }
    for (unsigned group = 0; group < 2; ++group)
      for (unsigned field = 0; field < region.fields; ++field)
        for (unsigned byte = 0; byte < 256; ++byte)
          if (written[group][field * 512 + group * 256 + byte] != 1)
            return nullptr;
    for (auto &memory : region.final)
      for (unsigned thread = 0; thread < 256; ++thread) {
        auto address = mla_row_groups::offset(memory.pointer, thread, layout);
        if (!address)
          return nullptr;
        auto byte = *address - 81920;
        if (byte < 0 || byte + 16 > region.fields * 512 || byte % 16 ||
            unsigned(byte % 512) / 256 != thread / 128 ||
            unsigned(byte % 256) + 16 > 256)
          return nullptr;
        for (unsigned i = 0; i < 16; ++i)
          if (written[thread / 128][byte + i] != 1)
            return nullptr;
      }
  }
  return tid;
}

inline bool groupBarrier(llvm::Instruction *instruction, unsigned first,
                         unsigned second, const llvm::DataLayout &layout) {
  using namespace llvm;
  if (first >= 256 || second >= 256 || first / 128 != second / 128 ||
      !instruction->getMetadata("ppu.mla.row.group.barrier") ||
      instruction->getFunction()
              ->getFnAttribute("ppu.mla.row.groups128")
              .getValueAsString() != "1")
    return false;
  auto *call = dyn_cast<CallInst>(instruction);
  auto *callee = call ? call->getCalledFunction() : nullptr;
  if (!callee || callee->getName() != "__ppukernel_barrier_sync_cnt_void" ||
      !callee->isDeclaration() || call->arg_size() != 3 ||
      !call->getType()->isVoidTy() || call->hasOperandBundles())
    return false;
  auto *count = dyn_cast<ConstantInt>(call->getArgOperand(1));
  auto *scope = dyn_cast<ConstantInt>(call->getArgOperand(2));
  if (!count || !count->getType()->isIntegerTy(32) || !count->equalsInt(128) ||
      !scope || !scope->getType()->isIntegerTy(32) || !scope->equalsInt(1))
    return false;
  for (unsigned tid : {first, second}) {
    DenseMap<Value *, Constant *> cache;
    auto *id = dyn_cast_or_null<ConstantInt>(mla_nope_prefetch::constantValue(
        call->getArgOperand(0), tid, layout, cache));
    if (!id || !id->getType()->isIntegerTy(32) || !id->equalsInt(1 + tid / 128))
      return false;
  }
  return true;
}

inline unsigned run(llvm::Function &function, bool subgroups = false) {
  using namespace llvm;
  if (function.getCallingConv() != CallingConv::PTX_Kernel ||
      function.hasFnAttribute("ppu.mla.row.gather"))
    return 0;
  for (auto [name, value] : {std::pair{"nvvm.reqntid", "256"},
                             {"ppu.mla.kv.pipeline", "1"},
                             {"ppu.mla.pv.batch", "4"},
                             {"ppu.mla.rope.prefetch", "1"}})
    if (!function.hasFnAttribute(name) ||
        function.getFnAttribute(name).getValueAsString() != value)
      return 0;
  const auto &layout = function.getParent()->getDataLayout();
  DominatorTree dominance(function);
  LoopInfo loops(dominance);
  SmallVector<mla_row_groups::Region> regions;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction)) {
        auto *callee = call->getCalledFunction();
        auto *loop = loops.getLoopFor(&block);
        if (!callee ||
            callee->getName() != "__ppukernel_barrier_sync_cnt_void" || !loop ||
            loop->getParentLoop() ||
            !readonly_shared_matrices::ctaBarrier(call, 256))
          continue;
        mla_row_groups::Region region;
        region.begin = call;
        if (mla_row_groups::collect(region, loop) &&
            mla_row_groups::prove(region, layout))
          regions.push_back(std::move(region));
      }
  if (regions.size() != 2 || regions[0].fields != 1 || regions[1].fields != 2 ||
      regions[0].end != regions[1].begin || !killed(regions[1], layout))
    return 0;
  SmallVector<Tree> trees;
  for (auto &region : regions) {
    for (unsigned field = 0; field < region.fields; ++field) {
      mla_row_groups::Access *read = nullptr, *write = nullptr;
      for (auto &memory : region.middle) {
        auto base = mla_row_groups::offset(memory.pointer, 0, layout);
        if (!base || *base != 81920 + field * 512)
          continue;
        (memory.write ? write : read) = &memory;
      }
      auto matched = read && write ? tree(*read, *write) : std::nullopt;
      if (!matched)
        return 0;
      trees.push_back(*matched);
    }
    for (auto &memory : region.final)
      for (unsigned tid = 0; tid < 256; ++tid) {
        auto address = mla_row_groups::offset(memory.pointer, tid, layout);
        if (!address || *address % 16 || (*address - 81920) % 512 + 16 > 512)
          return 0;
      }
  }
  auto *tid = subgroups ? groupThread(function, regions, layout) : nullptr;
  auto *sync = regions[0].begin->getCalledFunction();
  bool scoped =
      tid && sync && regions[0].begin->arg_size() == 3 &&
      isa<ConstantInt>(regions[0].begin->getArgOperand(2)) &&
      cast<ConstantInt>(regions[0].begin->getArgOperand(2))->equalsInt(1);
  SmallVector<CallInst *> scopeBarriers = {
      regions[0].barriers[0], regions[0].end, regions[1].barriers[0]};
  unsigned treeBase = 0;
  for (auto &region : regions) {
    SmallVector<Instruction *> middle;
    for (Instruction *instruction = region.barriers[0]->getNextNode();
         instruction != region.barriers[1];
         instruction = instruction->getNextNode())
      middle.push_back(instruction);
    for (auto &memory : region.final) {
      auto address = *mla_row_groups::offset(memory.pointer, 0, layout);
      auto &matched = trees[treeBase + (address - 81920) / 512];
      IRBuilder<NoFolder> builder(memory.instruction);
      auto *vector = builder.CreateAlignedLoad(
          FixedVectorType::get(builder.getFloatTy(), 4), memory.pointer,
          Align(16));
      Value *lanes[4];
      for (unsigned lane = 0; lane < 4; ++lane)
        lanes[lane] =
            builder.CreateExtractElement(vector, builder.getInt64(lane));
      Value *a = combine(builder, matched.first, lanes[0], lanes[2]);
      Value *b = combine(builder, matched.first, lanes[1], lanes[3]);
      Value *result = combine(builder, matched.last, a, b);
      memory.instruction->replaceAllUsesWith(result);
      memory.instruction->eraseFromParent();
    }
    for (auto &memory : region.middle)
      if (memory.write) {
        for (auto &instruction : middle)
          if (instruction == memory.instruction)
            instruction = nullptr;
        memory.instruction->eraseFromParent();
      }
    region.barriers[1]->eraseFromParent();
    SmallPtrSet<Instruction *, 8> reads;
    for (auto &memory : region.middle)
      if (!memory.write)
        reads.insert(memory.instruction);
    bool changed;
    do {
      changed = false;
      for (auto &instruction : middle)
        if (instruction && instruction->getParent() &&
            instruction->use_empty() &&
            (reads.contains(instruction) ||
             !instruction->mayHaveSideEffects() ||
             (isa<CallInst>(instruction) &&
              cast<CallInst>(instruction)->getCalledFunction() &&
              (cast<CallInst>(instruction)->getCalledFunction()->getName() ==
                   "llvm.nvvm.shfl.sync.bfly.i32" ||
               cast<CallInst>(instruction)->getCalledFunction()->getName() ==
                   "llvm.maxnum.f32")))) {
          instruction->eraseFromParent();
          instruction = nullptr;
          changed = true;
        }
    } while (changed);
    treeBase += region.fields;
  }
  if (scoped) {
    IRBuilder<> builder(tid->getNextNode());
    builder.SetCurrentDebugLocation(DebugLoc());
    auto *group =
        builder.CreateAdd(builder.CreateLShr(tid, builder.getInt32(7)),
                          builder.getInt32(1), "mla.row.group");
    for (auto *barrier : scopeBarriers) {
      builder.SetInsertPoint(barrier);
      builder.SetCurrentDebugLocation(barrier->getDebugLoc());
      auto *scopedBarrier = builder.CreateCall(
          sync, {group, builder.getInt32(128), builder.getInt32(1)});
      scopedBarrier->setTailCallKind(CallInst::TCK_Tail);
      scopedBarrier->setMetadata("ppu.mla.row.group.barrier",
                                 MDNode::get(function.getContext(), {}));
      barrier->eraseFromParent();
    }
    function.addFnAttr("ppu.mla.row.groups128", "1");
  }
  function.addFnAttr("ppu.mla.row.gather", "1");
  return scoped ? 9 : 6;
}
} // namespace mla_row_gather

inline unsigned gatherMlaRowReductions(llvm::Module &module, unsigned threads,
                                       unsigned shared,
                                       bool subgroups = false) {
  if (threads != 256 || shared != 94464)
    return 0;
  unsigned changed = 0;
  for (llvm::Function &function : module)
    if (!function.isDeclaration())
      changed += mla_row_gather::run(function, subgroups);
  return changed;
}
} // namespace triton::ppu
#endif
