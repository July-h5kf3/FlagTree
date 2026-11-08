// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_REPACK_MLA_LOW_PROBABILITY_H
#define TRITON_PPU_REPACK_MLA_LOW_PROBABILITY_H

#include "PrefetchMlaNope.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/KnownBits.h"

namespace triton::ppu {
namespace mla_low_probability {
using namespace llvm;

constexpr unsigned Threads = 256;
constexpr unsigned Shared = 106496;
constexpr unsigned Begin = 81920;
constexpr unsigned Bytes = 4096;
constexpr unsigned End = Begin + Bytes;
constexpr StringLiteral CopyAsm =
    "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;";

inline StringRef assembly(const CallBase &call) {
  auto *code = dyn_cast<InlineAsm>(call.getCalledOperand());
  return code && !code->canThrow() && !call.hasOperandBundles() &&
                 !code->getConstraintString().contains("memory")
             ? StringRef(code->getAsmString())
             : StringRef();
}

inline Value *matrixPointer(CallBase &call) {
  if (assembly(call) != mla_nope_prefetch::MatrixAsm || call.arg_size() != 1)
    return nullptr;
  auto *address = dyn_cast<PtrToIntInst>(call.getArgOperand(0));
  return address ? address->getPointerOperand() : nullptr;
}

struct Access {
  Value *pointer = nullptr;
  unsigned width = 0;
  bool copy = false;
};

inline std::optional<Access> access(Instruction &inst,
                                    const DataLayout &layout) {
  if (auto *load = dyn_cast<LoadInst>(&inst)) {
    auto size = layout.getTypeStoreSize(load->getType());
    if (load->isSimple() && !size.isScalable() && size.getFixedValue() <= 16)
      return Access{load->getPointerOperand(),
                    static_cast<unsigned>(size.getFixedValue())};
    return std::nullopt;
  }
  if (auto *store = dyn_cast<StoreInst>(&inst)) {
    auto size = layout.getTypeStoreSize(store->getValueOperand()->getType());
    if (store->isSimple() && !size.isScalable() && size.getFixedValue() <= 16)
      return Access{store->getPointerOperand(),
                    static_cast<unsigned>(size.getFixedValue())};
    return std::nullopt;
  }
  if (auto *call = dyn_cast<CallBase>(&inst)) {
    if (auto *callee = call->getCalledFunction()) {
      if (callee->getName() ==
              "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3" &&
          call->arg_size() == 1 && !call->hasOperandBundles())
        return Access{call->getArgOperand(0), 16};
      if (callee->getName() == "__nvvm_reflect" && callee->isDeclaration() &&
          call->arg_size() == 1 && !call->hasOperandBundles() &&
          call->getType()->isIntegerTy(32)) {
        auto *key = dyn_cast<GlobalVariable>(call->getArgOperand(0));
        auto *literal =
            key && key->isConstant() && key->hasInitializer()
                ? dyn_cast<ConstantDataSequential>(key->getInitializer())
                : nullptr;
        if (literal && literal->isCString() &&
            literal->getAsCString() == "__CUDA_FTZ")
          return Access{};
      }
      if (readonly_shared_matrices::ctaBarrier(call, Threads) ||
          callee->getName() == "llvm.nvvm.cp.async.commit.group" ||
          callee->getName() == "llvm.nvvm.cp.async.wait.group" ||
          ((call->doesNotAccessMemory() ||
            (callee->isIntrinsic() &&
             callee->getName() == "llvm.nvvm.shfl.sync.idx.i32")) &&
           !call->hasOperandBundles()))
        return Access{};
      return std::nullopt;
    }
    StringRef code = assembly(*call);
    if (auto *pointer = matrixPointer(*call))
      return Access{pointer, 16};
    if (code == CopyAsm && call->arg_size() == 3 &&
        cast<InlineAsm>(call->getCalledOperand())->getConstraintString() ==
            "r,l,r" &&
        call->getArgOperand(1)->getType()->isPointerTy() &&
        call->getArgOperand(1)->getType()->getPointerAddressSpace() == 1)
      return Access{call->getArgOperand(0), 16, true};
    if (code == "@$3 ppu.st.shared.v2.b32 [ $0 + 0 ], { $1, $2 };" &&
        call->arg_size() == 4)
      return Access{call->getArgOperand(0), 8};
    if (code == "@$2 ppu.st.shared.b32 [ $0 + 0 ], $1;" &&
        call->arg_size() == 3)
      return Access{call->getArgOperand(0), 4};
    if (code == "@$2 ppu.ld.shared.b32 $0, [ $1 + 0 ];" &&
        call->arg_size() == 3)
      return Access{call->getArgOperand(0), 4};
    if (code == "@$5 ppu.st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };" ||
        code == "@$2 ppu.st.global.b32 [ $1 + 0 ], { $0 };" ||
        (!code.empty() && !code.contains(".shared") &&
         readonly_shared_matrices::readOnlyAsm(code)))
      return Access{};
    return std::nullopt;
  }
  return inst.mayReadOrWriteMemory() ? std::nullopt : std::optional(Access{});
}

// Propagate possible bits across the epilogue's merges. LLVM's local
// KnownBits query can stop before reaching the lane masks through this DAG.
inline std::optional<uint64_t> indexMask(Value *value, const DataLayout &layout,
                                         unsigned depth = 0) {
  if (depth == 8 || !value->getType()->isIntegerTy() ||
      value->getType()->getIntegerBitWidth() > 64)
    return std::nullopt;
  KnownBits bits = computeKnownBits(value, layout);
  if (bits.getMaxValue().ule(Shared))
    return bits.getMaxValue().getZExtValue();
  SmallVector<Value *, 2> arms;
  if (auto *phi = dyn_cast<PHINode>(value)) {
    if (phi->getNumIncomingValues() != 2)
      return std::nullopt;
    arms.assign(phi->incoming_values().begin(), phi->incoming_values().end());
  } else if (auto *select = dyn_cast<SelectInst>(value))
    arms = {select->getTrueValue(), select->getFalseValue()};
  else if (auto *bin = dyn_cast<BinaryOperator>(value);
           bin && (bin->getOpcode() == Instruction::Or ||
                   bin->getOpcode() == Instruction::Xor))
    arms = {bin->getOperand(0), bin->getOperand(1)};
  if (arms.empty())
    return std::nullopt;
  uint64_t mask = 0;
  for (Value *arm : arms) {
    auto upper = indexMask(arm, layout, depth + 1);
    if (!upper)
      return std::nullopt;
    mask |= *upper;
  }
  return mask;
}

inline bool boundedDisjoint(Value *ptr, GlobalVariable *shared, unsigned width,
                            const DataLayout &layout) {
  uint64_t high = 0;
  while (auto *gep = dyn_cast<GEPOperator>(ptr)) {
    if (gep->getNumIndices() != 1)
      return false;
    auto size = layout.getTypeAllocSize(gep->getSourceElementType());
    auto mask = indexMask(gep->idx_begin()->get(), layout);
    if (size.isScalable() || size.getFixedValue() > 16 || !mask ||
        *mask > Shared)
      return false;
    high += *mask * size.getFixedValue();
    if (high > Shared)
      return false;
    ptr = gep->getPointerOperand();
  }
  return ptr == shared && high + width <= Begin;
}

class Geometry {
  GlobalVariable *shared;
  const DataLayout &layout;
  std::array<DenseMap<Value *, APInt>, Threads> integers;

public:
  Geometry(GlobalVariable *shared, const DataLayout &layout)
      : shared(shared), layout(layout) {}

  std::optional<unsigned> offset(Value *ptr, unsigned tid, unsigned width) {
    auto n =
        readonly_shared_matrices::sharedOffset(ptr, tid, layout, integers[tid]);
    if (!n || *n < 0 || *n > Shared - width)
      return std::nullopt;
    return *n;
  }

  bool disjoint(const Access &op) {
    if (!op.pointer)
      return true;
    if (!op.pointer->getType()->isPointerTy())
      return false;
    if (op.pointer->getType()->getPointerAddressSpace() == 1)
      return !op.copy;
    if (boundedDisjoint(op.pointer, shared, op.width, layout))
      return true;
    for (unsigned tid = 0; tid < Threads; ++tid) {
      auto n = offset(op.pointer, tid, op.width);
      if (!n || (*n < End && *n + op.width > Begin))
        return false;
    }
    return true;
  }

  bool matchesLane(Value *value) {
    for (unsigned tid = 0; tid < Threads; ++tid) {
      auto n = global_load_offsets::evaluate(value, tid, integers[tid]);
      unsigned expected = ((tid & 3) << 2) | ((tid & 12) >> 2) | (tid & 16);
      if (!n || *n != expected)
        return false;
    }
    return true;
  }
};

inline bool sizeOrZero(Value *bytes) {
  auto *choice = dyn_cast<SelectInst>(bytes);
  auto is = [](Value *value, unsigned n) {
    auto *constant = dyn_cast<ConstantInt>(value);
    return constant && constant->equalsInt(n);
  };
  return is(bytes, 16) ||
         (choice &&
          ((is(choice->getTrueValue(), 16) && is(choice->getFalseValue(), 0)) ||
           (is(choice->getTrueValue(), 0) && is(choice->getFalseValue(), 16))));
}

// A prior iteration leaves the low plane in the new layout. Prove that the
// next iteration publishes a complete overwrite before an old-layout reader.
inline bool restored(BasicBlock &header, Geometry &geometry,
                     const DataLayout &layout) {
  BitVector copied(Bytes);
  bool protectedEpoch = false, committed = false, waited = false;
  for (Instruction &inst : header) {
    auto *call = dyn_cast<CallBase>(&inst);
    auto *callee = call ? call->getCalledFunction() : nullptr;
    StringRef name = callee ? callee->getName() : StringRef();
    if (name == "llvm.nvvm.cp.async.commit.group") {
      committed = true;
      waited = false;
    } else if (name == "llvm.nvvm.cp.async.wait.group") {
      auto *count = dyn_cast<ConstantInt>(call->getArgOperand(0));
      waited = committed && count && count->isZero();
    }
    if (readonly_shared_matrices::ctaBarrier(&inst, Threads)) {
      if (waited && copied.all())
        return true;
      protectedEpoch = true;
    }
    auto op = access(inst, layout);
    if (!op)
      return false;
    if (geometry.disjoint(*op))
      continue;
    if (!protectedEpoch || !op->copy || !sizeOrZero(call->getArgOperand(2)))
      return false;
    committed = waited = false;
    for (unsigned tid = 0; tid < Threads; ++tid) {
      auto n = geometry.offset(op->pointer, tid, 16);
      if (!n || *n % 16)
        return false;
      unsigned lo = std::max(*n, Begin), hi = std::min(*n + 16, End);
      if (lo < hi)
        copied.set(lo - Begin, hi - Begin);
    }
  }
  return false;
}

struct Fragment {
  CallInst *call;
  std::array<ExtractValueInst *, 4> words{};
};

inline bool collect(BasicBlock &consumer, Geometry &geometry,
                    SmallVectorImpl<Fragment> &fragments,
                    std::array<unsigned, Bytes / 4> &remap) {
  BitVector seen(Bytes / 4);
  for (Instruction &inst : consumer) {
    auto *call = dyn_cast<CallInst>(&inst);
    auto *ptr = call ? matrixPointer(*call) : nullptr;
    if (!ptr || geometry.disjoint({ptr, 16}))
      continue;
    if (fragments.size() == 8)
      return false;
    Fragment fragment{call};
    if (!mla_nope_prefetch::extracts(call, fragment.words))
      return false;
    for (auto [slot, word] : llvm::enumerate(fragment.words)) {
      if (word->getParent() != &consumer || word->use_empty())
        return false;
      for (Use &use : word->uses()) {
        auto *mma = dyn_cast<CallInst>(use.getUser());
        if (!mma || mma->getParent() != &consumer ||
            assembly(*mma) != mla_nope_prefetch::ProductAsm ||
            !mma->isArgOperand(&use) || mma->getArgOperandNo(&use) != 8 + slot)
          return false;
      }
      for (unsigned lane = 0; lane < 32; ++lane) {
        auto row = geometry.offset(ptr, slot * 8 + lane / 4, 16);
        if (!row || *row % 16 || *row < Begin || *row + 16 > End)
          return false;
        for (unsigned warp = 1; warp < Threads / 32; ++warp)
          if (geometry.offset(ptr, warp * 32 + slot * 8 + lane / 4, 16) != row)
            return false;
        unsigned old = (*row - Begin) / 4 + lane % 4;
        if (seen.test(old))
          return false;
        seen.set(old);
        remap[old] = Begin + 512 * fragments.size() + 16 * lane + 4 * slot;
      }
    }
    fragments.push_back(fragment);
  }
  return fragments.size() == 8 && seen.all();
}

inline bool repack(Function &kernel) {
  auto *shared = kernel.getParent()->getGlobalVariable("global_smem");
  if (!shared || !shared->isDeclaration() || shared->getAddressSpace() != 3 ||
      shared->getAlign().valueOrOne() < Align(16))
    return false;
  const auto &layout = kernel.getParent()->getDataLayout();
  Geometry geometry(shared, layout);
  SmallVector<CallInst *> products;
  CallInst *tid = nullptr;
  for (Instruction &inst : instructions(kernel)) {
    auto *call = dyn_cast<CallInst>(&inst);
    if (!call)
      continue;
    if (assembly(*call) == mla_nope_prefetch::ProductAsm)
      products.push_back(call);
    if (auto *callee = call->getCalledFunction();
        callee && callee->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
      if (tid || call->getParent() != &kernel.getEntryBlock())
        return false;
      tid = call;
    }
  }
  if (!tid || products.size() != 96)
    return false;
  BasicBlock *consumer = products[32]->getParent();
  DominatorTree dominance(kernel);
  LoopInfo loops(dominance);
  auto *loop = loops.getLoopFor(consumer);
  if (!loop || loop->getLoopLatch() != consumer || loop->getParentLoop() ||
      !llvm::all_of(ArrayRef(products).drop_front(32), [&](CallInst *product) {
        return product->getParent() == consumer;
      }))
    return false;
  // The two incoming paths are the P publisher and its optional KV prefetch.
  auto *producer = consumer->getUniquePredecessor();
  if (producer)
    return false;
  BasicBlock *prefetch = nullptr;
  for (auto *block : predecessors(consumer)) {
    if (block->getSingleSuccessor() == consumer) {
      if (prefetch)
        return false;
      prefetch = block;
    } else {
      if (producer)
        return false;
      producer = block;
    }
  }
  if (!producer || !prefetch || prefetch->getSinglePredecessor() != producer ||
      !loop->contains(producer) || !loop->contains(prefetch) ||
      !dominance.dominates(producer, consumer) ||
      !restored(*loop->getHeader(), geometry, layout))
    return false;
  SmallVector<Fragment, 8> fragments;
  std::array<unsigned, Bytes / 4> remap{};
  if (!collect(*consumer, geometry, fragments, remap))
    return false;
  SmallVector<StoreInst *, 4> writes;
  SmallVector<unsigned, 4> bases;
  BitVector written(Bytes / 4);
  Instruction *selector = nullptr, *protect = nullptr, *publish = nullptr;
  for (Instruction &inst : *producer) {
    if (readonly_shared_matrices::ctaBarrier(&inst, Threads)) {
      if (writes.empty())
        protect = &inst;
      else if (!publish)
        publish = &inst;
      continue;
    }
    auto *store = dyn_cast<StoreInst>(&inst);
    if (!store || geometry.disjoint({store->getPointerOperand(), 4}))
      continue;
    if (!protect || publish || writes.size() == 4 || !store->isSimple() ||
        !store->getValueOperand()->getType()->isIntegerTy(32))
      return false;
    auto *transpose = dyn_cast<CallInst>(store->getValueOperand());
    auto *callee = transpose ? transpose->getCalledFunction() : nullptr;
    auto *shuffle =
        callee &&
                callee->getName() == "__flagtree_ppu_intrinsic__mat.trans.b8" &&
                transpose->arg_size() == 1
            ? dyn_cast<CallInst>(transpose->getArgOperand(0))
            : nullptr;
    auto *shfl = shuffle ? shuffle->getCalledFunction() : nullptr;
    auto *lane = shfl && shfl->getName() == "llvm.nvvm.shfl.sync.idx.i32" &&
                         shuffle->arg_size() == 4
                     ? dyn_cast<Instruction>(shuffle->getArgOperand(2))
                     : nullptr;
    if (!lane || !dominance.dominates(tid, lane) ||
        !dominance.dominates(lane, store) || (selector && selector != lane) ||
        !geometry.matchesLane(lane))
      return false;
    selector = lane;
    unsigned base = 0;
    for (unsigned thread = 0; thread < Threads; ++thread) {
      auto n = geometry.offset(store->getPointerOperand(), thread, 4);
      if (!n || *n % 4 || *n < Begin || *n + 4 > End)
        return false;
      unsigned word = (*n - Begin) / 4;
      if (written.test(word))
        return false;
      written.set(word);
      if (thread == 0)
        base = remap[word];
      unsigned lane =
          ((thread & 3) << 2) | ((thread & 12) >> 2) | (thread & 16);
      unsigned delta = 16 * lane + ((thread & 32) >> 3) + ((thread & 64) << 3) +
                       ((thread & 128) << 4);
      if (remap[word] != base + delta)
        return false;
    }
    bases.push_back(base);
    writes.push_back(store);
  }
  if (!publish || !selector || writes.size() != 4 || !written.all())
    return false;

  SmallPtrSet<Instruction *, 16> rewritten;
  rewritten.insert(writes.begin(), writes.end());
  for (const Fragment &fragment : fragments)
    rewritten.insert(fragment.call);
  // Verify the remainder of the live range, including exits. The header is
  // the only loop edge that can expose the old layout again.
  SmallVector<BasicBlock *> pending{producer};
  SmallPtrSet<BasicBlock *, 16> visited;
  while (!pending.empty()) {
    auto *block = pending.pop_back_val();
    if (block == loop->getHeader() || !visited.insert(block).second)
      continue;
    if (loop->contains(block) && block != producer && block != prefetch &&
        block != consumer)
      return false;
    auto *first = block == producer ? protect->getNextNode() : &block->front();
    for (auto *inst = first; inst; inst = inst->getNextNode()) {
      if (rewritten.contains(inst))
        continue;
      auto op = access(*inst, layout);
      if (!op || !geometry.disjoint(*op))
        return false;
    }
    for (auto *next : successors(block))
      pending.push_back(next);
  }

  IRBuilder<> b(tid->getNextNode());
  b.SetCurrentDebugLocation(DebugLoc());
  Value *lane = b.CreateAnd(tid, 31, "p.fragment.lane");
  Value *readLane = b.CreateShl(lane, 4, "p.fragment.read.lane");
  SmallVector<Value *, 8> pointers;
  for (unsigned i = 0; i < fragments.size(); ++i) {
    Value *offset = b.CreateAdd(readLane, b.getInt32(Begin + 512 * i),
                                "p.fragment.read.offset." + Twine(i));
    pointers.push_back(b.CreateGEP(b.getInt8Ty(), shared, offset,
                                   "p.fragment.read.ptr." + Twine(i)));
  }
  b.SetInsertPoint(selector->getNextNode());
  b.SetCurrentDebugLocation(DebugLoc());
  Value *writeLane = b.CreateShl(selector, 4, "p.fragment.write.lane");
  Value *slotBit = b.CreateAnd(tid, 32, "p.fragment.write.sbit");
  Value *slot = b.CreateLShr(slotBit, 3, "p.fragment.write.slot");
  Value *f0 = b.CreateAnd(tid, 64, "p.fragment.write.f0");
  Value *f1 = b.CreateAnd(tid, 128, "p.fragment.write.f1");
  Value *lo = b.CreateShl(f0, 3, "p.fragment.write.flo");
  Value *hi = b.CreateShl(f1, 4, "p.fragment.write.fhi");
  Value *inner = b.CreateOr(writeLane, slot, "p.fragment.write.inner");
  Value *outer = b.CreateOr(lo, hi, "p.fragment.write.outer");
  Value *dynamic = b.CreateOr(inner, outer, "p.fragment.write.dynamic");
  for (auto [i, store] : llvm::enumerate(writes)) {
    Value *offset = b.CreateAdd(dynamic, b.getInt32(bases[i]),
                                "p.fragment.write.offset." + Twine(i));
    store->setOperand(1, b.CreateGEP(b.getInt8Ty(), shared, offset,
                                     "p.fragment.write.ptr." + Twine(i)));
    store->setDebugLoc(DebugLoc());
  }
  for (auto [i, fragment] : llvm::enumerate(fragments)) {
    b.SetInsertPoint(fragment.call);
    b.SetCurrentDebugLocation(DebugLoc());
    Value *read = b.CreateAlignedLoad(FixedVectorType::get(b.getInt32Ty(), 4),
                                      pointers[i], Align(16));
    for (auto [slot, word] : llvm::enumerate(fragment.words)) {
      b.SetInsertPoint(word);
      b.SetCurrentDebugLocation(DebugLoc());
      word->replaceAllUsesWith(b.CreateExtractElement(read, b.getInt32(slot)));
      word->eraseFromParent();
    }
    fragment.call->eraseFromParent();
  }
  return true;
}
} // namespace mla_low_probability

inline unsigned repackMlaLowProbability(llvm::Module &module, unsigned threads,
                                        unsigned shared) {
  if (threads != mla_low_probability::Threads ||
      shared != mla_low_probability::Shared)
    return 0;
  unsigned changed = 0;
  for (auto &function : module)
    if (!function.isDeclaration() &&
        function.getCallingConv() == llvm::CallingConv::PTX_Kernel &&
        function.getFnAttribute("nvvm.reqntid").getValueAsString() == "256" &&
        function.getFnAttribute("ppu.mla.kv.pipeline").getValueAsString() ==
            "1" &&
        function.getFnAttribute("ppu.mla.pv.batch").getValueAsString() == "4")
      changed += mla_low_probability::repack(function);
  return changed;
}
} // namespace triton::ppu

#endif // TRITON_PPU_REPACK_MLA_LOW_PROBABILITY_H
