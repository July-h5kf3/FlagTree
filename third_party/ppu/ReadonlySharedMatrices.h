// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_READONLY_SHARED_MATRICES_H
#define TRITON_PPU_READONLY_SHARED_MATRICES_H

#include "GlobalLoadOffsets.h"
#include "TritonPPUGPUToLLVM/I8SharedStore.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Operator.h"

namespace triton::ppu {
namespace readonly_shared_matrices {

inline std::optional<int64_t>
sharedOffset(llvm::Value *value, unsigned tid, const llvm::DataLayout &layout,
             llvm::DenseMap<llvm::Value *, llvm::APInt> &cache,
             unsigned depth = 0) {
  using namespace llvm;
  if (depth > 32 || !value->getType()->isPointerTy() ||
      value->getType()->getPointerAddressSpace() != 3)
    return std::nullopt;
  if (auto *global = dyn_cast<GlobalVariable>(value)) {
    if (global->getName() == "global_smem" && global->isDeclaration() &&
        global->getAlign().valueOrOne() >= Align(16))
      return 0;
    return std::nullopt;
  }
  auto *gep = dyn_cast<GEPOperator>(value);
  if (!gep)
    return std::nullopt;
  auto base =
      sharedOffset(gep->getPointerOperand(), tid, layout, cache, depth + 1);
  if (!base)
    return std::nullopt;
  SmallVector<Value *> indices;
  for (Value *index : gep->indices()) {
    auto integer = global_load_offsets::evaluate(index, tid, cache);
    if (!integer)
      return std::nullopt;
    indices.push_back(ConstantInt::get(value->getContext(), *integer));
  }
  return *base +
         layout.getIndexedOffsetInType(gep->getSourceElementType(), indices);
}

inline bool readOnlyAsm(llvm::StringRef code) {
  llvm::SmallVector<llvm::StringRef> statements;
  code.split(statements, ';');
  for (auto statement : statements) {
    statement = statement.trim();
    statement.consume_front("{");
    statement = statement.trim();
    if (statement.empty() || statement == "}" || statement.starts_with(".reg "))
      continue;
    if (statement.starts_with("@"))
      statement = statement.split(' ').second.trim();
    auto opcode = statement.split(' ').first;
    if (!llvm::StringSwitch<bool>(opcode)
             .Cases("ppu.mov.u32", "ppu.mov.b32", true)
             .Cases("ppu.ld.global.b32", "ppu.ld.global.cg.v4.b32",
                    "ppu.ld.global.v4.b32", true)
             .Case("ppu.ld.global.v2.b32", true)
             .Cases("ppu.ld.shared.b32",
                    "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16", true)
             .Case("ppu.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8."
                   "s32",
                   true)
             .Cases("ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32",
                    "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32",
                    true)
             .Cases("ppu.div.full.f32", "ppu.add.f32", "ppu.sub.f32", true)
             .Cases("ppu.cvt.rmi.s32.f32", "ppu.cvt.rn.f32.s32", true)
             .Cases("ppu.cvt.rmi.f32.f32", "ppu.cvt.rzi.s32.f32", true)
             .Cases("ppu.fma.rn.f32", "ppu.prmt.b32", true)
             .Default(false))
      return false;
  }
  return true;
}

inline bool proveNoQueryWrites(llvm::Loop &loop, const llvm::BitVector &query,
                               const llvm::DataLayout &layout,
                               unsigned threads) {
  using namespace llvm;
  std::vector<DenseMap<Value *, APInt>> caches(threads);
  auto disjoint = [&](Value *pointer, unsigned width, bool aligned = false,
                      unsigned packets = 1) {
    for (unsigned tid = 0; tid < threads; ++tid) {
      auto offset = sharedOffset(pointer, tid, layout, caches[tid]);
      if (!offset)
        return false;
      int64_t base = aligned ? *offset & ~int64_t(15) : *offset;
      for (unsigned packet = 0; packet < packets; ++packet) {
        int64_t begin = base + packet * 8192;
        if (begin < 0 || begin + width > query.size())
          return false;
        for (unsigned byte = 0; byte < width; ++byte)
          if (query[begin + byte])
            return false;
      }
    }
    return true;
  };
  for (auto *block : loop.blocks()) {
    for (auto &instruction : *block) {
      if (isa<AtomicRMWInst, AtomicCmpXchgInst, FenceInst>(instruction))
        return false;
      if (auto *store = dyn_cast<StoreInst>(&instruction)) {
        unsigned space = store->getPointerAddressSpace();
        if (space != 1 &&
            (space != 3 ||
             !disjoint(
                 store->getPointerOperand(),
                 layout.getTypeStoreSize(store->getValueOperand()->getType())
                     .getFixedValue())))
          return false;
      } else if (auto *call = dyn_cast<CallBase>(&instruction)) {
        auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand());
        if (assembly) {
          StringRef code = assembly->getAsmString();
          StringRef constraints = assembly->getConstraintString();
          if (constraints.contains("memory"))
            return false;
          if (code == ::mlir::triton::ppu::groupedI8SharedStoreAsm()) {
            std::string expected = "r";
            for (unsigned i = 1; i < 65; ++i)
              expected += ",r";
            if (call->arg_size() != 65 || constraints != expected)
              return false;
            for (Value *argument : call->args())
              if (!argument->getType()->isIntegerTy(32))
                return false;
            auto *address = dyn_cast<PtrToIntInst>(call->getArgOperand(64));
            if (!address || !disjoint(address->getOperand(0), 16, true, 16))
              return false;
          } else if (code == "cp.async.cg.shared.global [ $0 + 0 ], "
                             "[ $1 + 0 ], 0x10, $2;") {
            if (call->arg_size() != 3 || constraints != "r,l,r" ||
                !call->getType()->isVoidTy() ||
                !call->getArgOperand(0)->getType()->isPointerTy() ||
                !call->getArgOperand(1)->getType()->isPointerTy() ||
                call->getArgOperand(1)->getType()->getPointerAddressSpace() !=
                    1 ||
                !call->getArgOperand(2)->getType()->isIntegerTy(32) ||
                !disjoint(call->getArgOperand(0), 16))
              return false;
          } else if (code == "@$2 ppu.st.shared.b32 [ $0 + 0 ], $1;") {
            if (call->arg_size() != 3 || constraints != "r,r,b" ||
                !disjoint(call->getArgOperand(0), 4))
              return false;
          } else if (!readOnlyAsm(code)) {
            return false;
          }
        } else {
          auto *callee = call->getCalledFunction();
          if (callee && callee->getName() == "__builtin_ppu_rcpf" &&
              (call->arg_size() != 1 || !call->getType()->isFloatTy() ||
               !call->getArgOperand(0)->getType()->isFloatTy()))
            return false;
          if (callee && callee->isDeclaration() &&
              callee->getName() == "llvm.nvvm.cp.async.commit.group" &&
              call->arg_empty() && call->getType()->isVoidTy())
            continue;
          if (callee && callee->isDeclaration() &&
              callee->getName() == "llvm.nvvm.cp.async.wait.group" &&
              call->arg_size() == 1 && call->getType()->isVoidTy()) {
            auto *groups = dyn_cast<ConstantInt>(call->getArgOperand(0));
            if (groups && groups->isZero())
              continue;
          }
          if (!callee || !callee->isDeclaration() ||
              !StringSwitch<bool>(callee->getName())
                   .Cases("__flagtree_ppu_intrinsic__mat.trans.b8",
                          "__nvvm_reflect", true)
                   .Cases("llvm.maxnum.f32", "llvm.maxnum.v2f32", true)
                   .Case("__builtin_ppu_rcpf", true)
                   .Cases("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3",
                          "llvm.nvvm.shfl.sync.bfly.i32",
                          "llvm.nvvm.shfl.sync.idx.i32", true)
                   .Cases("llvm.nvvm.barrier.cta.sync.aligned.all",
                          "llvm.nvvm.ex2.approx.ftz.f",
                          "llvm.nvvm.ex2.approx.f", true)
                   .Default(false))
            return false;
        }
      } else if (instruction.mayWriteToMemory()) {
        return false;
      }
    }
  }
  return true;
}

inline unsigned hoist(llvm::Function &function, unsigned count,
                      unsigned threads, unsigned sharedBytes) {
  using namespace llvm;
  SmallVector<CallInst *> products;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block)
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
          if (assembly->getAsmString().contains(".s32.s8.s8.s32"))
            products.push_back(call);
  if (products.size() != 96)
    return 0;
  DominatorTree dominators(function);
  LoopInfo loops(dominators);
  auto *header = products.front()->getParent();
  auto *loop = loops.getLoopFor(header);
  if (!loop || loop->getHeader() != header || loop->getParentLoop())
    return 0;
  auto *preheader = loop->getLoopPreheader();
  if (!preheader)
    return 0;
  DenseMap<CallInst *, unsigned> uses;
  for (auto *product : ArrayRef(products).take_front(32)) {
    if (product->getParent() != header || product->arg_size() != 16)
      return 0;
    CallInst *matrix = nullptr;
    for (unsigned word = 0; word < 4; ++word) {
      auto *extract =
          dyn_cast<ExtractValueInst>(product->getArgOperand(8 + word));
      if (!extract || extract->getIndices() != ArrayRef<unsigned>(word))
        return 0;
      auto *source = dyn_cast<CallInst>(extract->getAggregateOperand());
      if (!source || (matrix && matrix != source))
        return 0;
      matrix = source;
    }
    ++uses[matrix];
  }
  if (uses.size() != 16)
    return 0;
  SmallVector<CallInst *> matrices;
  CallInst *barrier = nullptr;
  for (Instruction &instruction : *header) {
    auto *call = dyn_cast<CallInst>(&instruction);
    if (!call)
      continue;
    if (call == products.front())
      break;
    if (auto *callee = call->getCalledFunction())
      if (callee->getName() == "llvm.nvvm.barrier.cta.sync.aligned.all")
        barrier = call;
    if (uses.contains(call))
      matrices.push_back(call);
  }
  if (!barrier || matrices.size() != 16)
    return 0;
  const auto &layout = function.getParent()->getDataLayout();
  BitVector query(sharedBytes);
  for (auto *matrix : matrices) {
    auto *assembly = dyn_cast<InlineAsm>(matrix->getCalledOperand());
    if (uses[matrix] != 2 || !assembly || matrix->arg_size() != 1 ||
        matrix->getNumUses() != 4 ||
        assembly->getConstraintString() != "=r,=r,=r,=r,r" ||
        assembly->getAsmString() != "ppu.ldmatrix.sync.aligned.m8n8.x4.shared."
                                    "b16 {$0, $1, $2, $3}, [$4];")
      return 0;
    auto *address = dyn_cast<PtrToIntInst>(matrix->getArgOperand(0));
    if (!address || !dominators.dominates(address, preheader->getTerminator()))
      return 0;
    for (User *user : matrix->users())
      if (!isa<ExtractValueInst>(user) ||
          cast<Instruction>(user)->getParent() != header)
        return 0;
    for (unsigned tid = 0; tid < threads; ++tid) {
      DenseMap<Value *, APInt> cache;
      auto offset = sharedOffset(address->getOperand(0), tid, layout, cache);
      if (!offset || *offset < 0 || *offset % 16 || *offset + 16 > sharedBytes)
        return 0;
      query.set(*offset, *offset + 16);
    }
  }
  if (query.count() != 16384 ||
      !proveNoQueryWrites(*loop, query, layout, threads))
    return 0;
  CallInst *wait = nullptr, *commit = nullptr;
  unsigned waits = 0, commits = 0;
  SmallVector<CallInst *> copies;
  for (Instruction &instruction : *header) {
    auto *call = dyn_cast<CallInst>(&instruction);
    if (!call)
      continue;
    if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand())) {
      if (assembly->getAsmString() ==
          "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;")
        copies.push_back(call);
    } else if (auto *callee = call->getCalledFunction()) {
      if (callee->getName() == "llvm.nvvm.cp.async.wait.group") {
        wait = call;
        ++waits;
      }
      if (callee->getName() == "llvm.nvvm.cp.async.commit.group") {
        commit = call;
        ++commits;
      }
    }
  }
  bool overlap = count == 12 && copies.size() == 16 && commits == 1 &&
                 waits == 1 && commit->comesBefore(wait);
  if (overlap) {
    for (auto *copy : copies)
      overlap &= copy->comesBefore(commit);
    for (auto *matrix : ArrayRef(matrices).drop_front(count))
      overlap &= wait->comesBefore(matrix);
  }
  // No mutation occurs until every loop writer and every thread is proved.
  auto *terminator = preheader->getTerminator();
  barrier->clone()->insertBefore(terminator->getIterator());
  for (unsigned index = 0; index < matrices.size(); ++index) {
    if (index >= count && !overlap)
      break;
    auto *matrix = matrices[index];
    Instruction *destination = index < count ? terminator : wait;
    SmallVector<Instruction *> extracts;
    for (User *user : matrix->users())
      extracts.push_back(cast<Instruction>(user));
    llvm::sort(extracts, [](Instruction *a, Instruction *b) {
      return cast<ExtractValueInst>(a)->getIndices()[0] <
             cast<ExtractValueInst>(b)->getIndices()[0];
    });
    matrix->moveBefore(destination->getIterator());
    for (auto *extract : extracts)
      extract->moveBefore(destination->getIterator());
  }
  return count;
}

} // namespace readonly_shared_matrices

inline unsigned hoistReadonlySharedMatrices(llvm::Module &module,
                                            unsigned count, unsigned threads,
                                            unsigned sharedBytes) {
  if ((count != 8 && count != 12 && count != 16) || threads != 512 ||
      (sharedBytes != 217088 && sharedBytes != 167936))
    return 0;
  unsigned changed = 0;
  for (auto &function : module)
    if (!function.isDeclaration())
      changed += readonly_shared_matrices::hoist(function, count, threads,
                                                 sharedBytes);
  return changed;
}

} // namespace triton::ppu
#endif
