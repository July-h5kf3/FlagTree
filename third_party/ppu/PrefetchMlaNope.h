// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_PREFETCH_MLA_NOPE_H
#define TRITON_PPU_PREFETCH_MLA_NOPE_H

#include "ReadonlySharedMatrices.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/IntrinsicInst.h"
#include <array>
#include <functional>

namespace triton::ppu {
namespace mla_nope_prefetch {

constexpr llvm::StringLiteral MatrixAsm =
    "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];";
constexpr llvm::StringLiteral ProductAsm =
    "ppu.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8.s32 "
    "{ $0, $1, $2, $3, $4, $5, $6, $7 }, { $16, $17, $18, $19 }, "
    "{ $20, $21, $22, $23 }, { $8, $9, $10, $11, $12, $13, $14, $15 };";

inline bool integerType(llvm::Type *type) {
  if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type))
    type = vector->getElementType();
  return type->isIntegerTy() && type->getIntegerBitWidth() <= 64;
}

inline llvm::Constant *
constantValue(llvm::Value *value, unsigned tid, const llvm::DataLayout &layout,
              llvm::DenseMap<llvm::Value *, llvm::Constant *> &cache,
              unsigned depth = 0) {
  using namespace llvm;
  if (depth > 64 || !integerType(value->getType()))
    return nullptr;
  if (auto *constant = dyn_cast<Constant>(value))
    return constant;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  auto *instruction = dyn_cast<Instruction>(value);
  if (!instruction)
    return nullptr;
  Constant *result = nullptr;
  if (auto *call = dyn_cast<CallInst>(instruction)) {
    auto *function = call->getCalledFunction();
    if (!function)
      return nullptr;
    if (function->getName() == "llvm.nvvm.read.ptx.sreg.tid.x") {
      result = ConstantInt::get(Type::getInt32Ty(value->getContext()), tid);
    } else if (function->getIntrinsicID() == Intrinsic::vector_reduce_or) {
      auto *source =
          constantValue(call->getArgOperand(0), tid, layout, cache, depth + 1);
      auto *type = dyn_cast<FixedVectorType>(call->getArgOperand(0)->getType());
      if (!source || !type)
        return nullptr;
      APInt bits(value->getType()->getIntegerBitWidth(), 0);
      for (unsigned index = 0; index < type->getNumElements(); ++index) {
        auto *element =
            dyn_cast_or_null<ConstantInt>(source->getAggregateElement(index));
        if (!element)
          return nullptr;
        bits |= element->getValue();
      }
      result = ConstantInt::get(value->getContext(), bits);
    }
  } else if (instruction->isBinaryOp() ||
             isa<CastInst, ICmpInst, SelectInst, InsertElementInst,
                 ExtractElementInst, ShuffleVectorInst>(instruction)) {
    SmallVector<Constant *> operands;
    for (Value *operand : instruction->operands()) {
      auto *constant = constantValue(operand, tid, layout, cache, depth + 1);
      if (!constant)
        return nullptr;
      operands.push_back(constant);
    }
    result =
        ConstantFoldInstOperands(instruction, operands, layout, nullptr, false);
  }
  if (result)
    cache.try_emplace(value, result);
  return result;
}

inline bool preparePointer(
    llvm::Value *pointer, unsigned tid, const llvm::DataLayout &layout,
    llvm::DenseMap<llvm::Value *, llvm::Constant *> &constants,
    llvm::DenseMap<llvm::Value *, llvm::APInt> &integers, unsigned depth = 0) {
  using namespace llvm;
  if (depth > 32)
    return false;
  if (isa<GlobalVariable>(pointer))
    return true;
  auto *gep = dyn_cast<GEPOperator>(pointer);
  if (!gep || !preparePointer(gep->getPointerOperand(), tid, layout, constants,
                              integers, depth + 1))
    return false;
  for (Value *index : gep->indices()) {
    auto *constant = dyn_cast_or_null<ConstantInt>(
        constantValue(index, tid, layout, constants));
    if (!constant)
      return false;
    integers.try_emplace(index, constant->getValue());
  }
  return true;
}

template <std::size_t Count>
inline bool extracts(llvm::CallInst *call,
                     std::array<llvm::ExtractValueInst *, Count> &words) {
  using namespace llvm;
  for (User *user : call->users()) {
    auto *word = dyn_cast<ExtractValueInst>(user);
    if (!word || word->getNumIndices() != 1 ||
        word->getIndices().front() >= Count ||
        !word->getType()->isIntegerTy(32))
      return false;
    unsigned index = word->getIndices().front();
    if (words[index])
      return false;
    words[index] = word;
  }
  return llvm::all_of(words, [](auto *word) { return word != nullptr; });
}

struct Fragment {
  llvm::CallInst *call = nullptr;
  llvm::Value *pointer = nullptr;
  std::array<llvm::ExtractValueInst *, 4> words{};
};

inline std::optional<Fragment> fragment(llvm::CallInst *product,
                                        unsigned first) {
  using namespace llvm;
  Fragment result;
  for (unsigned index = 0; index < 4; ++index) {
    auto *word =
        dyn_cast<ExtractValueInst>(product->getArgOperand(first + index));
    if (!word || word->getNumIndices() != 1 ||
        word->getIndices().front() != index)
      return std::nullopt;
    auto *call = dyn_cast<CallInst>(word->getAggregateOperand());
    if (!call || (result.call && result.call != call))
      return std::nullopt;
    result.call = call;
  }
  auto *assembly = dyn_cast<InlineAsm>(result.call->getCalledOperand());
  if (!assembly || assembly->getAsmString() != MatrixAsm ||
      assembly->getConstraintString() != "=r,=r,=r,=r,r" ||
      result.call->arg_size() != 1 || !extracts(result.call, result.words))
    return std::nullopt;
  auto *address = dyn_cast<PtrToIntInst>(result.call->getArgOperand(0));
  if (!address || address->getPointerAddressSpace() != 3)
    return std::nullopt;
  result.pointer = address->getPointerOperand();
  return result;
}

inline unsigned schedule(llvm::Function &function, unsigned shared) {
  using namespace llvm;
  SmallVector<CallInst *> products;
  DenseMap<Instruction *, unsigned> position;
  unsigned rope = 0, ropeFirst = 0;
  for (BasicBlock &block : function)
    for (Instruction &instruction : block) {
      position[&instruction] = position.size();
      auto *call = dyn_cast<CallInst>(&instruction);
      auto *assembly =
          call ? dyn_cast<InlineAsm>(call->getCalledOperand()) : nullptr;
      if (!assembly)
        continue;
      StringRef code = assembly->getAsmString();
      if (code == ProductAsm) {
        if (call->arg_size() != 16 ||
            assembly->getConstraintString() !=
                "=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,r,r,r,r,r,r,r,r")
          return 0;
        products.push_back(call);
      } else if (code.contains(".f32.bf16.bf16.f32") ||
                 code.contains(".f32.f16.f16.f32")) {
        if (!rope)
          ropeFirst = position[call];
        ++rope;
      }
    }
  if (products.size() != 96 || rope != 8)
    return 0;
  auto *block = products.front()->getParent();
  std::array<Fragment, 32> keys;
  std::array<Fragment, 32> queryInputs;
  std::array<std::array<ExtractValueInst *, 8>, 32> outputs{};
  DenseMap<CallInst *, unsigned> queries;
  SmallPtrSet<CallInst *, 32> keyCalls;
  SmallPtrSet<Instruction *, 32> allowedCalls;
  SmallPtrSet<Instruction *, 32> allowedExtracts;
  std::array<DenseMap<Value *, Constant *>, 512> constants;
  std::array<DenseMap<Value *, APInt>, 512> integers;
  const DataLayout &layout = function.getParent()->getDataLayout();
  DominatorTree dominance(function);
  Instruction *begin = nullptr, *last = nullptr;
  for (unsigned index = 0; index < 32; ++index) {
    auto *product = products[index];
    if (product->getParent() != block || position[product] >= ropeFirst ||
        !extracts(product, outputs[index]))
      return 0;
    auto query = fragment(product, 8), key = fragment(product, 12);
    if (!query || !key || key->call->getParent() != block ||
        position[key->call] >= position[products.front()] ||
        !keyCalls.insert(key->call).second)
      return 0;
    ++queries[query->call];
    for (unsigned tid = 0; tid < 512; ++tid) {
      for (unsigned operand = 0; operand < 2; ++operand) {
        auto *input = operand ? &*key : &*query;
        int64_t low = operand ? 0 : 131072;
        int64_t high = operand ? 131072 : shared;
        if (!preparePointer(input->pointer, tid, layout, constants[tid],
                            integers[tid]))
          return 0;
        auto offset = readonly_shared_matrices::sharedOffset(
            input->pointer, tid, layout, integers[tid]);
        if (!offset || *offset < low || *offset + 16 > high || *offset % 16)
          return 0;
      }
    }
    for (unsigned word = 0; word < 8; ++word) {
      Value *accumulator = product->getArgOperand(word);
      if (index < 2) {
        auto *zero = dyn_cast<ConstantInt>(accumulator);
        if (!zero || !zero->isZero())
          return 0;
      } else if (accumulator != outputs[index - 2][word]) {
        return 0;
      }
    }
    keys[index] = *key;
    queryInputs[index] = *query;
    allowedCalls.insert(product);
    allowedCalls.insert(key->call);
    allowedCalls.insert(query->call);
    for (auto *word : key->words)
      allowedExtracts.insert(word);
    for (auto *word : query->words)
      allowedExtracts.insert(word);
    for (auto *word : outputs[index]) {
      if (word->getParent() != block)
        return 0;
      allowedExtracts.insert(word);
      if (!last || position[word] > position[last])
        last = word;
    }
    if (!begin || position[key->call] < position[begin])
      begin = key->call;
  }
  if (queries.size() != 16 || llvm::any_of(queries, [](const auto &entry) {
        return entry.second != 2;
      }))
    return 0;
  Instruction *end = last->getNextNode();
  if (!end || !begin->comesBefore(last))
    return 0;
  SmallVector<Instruction *> region;
  SmallPtrSet<Instruction *, 32> members;
  for (Instruction *instruction = begin; instruction != end;
       instruction = instruction->getNextNode()) {
    if (!instruction)
      return 0;
    if (!allowedCalls.contains(instruction) &&
        !allowedExtracts.contains(instruction)) {
      bool pure = false;
      if (auto *gep = dyn_cast<GetElementPtrInst>(instruction))
        pure = gep->getPointerAddressSpace() == 3;
      else if (auto *cast = dyn_cast<CastInst>(instruction))
        pure = isa<PtrToIntInst>(cast) ||
               (integerType(cast->getType()) && integerType(cast->getSrcTy()));
      else if (auto *call = dyn_cast<IntrinsicInst>(instruction))
        pure = call->getIntrinsicID() == Intrinsic::vector_reduce_or;
      else
        pure = integerType(instruction->getType()) &&
               (instruction->isBinaryOp() ||
                isa<ICmpInst, SelectInst, InsertElementInst, ExtractElementInst,
                    ShuffleVectorInst>(instruction));
      if (!pure)
        return 0;
    }
    members.insert(instruction);
    region.push_back(instruction);
  }
  for (auto [query, uses] : queries) {
    std::array<ExtractValueInst *, 4> words{};
    if (!extracts(query, words) || llvm::any_of(words, [&](auto *word) {
          return !members.contains(word) && !dominance.dominates(word, begin);
        }))
      return 0;
  }
  SmallVector<Instruction *> ordered;
  SmallPtrSet<Instruction *, 32> emitted;
  SmallPtrSet<Instruction *, 32> productSet;
  for (unsigned index = 0; index < 32; ++index)
    productSet.insert(products[index]);
  std::function<bool(Value *, bool)> emit = [&](Value *value,
                                                bool allowProduct) {
    auto *instruction = dyn_cast<Instruction>(value);
    if (!instruction || !members.contains(instruction) ||
        emitted.contains(instruction))
      return true;
    if (productSet.contains(instruction) && !allowProduct)
      return false;
    for (Value *operand : instruction->operands())
      if (!emit(operand, false))
        return false;
    emitted.insert(instruction);
    ordered.push_back(instruction);
    return true;
  };
  auto load = [&](unsigned group) {
    if (group == 16)
      return true;
    for (unsigned index = group * 2; index < group * 2 + 2; ++index) {
      for (auto *word : queryInputs[index].words)
        if (!emit(word, false))
          return false;
      for (auto *word : keys[index].words)
        if (!emit(word, false))
          return false;
    }
    return true;
  };
  if (!load(0))
    return 0;
  for (unsigned group = 0; group < 16; ++group) {
    for (unsigned index = group * 2; index < group * 2 + 2; ++index) {
      if (!emit(products[index], true))
        return 0;
      for (auto *word : outputs[index])
        if (!emit(word, false))
          return 0;
    }
    if (!load(group + 1))
      return 0;
  }
  for (auto *instruction : region)
    if (!emit(instruction, false))
      return 0;
  if (ordered.size() != region.size())
    return 0;
  // The complete order is checked before changing the existing SSA graph.
  for (auto *instruction : ordered)
    instruction->moveBefore(*block, end->getIterator());
  return 1;
}

} // namespace mla_nope_prefetch

inline unsigned prefetchMlaNopeFragments(llvm::Module &module, unsigned threads,
                                         unsigned shared) {
  if (threads != 512 || (shared != 167936 && shared != 168192))
    return 0;
  unsigned changed = 0;
  for (llvm::Function &function : module)
    if (!function.isDeclaration() &&
        function.getCallingConv() == llvm::CallingConv::PTX_Kernel)
      changed += mla_nope_prefetch::schedule(function, shared);
  return changed;
}

} // namespace triton::ppu
#endif
