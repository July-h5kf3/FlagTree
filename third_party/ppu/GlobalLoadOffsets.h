// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_GLOBAL_LOAD_OFFSETS_H
#define TRITON_PPU_GLOBAL_LOAD_OFFSETS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace triton::ppu {
namespace global_load_offsets {

inline std::optional<llvm::APInt>
evaluate(llvm::Value *value, unsigned tid,
         llvm::DenseMap<llvm::Value *, llvm::APInt> &cache,
         unsigned depth = 0) {
  if (depth > 32 || !value->getType()->isIntegerTy() ||
      value->getType()->getIntegerBitWidth() > 64)
    return std::nullopt;
  if (auto *constant = llvm::dyn_cast<llvm::ConstantInt>(value))
    return constant->getValue();
  auto found = cache.find(value);
  if (found != cache.end())
    return found->second;
  std::optional<llvm::APInt> result;
  if (auto *call = llvm::dyn_cast<llvm::CallInst>(value)) {
    if (auto *function = call->getCalledFunction()) {
      if (function->getName() == "llvm.nvvm.read.ptx.sreg.tid.x")
        result = llvm::APInt(32, tid);
    }
  } else if (auto *cast = llvm::dyn_cast<llvm::CastInst>(value)) {
    auto source = evaluate(cast->getOperand(0), tid, cache, depth + 1);
    if (source) {
      unsigned width = cast->getType()->getIntegerBitWidth();
      switch (cast->getOpcode()) {
      case llvm::Instruction::ZExt:
        result = source->zext(width);
        break;
      case llvm::Instruction::SExt:
        result = source->sext(width);
        break;
      case llvm::Instruction::Trunc:
        result = source->trunc(width);
        break;
      default:
        break;
      }
    }
  } else if (auto *binary = llvm::dyn_cast<llvm::BinaryOperator>(value)) {
    auto left = evaluate(binary->getOperand(0), tid, cache, depth + 1);
    auto right = evaluate(binary->getOperand(1), tid, cache, depth + 1);
    if (left && right) {
      unsigned width = left->getBitWidth();
      switch (binary->getOpcode()) {
      case llvm::Instruction::Add:
        result = *left + *right;
        break;
      case llvm::Instruction::Sub:
        result = *left - *right;
        break;
      case llvm::Instruction::Mul:
        result = *left * *right;
        break;
      case llvm::Instruction::And:
        result = *left & *right;
        break;
      case llvm::Instruction::Or:
        result = *left | *right;
        break;
      case llvm::Instruction::Xor:
        result = *left ^ *right;
        break;
      case llvm::Instruction::Shl:
        if (right->ult(width))
          result = left->shl(right->getZExtValue());
        break;
      case llvm::Instruction::LShr:
        if (right->ult(width))
          result = left->lshr(right->getZExtValue());
        break;
      case llvm::Instruction::AShr:
        if (right->ult(width))
          result = left->ashr(right->getZExtValue());
        break;
      default:
        break;
      }
    }
  }
  if (result)
    cache.try_emplace(value, *result);
  return result;
}

inline llvm::GetElementPtrInst *byteGEP(llvm::Value *value) {
  auto *gep = llvm::dyn_cast<llvm::GetElementPtrInst>(value);
  if (!gep || gep->getPointerAddressSpace() != 1 ||
      !gep->getSourceElementType()->isIntegerTy(8) || gep->getNumIndices() != 1)
    return nullptr;
  return gep;
}

struct Group {
  llvm::Value *pointer;
  std::vector<llvm::APInt> rows;
};

} // namespace global_load_offsets

inline unsigned foldGlobalLoadOffsets(llvm::Module &module, unsigned threads) {
  using namespace llvm;
  using namespace global_load_offsets;
  if (!threads || threads > 1024)
    return 0;
  unsigned changed = 0;
  const std::string address = "[ $4 + 0 ]";
  for (Function &function : module) {
    for (BasicBlock &block : function) {
      DenseMap<std::pair<Value *, Value *>, Group> groups;
      for (Instruction &instruction : block) {
        auto *call = dyn_cast<CallInst>(&instruction);
        auto *assembly =
            call ? dyn_cast<InlineAsm>(call->getCalledOperand()) : nullptr;
        if (!assembly || call->arg_size() != 2 ||
            !call->getArgOperand(1)->getType()->isIntegerTy(1) ||
            assembly->getConstraintString() != "=r,=r,=r,=r,l,b")
          continue;
        std::string code = assembly->getAsmString().str();
        auto position = code.find(address);
        if (position == std::string::npos ||
            code.find(address, position + address.size()) !=
                std::string::npos ||
            code.find("ppu.ld.global.cg.v4.b32") == std::string::npos ||
            code.find("$4") != position + 2 ||
            code.find("$4", position + address.size()) != std::string::npos)
          continue;
        Value *pointer = call->getArgOperand(0);
        auto *outer = byteGEP(pointer);
        auto *inner = outer ? byteGEP(outer->getPointerOperand()) : nullptr;
        if (!outer || !inner)
          continue;
        std::vector<APInt> rows;
        rows.reserve(threads);
        for (unsigned tid = 0; tid < threads; ++tid) {
          DenseMap<Value *, APInt> cache;
          auto row = evaluate(inner->getOperand(1), tid, cache);
          if (!row || row->getBitWidth() != 64)
            break;
          rows.push_back(*row);
        }
        if (rows.size() != threads)
          continue;
        auto key =
            std::make_pair(inner->getPointerOperand(), outer->getOperand(1));
        auto found = groups.find(key);
        if (found == groups.end()) {
          groups.try_emplace(key, Group{pointer, std::move(rows)});
          continue;
        }
        APInt delta = rows.front() - found->second.rows.front();
        if (delta.isNegative() || delta.ugt(32767))
          continue;
        bool equal = true;
        for (unsigned tid = 0; tid < threads; ++tid)
          equal &= rows[tid] - found->second.rows[tid] == delta;
        if (!equal)
          continue;
        code.replace(position, address.size(),
                     "[ $4 + " + std::to_string(delta.getZExtValue()) + " ]");
        auto *replacement = InlineAsm::get(
            assembly->getFunctionType(), code, assembly->getConstraintString(),
            assembly->hasSideEffects(), assembly->isAlignStack(),
            assembly->getDialect(), assembly->canThrow());
        call->setArgOperand(0, found->second.pointer);
        call->setCalledOperand(replacement);
        ++changed;
      }
    }
  }
  return changed;
}

} // namespace triton::ppu
#endif
