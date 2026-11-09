// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_PPU_GUARD_MLA_SCORES_H
#define TRITON_PPU_GUARD_MLA_SCORES_H

#include "PrefetchMlaNope.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"

namespace triton::ppu::mla_score_guard {
using namespace llvm;

// Recognize only the masked score row: sixteen independent FP32 selects
// between their original score and negative infinity. Keep every memory
// access, group barrier, FP operation and loop exit outside the new branch.
struct Region {
  SmallVector<SelectInst *> selects;
  Value *common = nullptr;
  Value *page = nullptr;
  Value *end = nullptr;
  SmallVector<Value *, 2> limits;
};

inline bool isScore(Instruction *inst) {
  auto *s = dyn_cast_or_null<SelectInst>(inst);
  if (!s || !s->getType()->isFloatTy())
    return false;
  auto *c = dyn_cast<ConstantFP>(s->getFalseValue());
  return c && c->getValueAPF().isInfinity() && c->isNegative();
}

inline bool match(Region &r, const DataLayout &layout) {
  SmallPtrSet<ICmpInst *, 16> cmps;
  auto collect = [&](Value *v) {
    auto *c = dyn_cast<ICmpInst>(v);
    if (!c || c->getPredicate() != CmpInst::ICMP_SLT ||
        !c->getOperand(0)->getType()->isIntegerTy(32))
      return false;
    cmps.insert(c);
    return true;
  };
  for (auto *s : r.selects) {
    if (llvm::is_contained(r.selects, s->getTrueValue()))
      return false;
    auto *op = dyn_cast<BinaryOperator>(s->getCondition());
    if (op && op->getOpcode() == Instruction::And) {
      if (!collect(op->getOperand(0)) || !collect(op->getOperand(1)))
        return false;
    } else if (!collect(s->getCondition()))
      return false;
  }
  SmallVector<ICmpInst *, 4> pages;
  // Maintain lookup order independently of pointer addresses.
  SmallVector<ICmpInst *> ordered(cmps.begin(), cmps.end());
  if (llvm::any_of(ordered, [&](auto *c) {
        return c->getParent() != r.selects.front()->getParent();
      }))
    return false;
  llvm::sort(ordered, [](auto *a, auto *b) { return a->comesBefore(b); });
  for (auto *c : ordered) {
    auto *add = dyn_cast<BinaryOperator>(c->getOperand(0));
    if (!add || (add->getOpcode() != Instruction::Add &&
                 add->getOpcode() != Instruction::Or))
      return false;
    bool token = false;
    for (unsigned a : {0u, 1u}) {
      auto *sh = dyn_cast<BinaryOperator>(add->getOperand(a));
      if (!sh || sh->getOpcode() != Instruction::Shl ||
          !PatternMatch::match(sh->getOperand(1),
                               PatternMatch::m_SpecificInt(6)))
        continue;
      auto *offset = add->getOperand(1 - a);
      for (unsigned tid = 0; tid < 256; ++tid) {
        DenseMap<Value *, Constant *> cache;
        auto *v = dyn_cast_or_null<ConstantInt>(
            mla_nope_prefetch::constantValue(offset, tid, layout, cache));
        if (!v || !v->getValue().ult(128))
          return false;
      }
      if (r.page && r.page != sh->getOperand(0))
        return false;
      r.page = sh->getOperand(0);
      if (!llvm::is_contained(r.limits, c->getOperand(1)))
        r.limits.push_back(c->getOperand(1));
      token = true;
      break;
    }
    if (!token)
      pages.push_back(c);
  }
  if (!r.page || r.limits.empty() || r.limits.size() > 2 || pages.empty())
    return false;
  for (auto *c : pages) {
    using namespace llvm::PatternMatch;
    if (!llvm::PatternMatch::match(c->getOperand(0),
                                   m_c_Add(m_Specific(r.page), m_One())) ||
        (r.end && r.end != c->getOperand(1)))
      return false;
    r.end = c->getOperand(1);
  }
  return true;
}

// A shared OR operand is an exact sufficient predicate, including at tails.
inline bool commonMask(Region &r) {
  for (unsigned i : {0u, 1u}) {
    Value *common = nullptr;
    for (auto *sel : r.selects) {
      auto *mask = dyn_cast<BinaryOperator>(sel->getCondition());
      if (!mask || mask->getOpcode() != Instruction::Or ||
          llvm::is_contained(r.selects, sel->getTrueValue()))
        return false;
      if (!common)
        common = mask->getOperand(i);
      if (!isa<ICmpInst>(common) ||
          (mask->getOperand(0) != common && mask->getOperand(1) != common)) {
        common = nullptr;
        break;
      }
      auto *rest = mask->getOperand(mask->getOperand(0) == common ? 1 : 0);
      if (isa<Constant>(rest) ||
          (isa<BinaryOperator>(rest) &&
           llvm::any_of(cast<BinaryOperator>(rest)->operands(),
                        [](Value *v) { return isa<Constant>(v); }))) {
        common = nullptr;
        break;
      }
    }
    if (common) {
      r.common = common;
      return true;
    }
  }
  return false;
}

inline unsigned run(Module &module, unsigned mode) {
  if (mode < 1 || mode > 3)
    return 0;
  auto *f = module.getFunction("dense_int8_mla_kernel");
  if (!f || f->getCallingConv() != CallingConv::PTX_Kernel ||
      !f->hasFnAttribute("nvvm.reqntid") ||
      f->getFnAttribute("nvvm.reqntid").getValueAsString() != "256" ||
      (!f->hasFnAttribute("ppu.mla.row.gather") &&
       !(mode == 3 && f->hasFnAttribute("ppu.mla.h16.kr.copy"))) ||
      f->hasFnAttribute("ppu.mla.score.guard"))
    return 0;
  Region found;
  for (auto &bb : *f) {
    Region r;
    for (auto it = bb.begin(); it != bb.end();) {
      if (!isScore(&*it)) {
        ++it;
        continue;
      }
      r = Region{};
      while (it != bb.end() && isScore(&*it))
        r.selects.push_back(cast<SelectInst>(&*it++));
      if (r.selects.size() ==
              (f->hasFnAttribute("ppu.mla.h16.kr.copy") ? 8 : 16) &&
          (mode == 3 ? commonMask(r) : match(r, module.getDataLayout()))) {
        if (!found.selects.empty())
          return 0;
        found = std::move(r);
      }
    }
  }
  if (found.selects.empty())
    return 0;
  auto *first = found.selects.front(), *last = found.selects.back();
  auto *before = first->getParent();
  // These values previously controlled data selection. Freeze each bound
  // independently before using it for control flow; a defined, failing
  // bound must still prevent the fast path when another bound is poison.
  Value *cutoff = nullptr;
  if (mode == 2) {
    DominatorTree dom(*f);
    LoopInfo loops(dom);
    auto *loop = loops.getLoopFor(before);
    auto *pre = loop ? loop->getLoopPreheader() : nullptr;
    if (!pre)
      return 0;
    auto available = [&](Value *value) {
      auto *i = dyn_cast<Instruction>(value);
      return !i || dom.dominates(i, pre->getTerminator());
    };
    if (!available(found.end) || !llvm::all_of(found.limits, available))
      return 0;
    IRBuilder<> preBuilder(pre->getTerminator());
    preBuilder.SetCurrentDebugLocation(DebugLoc());
    auto *limit = preBuilder.CreateFreeze(found.limits.front());
    if (found.limits.size() == 2) {
      auto *other = preBuilder.CreateFreeze(found.limits[1]);
      limit = preBuilder.CreateSelect(preBuilder.CreateICmpSLT(limit, other),
                                      limit, other);
    }
    auto *pages = preBuilder.CreateAShr(limit, 6);
    auto *end = preBuilder.CreateFreeze(found.end);
    auto *common = preBuilder.CreateSelect(preBuilder.CreateICmpSLT(pages, end),
                                           pages, end);
    // page + 2 <= common iff unsigned(page) < max(common, 1) - 1.
    // The token limit is signed i32, so this cutoff is at most 2^25.
    auto *clamp = preBuilder.CreateSelect(
        preBuilder.CreateICmpSGT(common, preBuilder.getInt32(1)), common,
        preBuilder.getInt32(1));
    cutoff = preBuilder.CreateSub(clamp, preBuilder.getInt32(1),
                                  "mla.score.cutoff", false, true);
  }
  auto *slow = before->splitBasicBlock(first->getIterator(), "mla.score.mask");
  auto *join =
      slow->splitBasicBlock(std::next(last->getIterator()), "mla.score.ready");
  IRBuilder<> b(before->getTerminator());
  b.SetCurrentDebugLocation(DebugLoc());
  Value *full;
  if (found.common)
    full = b.CreateFreeze(found.common);
  else if (cutoff)
    full = b.CreateICmpULT(b.CreateFreeze(found.page), cutoff);
  else {
    auto *pageEnd =
        b.CreateAdd(b.CreateZExt(b.CreateFreeze(found.page), b.getInt64Ty()),
                    b.getInt64(2));
    auto *tokenEnd = b.CreateShl(pageEnd, 6);
    auto *limit = b.CreateFreeze(found.limits.front());
    if (found.limits.size() == 2) {
      auto *other = b.CreateFreeze(found.limits[1]);
      limit = b.CreateSelect(b.CreateICmpSLT(limit, other), limit, other);
    }
    auto *visible =
        b.CreateICmpSLE(tokenEnd, b.CreateSExt(limit, b.getInt64Ty()));
    auto *inSplit = b.CreateICmpSLE(
        pageEnd, b.CreateSExt(b.CreateFreeze(found.end), b.getInt64Ty()));
    full = b.CreateAnd(visible, inSplit);
  }
  b.CreateCondBr(full, join, slow);
  before->getTerminator()->eraseFromParent();
  auto insertion = join->begin();
  for (auto *s : found.selects) {
    if (found.common) {
      auto *mask = cast<BinaryOperator>(s->getCondition());
      s->setCondition(
          mask->getOperand(mask->getOperand(0) == found.common ? 1 : 0));
    }
    auto *phi = PHINode::Create(s->getType(), 2, "mla.score", insertion);
    phi->setDebugLoc(s->getDebugLoc());
    s->replaceAllUsesWith(phi);
    phi->addIncoming(s->getTrueValue(), before);
    phi->addIncoming(s, slow);
  }
  f->addFnAttr("ppu.mla.score.guard", "1");
  return found.selects.size();
}
} // namespace triton::ppu::mla_score_guard
#endif
