// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "VersionMlaTiles.h"
#include "PrefetchMlaNope.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace triton::ppu::mla_full_tiles {
// The pipeline has a uniform page IV. On complete pairs of pages we can
// specialize the existing loop once per CTA, including its prefetch path.
unsigned run(llvm::Function &f, bool q2Tail) {
  using namespace llvm;
  if (f.getCallingConv() != CallingConv::PTX_Kernel ||
      f.getFnAttribute("nvvm.reqntid").getValueAsString() != "256" ||
      !f.hasFnAttribute("ppu.mla.kv.pipeline") ||
      !f.hasFnAttribute("ppu.mla.row.gather") ||
      f.hasFnAttribute("ppu.mla.full.tiles"))
    return 0;
  DominatorTree dt(f);
  LoopInfo li(dt);
  if (li.getTopLevelLoops().size() != 1)
    return 0;
  auto *loop = li.getTopLevelLoops().front();
  auto *pre = loop->getLoopPreheader(), *latch = loop->getLoopLatch(),
       *exit = loop->getExitBlock();
  if (!pre || !latch || !exit || !loop->isInnermost() || !pre->phis().empty() ||
      !pre->getSinglePredecessor() || exit->getSinglePredecessor() != latch ||
      !exit->getSingleSuccessor())
    return 0;
  auto *back = dyn_cast<BranchInst>(latch->getTerminator());
  if (!back || !back->isConditional() ||
      back->getSuccessor(0) != loop->getHeader() ||
      back->getSuccessor(1) != exit)
    return 0;
  auto *cond = dyn_cast<ICmpInst>(back->getCondition());
  if (!cond || cond->getPredicate() != CmpInst::ICMP_SLT)
    return 0;
  Value *end = cond->getOperand(1);
  auto *next = dyn_cast<BinaryOperator>(cond->getOperand(0));
  if (!next || next->getOpcode() != Instruction::Add ||
      !next->getType()->isIntegerTy(32) ||
      !match(next->getOperand(1), PatternMatch::m_SpecificInt(2)))
    return 0;
  auto *iv = dyn_cast<PHINode>(next->getOperand(0));
  if (!iv || iv->getParent() != loop->getHeader() ||
      iv->getIncomingValueForBlock(latch) != next ||
      iv->getNumIncomingValues() != 2)
    return 0;
  Value *start = iv->getIncomingValueForBlock(pre);
  auto available = [&](Value *v) {
    auto *i = dyn_cast<Instruction>(v);
    return !i || dt.dominates(i, pre->getSinglePredecessor()->getTerminator());
  };
  if (!available(start) || !available(end))
    return 0;
  using namespace llvm::PatternMatch;
  Value *pages = nullptr, *split = nullptr, *length = nullptr;
  if (!match(start, m_c_Mul(m_Value(pages), m_Value(split))))
    return 0;
  if (!match(end,
             m_c_Mul(m_Specific(pages), m_c_Add(m_Specific(split), m_One())))) {
    std::swap(pages, split);
    if (!match(end,
               m_c_Mul(m_Specific(pages), m_c_Add(m_Specific(split), m_One()))))
      return 0;
  }
  if (!match(pages, m_SDiv(m_c_Add(m_Value(length), m_SpecificInt(63)),
                           m_SpecificInt(64))))
    return 0;
  auto *splitId = dyn_cast<CallInst>(split);
  if (!splitId || !splitId->getCalledFunction() ||
      splitId->getCalledFunction()->getName() !=
          "llvm.nvvm.read.ptx.sreg.ctaid.z")
    return 0;
  SmallVector<BasicBlock *> region;
  for (auto &bb : f)
    if (&bb == pre || &bb == exit || loop->contains(&bb))
      region.push_back(&bb);
  SmallPtrSet<BasicBlock *, 32> inside(region.begin(), region.end());
  auto *join = exit->getSingleSuccessor();
  for (auto *bb : region) {
    if (bb->hasAddressTaken() || bb->isEHPad())
      return 0;
    for (auto &i : *bb)
      for (Use &u : i.uses()) {
        auto *user = dyn_cast<Instruction>(u.getUser());
        if (!user || inside.contains(user->getParent()))
          continue;
        auto *phi = dyn_cast<PHINode>(user);
        if (!phi || phi->getParent() != join ||
            phi->getIncomingBlock(u) != exit)
          return 0;
      }
  }
  auto &dl = f.getParent()->getDataLayout();
  DenseMap<Value *, unsigned> offsets;
  auto maxOffset = [&](Value *v) {
    if (!offsets.count(v)) {
      unsigned upper = 0;
      for (unsigned tid = 0; tid < 256; ++tid) {
        DenseMap<Value *, Constant *> cache;
        auto *c = dyn_cast_or_null<ConstantInt>(
            mla_nope_prefetch::constantValue(v, tid, dl, cache));
        if (!c || !c->getValue().ult(128)) {
          upper = 128;
          break;
        }
        upper = std::max(upper, unsigned(c->getZExtValue()));
      }
      offsets[v] = upper;
    }
    return offsets.lookup(v);
  };
  auto validPage = [&](Value *page, Instruction &at) {
    if (page == start || page == iv)
      return true;
    if (page != next)
      return false;
    for (auto *u : cond->users())
      if (auto *br = dyn_cast<BranchInst>(u);
          br && br != back && br->isConditional() &&
          br->getCondition() == cond &&
          br->getSuccessor(0) != br->getSuccessor(1) &&
          dt.dominates(BasicBlockEdge(br->getParent(), br->getSuccessor(0)),
                       at.getParent()))
        return true;
    return false;
  };
  SmallVector<ICmpInst *> checks;
  CallInst *head = nullptr;
  for (auto &i : f.getEntryBlock())
    if (auto *c = dyn_cast<CallInst>(&i);
        c && c->getCalledFunction() &&
        c->getCalledFunction()->getName() ==
            "llvm.nvvm.read.ptx.sreg.ctaid.x") {
      if (head)
        return 0;
      head = c;
    }
  if (!head || !available(head))
    return 0;
  bool boundedHead = false;
  DenseMap<Value *, unsigned> allowed;
  DenseMap<Value *, Value *> queryOffsets;
  struct Tail {
    ICmpInst *cmp;
    Value *page, *off, *query;
  };
  SmallVector<Tail> tails;
  auto tokenCount = [&](Value *rhs) {
    if (rhs == length)
      return 128u;
    if (allowed.count(rhs))
      return allowed.lookup(rhs);
    auto *clamp = dyn_cast<IntrinsicInst>(rhs);
    Value *offset = nullptr;
    if (!clamp || clamp->getIntrinsicID() != Intrinsic::smax ||
        !match(clamp->getArgOperand(1), m_Zero()) ||
        !match(clamp->getArgOperand(0),
               m_c_Add(m_Specific(length), m_Value(offset))))
      return allowed[rhs] = 0u;
    unsigned count = 128;
    for (unsigned tile = 0; tile < 8; ++tile)
      for (unsigned tid = 0; tid < 256; ++tid) {
        DenseMap<Value *, Constant *> cache;
        cache[head] = ConstantInt::get(head->getType(), tile);
        auto *v = dyn_cast_or_null<ConstantInt>(
            mla_nope_prefetch::constantValue(offset, tid, dl, cache));
        if (!v || v->getSExtValue() < -1 || v->getSExtValue() >= 32)
          return allowed[rhs] = 0u;
        if (v->isMinusOne()) {
          if (!q2Tail)
            return allowed[rhs] = 0u;
          count = 127;
        }
      }
    boundedHead = true;
    queryOffsets[rhs] = offset;
    return allowed[rhs] = count;
  };
  for (auto *bb : region)
    for (auto &i : *bb) {
      auto *c = dyn_cast<ICmpInst>(&i);
      if (!c || c == cond || c->getPredicate() != CmpInst::ICMP_SLT ||
          !c->getOperand(0)->getType()->isIntegerTy(32) ||
          !available(c->getOperand(1)))
        continue;
      auto *b = dyn_cast<BinaryOperator>(c->getOperand(0));
      if (!b || (b->getOpcode() != Instruction::Add &&
                 b->getOpcode() != Instruction::Or))
        continue;
      for (unsigned a : {0u, 1u}) {
        auto *base = b->getOperand(a), *off = b->getOperand(1 - a);
        if (c->getOperand(1) == end && b->getOpcode() == Instruction::Add &&
            match(off, PatternMatch::m_SpecificInt(1)) && validPage(base, i)) {
          checks.push_back(c);
          break;
        }
        auto *sh = dyn_cast<BinaryOperator>(base);
        if (!sh || sh->getOpcode() != Instruction::Shl ||
            !match(sh->getOperand(1), PatternMatch::m_SpecificInt(6)) ||
            !validPage(sh->getOperand(0), i))
          continue;
        unsigned upper = maxOffset(off), count = tokenCount(c->getOperand(1));
        if (upper >= count) {
          if (upper == 127 && count == 127 && sh->getOperand(0) == iv &&
              available(off) &&
              available(queryOffsets.lookup(c->getOperand(1))))
            tails.push_back(
                {c, iv, off, queryOffsets.lookup(c->getOperand(1))});
          continue;
        }
        checks.push_back(c);
        break;
      }
    }
  if (checks.size() < 16)
    return 0;
  auto &ctx = f.getContext();
  // The selected length must be the same in every lane. Reject unknown
  // address producers, divergent PHIs and unrecognized loads.
  DenseMap<Value *, bool> proved;
  std::function<bool(Value *)> uniform = [&](Value *v) -> bool {
    if (isa<Argument, GlobalVariable, ConstantInt>(v))
      return true;
    if (auto it = proved.find(v); it != proved.end())
      return it->second;
    proved[v] = false;
    auto *i = dyn_cast<Instruction>(v);
    if (!i)
      return false;
    bool ok = false;
    if (auto *phi = dyn_cast<PHINode>(i);
        phi && phi->getNumIncomingValues() == 2) {
      for (unsigned k : {0u, 1u}) {
        auto *arm = phi->getIncomingBlock(k),
             *entry = phi->getIncomingBlock(1 - k);
        auto *br = dyn_cast<BranchInst>(entry->getTerminator());
        if (arm->getSinglePredecessor() == entry &&
            arm->getSingleSuccessor() == phi->getParent() && br &&
            br->isConditional() && uniform(br->getCondition()) &&
            uniform(phi->getIncomingValue(0)) &&
            uniform(phi->getIncomingValue(1)))
          ok = true;
      }
    } else if (auto *ld = dyn_cast<LoadInst>(i)) {
      auto *bar = dyn_cast_or_null<CallInst>(i->getPrevNode());
      ok = ld->isSimple() && ld->getPointerAddressSpace() == 3 &&
           isa<GlobalVariable>(ld->getPointerOperand()) && bar &&
           bar->getCalledFunction() &&
           bar->getCalledFunction()->getName() ==
               "llvm.nvvm.barrier.cta.sync.aligned.all";
    } else if (auto *c = dyn_cast<CallInst>(i)) {
      if (auto *callee = c->getCalledFunction()) {
        ok = callee->getName().starts_with("llvm.nvvm.read.ptx.sreg.ctaid.") ||
             (callee->getIntrinsicID() == Intrinsic::smax &&
              uniform(c->getArgOperand(0)) && uniform(c->getArgOperand(1)));
      } else if (auto *asmOp = dyn_cast<InlineAsm>(c->getCalledOperand())) {
        ok = asmOp->getAsmString() ==
                 "ppu.mov.u32 $0, 0x0;\n\tppu.ld.global.b32 { $0 }, [ $1 + 0 "
                 "];" &&
             asmOp->getConstraintString() == "=r,l" && c->arg_size() == 1 &&
             uniform(c->getArgOperand(0));
      }
    } else if (i->isBinaryOp() ||
               isa<CastInst, ICmpInst, GetElementPtrInst>(i)) {
      ok = llvm::all_of(i->operands(), [&](Value *op) { return uniform(op); });
    }
    return proved[v] = ok;
  };
  if (!uniform(length))
    return 0;
  auto *pred = pre->getSinglePredecessor();
  ValueToValueMapTy map;
  for (auto *bb : region) {
    auto *clone = CloneBasicBlock(bb, map, ".full", &f);
    map[bb] = clone;
    clone->moveBefore(join);
  }
  for (auto *bb : region)
    for (auto &i : *cast<BasicBlock>(map[bb]))
      RemapInstruction(&i, map,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
  for (auto &phi : join->phis()) {
    auto *value = phi.getIncomingValueForBlock(exit);
    phi.addIncoming(map.count(value) ? static_cast<Value *>(map[value]) : value,
                    cast<BasicBlock>(map[exit]));
  }
  for (auto *c : checks)
    cast<Instruction>(map[c])->replaceAllUsesWith(ConstantInt::getTrue(ctx));
  auto *dispatch = BasicBlock::Create(ctx, "mla.tile.dispatch", &f, pre);
  pred->getTerminator()->replaceSuccessorWith(pre, dispatch);
  IRBuilder<> b(dispatch);
  // With split zero and an aligned nonnegative length, the loop visits
  // page pairs [0, length/64). The upper bound also excludes signed
  // overflow in ceil-div and in the proven nonnegative query offset.
  Value *aligned =
      b.CreateICmpEQ(b.CreateAnd(length, b.getInt32(127)), b.getInt32(0));
  Value *fits = b.CreateICmpULE(length, b.getInt32(1 << 30));
  Value *zero = b.CreateICmpEQ(split, b.getInt32(0));
  Value *valid = boundedHead
                     ? b.CreateAnd(zero, b.CreateICmpULT(head, b.getInt32(8)))
                     : zero;
  Value *ok = b.CreateAnd(b.CreateAnd(fits, aligned), valid);
  for (auto tail : tails) {
    // In a full tile the sole excluded token is length-1 for query offset -1.
    auto *inLane = b.CreateAnd(b.CreateICmpEQ(tail.off, b.getInt32(127)),
                               b.CreateICmpEQ(tail.query, b.getInt32(-1)));
    auto *lastPage = b.CreateSub(end, b.getInt32(2));
    auto *cloned = cast<Instruction>(map[tail.cmp]);
    IRBuilder<> at(cloned);
    auto *inLast = at.CreateICmpEQ(map[tail.page], lastPage);
    cloned->replaceAllUsesWith(at.CreateNot(at.CreateAnd(inLast, inLane)));
  }
  b.CreateCondBr(ok, cast<BasicBlock>(map[pre]), pre);
  if (!tails.empty())
    f.addFnAttr("ppu.mla.last.token", "1");
  f.addFnAttr("ppu.mla.full.tiles", "1");
  return checks.size();
}
} // namespace triton::ppu::mla_full_tiles
