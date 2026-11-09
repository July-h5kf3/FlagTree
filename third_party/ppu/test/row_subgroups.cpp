// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "GatherMlaRowReductions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
using namespace llvm;
using namespace triton::ppu;
static std::string text(Module &m) {
  std::string s;
  raw_string_ostream o(s);
  m.print(o, nullptr);
  return s;
}
static Function *kernel(Module &m) {
  for (auto &f : m)
    if (f.getCallingConv() == CallingConv::PTX_Kernel)
      return &f;
  return nullptr;
}
static SmallVector<mla_row_groups::Region> regions(Function &f) {
  DominatorTree dt(f);
  LoopInfo loops(dt);
  SmallVector<mla_row_groups::Region> result;
  for (auto &block : f)
    for (auto &inst : block)
      if (auto *call = dyn_cast<CallInst>(&inst)) {
        auto *callee = call->getCalledFunction();
        auto *loop = loops.getLoopFor(&block);
        if (callee &&
            callee->getName() == "__ppukernel_barrier_sync_cnt_void" && loop &&
            readonly_shared_matrices::ctaBarrier(call, 256)) {
          mla_row_groups::Region r;
          r.begin = call;
          if (mla_row_groups::collect(r, loop) &&
              mla_row_groups::prove(r, f.getParent()->getDataLayout()))
            result.push_back(std::move(r));
        }
      }
  return result;
}
int main(int argc, char **argv) {
  assert(argc == 4);
  unsigned passed = 0;
  for (int file = 1; file < argc; ++file)
    for (unsigned test = 0; test < 11; ++test) {
      std::string outputs[2];
      for (bool scoped : {false, true}) {
        LLVMContext ctx;
        SMDiagnostic diag;
        auto m = parseIRFile(argv[file], diag, ctx);
        assert(m && !verifyModule(*m, &errs()));
        auto *f = kernel(*m);
        assert(f);
        auto r = regions(*f);
        assert(r.size() == 2);
        IRBuilder<> b(r[0].begin);
        if (test == 3) {
          b.SetInsertPoint(&*f->getEntryBlock().getFirstInsertionPt());
          auto *tid = m->getFunction("llvm.nvvm.read.ptx.sreg.tid.x");
          assert(tid);
          b.CreateCall(tid);
        }
        if (test == 4) {
          b.SetInsertPoint(&*f->getEntryBlock().getFirstInsertionPt());
          b.CreateCall(r[0].begin->getCalledFunction(),
                       {b.getInt32(3), b.getInt32(128), b.getInt32(1)});
        }
        if (test == 5) {
          b.SetInsertPoint(&*f->getEntryBlock().getFirstInsertionPt());
          b.CreateCall(InlineAsm::get(FunctionType::get(b.getVoidTy(), false),
                                      "membar.cta;", "~{memory}", true));
        }
        if (test == 6 || test == 9) {
          auto &a = r[0].partial[0];
          b.SetInsertPoint(a.instruction);
          cast<CallInst>(a.instruction)
              ->setArgOperand(0, b.CreateGEP(b.getInt8Ty(), a.pointer,
                                             b.getInt32(test == 6 ? 256 : 2)));
        }
        if (test == 7)
          cast<CallInst>(r[0].partial[0].instruction)
              ->setArgOperand(2, b.getFalse());
        if (test == 8) {
          b.SetInsertPoint(r[0].partial[0].instruction);
          b.CreateStore(b.getInt32(0), f->getArg(0));
        }
        if (test == 10)
          f->addFnAttr("nvvm.reqntid", "512");
        assert(!verifyModule(*m, &errs()));
        auto before = text(*m);
        SmallVector<std::pair<CallInst *, SmallVector<Value *>>> mma;
        for (auto &block : *f)
          for (auto &inst : block)
            if (auto *c = dyn_cast<CallInst>(&inst))
              if (auto *a = dyn_cast<InlineAsm>(c->getCalledOperand());
                  a && a->getAsmString().contains("ppu.mma.sync.")) {
                SmallVector<Value *> v;
                for (Value *x : c->args())
                  v.push_back(x);
                mma.push_back({c, v});
              }
        unsigned result = gatherMlaRowReductions(*m, 256, 94464, scoped);
        unsigned expected = test <= 2 ? (scoped ? 9 : 6) : (test <= 5 ? 6 : 0);
        assert(result == expected);
        assert(!verifyModule(*m, &errs()));
        for (auto &[c, v] : mma)
          assert(llvm::equal(c->args(), v));
        if (!expected)
          assert(text(*m) == before);
        if (test == 0 && scoped) {
          SmallVector<CallInst *> barriers;
          for (auto &block : *f)
            for (auto &inst : block)
              if (inst.getMetadata("ppu.mla.row.group.barrier"))
                barriers.push_back(cast<CallInst>(&inst));
          assert(barriers.size() == 3);
          for (auto *bar : barriers) {
            for (unsigned tid = 0; tid < 256; ++tid) {
              assert(mla_row_gather::groupBarrier(bar, tid, tid & ~99u,
                                                  m->getDataLayout()));
              assert(!mla_row_gather::groupBarrier(bar, tid, tid ^ 128,
                                                   m->getDataLayout()));
            }
            auto *md = bar->getMetadata("ppu.mla.row.group.barrier");
            bar->setMetadata("ppu.mla.row.group.barrier", nullptr);
            assert(
                !mla_row_gather::groupBarrier(bar, 0, 4, m->getDataLayout()));
            bar->setMetadata("ppu.mla.row.group.barrier", md);
            auto *id = bar->getArgOperand(0);
            bar->setArgOperand(0, b.getInt32(3));
            assert(
                !mla_row_gather::groupBarrier(bar, 0, 4, m->getDataLayout()));
            bar->setArgOperand(0, id);
            bar->setArgOperand(1, b.getInt32(256));
            assert(
                !mla_row_gather::groupBarrier(bar, 0, 4, m->getDataLayout()));
            bar->setArgOperand(1, b.getInt32(128));
            bar->setArgOperand(2, b.getInt32(31));
            assert(
                !mla_row_gather::groupBarrier(bar, 0, 4, m->getDataLayout()));
            bar->setArgOperand(2, b.getInt32(1));
          }
          f->removeFnAttr("ppu.mla.row.groups128");
          assert(!mla_row_gather::groupBarrier(barriers[0], 0, 4,
                                               m->getDataLayout()));
          f->addFnAttr("ppu.mla.row.groups128", "1");
        }
        auto once = text(*m);
        assert(gatherMlaRowReductions(*m, 256, 94464, scoped) == 0);
        assert(text(*m) == once);
        outputs[scoped] = text(*m);
      }
      if (test >= 3)
        assert(outputs[0] == outputs[1]);
      ++passed;
    }
  outs() << "real-IR subgroup checks: " << passed << " passed (both modes)\n";
}
