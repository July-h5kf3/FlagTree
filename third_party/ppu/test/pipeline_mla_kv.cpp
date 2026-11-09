// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "PipelineMlaKv.h"
#include "PrefetchMlaRope.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
using namespace llvm;
static std::string text(Module &module) {
  std::string result;
  raw_string_ostream out(result);
  module.print(out, nullptr);
  return result;
}
static std::unique_ptr<Module> fixture(LLVMContext &context, bool h32 = false,
                                       bool wideIV = false, bool h16 = false) {
  bool narrow = h32 || h16;
  auto module = std::make_unique<Module>("mla-kv-pipeline", context);
  module->setDataLayout("e-p:64:64-p3:32:32");
  IRBuilder<NoFolder> builder(context);
  auto *shared =
      new GlobalVariable(*module, ArrayType::get(builder.getInt8Ty(), 0), false,
                         GlobalValue::ExternalLinkage, nullptr, "global_smem",
                         nullptr, GlobalVariable::NotThreadLocal, 3);
  shared->setAlignment(Align(16));
  auto *pointer = builder.getPtrTy(1);
  auto *function = Function::Create(
      FunctionType::get(builder.getVoidTy(), std::vector<Type *>(7, pointer),
                        false),
      GlobalValue::ExternalLinkage, "decode", *module);
  function->setCallingConv(CallingConv::PTX_Kernel);
  if (h32)
    function->addFnAttr("ppu.mla.pv.batch", "4");
  auto *entry = BasicBlock::Create(context, "entry", function);
  auto *header = BasicBlock::Create(context, "header", function);
  auto *body = BasicBlock::Create(context, "body", function);
  auto *exit = BasicBlock::Create(context, "exit", function);
  builder.SetInsertPoint(entry);
  auto tidFunction = module->getOrInsertFunction(
      "llvm.nvvm.read.ptx.sreg.tid.x",
      FunctionType::get(builder.getInt32Ty(), false));
  Value *tid = builder.CreateCall(tidFunction);
  Value *byte16 = builder.CreateMul(tid, builder.getInt32(16));
  Value *byte4 = builder.CreateMul(tid, builder.getInt32(4));
  builder.CreateBr(header);
  builder.SetInsertPoint(header);
  auto *ivType = narrow && wideIV ? builder.getInt64Ty() : builder.getInt32Ty();
  auto *iv = builder.CreatePHI(ivType, 2);
  iv->addIncoming(ConstantInt::get(ivType, 0), entry);
  // A second PHI supplies the deliberately invalid floating-dependency test.
  auto *stateType = FixedVectorType::get(builder.getFloatTy(), 2);
  auto *state = builder.CreatePHI(stateType, 2);
  state->addIncoming(Constant::getNullValue(stateType), entry);
  auto barrierType =
      FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()}, false);
  builder.CreateCall(module->getOrInsertFunction(
                         "llvm.nvvm.barrier.cta.sync.aligned.all", barrierType),
                     {builder.getInt32(0)});
  auto *pageType = FunctionType::get(builder.getInt32Ty(),
                                     {pointer, builder.getInt1Ty()}, false);
  auto *pageAsm = InlineAsm::get(
      pageType,
      "ppu.mov.u32 $0, 0x0;\n\t@$2 ppu.ld.global.b32 { $0 }, [ $1 + 0 ];",
      "=r,l,b", true);
  auto *copyType = FunctionType::get(
      builder.getVoidTy(), {builder.getPtrTy(3), pointer, builder.getInt32Ty()},
      false);
  auto *copyAsm = InlineAsm::get(
      copyType, "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;",
      "r,l,r", true);
  SmallVector<Value *> pages(narrow ? 2 : 4);
  for (unsigned index = 0; index < pages.size(); ++index) {
    Value *offset =
        index ? builder.CreateAdd(iv, ConstantInt::get(ivType, index)) : iv;
    auto *table =
        builder.CreateGEP(builder.getInt32Ty(), function->getArg(6), offset);
    pages[index] =
        builder.CreateCall(pageType, pageAsm, {table, builder.getTrue()});
  }
  for (unsigned packet = 0; packet < 16; ++packet) {
    auto *source = builder.CreateGEP(builder.getInt8Ty(), function->getArg(2),
                                     pages[packet / (narrow ? 8 : 4)]);
    auto *destination = builder.CreateGEP(
        builder.getInt8Ty(), shared,
        builder.CreateAdd(byte16,
                          builder.getInt32(packet * (narrow ? 4096 : 8192))));
    builder.CreateCall(copyType, copyAsm,
                       {destination, source, builder.getInt32(16)});
  }
  builder.CreateCall(module->getOrInsertFunction(
      "llvm.nvvm.cp.async.commit.group",
      FunctionType::get(builder.getVoidTy(), false)));
  builder.CreateCall(
      module->getOrInsertFunction("llvm.nvvm.cp.async.wait.group", barrierType),
      {builder.getInt32(0)});
  SmallVector<Value *> krPages;
  if (h32)
    for (auto *page : pages)
      krPages.push_back(builder.CreateCall(
          pageType, pageAsm,
          {cast<CallInst>(page)->getArgOperand(0), builder.getTrue()}));
  auto *mmaResult =
      StructType::get(context, std::vector<Type *>(8, builder.getInt32Ty()));
  auto *mmaType = FunctionType::get(
      mmaResult, std::vector<Type *>(16, builder.getInt32Ty()), false);
  auto *mma = InlineAsm::get(
      mmaType, triton::ppu::mla_nope_prefetch::ProductAsm,
      "=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,r,r,r,r,r,r,r,r", true);
  SmallVector<Value *> zero(16, builder.getInt32(0));
  for (unsigned index = 0; index < (h16 ? 16 : 32); ++index)
    builder.CreateCall(mmaType, mma, zero);
  if (h32) {
    auto *krType = FunctionType::get(
        StructType::get(context, SmallVector<Type *>(4, builder.getInt32Ty())),
        {builder.getInt32Ty(), builder.getInt32Ty(), builder.getInt32Ty(),
         builder.getInt32Ty(), pointer, builder.getInt1Ty()},
        false);
    auto *krAsm =
        InlineAsm::get(krType, triton::ppu::mla_rope_prefetch::LoadAsm,
                       "=r,=r,=r,=r,r,r,r,r,l,b", true);
    for (unsigned index = 0; index < 4; ++index) {
      auto *source = builder.CreateGEP(builder.getInt8Ty(), function->getArg(3),
                                       krPages[index / 2]);
      source = builder.CreateGEP(builder.getInt8Ty(), source,
                                 builder.getInt32(index * 16));
      builder.CreateCall(krType, krAsm,
                         {builder.getInt32(0), builder.getInt32(0),
                          builder.getInt32(0), builder.getInt32(0), source,
                          builder.getTrue()});
    }
  }
  builder.CreateBr(body);
  builder.SetInsertPoint(body);
  auto transpose = module->getOrInsertFunction(
      "__flagtree_ppu_intrinsic__mat.trans.b8",
      FunctionType::get(builder.getInt32Ty(), {builder.getInt32Ty()}, false));
  cast<Function>(transpose.getCallee())->setDoesNotAccessMemory();
  std::array<Value *, 64> words{};
  auto read = [&](unsigned index) {
    auto *address = builder.CreateGEP(
        builder.getInt8Ty(), shared,
        builder.CreateAdd(byte4,
                          builder.getInt32(index * (narrow ? 1024 : 2048))));
    auto *load = builder.CreateLoad(builder.getInt32Ty(), address);
    words[index] = builder.CreateCall(transpose, {load});
  };
  for (unsigned index = 0; index < (h16 ? 64 : 60); ++index)
    read(index);
  unsigned tailStart = h32 ? 15 : 29;
  for (unsigned index = 0; index < (h16 ? 32 : 64); ++index) {
    if (!h16 && index == tailStart)
      for (unsigned word = 60; word < 64; ++word)
        read(word);
    SmallVector<Value *> args(12, builder.getInt32(0));
    unsigned group =
        h16 ? index % 16 : (index < tailStart ? index % 15 : index % 16);
    for (unsigned word = 0; word < 4; ++word)
      args.push_back(words[group * 4 + word]);
    builder.CreateCall(mmaType, mma, args);
  }
  auto *next = builder.CreateAdd(iv, ConstantInt::get(ivType, narrow ? 2 : 4));
  auto *condition = builder.CreateICmpSLT(next, ConstantInt::get(ivType, 16));
  iv->addIncoming(next, body);
  state->addIncoming(state, body);
  builder.CreateCondBr(condition, header, exit);
  builder.SetInsertPoint(exit);
  builder.CreateRetVoid();
  return module;
}

static unsigned testH32Pages(StringRef filename = {}, bool wideIV = false) {
  unsigned passed = 0;
  for (bool reusePages : {false, true})
    for (unsigned test = 0; test < 4; ++test) {
      LLVMContext context;
      SMDiagnostic diagnostic;
      auto module = filename.empty()
                        ? fixture(context, true, wideIV)
                        : parseIRFile(filename, diagnostic, context);
      assert(module);
      SmallVector<CallInst *> products, copies, sourcePages, aliases;
      Function *kernel = nullptr;
      for (Function &function : *module)
        for (BasicBlock &block : function)
          for (Instruction &instruction : block)
            if (auto *call = dyn_cast<CallInst>(&instruction))
              if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand())) {
                if (code->getAsmString() ==
                    triton::ppu::mla_nope_prefetch::ProductAsm) {
                  products.push_back(call);
                  kernel = &function;
                }
                if (code->getAsmString() == "cp.async.cg.shared.global [ $0 + "
                                            "0 ], [ $1 + 0 ], 0x10, $2;")
                  copies.push_back(call);
              }
      assert(kernel && products.size() == 96 && copies.size() == 16);
      auto *header = products.front()->getParent();
      SmallPtrSet<Value *, 32> seen;
      std::function<void(Value *)> collect = [&](Value *value) {
        auto *instruction = dyn_cast<Instruction>(value);
        if (!instruction || instruction->getParent() != header ||
            isa<PHINode>(instruction) || !seen.insert(value).second)
          return;
        if (auto *call = dyn_cast<CallInst>(instruction))
          if (triton::ppu::mla_kv_pipeline::pageRead(call)) {
            sourcePages.push_back(call);
            return;
          }
        for (Value *operand : instruction->operands())
          collect(operand);
      };
      for (auto *copy : copies)
        collect(copy->getArgOperand(1));
      assert(sourcePages.size() == 2);
      for (Instruction &instruction : *header)
        if (auto *call = dyn_cast<CallInst>(&instruction))
          if (triton::ppu::mla_kv_pipeline::pageRead(call))
            aliases.push_back(call);
      DominatorTree dominance(*kernel);
      LoopInfo loops(dominance);
      auto *loop = loops.getLoopFor(header);
      auto *preheader = loop->getLoopPreheader();
      auto *latch = loop->getLoopLatch();
      auto *originalBranch = cast<BranchInst>(latch->getTerminator());
      auto *exit = originalBranch->getSuccessor(1);
      auto *condition = cast<ICmpInst>(originalBranch->getCondition());
      auto *next = cast<BinaryOperator>(condition->getOperand(0));
      if (test == 1) {
        sourcePages[1]->setArgOperand(0, sourcePages[0]->getArgOperand(0));
        sourcePages[1]->setArgOperand(1, sourcePages[0]->getArgOperand(1));
      }
      if (test == 2) {
        IRBuilder<> builder(copies.front());
        auto *iteration = next->getOperand(0);
        auto *index = builder.CreateAdd(
            iteration, ConstantInt::get(iteration->getType(), 2));
        auto *table =
            builder.CreateGEP(builder.getInt32Ty(), kernel->getArg(6), index);
        auto *page = builder.CreateCall(sourcePages[0]->getFunctionType(),
                                        sourcePages[0]->getCalledOperand(),
                                        {table, builder.getTrue()});
        copies.front()->setArgOperand(
            1, builder.CreateGEP(builder.getInt8Ty(), kernel->getArg(2), page));
      }
      if (test == 3) {
        IRBuilder<> builder(latch->getTerminator());
        builder.CreateStore(builder.getInt32(0), kernel->getArg(6));
      }
      assert(!verifyModule(*module, &errs()));
      auto before = text(*module);
      unsigned changed =
          triton::ppu::pipelineMlaKv(*module, 256, 94464, reusePages);
      unsigned expected =
          test == 3 || (reusePages && test != 0) ? 0 : (reusePages ? 3 : 1);
      assert(changed == expected);
      assert(!verifyModule(*module, &errs()));
      if (!changed) {
        assert(text(*module) == before);
        if (reusePages) {
          changed = triton::ppu::pipelineMlaKv(*module, 256, 94464, false);
          assert(changed == (test == 3 ? 0 : 1));
          assert(!verifyModule(*module, &errs()));
          if (!changed) {
            assert(text(*module) == before);
            ++passed;
            continue;
          }
        } else {
          ++passed;
          continue;
        }
      }
      bool pagesReused = changed == 3;
      BasicBlock *prefetch = nullptr, *continuation = nullptr;
      unsigned groupCopies = 0, commits = 0, current = 0, following = 0;
      for (BasicBlock &block : *kernel) {
        if (block.getName() == "mla.kv.prefetch")
          prefetch = &block;
        if (block.getName() == "mla.pv.cont")
          continuation = &block;
        for (Instruction &instruction : block)
          if (auto *call = dyn_cast<CallInst>(&instruction)) {
            if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand()))
              groupCopies +=
                  code->getAsmString() ==
                  "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, $2;";
            else if (call->getCalledFunction() &&
                     call->getCalledFunction()->getName() ==
                         "llvm.nvvm.cp.async.commit.group")
              ++commits;
          }
      }
      assert(prefetch && continuation && groupCopies == 32 && commits == 2);
      for (BasicBlock &block : *kernel)
        for (PHINode &phi : block.phis()) {
          if (phi.getName().starts_with("mla.kv.page.current")) {
            ++current;
            assert(&block == header && phi.getNumIncomingValues() == 2);
            auto *initial =
                cast<CallInst>(phi.getIncomingValueForBlock(preheader));
            assert(initial->getParent() == preheader &&
                   triton::ppu::mla_kv_pipeline::pageRead(initial));
            assert(isa<PHINode>(phi.getIncomingValueForBlock(continuation)));
          }
          if (phi.getName().starts_with("mla.kv.page.next")) {
            ++following;
            assert(&block == continuation && phi.getNumIncomingValues() == 2);
            auto *page = cast<CallInst>(phi.getIncomingValueForBlock(prefetch));
            assert(page->getParent() == prefetch &&
                   triton::ppu::mla_kv_pipeline::pageRead(page));
            assert(isa<PoisonValue>(phi.getIncomingValueForBlock(latch)));
          }
        }
      assert(current == (pagesReused ? 2 : 0) && following == current);
      if (pagesReused && filename.empty())
        for (auto *alias : aliases)
          assert(alias->use_empty());
      // The original next/end test gates the only following-copy block.
      // Its false edge skips that block and reaches the original loop exit.
      assert(prefetch->getSinglePredecessor() == latch);
      auto *guard = cast<BranchInst>(latch->getTerminator());
      assert(guard->isConditional() && guard->getCondition() == condition &&
             guard->getSuccessor(0) == prefetch &&
             guard->getSuccessor(1) == continuation);
      auto *backedge = cast<BranchInst>(continuation->getTerminator());
      assert(backedge->getCondition() == condition &&
             backedge->getSuccessor(0) == header &&
             backedge->getSuccessor(1) == exit);
      assert(triton::ppu::prefetchMlaRope(*module, 256, 94464) == 4);
      assert(!verifyModule(*module, &errs()));
      auto once = text(*module);
      assert(triton::ppu::pipelineMlaKv(*module, 256, 94464, reusePages) == 0);
      assert(text(*module) == once);
      ++passed;
    }
  return passed;
}

static unsigned testH16() {
  unsigned passed = 0;
  for (bool wideIV : {false, true})
    for (bool pages : {false, true})
      for (unsigned test = 0; test < 11; ++test) {
        LLVMContext context;
        auto module = fixture(context, false, wideIV, true);
        SmallVector<CallInst *> products, copies;
        CallInst *wait = nullptr;
        for (Function &function : *module)
          for (BasicBlock &block : function)
            for (Instruction &instruction : block)
              if (auto *call = dyn_cast<CallInst>(&instruction)) {
                if (auto *assembly =
                        dyn_cast<InlineAsm>(call->getCalledOperand())) {
                  if (assembly->getAsmString().contains(".s32.s8.s8.s32"))
                    products.push_back(call);
                  if (assembly->getAsmString().starts_with(
                          "cp.async.cg.shared.global"))
                    copies.push_back(call);
                } else if (call->getCalledFunction() &&
                           call->getCalledFunction()->getName() ==
                               "llvm.nvvm.cp.async.wait.group") {
                  wait = call;
                }
              }
        assert(products.size() == 48 && copies.size() == 16 && wait);
        auto *function = products.front()->getFunction();
        DominatorTree dominance(*function);
        LoopInfo loops(dominance);
        auto *loop = loops.getLoopFor(products.front()->getParent());
        auto *branch = cast<BranchInst>(loop->getLoopLatch()->getTerminator());
        auto *condition = cast<ICmpInst>(branch->getCondition());
        auto *next = cast<BinaryOperator>(condition->getOperand(0));
        auto *transpose = cast<CallInst>(products[16]->getArgOperand(12));
        auto *read = cast<LoadInst>(transpose->getArgOperand(0));
        IRBuilder<> builder(products[16]);
        if (test == 1)
          builder.CreateLoad(builder.getInt32Ty(), read->getPointerOperand());
        if (test == 2)
          builder.CreateStore(builder.getInt32(0), function->getArg(6));
        if (test == 3)
          builder.CreateCall(module->getOrInsertFunction(
              "unknown_h16_effect",
              FunctionType::get(builder.getVoidTy(), false)));
        if (test == 4)
          wait->setArgOperand(0, builder.getInt32(1));
        if (test == 5)
          next->setOperand(1, ConstantInt::get(next->getType(), 4));
        if (test == 6) {
          IRBuilder<> before(copies.front());
          copies.front()->setArgOperand(
              0, before.CreateGEP(before.getInt8Ty(),
                                  copies.front()->getArgOperand(0),
                                  before.getInt32(4)));
        }
        if (test == 7)
          read->setVolatile(true);
        if (test == 10)
          copies.front()->eraseFromParent();
        assert(!verifyModule(*module, &errs()));
        auto before = text(*module);
        unsigned rewritten = triton::ppu::pipelineMlaKv(
            *module, test == 8 ? 512 : 256, test == 9 ? 83969 : 83968, pages);
        assert(rewritten == (test == 0 ? (pages ? 3u : 1u) : 0u));
        assert(!verifyModule(*module, &errs()));
        if (!rewritten)
          assert(text(*module) == before);
        auto once = text(*module);
        assert(triton::ppu::pipelineMlaKv(*module, test == 8 ? 512 : 256,
                                          test == 9 ? 83969 : 83968,
                                          pages) == 0);
        assert(text(*module) == once);
        ++passed;
      }
  return passed;
}

static void testEqualPhis() {
  LLVMContext context;
  SMDiagnostic diagnostic;
  for (StringRef right :
       {"and i32 %tid, 3", "xor i32 %tid, -1", "add i32 %unknown, 0"}) {
    std::string source = R"(
      declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
      define i32 @test(i1 %condition, i32 %unknown) {
      entry:
        %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
        br i1 %condition, label %left, label %right
      left:
        %a = and i32 %tid, 3
        br label %merge
      right:
        %b = )" + right.str() +
                         R"(
        br label %merge
      merge:
        %joined = phi i32 [ %a, %left ], [ %b, %right ]
        ret i32 %joined
      }
    )";
    auto module = parseAssemblyString(source, diagnostic, context);
    assert(module && !verifyModule(*module, &errs()));
    auto *phi = &module->getFunction("test")->back().front();
    for (unsigned tid = 0; tid < 256; ++tid) {
      DenseMap<Value *, APInt> cache;
      auto value = triton::ppu::global_load_offsets::evaluate(phi, tid, cache);
      if (right == "and i32 %tid, 3")
        assert(value && value->getZExtValue() == (tid & 3));
      else
        assert(!value);
    }
  }
  auto module = parseAssemblyString(R"(
    define i32 @cycle(i1 %condition) {
    entry:
      br label %loop
    loop:
      %joined = phi i32 [ 0, %entry ], [ %joined, %loop ]
      br i1 %condition, label %loop, label %exit
    exit:
      ret i32 %joined
    }
  )",
                                    diagnostic, context);
  assert(module && !verifyModule(*module, &errs()));
  DenseMap<Value *, APInt> cache;
  auto *phi = &std::next(module->getFunction("cycle")->begin())->front();
  assert(!triton::ppu::global_load_offsets::evaluate(phi, 0, cache));
}

static void testEarlyCondition() {
  for (bool complete : {false, true}) {
    LLVMContext context;
    auto module = fixture(context, true);
    auto *f = module->getFunction("decode");
    DominatorTree dt(*f);
    LoopInfo loops(dt);
    auto *loop = loops.getTopLevelLoops().front();
    auto *header = loop->getHeader();
    auto *branch = cast<BranchInst>(loop->getLoopLatch()->getTerminator());
    auto *condition = cast<ICmpInst>(branch->getCondition());
    auto *next = cast<BinaryOperator>(condition->getOperand(0));
    next->moveBefore(header->getTerminator()->getIterator());
    if (complete) {
      condition->moveBefore(header->getTerminator()->getIterator());
      IRBuilder<NoFolder> b(header->getTerminator());
      b.CreateOr(condition, b.getFalse());
    }
    auto before = text(*module);
    assert(!verifyModule(*module, &errs()));
    assert(triton::ppu::pipelineMlaKv(*module, 256, 94464, true) ==
           (complete ? 3u : 0u));
    assert(!verifyModule(*module, &errs()));
    if (complete)
      assert(next->getParent() == header && condition->getParent() == header);
    else
      assert(text(*module) == before);
  }
}

int main(int argc, char **argv) {
  assert(argc == 1 || argc == 2 ||
         (argc == 3 && StringRef(argv[1]) == "--h32"));
  testEarlyCondition();
  testEqualPhis();
  outs() << "H16 pipeline:" << testH16() << " synthetic checks passed\n";
  unsigned h32Passed = testH32Pages() + testH32Pages({}, true);
  if (argc == 3) {
    unsigned realPassed = testH32Pages(argv[2]);
    outs() << "H32 page reuse:" << h32Passed << " synthetic and " << realPassed
           << " real checks passed\n";
    return 0;
  }
  for (unsigned barrierMode = 0; barrierMode < 6; ++barrierMode)
    for (bool reusePages : {false, true})
      for (unsigned test = 0; test < 13; ++test) {
        LLVMContext context;
        SMDiagnostic diagnostic;
        auto module = argc == 1 ? fixture(context)
                                : parseIRFile(argv[1], diagnostic, context);
        assert(module);
        if (barrierMode) {
          SmallVector<CallInst *> barriers;
          for (Function &function : *module)
            for (BasicBlock &block : function)
              for (Instruction &instruction : block)
                if (auto *call = dyn_cast<CallInst>(&instruction))
                  if (triton::ppu::readonly_shared_matrices::ctaBarrier(call))
                    barriers.push_back(call);
          assert(!barriers.empty());
          for (auto *barrier : barriers) {
            IRBuilder<> builder(barrier);
            auto type =
                FunctionType::get(builder.getVoidTy(),
                                  {builder.getInt32Ty(), builder.getInt32Ty(),
                                   builder.getInt32Ty()},
                                  false);
            auto builtin = module->getOrInsertFunction(
                "__ppukernel_barrier_sync_cnt_void", type);
            builder.CreateCall(builtin,
                               {builder.getInt32(barrierMode == 3 ? 1 : 0),
                                builder.getInt32(barrierMode == 4 ? 256 : 512),
                                builder.getInt32(barrierMode == 5   ? 0
                                                 : barrierMode == 2 ? 31
                                                                    : 1)});
            barrier->eraseFromParent();
          }
        }
        SmallVector<CallInst *> products, copies;
        CallInst *wait = nullptr;
        Function *kernel = nullptr;
        for (Function &f : *module)
          for (BasicBlock &b : f)
            for (Instruction &i : b)
              if (auto *call = dyn_cast<CallInst>(&i)) {
                if (auto *assembly =
                        dyn_cast<InlineAsm>(call->getCalledOperand())) {
                  if (assembly->getAsmString().contains(".s32.s8.s8.s32")) {
                    products.push_back(call);
                    kernel = &f;
                  }
                  if (assembly->getAsmString() ==
                      "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x10, "
                      "$2;")
                    copies.push_back(call);
                } else if (call->getCalledFunction() &&
                           call->getCalledFunction()->getName() ==
                               "llvm.nvvm.cp.async.wait.group")
                  wait = call;
              }
        assert(kernel && products.size() == 96 && copies.size() == 16 && wait);
        DominatorTree dominance(*kernel);
        LoopInfo loops(dominance);
        auto *loop = loops.getLoopFor(products.front()->getParent());
        auto *latch = loop->getLoopLatch();
        auto *branch = cast<BranchInst>(latch->getTerminator());
        auto *condition = cast<ICmpInst>(branch->getCondition());
        auto *next = cast<BinaryOperator>(condition->getOperand(0));
        LoadInst *last = nullptr;
        for (auto *product : ArrayRef(products).drop_front(32))
          for (unsigned index = 12; index < 16; ++index) {
            auto *call = cast<CallInst>(product->getArgOperand(index));
            auto *load = cast<LoadInst>(call->getArgOperand(0));
            if (!last || last->comesBefore(load))
              last = load;
          }
        assert(last);
        if (test == 12) {
          // The PV batch scheduler can materialize the last V fragment early.
          // A longer independent product tail must remain pipelineable.
          auto *anchor = &last->getParent()->front();
          SmallVector<Instruction *> dependencies;
          if (auto *address =
                  dyn_cast<Instruction>(last->getPointerOperand())) {
            if (auto *offset = dyn_cast<Instruction>(address->getOperand(1)))
              dependencies.push_back(offset);
            dependencies.push_back(address);
          }
          dependencies.push_back(last);
          for (User *user : last->users())
            dependencies.push_back(cast<Instruction>(user));
          for (Instruction *instruction : dependencies)
            instruction->moveBefore(anchor->getIterator());
        }
        IRBuilder<> builder(last->getNextNode());
        if (test == 1)
          builder.CreateLoad(builder.getInt32Ty(), last->getPointerOperand());
        if (test == 2)
          builder.CreateStore(builder.getInt32(0), last->getPointerOperand());
        if (test == 3) {
          auto *type = FunctionType::get(builder.getVoidTy(), false);
          builder.CreateCall(
              module->getOrInsertFunction("unknown_loop_effect", type));
        }
        if (test == 4)
          builder.CreateStore(builder.getInt32(0), kernel->getArg(6));
        if (test == 5)
          wait->setArgOperand(0, builder.getInt32(1));
        if (test == 6)
          next->setOperand(1, builder.getInt32(8));
        if (test == 7) {
          IRBuilder<> before(copies.front());
          copies.front()->setArgOperand(
              0, before.CreateGEP(before.getInt8Ty(),
                                  copies.front()->getArgOperand(0),
                                  before.getInt32(4)));
        }
        if (test == 8) {
          IRBuilder<> before(copies.front());
          auto *floating = &*std::next(loop->getHeader()->phis().begin());
          auto *value =
              before.CreateExtractElement(floating, before.getInt32(0));
          auto *index = before.CreateFPToSI(value, before.getInt64Ty());
          copies.front()->setArgOperand(
              1,
              before.CreateGEP(before.getInt8Ty(), kernel->getArg(2), index));
        }
        if (test == 11)
          condition->setPredicate(CmpInst::ICMP_SLE);
        assert(!verifyModule(*module, &errs()));
        auto before = text(*module);
        unsigned changed = triton::ppu::pipelineMlaKv(
            *module, test == 9 ? 256 : 512, test == 10 ? 167936 : 168192,
            reusePages);
        unsigned expected = (test == 0 || test == 12) && barrierMode <= 2
                                ? (reusePages ? 5u : 1u)
                                : 0u;
        if (changed != expected)
          errs() << "case " << test << " changed " << changed << "\n";
        assert(changed == expected);
        assert(!verifyModule(*module, &errs()));
        if (!changed)
          assert(text(*module) == before);
        else {
          DominatorTree tree(*kernel);
          assert(tree.verify());
          LoopInfo info(tree);
          auto *updated = info.getLoopFor(products.front()->getParent());
          assert(updated && updated->getLoopPreheader() &&
                 updated->getLoopLatch());
          unsigned groupCopies = 0, groups = 0, integerProducts = 0;
          BasicBlock *prefetch = nullptr;
          for (BasicBlock &b : *kernel) {
            if (b.getName() == "mla.kv.prefetch")
              prefetch = &b;
            for (Instruction &i : b)
              if (auto *call = dyn_cast<CallInst>(&i)) {
                if (auto *assembly =
                        dyn_cast<InlineAsm>(call->getCalledOperand())) {
                  groupCopies +=
                      assembly->getAsmString() ==
                      "cp.async.cg.shared.global [ $0 + 0 ], [ $1 + 0 "
                      "], 0x10, $2;";
                  integerProducts +=
                      assembly->getAsmString().contains(".s32.s8.s8.s32");
                } else if (call->getCalledFunction() &&
                           call->getCalledFunction()->getName() ==
                               "llvm.nvvm.cp.async.commit.group")
                  ++groups;
              }
          }
          assert(groupCopies == 32 && groups == 2 && integerProducts == 96 &&
                 prefetch);
          if (reusePages) {
            unsigned current = 0, following = 0;
            for (BasicBlock &block : *kernel)
              for (auto &phi : block.phis()) {
                current += phi.getName().starts_with("mla.kv.page.current");
                following += phi.getName().starts_with("mla.kv.page.next");
              }
            assert(current == 4 && following == 4);
          }
          assert(prefetch->getSinglePredecessor() == latch);
          auto *guard = cast<BranchInst>(latch->getTerminator());
          assert(guard->isConditional() && guard->getCondition() == condition &&
                 guard->getSuccessor(0) == prefetch);
          auto once = text(*module);
          assert(triton::ppu::pipelineMlaKv(*module, 512, 168192, reusePages) ==
                 0);
          assert(text(*module) == once);
        }
      }
  outs() << "KV pipeline:156 checks passed; H32 pages:" << h32Passed
         << " checks passed\n";
}
