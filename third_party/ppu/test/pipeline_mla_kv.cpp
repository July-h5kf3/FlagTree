// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "PipelineMlaKv.h"
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
static std::unique_ptr<Module> fixture(LLVMContext &context) {
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
  auto *iv = builder.CreatePHI(builder.getInt32Ty(), 2);
  iv->addIncoming(builder.getInt32(0), entry);
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
  std::array<Value *, 4> pages;
  for (unsigned index = 0; index < pages.size(); ++index) {
    Value *offset = index ? builder.CreateAdd(iv, builder.getInt32(index)) : iv;
    auto *table =
        builder.CreateGEP(builder.getInt32Ty(), function->getArg(6), offset);
    pages[index] =
        builder.CreateCall(pageType, pageAsm, {table, builder.getTrue()});
  }
  for (unsigned packet = 0; packet < 16; ++packet) {
    auto *source = builder.CreateGEP(builder.getInt8Ty(), function->getArg(2),
                                     pages[packet / 4]);
    auto *destination = builder.CreateGEP(
        builder.getInt8Ty(), shared,
        builder.CreateAdd(byte16, builder.getInt32(packet * 8192)));
    builder.CreateCall(copyType, copyAsm,
                       {destination, source, builder.getInt32(16)});
  }
  builder.CreateCall(module->getOrInsertFunction(
      "llvm.nvvm.cp.async.commit.group",
      FunctionType::get(builder.getVoidTy(), false)));
  builder.CreateCall(
      module->getOrInsertFunction("llvm.nvvm.cp.async.wait.group", barrierType),
      {builder.getInt32(0)});
  auto *mmaResult =
      StructType::get(context, std::vector<Type *>(8, builder.getInt32Ty()));
  auto *mmaType = FunctionType::get(
      mmaResult, std::vector<Type *>(16, builder.getInt32Ty()), false);
  auto *mma = InlineAsm::get(
      mmaType, triton::ppu::mla_nope_prefetch::ProductAsm,
      "=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,r,r,r,r,r,r,r,r", true);
  SmallVector<Value *> zero(16, builder.getInt32(0));
  for (unsigned index = 0; index < 32; ++index)
    builder.CreateCall(mmaType, mma, zero);
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
        builder.CreateAdd(byte4, builder.getInt32(index * 2048)));
    auto *load = builder.CreateLoad(builder.getInt32Ty(), address);
    words[index] = builder.CreateCall(transpose, {load});
  };
  for (unsigned index = 0; index < 60; ++index)
    read(index);
  for (unsigned index = 0; index < 64; ++index) {
    if (index == 29)
      for (unsigned word = 60; word < 64; ++word)
        read(word);
    SmallVector<Value *> args(12, builder.getInt32(0));
    unsigned group = index < 29 ? index % 15 : index % 16;
    for (unsigned word = 0; word < 4; ++word)
      args.push_back(words[group * 4 + word]);
    builder.CreateCall(mmaType, mma, args);
  }
  auto *next = builder.CreateAdd(iv, builder.getInt32(4));
  auto *condition = builder.CreateICmpSLT(next, builder.getInt32(16));
  iv->addIncoming(next, body);
  state->addIncoming(state, body);
  builder.CreateCondBr(condition, header, exit);
  builder.SetInsertPoint(exit);
  builder.CreateRetVoid();
  return module;
}

int main(int argc, char **argv) {
  assert(argc == 1 || argc == 2);
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
  outs() << "KV pipeline:156 checks passed\n";
}
