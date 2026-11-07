// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "GatherMlaRowReductions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include <functional>

using namespace llvm;
using namespace triton::ppu;

std::string dump(Module &module) {
  std::string text;
  raw_string_ostream stream(text);
  module.print(stream, nullptr);
  return text;
}

SmallVector<mla_row_groups::Region> regions(Module &module) {
  SmallVector<mla_row_groups::Region> result;
  for (Function &function : module) {
    if (function.isDeclaration())
      continue;
    DominatorTree dominance(function);
    LoopInfo loops(dominance);
    for (BasicBlock &block : function)
      for (Instruction &instruction : block) {
        auto *call = dyn_cast<CallInst>(&instruction);
        auto *callee = call ? call->getCalledFunction() : nullptr;
        auto *loop = loops.getLoopFor(&block);
        if (!callee ||
            callee->getName() != "__ppukernel_barrier_sync_cnt_void" || !loop ||
            !readonly_shared_matrices::ctaBarrier(call, 256))
          continue;
        mla_row_groups::Region region;
        region.begin = call;
        if (mla_row_groups::collect(region, loop) &&
            mla_row_groups::prove(region, module.getDataLayout()))
          result.push_back(std::move(region));
      }
  }
  return result;
}

std::unique_ptr<Module> fixture(LLVMContext &context) {
  auto module = std::make_unique<Module>("row-gather", context);
  module->setDataLayout("e-p:64:64-p1:64:64-p3:32:32");
  auto *i32 = Type::getInt32Ty(context);
  auto *f32 = Type::getFloatTy(context);
  auto *word = FixedVectorType::get(i32, 1);
  auto *smem = new GlobalVariable(
      *module, ArrayType::get(Type::getInt8Ty(context), 0), false,
      GlobalValue::ExternalLinkage, nullptr, "global_smem", nullptr,
      GlobalVariable::NotThreadLocal, 3);
  smem->setAlignment(Align(16));
  auto *function = Function::Create(
      FunctionType::get(Type::getVoidTy(context), false),
      GlobalValue::ExternalLinkage, "dense_int8_mla_kernel", *module);
  function->setCallingConv(CallingConv::PTX_Kernel);
  for (auto [name, value] : {std::pair{"nvvm.reqntid", "256"},
                             {"ppu.mla.kv.pipeline", "1"},
                             {"ppu.mla.pv.batch", "4"},
                             {"ppu.mla.rope.prefetch", "1"}})
    function->addFnAttr(name, value);
  auto sync = module->getOrInsertFunction(
      "__ppukernel_barrier_sync_cnt_void",
      FunctionType::get(Type::getVoidTy(context), {i32, i32, i32}, false));
  cast<Function>(sync.getCallee())->addFnAttr(Attribute::Convergent);
  auto barrier = module->getOrInsertFunction(
      "llvm.nvvm.barrier.cta.sync.aligned.all",
      FunctionType::get(Type::getVoidTy(context), {i32}, false));
  auto shuffle = module->getOrInsertFunction(
      "llvm.nvvm.shfl.sync.bfly.i32",
      FunctionType::get(i32, {i32, i32, i32, i32}, false));
  auto maximum = module->getOrInsertFunction(
      "llvm.maxnum.f32", FunctionType::get(f32, {f32, f32}, false));
  auto *pointer = PointerType::get(context, 3);
  auto *storeAsm = InlineAsm::get(
      FunctionType::get(Type::getVoidTy(context),
                        {pointer, word, Type::getInt1Ty(context)}, false),
      "@$2 ppu.st.shared.b32 [ $0 + 0 ], $1;", "r,r,b", true);
  auto *loadAsm = InlineAsm::get(
      FunctionType::get(i32, {pointer, Type::getInt1Ty(context)}, false),
      "@$2 ppu.ld.shared.b32 $0, [ $1 + 0 ];", "=r,r,b", true);
  auto *entry = BasicBlock::Create(context, "entry", function);
  auto *loop = BasicBlock::Create(context, "loop", function);
  auto *exit = BasicBlock::Create(context, "exit", function);
  IRBuilder<NoFolder> builder(entry);
  auto tidFn = module->getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                           FunctionType::get(i32, false));
  Value *tid = builder.CreateCall(tidFn);
  Value *head = builder.CreateOr(
      builder.CreateLShr(builder.CreateAnd(tid, builder.getInt32(128)),
                         builder.getInt32(3)),
      builder.CreateLShr(builder.CreateAnd(tid, builder.getInt32(28)),
                         builder.getInt32(2)));
  Value *warp = builder.CreateAnd(builder.CreateLShr(tid, builder.getInt32(5)),
                                  builder.getInt32(3));
  Value *writer = builder.CreateICmpEQ(
      builder.CreateAnd(tid, builder.getInt32(3)), builder.getInt32(0));
  Value *active = builder.CreateICmpULT(tid, builder.getInt32(128));
  Value *finalWriter = builder.CreateAnd(active, writer);
  Value *sample = builder.CreateBitCast(builder.CreateSIToFP(tid, f32), word);
  builder.CreateBr(loop);
  builder.SetInsertPoint(loop);
  auto *iteration = builder.CreatePHI(i32, 2);
  iteration->addIncoming(builder.getInt32(0), entry);
  auto address = [&](unsigned field, Value *index) {
    auto *base = builder.CreateGEP(builder.getInt8Ty(), smem,
                                   builder.getInt32(81920 + field * 512));
    return builder.CreateGEP(f32, base, index);
  };
  auto sdk = [&] {
    builder.CreateCall(sync, {builder.getInt32(0), builder.getInt32(256),
                              builder.getInt32(1)});
  };
  auto emit = [&](unsigned fields) {
    sdk();
    for (unsigned field = 0; field < fields; ++field)
      for (unsigned row = 0; row < 2; ++row) {
        Value *index = builder.CreateAdd(
            builder.CreateMul(head, builder.getInt32(4)), warp);
        index = builder.CreateAdd(index, builder.getInt32(row * 32));
        builder.CreateCall(storeAsm, {address(field, index), sample, writer});
      }
    builder.CreateCall(barrier, {builder.getInt32(0)});
    for (unsigned field = 0; field < fields; ++field) {
      Value *middle = address(field, tid);
      Value *raw = builder.CreateCall(loadAsm, {middle, active});
      Value *input = builder.CreateBitCast(raw, f32);
      Value *other = builder.CreateBitCast(
          builder.CreateCall(shuffle,
                             {builder.getInt32(-1), raw, builder.getInt32(2),
                              builder.getInt32(31)}),
          f32);
      bool sum = fields == 2 && field == 0;
      Value *first = sum ? builder.CreateFAdd(input, other)
                         : builder.CreateCall(maximum, {input, other});
      Value *neighbor = builder.CreateBitCast(
          builder.CreateCall(
              shuffle, {builder.getInt32(-1), builder.CreateBitCast(first, i32),
                        builder.getInt32(1), builder.getInt32(31)}),
          f32);
      Value *last = sum ? builder.CreateFAdd(first, neighbor)
                        : builder.CreateCall(maximum, {first, neighbor});
      builder.CreateCall(
          storeAsm, {middle, builder.CreateBitCast(last, word), finalWriter});
    }
    builder.CreateCall(barrier, {builder.getInt32(0)});
    for (unsigned field = 0; field < fields; ++field)
      for (unsigned row = 0; row < 2; ++row) {
        Value *index =
            builder.CreateAdd(builder.CreateMul(head, builder.getInt32(4)),
                              builder.getInt32(row * 32));
        builder.CreateAlignedLoad(f32, address(field, index), Align(16));
      }
  };
  emit(1);
  emit(2);
  sdk();
  builder.CreateStore(builder.getInt32(0), address(0, tid));
  sdk();
  Value *next = builder.CreateAdd(iteration, builder.getInt32(1));
  iteration->addIncoming(next, loop);
  builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt32(2)), loop,
                       exit);
  builder.SetInsertPoint(exit);
  builder.CreateRetVoid();
  return module;
}

int main(int argc, char **argv) {
  if (argc != 1 && argc != 2)
    return 2;
  unsigned passed = 0;
  auto test = [&](StringRef name, std::function<void(Module &)> mutate,
                  unsigned expected, unsigned threads = 256,
                  unsigned shared = 94464) {
    LLVMContext context;
    SMDiagnostic error;
    auto module =
        argc == 2 ? parseIRFile(argv[1], error, context) : fixture(context);
    if (!module)
      report_fatal_error("cannot read real IR");
    auto original = regions(*module);
    if (original.size() != 2)
      report_fatal_error("missing original row regions");
    if (name == "actual gather and lifetime")
      errs() << "KILL "
             << mla_row_gather::killed(original[1], module->getDataLayout())
             << "\n";
    mutate(*module);
    auto before = dump(*module);
    SmallVector<std::pair<CallInst *, SmallVector<Value *>>> products;
    for (Function &function : *module)
      for (BasicBlock &block : function)
        for (Instruction &instruction : block)
          if (auto *call = dyn_cast<CallInst>(&instruction))
            if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand());
                code && code->getAsmString().contains("ppu.mma"))
              products.emplace_back(
                  call, SmallVector<Value *>(call->operands().begin(),
                                             call->operands().end()));
    unsigned beforeBarriers = 0;
    for (Function &function : *module)
      for (BasicBlock &block : function)
        for (Instruction &instruction : block)
          beforeBarriers +=
              readonly_shared_matrices::ctaBarrier(&instruction, 256);
    unsigned actual = gatherMlaRowReductions(*module, threads, shared);
    if (actual != expected || verifyModule(*module, &errs()))
      report_fatal_error(Twine("unexpected result: ") + name);
    if (!expected && dump(*module) != before)
      report_fatal_error("rejection mutated IR");
    if (expected) {
      for (auto &[call, operands] : products)
        if (!llvm::equal(call->operands(), operands))
          report_fatal_error("MMA operands changed");
      unsigned barriers = 0, vectors = 0;
      for (Function &function : *module)
        for (BasicBlock &block : function)
          for (Instruction &instruction : block) {
            barriers += readonly_shared_matrices::ctaBarrier(&instruction, 256);
            if (auto *load = dyn_cast<LoadInst>(&instruction))
              if (auto *type = dyn_cast<FixedVectorType>(load->getType());
                  type && type->getNumElements() == 4 &&
                  type->getElementType()->isFloatTy()) {
                auto address = mla_row_groups::offset(
                    load->getPointerOperand(), 0, module->getDataLayout());
                if (address && *address >= 81920 && *address < 82944) {
                  ++vectors;
                  for (unsigned tid = 0; tid < 256; ++tid) {
                    auto offset =
                        mla_row_groups::offset(load->getPointerOperand(), tid,
                                               module->getDataLayout());
                    if (!offset || *offset % 16 || *offset < 81920 ||
                        *offset + 16 > 82944)
                      report_fatal_error("gather address invalid");
                  }
                }
              }
          }
      if (barriers != beforeBarriers - 2 || vectors != 6)
        report_fatal_error("incorrect barrier/load rewrite");
      for (auto &region : original)
        for (auto &memory : region.middle)
          for (Function &function : *module)
            for (BasicBlock &block : function)
              for (Instruction &instruction : block)
                if (&instruction == memory.instruction &&
                    mla_row_groups::access(instruction))
                  report_fatal_error("middle memory survived");
      auto after = dump(*module);
      if (gatherMlaRowReductions(*module, threads, shared) ||
          dump(*module) != after)
        report_fatal_error("not idempotent");
    }
    outs() << "PASS " << name << "\n";
    ++passed;
  };
  auto unchanged = [](Module &) {};
  test("actual gather and lifetime", unchanged, 6);
  test("wrong threads", unchanged, 0, 512);
  test("wrong shared", unchanged, 0, 256, 94208);
  for (StringRef attr : {"nvvm.reqntid", "ppu.mla.kv.pipeline",
                         "ppu.mla.pv.batch", "ppu.mla.rope.prefetch"})
    test(
        attr,
        [&](Module &module) {
          regions(module)[0].begin->getFunction()->removeFnAttr(attr);
        },
        0);
  test(
      "changed shuffle",
      [](Module &module) {
        auto region = regions(module)[0];
        for (auto it = region.barriers[0]->getIterator();
             &*it != region.barriers[1]; ++it)
          if (auto *call = dyn_cast<CallInst>(&*it))
            if (auto *callee = call->getCalledFunction();
                callee && callee->getName().contains("shfl")) {
              call->setArgOperand(
                  2,
                  ConstantInt::get(Type::getInt32Ty(module.getContext()), 4));
              break;
            }
      },
      0);
  test(
      "published value escapes",
      [](Module &module) {
        auto region = regions(module)[1];
        IRBuilder<NoFolder> builder(region.end->getNextNode());
        builder.CreateLoad(builder.getFloatTy(), region.final[0].pointer);
      },
      0);
  test(
      "incomplete overwrite",
      [](Module &module) {
        auto region = regions(module)[1];
        for (Instruction *instruction = region.end->getNextNode(); instruction;
             instruction = instruction->getNextNode())
          if (auto *store = dyn_cast<StoreInst>(instruction)) {
            store->eraseFromParent();
            break;
          }
      },
      0);
  outs() << passed << " row-reduction checks passed\n";
}
