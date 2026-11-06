// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "PrefetchMlaRope.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

static std::string print(Module &module) {
  std::string text;
  raw_string_ostream stream(text);
  module.print(stream, nullptr);
  return text;
}

static SmallVector<CallInst *> products(Module &module) {
  SmallVector<CallInst *> calls;
  for (Function &function : module)
    for (BasicBlock &block : function)
      for (Instruction &instruction : block)
        if (auto *call = dyn_cast<CallInst>(&instruction))
          if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
            if (assembly->getAsmString() ==
                triton::ppu::mla_nope_prefetch::ProductAsm)
              calls.push_back(call);
  return calls;
}

static std::unique_ptr<Module> fixture(LLVMContext &context, unsigned variant,
                                       unsigned threads) {
  auto module = std::make_unique<Module>("mla-rope-prefetch", context);
  IRBuilder<NoFolder> builder(context);
  auto *word = builder.getInt32Ty();
  auto *pointer = PointerType::get(context, 1);
  auto *function = Function::Create(
      FunctionType::get(builder.getVoidTy(), SmallVector<Type *>(4, pointer),
                        false),
      GlobalValue::ExternalLinkage, "decode", *module);
  function->setCallingConv(CallingConv::PTX_Kernel);
  function->addFnAttr("ppu.mla.kv.pipeline", "1");
  if (variant != 8)
    function->addFnAttr("ppu.mla.pv.batch", "4");
  auto *entry = BasicBlock::Create(context, "entry", function);
  auto *loop = BasicBlock::Create(context, "loop", function);
  auto *tail = BasicBlock::Create(context, "tail", function);
  auto *exit = BasicBlock::Create(context, "exit", function);
  builder.SetInsertPoint(entry);
  auto *base = builder.CreateGEP(builder.getInt8Ty(), function->getArg(3),
                                 builder.getInt64(32), "external.base");
  builder.CreateBr(loop);
  builder.SetInsertPoint(loop);
  auto *iteration = builder.CreatePHI(builder.getInt64Ty(), 2, "iteration");
  iteration->addIncoming(builder.getInt64(0), entry);
  auto *mma = InlineAsm::get(
      FunctionType::get(StructType::get(context, SmallVector<Type *>(8, word)),
                        SmallVector<Type *>(16, word), false),
      triton::ppu::mla_nope_prefetch::ProductAsm,
      "=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,r,r,r,r,r,r,r,r", true);
  SmallVector<Value *> arguments(16, builder.getInt32(0));
  CallInst *last = nullptr;
  for (unsigned product = 0; product < 32; ++product)
    last = builder.CreateCall(mma, arguments);
  auto *barrier = InlineAsm::get(FunctionType::get(builder.getVoidTy(), false),
                                 "barrier", "", true);
  if (variant == 1)
    builder.CreateStore(builder.getInt32(1), function->getArg(3));
  else if (variant == 2)
    builder.CreateCall(barrier);
  else if (variant == 6)
    builder.CreateCall(module->getOrInsertFunction(
        "unknown", FunctionType::get(builder.getVoidTy(), false)));
  else if (variant == 7)
    builder.CreateLoad(word, function->getArg(3))->setVolatile(true);
  Value *index = iteration;
  if (variant == 3)
    index = builder.CreateSExt(builder.CreateExtractValue(last, 0),
                               builder.getInt64Ty());
  if (variant == 4)
    base = function->getArg(2);
  auto *offset = builder.CreateShl(index, builder.getInt64(13), "late.offset");
  auto *address =
      builder.CreateGEP(builder.getInt8Ty(), base, offset, "late.address");
  SmallVector<Type *> types(4, word);
  types.push_back(pointer);
  types.push_back(builder.getInt1Ty());
  std::string constraints = "=r,=r,=r,=r,r,r,r,r,l,b";
  if (variant == 5)
    constraints += ",~{memory}";
  auto *loadAsm = InlineAsm::get(
      FunctionType::get(StructType::get(context, SmallVector<Type *>(4, word)),
                        types, false),
      triton::ppu::mla_rope_prefetch::LoadAsm, constraints, true);
  SmallVector<CallInst *> loads;
  for (unsigned index = 0; index < (threads == 256 ? 4 : 2); ++index) {
    auto *source = builder.CreateGEP(builder.getInt8Ty(), address,
                                     builder.getInt64(index * 16));
    auto *load = builder.CreateCall(
        loadAsm, {builder.getInt32(0), builder.getInt32(0), builder.getInt32(0),
                  builder.getInt32(0), source, builder.getTrue()});
    loads.push_back(load);
  }
  auto *shared = new GlobalVariable(
      *module, word, false, GlobalValue::InternalLinkage, builder.getInt32(0),
      "shared", nullptr, GlobalValue::NotThreadLocal, 3);
  for (auto *load : loads)
    builder.CreateStore(builder.CreateExtractValue(load, 0), shared);
  builder.CreateCall(barrier);
  builder.CreateBr(tail);
  builder.SetInsertPoint(tail);
  for (unsigned product = 0; product < 64; ++product)
    builder.CreateCall(mma, arguments);
  auto *next = builder.CreateAdd(iteration, builder.getInt64(2));
  iteration->addIncoming(next, tail);
  builder.CreateCondBr(builder.CreateICmpSLT(next, builder.getInt64(8)), loop,
                       exit);
  builder.SetInsertPoint(exit);
  builder.CreateRetVoid();
  return module;
}

static void check(Module &module, unsigned threads, unsigned expected) {
  assert(!verifyModule(module, &errs()));
  const auto original = print(module);
  auto originalProducts = products(module);
  assert(originalProducts.size() == 96);
  SmallVector<CallInst *> loads;
  SmallVector<SmallVector<Value *>> arguments;
  SmallVector<Instruction *> stores;
  for (BasicBlock &block : *originalProducts.front()->getFunction())
    for (Instruction &instruction : block) {
      if (auto *call = dyn_cast<CallInst>(&instruction))
        if (triton::ppu::mla_rope_prefetch::globalLoad(call)) {
          loads.push_back(call);
          arguments.emplace_back(call->args().begin(), call->args().end());
        }
      if (isa<StoreInst>(instruction))
        stores.push_back(&instruction);
    }
  auto result = triton::ppu::prefetchMlaRope(module, threads,
                                             threads == 256 ? 94464 : 123392);
  assert(result == expected);
  assert(!verifyModule(module, &errs()));
  assert(products(module) == originalProducts);
  if (!expected) {
    assert(print(module) == original);
    return;
  }
  for (unsigned index = 0; index < loads.size(); ++index) {
    assert(loads[index]->comesBefore(originalProducts.front()));
    for (unsigned argument = 0; argument < arguments[index].size(); ++argument)
      assert(loads[index]->getArgOperand(argument) ==
             arguments[index][argument]);
  }
  for (auto *store : stores)
    if (store->getParent() == loads.front()->getParent())
      assert(loads.back()->comesBefore(store));
  auto transformed = print(module);
  assert(triton::ppu::prefetchMlaRope(module, threads,
                                      threads == 256 ? 94464 : 123392) == 0);
  assert(print(module) == transformed);
}

int main(int argc, char **argv) {
  LLVMContext context;
  for (unsigned threads : {512u, 256u})
    for (unsigned variant = 0; variant < 9; ++variant) {
      auto module = fixture(context, variant, threads);
      check(*module, threads, variant == 0 && threads == 256 ? 4 : 0);
    }
  for (auto [threads, shared] :
       {std::pair{512u, 123392u}, std::pair{512u, 168192u},
        std::pair{256u, 123392u}}) {
    auto module = fixture(context, 0, threads);
    assert(triton::ppu::prefetchMlaRope(*module, threads, shared) == 0);
  }
  if (argc > 1) {
    SMDiagnostic error;
    auto module = parseIRFile(argv[1], error, context);
    if (!module) {
      error.print(argv[0], errs());
      return 1;
    }
    unsigned threads = argc > 2 ? std::stoul(argv[2]) : 512;
    check(*module, threads, threads == 256 ? 4 : 0);
    outs() << "real KR loads prefetched: " << (threads == 256 ? 4 : 0) << '\n';
  }
  outs() << "KR prefetch checks passed\n";
}
