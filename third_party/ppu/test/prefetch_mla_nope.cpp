// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "PrefetchMlaNope.h"
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

// Self-contained input for CTest; real lowered modules can also be passed
// below.
static std::unique_ptr<Module> fixture(LLVMContext &context) {
  auto module = std::make_unique<Module>("mla-nope-prefetch", context);
  module->setDataLayout("e-p:64:64-p3:32:32");
  IRBuilder<NoFolder> builder(context);
  auto *shared =
      new GlobalVariable(*module, ArrayType::get(builder.getInt8Ty(), 0), false,
                         GlobalValue::ExternalLinkage, nullptr, "global_smem",
                         nullptr, GlobalVariable::NotThreadLocal, 3);
  shared->setAlignment(Align(16));
  auto *function =
      Function::Create(FunctionType::get(builder.getVoidTy(), false),
                       GlobalValue::ExternalLinkage, "decode", *module);
  function->setCallingConv(CallingConv::PTX_Kernel);
  module->getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(MDNode::get(
          context,
          {ValueAsMetadata::get(function), MDString::get(context, "smemsize"),
           ConstantAsMetadata::get(builder.getInt32(167936))}));
  builder.SetInsertPoint(BasicBlock::Create(context, "entry", function));
  auto *fragmentType =
      StructType::get(context, std::vector<Type *>(4, builder.getInt32Ty()));
  auto *loadType =
      FunctionType::get(fragmentType, {builder.getInt32Ty()}, false);
  auto *load =
      InlineAsm::get(loadType, triton::ppu::mla_nope_prefetch::MatrixAsm,
                     "=r,=r,=r,=r,r", true);
  auto matrix = [&](unsigned offset) {
    auto *pointer = builder.CreateGEP(builder.getInt8Ty(), shared,
                                      builder.getInt32(offset));
    auto *call = builder.CreateCall(
        loadType, load,
        {builder.CreatePtrToInt(pointer, builder.getInt32Ty())});
    std::array<Value *, 4> words;
    for (unsigned word = 0; word < 4; ++word)
      words[word] = builder.CreateExtractValue(call, {word});
    return words;
  };
  std::array<std::array<Value *, 4>, 16> queries;
  std::array<std::array<Value *, 4>, 32> keys;
  for (unsigned index = 0; index < queries.size(); ++index)
    queries[index] = matrix(131072 + index * 16);
  for (unsigned index = 0; index < keys.size(); ++index)
    keys[index] = matrix(index * 16);
  auto *productType =
      StructType::get(context, std::vector<Type *>(8, builder.getInt32Ty()));
  auto *mmaType = FunctionType::get(
      productType, std::vector<Type *>(16, builder.getInt32Ty()), false);
  auto *mma = InlineAsm::get(
      mmaType, triton::ppu::mla_nope_prefetch::ProductAsm,
      "=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,r,r,r,r,r,r,r,r", true);
  std::array<std::array<Value *, 8>, 96> results;
  for (unsigned index = 0; index < results.size(); ++index) {
    SmallVector<Value *> inputs;
    for (unsigned word = 0; word < 8; ++word)
      inputs.push_back(index < 2 ? builder.getInt32(0)
                                 : results[index - 2][word]);
    llvm::append_range(inputs, queries[index % 16]);
    llvm::append_range(inputs, keys[index % 32]);
    auto *call = builder.CreateCall(mmaType, mma, inputs);
    for (unsigned word = 0; word < 8; ++word)
      results[index][word] = builder.CreateExtractValue(call, {word});
  }
  auto *ropeType = FunctionType::get(builder.getVoidTy(), false);
  auto *rope = InlineAsm::get(ropeType, "ppu.mma.f32.bf16.bf16.f32", "", true);
  for (unsigned index = 0; index < 8; ++index)
    builder.CreateCall(ropeType, rope, {});
  builder.CreateRetVoid();
  return module;
}

int main(int argc, char **argv) {
  int files = std::max(argc - 1, 1);
  for (int file = 0; file < files; ++file) {
    for (unsigned test = 0; test < 10; ++test) {
      LLVMContext context;
      SMDiagnostic diagnostic;
      auto module = argc == 1
                        ? fixture(context)
                        : parseIRFile(argv[file + 1], diagnostic, context);
      if (!module) {
        diagnostic.print(argv[0], errs());
        return 1;
      }
      unsigned shared = 0;
      for (MDNode *node :
           module->getNamedMetadata("nvvm.annotations")->operands())
        if (node->getNumOperands() == 3)
          if (auto *key = dyn_cast<MDString>(node->getOperand(1)))
            if (key->getString() == "smemsize")
              shared =
                  cast<ConstantInt>(
                      cast<ConstantAsMetadata>(node->getOperand(2))->getValue())
                      ->getZExtValue();
      assert(shared == 167936 || shared == 168192);
      auto calls = products(*module);
      assert(calls.size() == 96);
      auto query = triton::ppu::mla_nope_prefetch::fragment(calls.front(), 8);
      auto key = triton::ppu::mla_nope_prefetch::fragment(calls.front(), 12);
      assert(query && key);
      unsigned threads = 512;
      if (test == 1)
        module->getGlobalVariable("global_smem")->setAlignment(Align(8));
      if (test == 2) {
        IRBuilder<> builder(calls.front());
        builder.CreateStore(builder.getInt32(0), key->pointer);
      }
      if (test == 3)
        calls[2]->setArgOperand(0,
                                ConstantInt::get(Type::getInt32Ty(context), 1));
      if (test == 4) {
        auto *assembly = cast<InlineAsm>(key->call->getCalledOperand());
        key->call->setCalledOperand(InlineAsm::get(
            assembly->getFunctionType(), "ppu.ldmatrix.unknown $0,[$4];",
            assembly->getConstraintString(), true));
      }
      if (test == 5)
        key->call->setArgOperand(0, query->call->getArgOperand(0));
      if (test == 6)
        threads = 256;
      if (test == 7)
        shared = 167680;
      if (test == 8) {
        IRBuilder<> builder(calls.front());
        auto *type = FunctionType::get(builder.getVoidTy(), false);
        builder.CreateCall(
            module->getOrInsertFunction("unknown_memory_effect", type));
      }
      if (test == 9) {
        IRBuilder<> builder(key->call->getNextNode());
        builder.CreateExtractValue(key->call, {0});
      }
      assert(!verifyModule(*module, &errs()));
      std::string before = print(*module);
      DenseMap<Instruction *, SmallVector<Value *>> operands;
      for (Function &function : *module)
        for (BasicBlock &block : function)
          for (Instruction &instruction : block)
            operands[&instruction].append(instruction.op_begin(),
                                          instruction.op_end());
      unsigned changed =
          triton::ppu::prefetchMlaNopeFragments(*module, threads, shared);
      if (changed != (test == 0 ? 1 : 0))
        errs() << module->getName() << ": case " << test
               << ", changed=" << changed << "\n";
      assert(changed == (test == 0 ? 1 : 0));
      assert(!verifyModule(*module, &errs()));
      if (test != 0) {
        assert(print(*module) == before);
      } else {
        assert(products(*module) == calls);
        unsigned count = 0;
        for (Function &function : *module)
          for (BasicBlock &block : function)
            for (Instruction &instruction : block) {
              assert(operands.contains(&instruction));
              assert(
                  llvm::equal(instruction.operands(), operands[&instruction]));
              ++count;
            }
        assert(count == operands.size());
        std::string once = print(*module);
        assert(triton::ppu::prefetchMlaNopeFragments(*module, threads,
                                                     shared) == 0);
        assert(print(*module) == once);
      }
    }
  }
  outs() << "Mla NoPe prefetch: " << files * 10 << " checks passed\n";
}
