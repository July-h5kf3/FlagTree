// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "ScheduleMlaPvTail.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
template <class T> static std::string text(const T &value) {
  std::string out;
  raw_string_ostream stream(out);
  value.print(stream);
  return out;
}
static std::string text(const Module &module) {
  std::string out;
  raw_string_ostream stream(out);
  module.print(stream, nullptr);
  return out;
}
static std::unique_ptr<Module> fixture(LLVMContext &context) {
  auto module = std::make_unique<Module>("mla-pv-tail", context);
  IRBuilder<NoFolder> builder(context);
  auto *type = FunctionType::get(
      builder.getVoidTy(),
      {builder.getFloatTy(), builder.getFloatTy(), builder.getPtrTy(1)}, false);
  auto *kernel =
      Function::Create(type, GlobalValue::ExternalLinkage, "decode", *module);
  kernel->setCallingConv(CallingConv::PTX_Kernel);
  kernel->addFnAttr("ppu.mla.kv.pipeline", "1");
  auto *entry = BasicBlock::Create(context, "entry", kernel);
  auto *tail = BasicBlock::Create(context, "mla.pv.cont", kernel);
  auto *exit = BasicBlock::Create(context, "exit", kernel);
  builder.SetInsertPoint(entry);
  builder.CreateBr(tail);
  builder.SetInsertPoint(tail);
  auto *resultType =
      StructType::get(context, std::vector<Type *>(8, builder.getInt32Ty()));
  auto *mmaType = FunctionType::get(
      resultType, std::vector<Type *>(16, builder.getInt32Ty()), false);
  auto *mma = InlineAsm::get(
      mmaType, triton::ppu::mla_nope_prefetch::ProductAsm,
      "=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,r,r,r,r,r,r,r,r", true);
  std::array<std::array<Value *, 8>, 4> chains;
  for (auto &chain : chains)
    chain.fill(builder.getInt32(0));
  for (unsigned index = 0; index < 35; ++index) {
    auto &chain = chains[index % 4];
    SmallVector<Value *> arguments(chain.begin(), chain.end());
    for (unsigned word = 0; word < 8; ++word)
      arguments.push_back(builder.getInt32(word + 1));
    auto *product = builder.CreateCall(mmaType, mma, arguments);
    for (unsigned word = 0; word < 8; ++word)
      chain[word] = builder.CreateExtractValue(product, {word});
  }
  Value *packed =
      PoisonValue::get(FixedVectorType::get(builder.getFloatTy(), 32));
  for (unsigned lane = 0; lane < 32; ++lane) {
    Value *integer = chains[lane / 8][lane % 8];
    Value *converted = builder.CreateSIToFP(integer, builder.getFloatTy());
    Value *scaled = builder.CreateFMul(converted, kernel->getArg(0));
    Value *previous = builder.CreateFMul(
        ConstantFP::get(builder.getFloatTy(), 2.0), kernel->getArg(1));
    Value *updated = builder.CreateFAdd(previous, scaled, "pv_stream_scalar");
    FastMathFlags flags;
    flags.setAllowContract();
    cast<Instruction>(updated)->setFastMathFlags(flags);
    packed = builder.CreateInsertElement(
        packed, updated, builder.getInt64(lane), "pv_stream_pack");
  }
  builder.CreateBr(exit);
  builder.SetInsertPoint(exit);
  builder.CreateStore(packed, kernel->getArg(2));
  builder.CreateRetVoid();
  return module;
}

int main(int argc, char **argv) {

  unsigned checks = 0;
  for (int file = 0; file < std::max(argc - 1, 1); ++file)
    for (unsigned test = 0; test < 8; ++test) {
      LLVMContext context;
      SMDiagnostic diagnostic;
      auto module = argc == 1
                        ? fixture(context)
                        : parseIRFile(argv[file + 1], diagnostic, context);
      if (!module) {
        diagnostic.print(argv[0], errs());
        return 1;
      }
      Function *kernel = nullptr;
      BasicBlock *tail = nullptr;
      InsertElementInst *pack = nullptr;
      for (Function &function : *module)
        if (function.hasFnAttribute("ppu.mla.kv.pipeline")) {
          kernel = &function;
          for (BasicBlock &block : function)
            if (block.getName() == "mla.pv.cont")
              tail = &block;
        }
      assert(kernel && tail);
      for (Instruction &instruction : *tail)
        if (auto *insert = dyn_cast<InsertElementInst>(&instruction))
          if (insert->getName().starts_with("pv_stream_pack"))
            pack = insert;
      assert(pack);
      if (test == 3)
        kernel->removeFnAttr("ppu.mla.kv.pipeline");
      IRBuilder<> builder(&*tail->getFirstNonPHIIt());
      if (test == 4)
        builder.CreateCall(module->getOrInsertFunction(
            "unknown_effect", FunctionType::get(builder.getVoidTy(), false)));
      if (test == 5)
        builder.CreateStore(builder.getInt32(0),
                            PoisonValue::get(builder.getPtrTy(3)));
      if (test == 6)
        builder.CreateCall(
            module->getOrInsertFunction(
                "llvm.nvvm.barrier.cta.sync.aligned.all",
                FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()},
                                  false)),
            {builder.getInt32(0)});
      if (test == 7)
        pack->setOperand(2, builder.getInt64(30));
      assert(!verifyModule(*module, &errs()));
      std::string before = text(*module);
      DenseMap<Instruction *, Instruction *> instructions;
      for (BasicBlock &block : *kernel)
        for (Instruction &instruction : block)
          instructions[&instruction] = instruction.clone();
      unsigned changed = triton::ppu::scheduleMlaPvTail(
          *module, test == 1 ? 256 : 512, test == 2 ? 168196 : 168192);
      assert(changed == (test == 0 ? 1u : 0u));
      unsigned count = 0;
      for (BasicBlock &block : *kernel)
        for (Instruction &instruction : block) {
          assert(instruction.isIdenticalTo(instructions.lookup(&instruction)));
          ++count;
        }
      assert(count == instructions.size());
      for (auto &entry : instructions)
        entry.second->dropAllReferences();
      for (auto &entry : instructions)
        entry.second->deleteValue();
      assert(!verifyModule(*module, &errs()));
      if (!changed) {
        assert(text(*module) == before);
      } else {
        DominatorTree tree(*kernel);
        assert(tree.verify());
        std::string once = text(*module);
        assert(triton::ppu::scheduleMlaPvTail(*module, 512, 168192) == 0);
        assert(text(*module) == once);
      }
      ++checks;
    }
  outs() << "PV tail schedule: " << checks << " checks passed\n";
}
