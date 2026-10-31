// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "EarlySharedFloatScales.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <fstream>

using namespace llvm;

static std::string printModule(Module &module) {
  std::string text;
  raw_string_ostream stream(text);
  module.print(stream, nullptr);
  return text;
}

int main(int argc, char **argv) {
  assert(argc == 3);
  unsigned passed = 0;
  for (unsigned test = 0; test < 15; ++test) {
    LLVMContext context;
    SMDiagnostic diagnostic;
    auto module = parseIRFile(argv[1], diagnostic, context);
    if (!module) {
      diagnostic.print(argv[0], errs());
      return 1;
    }
    SmallVector<CallInst *> products;
    for (Function &function : *module)
      for (BasicBlock &block : function)
        for (Instruction &instruction : block)
          if (auto *call = dyn_cast<CallInst>(&instruction))
            if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
              if (assembly->getAsmString().contains(".s32.s8.s8.s32"))
                products.push_back(call);
    assert(products.size() == 96);
    auto *pv = products.back()->getParent();
    SmallVector<StoreInst *> stores;
    SmallVector<LoadInst *> reads;
    for (Instruction *instruction = products.back()->getNextNode();
         instruction != pv->getTerminator();
         instruction = instruction->getNextNode()) {
      if (auto *store = dyn_cast<StoreInst>(instruction))
        stores.push_back(store);
      if (auto *load = dyn_cast<LoadInst>(instruction))
        reads.push_back(load);
    }
    assert(stores.size() == 2 && !reads.empty());
    unsigned threads = 512, shared = 217088;
    IRBuilder<> builder(products.front());
    switch (test) {
    case 0:
      break;
    case 1:
      threads = 256;
      break;
    case 2:
      shared = 217056;
      break;
    case 3:
      stores[0]->setVolatile(true);
      break;
    case 4:
      reads[0]->setVolatile(true);
      break;
    case 5: {
      auto *type = FunctionType::get(builder.getVoidTy(), false);
      auto *clobber = InlineAsm::get(type, "", "~{memory}", true);
      builder.CreateCall(type, clobber, {});
      break;
    }
    case 6: {
      auto writer =
          module->getOrInsertFunction("unknown_writer", builder.getVoidTy());
      builder.CreateCall(writer);
      break;
    }
    case 7: {
      auto *base = module->getGlobalVariable("global_smem");
      auto *outside = builder.CreateGEP(builder.getInt8Ty(), base,
                                        builder.getInt32(217088));
      builder.CreateStore(builder.getInt32(0), outside);
      break;
    }
    case 8: {
      SmallVector<Instruction *> barriers;
      for (Instruction &instruction : *products.front()->getParent()) {
        if (&instruction == products.front())
          break;
        if (triton::ppu::early_shared_float_scales::barrier(&instruction))
          barriers.push_back(&instruction);
      }
      for (auto *barrier : barriers)
        barrier->eraseFromParent();
      break;
    }
    case 9: {
      IRBuilder<> late(stores[1]);
      late.CreateStore(ConstantFP::get(builder.getFloatTy(), 0),
                       stores[1]->getPointerOperand());
      break;
    }
    case 10: {
      BinaryOperator *multiply = nullptr;
      for (Instruction *instruction = products.back()->getNextNode();
           instruction != pv->getTerminator();
           instruction = instruction->getNextNode())
        if (instruction->getOpcode() == Instruction::FMul &&
            instruction->getType()->isFloatTy()) {
          multiply = cast<BinaryOperator>(instruction);
          break;
        }
      assert(multiply);
      auto *add = BinaryOperator::CreateFAdd(multiply->getOperand(0),
                                             multiply->getOperand(1), "changed",
                                             multiply);
      multiply->replaceAllUsesWith(add);
      multiply->eraseFromParent();
      break;
    }
    case 11: {
      auto *annotations = module->getNamedMetadata("nvvm.annotations");
      MDNode *shared = nullptr;
      for (MDNode *entry : annotations->operands())
        if (entry->getNumOperands() == 3)
          if (auto *name = dyn_cast_or_null<MDString>(entry->getOperand(1)))
            if (name->getString() == "smemsize")
              shared = entry;
      assert(shared);
      annotations->addOperand(shared);
      break;
    }
    case 12: {
      auto *call = products.front();
      auto *old = cast<InlineAsm>(call->getCalledOperand());
      std::string text = old->getAsmString().str();
      auto at = text.find(".s32.s8.s8.s32");
      text.replace(at, std::string(".s32.s8.s8.s32").size(), ".s32.u8.s8.s32");
      auto *changed = InlineAsm::get(old->getFunctionType(), text,
                                     old->getConstraintString(), true);
      call->setCalledOperand(changed);
      break;
    }
    case 14: {
      IRBuilder<> inside(products[40]);
      auto *output = products[40]->getFunction()->getArg(11);
      assert(output->getType()->getPointerAddressSpace() == 1);
      inside.CreateStore(inside.getInt32(0), output);
      break;
    }
    case 13:
      reads[0]->setAlignment(Align(32));
      break;
    }
    assert(!verifyModule(*module, &errs()));
    auto before = printModule(*module);
    auto result =
        triton::ppu::publishSharedFloatScalesEarly(*module, threads, shared);
    if (test == 0) {
      assert(result.first == 217344 && result.second == 1);
      assert(!verifyModule(*module, &errs()));
      auto after = printModule(*module);
      assert(after != before);
      auto repeated = triton::ppu::publishSharedFloatScalesEarly(*module, 512,
                                                                 result.first);
      assert(repeated.first == result.first && repeated.second == 0);
      assert(printModule(*module) == after);
      std::ofstream(argv[2]) << after;
    } else {
      assert(result.second == 0 && printModule(*module) == before);
    }
    ++passed;
  }
  outs() << "{\"passed\":" << passed
         << ",\"positive\":1,\"rejected_without_mutation\":14,\"hardware_"
            "dispatch\":false}\n";
}
