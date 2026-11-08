// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "AsyncMlaH16Kr.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
using namespace llvm;

static std::string text(Module &module) {
  std::string result;
  raw_string_ostream stream(result);
  module.print(stream, nullptr);
  return result;
}

// Pass actual pre-transform LLVM IR fixtures as arguments. Each rejected
// mutation must leave the whole module unchanged.
int main(int argc, char **argv) {
  assert(argc > 1);
  unsigned passed = 0;
  for (int file = 1; file < argc; ++file)
    for (unsigned test = 0; test < 17; ++test) {
      LLVMContext context;
      SMDiagnostic diagnostic;
      auto module = parseIRFile(argv[file], diagnostic, context);
      assert(module && !verifyModule(*module, &errs()));
      Function *kernel = nullptr;
      SmallVector<CallInst *> loads, matrices, products;
      SmallVector<SmallVector<Value *>> operands;
      unsigned barriers = 0, copies = 0;
      for (auto &function : *module)
        for (auto &block : function)
          for (auto &instruction : block)
            if (auto *call = dyn_cast<CallInst>(&instruction)) {
              barriers +=
                  triton::ppu::readonly_shared_matrices::ctaBarrier(call, 256);
              if (triton::ppu::mla_rope_prefetch::globalLoad(call))
                loads.push_back(call);
              if (triton::ppu::mla_kr_copy::matrix(call))
                matrices.push_back(call);
              if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand())) {
                copies += code->getAsmString().starts_with("cp.async.cg.");
                if (code->getAsmString().contains("ppu.mma.sync.")) {
                  products.push_back(call);
                  SmallVector<Value *> values;
                  for (Value *value : call->args())
                    values.push_back(value);
                  operands.push_back(values);
                  kernel = &function;
                }
              }
            }
      assert(kernel && loads.size() == 4 && matrices.size() >= 4);
      auto *first = products.front();
      llvm::erase_if(matrices, [&](auto *call) {
        return call->getParent() != first->getParent() ||
               !call->comesBefore(first);
      });
      assert(matrices.size() >= 5);
      IRBuilder<> builder(first);
      unsigned threads = 256, shared = 83968;
      if (test == 1)
        threads = 512;
      if (test == 2)
        shared = 83984;
      if (test == 3)
        kernel->removeFnAttr("ppu.mla.kv.pipeline");
      if (test == 4)
        kernel->addFnAttr("nvvm.reqntid", "128");
      if (test == 5)
        loads[0]->setArgOperand(0, builder.getInt32(1));
      if (test == 6)
        loads[0]->setArgOperand(4, kernel->getArg(2));
      if (test == 7) {
        auto *load = loads[0];
        builder.SetInsertPoint(load);
        load->setArgOperand(4, builder.CreateGEP(builder.getInt8Ty(),
                                                 load->getArgOperand(4),
                                                 builder.getInt64(1)));
      }
      SmallVector<StoreInst *> stores;
      for (auto &instruction : *first->getParent())
        if (auto *store = dyn_cast<StoreInst>(&instruction))
          for (auto *load : loads) {
            SmallVector<Instruction *> dead;
            if (triton::ppu::mla_kr_copy::matchStore(store, load, dead))
              stores.push_back(store);
          }
      if (test == 8) {
        assert(stores.size() == 4);
        stores[1]->setOperand(1, stores[0]->getPointerOperand());
      }
      if (test == 9)
        matrices[0]->setArgOperand(0, matrices[4]->getArgOperand(0));
      if (test == 10)
        matrices[4]->setArgOperand(0, matrices[0]->getArgOperand(0));
      if (test == 11)
        builder.CreateStore(builder.getInt32(0), kernel->getArg(0));
      if (test == 12)
        builder.CreateCall(module->getOrInsertFunction(
            "unknown", FunctionType::get(builder.getVoidTy(), false)));
      if (test == 13)
        builder.CreateLoad(builder.getInt32Ty(), kernel->getArg(0))
            ->setVolatile(true);
      if (test == 14) {
        for (auto &instruction : *first->getParent())
          if (triton::ppu::readonly_shared_matrices::ctaBarrier(&instruction,
                                                                256)) {
            builder.Insert(instruction.clone());
            break;
          }
      }
      if (test == 15) {
        for (auto &instruction : *first->getParent())
          if (auto *call = dyn_cast<CallInst>(&instruction))
            if (call->getCalledFunction() &&
                call->getCalledFunction()->getName() ==
                    "llvm.nvvm.cp.async.wait.group") {
              call->setArgOperand(0, builder.getInt32(1));
              break;
            }
      }
      if (test == 16) {
        auto *word = dyn_cast<ExtractValueInst>(*matrices[0]->user_begin());
        assert(word);
        first->setArgOperand(8, word);
      }
      assert(!verifyModule(*module, &errs()));
      auto before = text(*module);
      auto changed = triton::ppu::asyncMlaH16Kr(*module, threads, shared);
      assert(changed == (test == 0));
      assert(!verifyModule(*module, &errs()));
      if (test) {
        assert(text(*module) == before);
      } else {
        for (auto [index, product] : llvm::enumerate(products))
          assert(llvm::equal(product->args(), operands[index]));
        unsigned newBarriers = 0, newCopies = 0;
        for (auto &block : *kernel)
          for (auto &instruction : block)
            if (auto *call = dyn_cast<CallInst>(&instruction)) {
              newBarriers +=
                  triton::ppu::readonly_shared_matrices::ctaBarrier(call, 256);
              assert(!triton::ppu::mla_rope_prefetch::globalLoad(call));
              if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand()))
                newCopies += code->getAsmString().starts_with("cp.async.cg.");
            }
        assert(newBarriers == barriers && newCopies == copies + 4);
        auto after = text(*module);
        assert(triton::ppu::asyncMlaH16Kr(*module, threads, shared) == 0);
        assert(text(*module) == after);
      }
      ++passed;
    }
  outs() << "H16 KR copy checks: " << passed << " passed\n";
}
