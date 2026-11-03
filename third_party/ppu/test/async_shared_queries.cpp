// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "ReadonlySharedMatrices.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IRBuilder.h"
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

int main(int argc, char **argv) {
  assert(argc == 2);
  for (unsigned test = 0; test < 6; ++test) {
    LLVMContext context;
    SMDiagnostic diagnostic;
    auto module = parseIRFile(argv[1], diagnostic, context);
    if (!module) {
      diagnostic.print(argv[0], errs());
      return 1;
    }
    CallInst *copy = nullptr, *wait = nullptr, *product = nullptr;
    for (Function &function : *module)
      for (BasicBlock &block : function)
        for (Instruction &instruction : block)
          if (auto *call = dyn_cast<CallInst>(&instruction)) {
            if (auto *assembly =
                    dyn_cast<InlineAsm>(call->getCalledOperand())) {
              if (!copy && assembly->getAsmString().starts_with("cp.async."))
                copy = call;
              if (!product &&
                  assembly->getAsmString().contains(".s32.s8.s8.s32"))
                product = call;
            } else if (auto *callee = call->getCalledFunction()) {
              if (callee->getName() == "llvm.nvvm.cp.async.wait.group")
                wait = call;
            }
          }
    assert(copy && wait && product);
    auto *word = cast<ExtractValueInst>(product->getArgOperand(8));
    auto *query = cast<CallInst>(word->getAggregateOperand());
    auto *address = cast<PtrToIntInst>(query->getArgOperand(0));
    if (test == 1)
      copy->setArgOperand(0, address->getPointerOperand());
    if (test == 2)
      wait->setArgOperand(0, ConstantInt::get(Type::getInt32Ty(context), 1));
    if (test == 3)
      module->getGlobalVariable("global_smem")->setAlignment(Align(8));
    if (test == 4 || test == 5) {
      auto *assembly = cast<InlineAsm>(copy->getCalledOperand());
      auto *replacement = InlineAsm::get(
          assembly->getFunctionType(),
          test == 4 ? "cp.async.unknown [$0], [$1], $2;"
                    : assembly->getAsmString(),
          test == 5 ? "r,l,r,~{memory}" : assembly->getConstraintString(),
          true);
      copy->setCalledOperand(replacement);
    }
    assert(!verifyModule(*module, &errs()));
    std::string before = print(*module);
    unsigned changed =
        triton::ppu::hoistReadonlySharedMatrices(*module, 12, 512, 167936);
    assert(changed == (test == 0 ? 12 : 0));
    assert(!verifyModule(*module, &errs()));
    if (test != 0)
      assert(print(*module) == before);
    else {
      unsigned resident = 0, overlap = 0;
      SmallPtrSet<CallInst *, 16> queries;
      for (Instruction &instruction : *product->getParent())
        if (auto *call = dyn_cast<CallInst>(&instruction))
          if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
            if (assembly->getAsmString().contains(".s32.s8.s8.s32"))
              for (unsigned index = 0; index < 4; ++index) {
                auto *extract =
                    cast<ExtractValueInst>(call->getArgOperand(8 + index));
                auto *matrix = cast<CallInst>(extract->getAggregateOperand());
                queries.insert(matrix);
              }
      for (auto *matrix : queries) {
        if (matrix->getParent() == wait->getParent()) {
          assert(matrix->comesBefore(wait));
          ++overlap;
        } else
          ++resident;
      }
      // The four remaining page-loop Q reads precede wait_group(0).
      assert(overlap == 4 && resident == 12);
    }
  }
  outs() << "ASYNC_Q_PROOF_CHECKS_PASSED 6\n";
}
