// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Run against an unhoisted grouped-store MLA LLVM module. This is CPU-only.
#include "ReadonlySharedMatrices.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <fstream>
#include <sstream>

using namespace llvm;

static std::string printModule(Module &module) {
  std::string text;
  raw_string_ostream stream(text);
  module.print(stream, nullptr);
  return text;
}

static CallInst *queryMatrix(Module &module) {
  for (Function &function : module)
    for (BasicBlock &block : function)
      for (Instruction &instruction : block)
        if (auto *call = dyn_cast<CallInst>(&instruction))
          if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand()))
            if (assembly->getAsmString().contains(".s32.s8.s8.s32")) {
              auto *word = cast<ExtractValueInst>(call->getArgOperand(8));
              return cast<CallInst>(word->getAggregateOperand());
            }
  llvm_unreachable("No native INT8 QK in test fixture");
}

int main(int argc, char **argv) {
  assert(argc == 3 && "usage: readonly-shared-matrices input.llir output.llir");
  assert(triton::ppu::readonly_shared_matrices::readOnlyAsm(
      "ppu.cvt.rmi.f32.f32 x, y;"));
  assert(triton::ppu::readonly_shared_matrices::readOnlyAsm(
      "ppu.cvt.rzi.s32.f32 x, y;"));
  assert(!triton::ppu::readonly_shared_matrices::readOnlyAsm(
      "ppu.st.shared.b32 [x], y;"));
  assert(!triton::ppu::readonly_shared_matrices::readOnlyAsm(
      "ppu.atomic.or.b32 x, y;"));
  unsigned passed = 0;
  for (unsigned test = 0; test < 16; ++test) {
    LLVMContext context;
    SMDiagnostic error;
    auto module = parseIRFile(argv[1], error, context);
    if (!module) {
      error.print(argv[0], errs());
      return 1;
    }
    auto *matrix = queryMatrix(*module);
    auto *address = cast<PtrToIntInst>(matrix->getArgOperand(0));
    auto *query = address->getPointerOperand();
    IRBuilder<> builder(matrix);
    if (test == 1)
      builder.CreateStore(builder.getInt32(0), query);
    if (test == 2) {
      auto unknown = module->getOrInsertFunction("unknown_shared_writer",
                                                 builder.getVoidTy());
      builder.CreateCall(unknown);
    }
    if (test == 3)
      module->getGlobalVariable("global_smem")->setAlignment(Align(8));
    if (test == 4) {
      auto *clobber = InlineAsm::get(
          FunctionType::get(builder.getVoidTy(), false), "", "~{memory}", true);
      builder.CreateCall(clobber->getFunctionType(), clobber, {});
    }
    if (test == 5) {
      auto *base = module->getGlobalVariable("global_smem");
      auto *outside = builder.CreateGEP(builder.getInt8Ty(), base,
                                        builder.getInt32(217088));
      builder.CreateStore(builder.getInt32(0), outside);
    }
    if (test == 6) {
      auto *pointer = ConstantPointerNull::get(builder.getPtrTy(0));
      builder.CreateStore(builder.getInt32(0), pointer);
    }
    if (test == 9) {
      auto *type = FunctionType::get(
          builder.getVoidTy(),
          {builder.getInt1Ty(), builder.getInt32Ty(), builder.getInt32Ty()},
          false);
      auto *writer = InlineAsm::get(type, "@$0 ppu.st.shared.b32 [$1], $2;",
                                    "b,r,r", true);
      builder.CreateCall(
          type, writer,
          {builder.getTrue(), matrix->getArgOperand(0), builder.getInt32(0)});
    }
    if (test == 10 || test == 14) {
      auto *type = FunctionType::get(builder.getFloatTy(),
                                     {builder.getFloatTy()}, false);
      auto function = module->getOrInsertFunction(
          test == 10 ? "__builtin_ppu_rcpf" : "__builtin_ppu_rcpf_writer",
          type);
      builder.CreateCall(function,
                         {ConstantFP::get(builder.getFloatTy(), 1.0)});
    }
    if (test == 11) {
      auto *type = FunctionType::get(builder.getDoubleTy(),
                                     {builder.getDoubleTy()}, false);
      auto function = module->getOrInsertFunction("__builtin_ppu_rcpf", type);
      builder.CreateCall(function,
                         {ConstantFP::get(builder.getDoubleTy(), 1.0)});
    }
    if (test == 12) {
      auto *type =
          FunctionType::get(builder.getFloatTy(), {query->getType()}, false);
      auto function = module->getOrInsertFunction("__builtin_ppu_rcpf", type);
      builder.CreateCall(function, {query});
    }
    if (test == 13) {
      auto *type = FunctionType::get(
          builder.getFloatTy(), {builder.getFloatTy(), builder.getFloatTy()},
          false);
      auto function = module->getOrInsertFunction("__builtin_ppu_rcpf", type);
      auto *one = ConstantFP::get(builder.getFloatTy(), 1.0);
      builder.CreateCall(function, {one, one});
    }
    if (test == 15) {
      auto *type = FunctionType::get(builder.getFloatTy(),
                                     {builder.getFloatTy()}, false);
      auto *function = Function::Create(type, Function::ExternalLinkage,
                                        "__builtin_ppu_rcpf", *module);
      IRBuilder<> body(BasicBlock::Create(context, "entry", function));
      body.CreateRet(function->getArg(0));
      builder.CreateCall(function,
                         {ConstantFP::get(builder.getFloatTy(), 1.0)});
    }
    assert(!verifyModule(*module, &errs()));
    auto before = printModule(*module);
    unsigned changed = triton::ppu::hoistReadonlySharedMatrices(
        *module, test == 7 ? 7 : 12, test == 8 ? 256 : 512, 217088);
    if (test == 0 || test == 10) {
      assert(changed == 12);
      assert(!verifyModule(*module, &errs()));
      auto after = printModule(*module);
      assert(after != before);
      assert(triton::ppu::hoistReadonlySharedMatrices(*module, 12, 512,
                                                      217088) == 0);
      assert(printModule(*module) == after);
      std::ofstream(argv[2]) << after;
    } else {
      assert(changed == 0 && "An unproved matrix lifetime was accepted");
      assert(printModule(*module) == before &&
             "Rejected input was partially modified");
    }
    ++passed;
  }
  outs() << "{\"passed\":" << passed
         << ",\"positive\":2,\"rejected_without_mutation\":14,\"hardware_"
            "dispatch\":false}\n";
  return 0;
}
