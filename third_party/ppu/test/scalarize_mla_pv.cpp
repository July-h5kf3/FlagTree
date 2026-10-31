// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "ScalarizeMlaPv.h"
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

static SmallPtrSet<Instruction *, 32> effects(Module &module) {
  SmallPtrSet<Instruction *, 32> result;
  for (Function &function : module)
    for (BasicBlock &block : function)
      for (Instruction &instruction : block)
        if (instruction.mayReadOrWriteMemory())
          result.insert(&instruction);
  return result;
}

static std::unique_ptr<Module> fixture(LLVMContext &context, unsigned variant) {
  auto module = std::make_unique<Module>("mla-pv-scalarization", context);
  module->setDataLayout("e-p:64:64-p3:32:32");
  IRBuilder<NoFolder> builder(context);
  auto *function =
      Function::Create(FunctionType::get(builder.getVoidTy(), false),
                       GlobalValue::ExternalLinkage, "decode", *module);
  function->setCallingConv(CallingConv::PTX_Kernel);
  auto *entry = BasicBlock::Create(context, "entry", function);
  auto *loop = BasicBlock::Create(context, "loop", function);
  auto *exit = BasicBlock::Create(context, "exit", function);
  builder.SetInsertPoint(entry);
  builder.CreateBr(loop);
  builder.SetInsertPoint(loop);
  auto *floatType = FixedVectorType::get(builder.getFloatTy(), 32);
  auto *integerType = FixedVectorType::get(builder.getInt32Ty(), 32);
  auto *carry = builder.CreatePHI(floatType, 2);
  carry->addIncoming(Constant::getNullValue(floatType), entry);
  Value *old30 = builder.CreateExtractElement(carry, builder.getInt64(30));
  Value *old31 = builder.CreateExtractElement(carry, builder.getInt64(31));
  auto *resultType =
      StructType::get(context, std::vector<Type *>(8, builder.getInt32Ty()));
  auto *mmaType = FunctionType::get(
      resultType, std::vector<Type *>(16, builder.getInt32Ty()), false);
  auto *mma = InlineAsm::get(
      mmaType, triton::ppu::mla_nope_prefetch::ProductAsm,
      "=r,=r,=r,=r,=r,=r,=r,=r,0,1,2,3,4,5,6,7,r,r,r,r,r,r,r,r", true);
  using Words = std::array<Value *, 8>;
  Words zeros;
  zeros.fill(builder.getInt32(0));
  auto multiply = [&](const Words &accumulator) {
    SmallVector<Value *> operands(accumulator.begin(), accumulator.end());
    for (unsigned index = 0; index < 8; ++index)
      operands.push_back(builder.getInt32(index + 1));
    auto *call = builder.CreateCall(mmaType, mma, operands);
    Words result;
    for (unsigned word = 0; word < 8; ++word)
      result[word] = builder.CreateExtractValue(call, {word});
    return result;
  };
  for (unsigned index = 0; index < 32; ++index)
    multiply(zeros);
  auto barrierType =
      FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()}, false);
  builder.CreateCall(module->getOrInsertFunction(
                         "llvm.nvvm.barrier.cta.sync.aligned.all", barrierType),
                     {builder.getInt32(0)});
  std::array<Words, 4> high, low, seeds;
  high.fill(zeros);
  low.fill(zeros);
  for (unsigned k = 0; k < 8; ++k)
    for (auto &words : high)
      words = multiply(words);
  Value *packedHigh = PoisonValue::get(integerType);
  for (unsigned lane = 0; lane < 32; ++lane)
    packedHigh = builder.CreateInsertElement(
        packedHigh, high[lane / 8][lane % 8], builder.getInt64(lane));
  auto *factor = ConstantVector::getSplat(ElementCount::getFixed(32),
                                          builder.getInt32(127));
  Value *packedSeed =
      variant == 1 ? builder.CreateMul(packedHigh, factor) : nullptr;
  for (unsigned lane = 0; lane < 32; ++lane)
    seeds[lane / 8][lane % 8] =
        packedSeed
            ? builder.CreateExtractElement(packedSeed, builder.getInt64(lane))
            : builder.CreateMul(high[lane / 8][lane % 8],
                                builder.getInt32(127));
  for (unsigned k = 0; k < 8; ++k)
    for (unsigned chain = 0; chain < 4; ++chain)
      low[chain] = multiply(k ? low[chain] : seeds[chain]);
  Value *packedLow = PoisonValue::get(integerType);
  for (unsigned lane = 0; lane < 32; ++lane)
    packedLow = builder.CreateInsertElement(packedLow, low[lane / 8][lane % 8],
                                            builder.getInt64(lane));
  Value *contribution = builder.CreateSIToFP(packedLow, floatType);
  if (variant == 2) {
    contribution =
        builder.CreateInsertElement(contribution, old30, builder.getInt64(30));
    contribution =
        builder.CreateInsertElement(contribution, old31, builder.getInt64(31));
  }
  if (variant == 3) {
    SmallVector<int> mask;
    for (unsigned lane = 0; lane < 32; ++lane)
      mask.push_back(lane < 30 ? int(lane) : -1);
    contribution = builder.CreateShuffleVector(
        contribution, PoisonValue::get(floatType), mask);
  }
  auto *scale = ConstantVector::getSplat(
      ElementCount::getFixed(32), ConstantFP::get(builder.getFloatTy(), 0.125));
  auto *root = builder.CreateFAdd(builder.CreateFMul(carry, scale),
                                  builder.CreateFMul(contribution, scale));
  carry->addIncoming(root, loop);
  builder.CreateCondBr(builder.getFalse(), loop, exit);
  builder.SetInsertPoint(exit);
  builder.CreateRetVoid();
  return module;
}

int main(int argc, char **argv) {
  unsigned checks = 0;
  for (int file = 0; file < std::max(argc - 1, 1); ++file)
    for (unsigned test = 0; test < 9; ++test) {
      if (argc > 1 && test > 0 && test < 4)
        continue;
      LLVMContext context;
      SMDiagnostic diagnostic;
      auto module = argc == 1
                        ? fixture(context, test)
                        : parseIRFile(argv[file + 1], diagnostic, context);
      if (!module) {
        diagnostic.print(argv[0], errs());
        return 1;
      }
      auto calls = products(*module);
      assert(calls.size() == 96);
      if (test == 6) {
        IRBuilder<> builder(calls[32]->getNextNode());
        auto *type = FunctionType::get(builder.getVoidTy(),
                                       {builder.getInt32Ty()}, false);
        builder.CreateCall(module->getOrInsertFunction(
                               "llvm.nvvm.barrier.cta.sync.aligned.all", type),
                           {builder.getInt32(0)});
      }
      if (test == 7) {
        IRBuilder<> builder(calls[32]->getNextNode());
        builder.CreateCall(module->getOrInsertFunction(
            "unknown_memory_effect",
            FunctionType::get(builder.getVoidTy(), false)));
      }
      if (test == 8) {
        Value *seed = calls[64]->getArgOperand(0);
        if (auto *extract = dyn_cast<ExtractElementInst>(seed))
          seed = extract->getVectorOperand();
        auto *multiply = cast<BinaryOperator>(seed);
        auto *type = multiply->getType();
        Constant *factor = ConstantInt::get(Type::getInt32Ty(context), 126);
        if (auto *vector = dyn_cast<FixedVectorType>(type))
          factor = ConstantVector::getSplat(vector->getElementCount(), factor);
        multiply->setOperand(1, factor);
      }
      assert(!verifyModule(*module, &errs()));
      std::string before = print(*module);
      auto originalEffects = effects(*module);
      std::array<std::array<Value *, 8>, 96> inputs;
      for (unsigned index = 0; index < 96; ++index)
        for (unsigned word = 0; word < 8; ++word)
          inputs[index][word] = calls[index]->getArgOperand(word + 8);
      unsigned changed = triton::ppu::scalarizeMlaPv(
          *module, test == 4 ? 256 : 512, test == 5 ? 167936 : 168192);
      assert(changed == (test < 4 ? 1 : 0));
      assert(!verifyModule(*module, &errs()));
      if (!changed) {
        assert(print(*module) == before);
      } else {
        assert(products(*module) == calls);
        assert(effects(*module) == originalEffects);
        for (unsigned index = 0; index < 96; ++index)
          for (unsigned word = 0; word < 8; ++word)
            assert(calls[index]->getArgOperand(word + 8) ==
                   inputs[index][word]);
        std::string once = print(*module);
        assert(triton::ppu::scalarizeMlaPv(*module, 512, 168192) == 0);
        assert(print(*module) == once);
      }
      ++checks;
    }
  outs() << "Mla PV scalarization: " << checks << " checks passed\n";
}
