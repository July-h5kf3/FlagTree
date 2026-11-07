// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "UniqueMlaScaleWriters.h"
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

SmallVector<StoreInst *> stores(Module &module) {
  SmallVector<StoreInst *> result;
  for (Function &function : module)
    for (BasicBlock &block : function)
      for (Instruction &instruction : block)
        if (auto *store = dyn_cast<StoreInst>(&instruction)) {
          auto address = mla_row_groups::offset(store->getPointerOperand(), 0,
                                                module.getDataLayout());
          if (address && (*address == 94208 || *address == 94336))
            result.push_back(store);
        }
  return result;
}

CallInst *thread(Module &module) {
  return cast<CallInst>(
      *module.getFunction("llvm.nvvm.read.ptx.sreg.tid.x")->user_begin());
}

std::unique_ptr<Module> fixture(LLVMContext &context) {
  auto module = std::make_unique<Module>("unique-scale-writers", context);
  module->setDataLayout("e-p:64:64-p1:64:64-p3:32:32");
  auto *i32 = Type::getInt32Ty(context);
  auto *f32 = Type::getFloatTy(context);
  auto *pairType = FixedVectorType::get(f32, 2);
  auto *smem = new GlobalVariable(
      *module, ArrayType::get(Type::getInt8Ty(context), 0), false,
      GlobalValue::ExternalLinkage, nullptr, "global_smem", nullptr,
      GlobalVariable::NotThreadLocal, 3);
  smem->setAlignment(Align(16));
  auto *function = Function::Create(
      FunctionType::get(Type::getVoidTy(context), false),
      GlobalValue::ExternalLinkage, "dense_int8_mla_kernel", *module);
  function->setCallingConv(CallingConv::PTX_Kernel);
  for (auto [name, expected] : {std::pair{"nvvm.reqntid", "256"},
                                {"ppu.mla.kv.pipeline", "1"},
                                {"ppu.mla.pv.batch", "4"},
                                {"ppu.mla.rope.prefetch", "1"}})
    function->addFnAttr(name, expected);
  auto tidFn = module->getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                           FunctionType::get(i32, false));
  auto sync = module->getOrInsertFunction(
      "__ppukernel_barrier_sync_cnt_void",
      FunctionType::get(Type::getVoidTy(context), {i32, i32, i32}, false));
  auto maximum = module->getOrInsertFunction(
      "llvm.maxnum.v2f32",
      FunctionType::get(pairType, {pairType, pairType}, false));
  auto exp = module->getOrInsertFunction("llvm.nvvm.ex2.approx.f",
                                         FunctionType::get(f32, {f32}, false));
  auto expFtz = module->getOrInsertFunction(
      "llvm.nvvm.ex2.approx.ftz.f", FunctionType::get(f32, {f32}, false));
  auto *entry = BasicBlock::Create(context, "entry", function);
  auto *header = BasicBlock::Create(context, "header", function);
  auto *latch = BasicBlock::Create(context, "latch", function);
  auto *exit = BasicBlock::Create(context, "exit", function);
  IRBuilder<NoFolder> builder(entry);
  auto *tid = builder.CreateCall(tidFn);
  Value *low = builder.CreateAnd(tid, builder.getInt32(28));
  Value *high = builder.CreateAnd(tid, builder.getInt32(128));
  Value *offset =
      builder.CreateAdd(builder.CreateShl(low, 2), builder.CreateLShr(high, 4));
  Value *write =
      builder.CreateGEP(builder.getInt8Ty(), smem,
                        builder.CreateAdd(offset, builder.getInt32(94208)));
  Value *writeScale =
      builder.CreateGEP(builder.getInt8Ty(), write, builder.getInt32(128));
  Value *row = builder.CreateAdd(builder.CreateLShr(low, 2),
                                 builder.CreateLShr(high, 3));
  Value *base =
      builder.CreateGEP(builder.getInt8Ty(), smem, builder.getInt32(81920));
  Value *reads[2];
  for (unsigned index = 0; index < 2; ++index)
    reads[index] = builder.CreateGEP(
        builder.getInt8Ty(), base,
        builder.CreateMul(builder.CreateAdd(row, builder.getInt32(index * 8)),
                          builder.getInt32(16)));
  builder.CreateBr(header);
  builder.SetInsertPoint(header);
  auto *iteration = builder.CreatePHI(i32, 2);
  iteration->addIncoming(builder.getInt32(0), entry);
  auto *state = builder.CreatePHI(pairType, 2);
  state->addIncoming(
      ConstantVector::getSplat(ElementCount::getFixed(2),
                               ConstantFP::getInfinity(f32, true)),
      entry);
  builder.CreateCall(
      sync, {builder.getInt32(0), builder.getInt32(256), builder.getInt32(1)});
  Value *values[2];
  for (unsigned index = 0; index < 2; ++index) {
    auto *load = builder.CreateAlignedLoad(FixedVectorType::get(f32, 4),
                                           reads[index], Align(16));
    Value *lanes[4];
    for (unsigned lane = 0; lane < 4; ++lane)
      lanes[lane] = builder.CreateExtractElement(load, builder.getInt64(lane));
    values[index] = builder.CreateFAdd(builder.CreateFAdd(lanes[0], lanes[2]),
                                       builder.CreateFAdd(lanes[1], lanes[3]));
  }
  Value *pair = builder.CreateInsertElement(PoisonValue::get(pairType),
                                            values[0], builder.getInt64(0));
  pair = builder.CreateInsertElement(pair, values[1], builder.getInt64(1));
  Value *nextState = builder.CreateCall(maximum, {state, pair});
  Value *alpha[2];
  for (unsigned index = 0; index < 2; ++index) {
    Value *old = builder.CreateExtractElement(state, builder.getInt64(index));
    Value *current =
        builder.CreateExtractElement(nextState, builder.getInt64(index));
    Value *argument = builder.CreateFSub(old, current);
    auto *fast = BasicBlock::Create(context, "fast", function);
    auto *slow = BasicBlock::Create(context, "slow", function);
    auto *join = BasicBlock::Create(context, "join", function);
    builder.CreateCondBr(
        builder.CreateFCmpOGT(argument, ConstantFP::get(f32, -126.0)), fast,
        slow);
    builder.SetInsertPoint(fast);
    Value *a = builder.CreateCall(expFtz, {argument});
    builder.CreateBr(join);
    builder.SetInsertPoint(slow);
    Value *b = builder.CreateCall(exp, {argument});
    builder.CreateBr(join);
    builder.SetInsertPoint(join);
    auto *phi = builder.CreatePHI(f32, 2);
    phi->addIncoming(a, fast);
    phi->addIncoming(b, slow);
    alpha[index] = phi;
  }
  Value *alphaPair = builder.CreateInsertElement(PoisonValue::get(pairType),
                                                 alpha[0], builder.getInt64(0));
  alphaPair =
      builder.CreateInsertElement(alphaPair, alpha[1], builder.getInt64(1));
  Value *scalePair = builder.CreateFMul(
      pair, ConstantVector::getSplat(ElementCount::getFixed(2),
                                     ConstantFP::get(f32, 0.00390625)));
  builder.CreateAlignedStore(alphaPair, write, Align(8));
  builder.CreateAlignedStore(scalePair, writeScale, Align(8));
  builder.CreateCall(
      sync, {builder.getInt32(0), builder.getInt32(256), builder.getInt32(1)});
  builder.CreateBr(latch);
  builder.SetInsertPoint(latch);
  Value *next = builder.CreateAdd(iteration, builder.getInt32(1));
  iteration->addIncoming(next, latch);
  state->addIncoming(nextState, latch);
  builder.CreateCondBr(builder.CreateICmpULT(next, builder.getInt32(2)), header,
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
    SMDiagnostic diagnostic;
    auto module = argc == 2 ? parseIRFile(argv[1], diagnostic, context)
                            : fixture(context);
    if (!module || stores(*module).size() != 2)
      report_fatal_error("missing scale stores");
    mutate(*module);
    if (verifyModule(*module, &errs()))
      report_fatal_error("invalid test input");
    auto before = dump(*module);
    auto original = stores(*module);
    SmallPtrSet<Instruction *, 2> removed;
    for (auto *store : original)
      removed.insert(store);
    SmallVector<std::pair<Instruction *, SmallVector<Value *>>> retained;
    SmallVector<Value *> payloads;
    for (auto *store : original)
      payloads.push_back(store->getValueOperand());
    for (Function &function : *module)
      for (BasicBlock &block : function)
        for (Instruction &instruction : block)
          if (!removed.contains(&instruction))
            retained.emplace_back(
                &instruction,
                SmallVector<Value *>(instruction.operands().begin(),
                                     instruction.operands().end()));
    unsigned changed = uniqueMlaScaleWriters(*module, threads, shared);
    if (changed != expected) {
      if (expected) {
        auto *function = original[0]->getFunction();
        DominatorTree dominance(*function);
        LoopInfo loops(dominance);
        mla_scale_writers::Equivalent proof(1, 0, module->getDataLayout(),
                                            loops);
        proof.value(payloads[0]);
        for (auto [value, equal] : proof.known)
          if (!equal)
            errs() << "NOT EQUIVALENT " << *value << "\n";
      }
      report_fatal_error(Twine("unexpected result: ") + name);
    }
    if (verifyModule(*module, &errs()))
      report_fatal_error("invalid output");
    if (!changed && dump(*module) != before)
      report_fatal_error("rejection changed module");
    if (changed) {
      SmallPtrSet<Instruction *, 32> live;
      SmallVector<CallInst *> writes;
      for (Function &function : *module)
        for (BasicBlock &block : function)
          for (Instruction &instruction : block) {
            live.insert(&instruction);
            if (auto *call = dyn_cast<CallInst>(&instruction))
              if (auto *code = dyn_cast<InlineAsm>(call->getCalledOperand());
                  code && code->getAsmString().contains("ppu.st.shared.v2.b32"))
                writes.push_back(call);
          }
      for (auto &[instruction, operands] : retained)
        if (!live.contains(instruction) ||
            !llvm::equal(instruction->operands(), operands))
          report_fatal_error("original non-store instruction changed");
      if (writes.size() != 2)
        report_fatal_error("missing unique stores");
      for (unsigned phase = 0; phase < writes.size(); ++phase) {
        BitVector bytes(128);
        unsigned writers = 0;
        for (unsigned tid = 0; tid < 256; ++tid) {
          auto predicate = mla_row_groups::predicate(
              writes[phase]->getArgOperand(3), tid, module->getDataLayout());
          auto offset = mla_row_groups::offset(writes[phase]->getArgOperand(0),
                                               tid, module->getDataLayout());
          unsigned expectedAddress =
              94208 + phase * 128 + 4 * (tid & 28) + ((tid & 128) >> 4);
          if (!predicate || !offset || *offset != expectedAddress ||
              *predicate != ((tid & 99) == 0))
            report_fatal_error("writer relation changed");
          if (*predicate) {
            ++writers;
            for (unsigned byte = 0; byte < 8; ++byte) {
              unsigned index = *offset + byte - 94208 - phase * 128;
              if (bytes[index])
                report_fatal_error("duplicate writer");
              bytes.set(index);
            }
          }
        }
        if (!bytes.all() || writers != 16)
          report_fatal_error("incomplete table coverage");
        for (unsigned word = 0; word < 2; ++word) {
          auto *bits =
              dyn_cast<BitCastInst>(writes[phase]->getArgOperand(word + 1));
          auto *element =
              bits ? dyn_cast<ExtractElementInst>(bits->getOperand(0))
                   : nullptr;
          if (!element || element->getVectorOperand() != payloads[phase] ||
              cast<ConstantInt>(element->getIndexOperand())->getZExtValue() !=
                  word)
            report_fatal_error("store payload changed");
        }
      }
      auto after = dump(*module);
      if (uniqueMlaScaleWriters(*module, threads, shared) ||
          dump(*module) != after)
        report_fatal_error("not idempotent");
    }
    outs() << "PASS " << name << "\n";
    ++passed;
  };
  auto unchanged = [](Module &) {};
  test("actual unique writers", unchanged, 2);
  test("wrong threads", unchanged, 0, 512);
  test("wrong shared", unchanged, 0, 256, 94208);
  for (StringRef attr : {"nvvm.reqntid", "ppu.mla.kv.pipeline",
                         "ppu.mla.pv.batch", "ppu.mla.rope.prefetch"})
    test(
        attr,
        [&](Module &module) {
          stores(module)[0]->getFunction()->removeFnAttr(attr);
        },
        0);
  test(
      "volatile", [](Module &module) { stores(module)[0]->setVolatile(true); },
      0);
  test(
      "misaligned",
      [](Module &module) { stores(module)[0]->setAlignment(Align(4)); }, 0);
  test(
      "nonuniform payload",
      [](Module &module) {
        auto *store = stores(module)[0];
        IRBuilder<NoFolder> builder(store);
        auto *value =
            builder.CreateBitCast(thread(module), builder.getFloatTy());
        store->setOperand(
            0, builder.CreateInsertElement(store->getValueOperand(), value,
                                           builder.getInt64(0)));
      },
      0);
  test(
      "unknown payload call",
      [](Module &module) {
        auto *store = stores(module)[0];
        IRBuilder<NoFolder> builder(store);
        auto unknown = module.getOrInsertFunction(
            "unknown", FunctionType::get(builder.getFloatTy(), false));
        auto *value = builder.CreateCall(unknown);
        store->setOperand(
            0, builder.CreateInsertElement(store->getValueOperand(), value,
                                           builder.getInt64(0)));
      },
      0);
  test(
      "row load unpublished",
      [](Module &module) {
        for (Function &function : module)
          for (BasicBlock &block : function)
            for (Instruction &instruction : block)
              if (auto *load = dyn_cast<LoadInst>(&instruction)) {
                auto offset = mla_row_groups::offset(load->getPointerOperand(),
                                                     0, module.getDataLayout());
                if (offset && *offset == 81920 &&
                    load->getType()->isVectorTy()) {
                  IRBuilder<NoFolder> builder(load);
                  builder.CreateAlignedStore(
                      ConstantFP::get(builder.getFloatTy(), 1.0),
                      load->getPointerOperand(), Align(4));
                  return;
                }
              }
      },
      0);
  test(
      "nonuniform exp2 selector",
      [](Module &module) {
        auto *pack =
            cast<InsertElementInst>(stores(module)[0]->getValueOperand());
        auto *phi = cast<PHINode>(pack->getOperand(1));
        auto *branchBlock = phi->getIncomingBlock(0)->getSinglePredecessor();
        auto *branch = cast<BranchInst>(branchBlock->getTerminator());
        IRBuilder<NoFolder> builder(branch);
        branch->setCondition(builder.CreateICmpNE(
            builder.CreateAnd(thread(module), builder.getInt32(1)),
            builder.getInt32(0)));
      },
      0);
  test(
      "nonuniform loop initializer",
      [](Module &module) {
        auto *function = stores(module)[0]->getFunction();
        DominatorTree dominance(*function);
        LoopInfo loops(dominance);
        auto *loop = loops.getLoopFor(stores(module)[0]->getParent());
        for (PHINode &phi : loop->getHeader()->phis()) {
          auto *initial = dyn_cast<Constant>(
              phi.getIncomingValueForBlock(loop->getLoopPreheader()));
          if (early_shared_float_scales::vector2(phi.getType()) && initial &&
              !initial->isNullValue()) {
            IRBuilder<NoFolder> builder(
                loop->getLoopPreheader()->getTerminator());
            Value *scalar =
                builder.CreateBitCast(thread(module), builder.getFloatTy());
            Value *value = builder.CreateInsertElement(
                PoisonValue::get(phi.getType()), scalar, builder.getInt64(0));
            value =
                builder.CreateInsertElement(value, scalar, builder.getInt64(1));
            phi.setIncomingValue(
                phi.getBasicBlockIndex(loop->getLoopPreheader()), value);
            return;
          }
        }
        report_fatal_error("missing loop row-state phi");
      },
      0);
  outs() << passed << " unique-writer checks passed\n";
}
