#include "UniqueMlaFinalRows.h"
#include "llvm/AsmParser/Parser.h"
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
mla_final_rows::Match inspect(Module &module) {
  auto result =
      mla_final_rows::match(*module.getFunction("dense_int8_mla_kernel"), true);
  if (!result)
    report_fatal_error("unsupported source");
  return *result;
}
std::unique_ptr<Module> fixture(LLVMContext &context) {
  SMDiagnostic error;
  return parseAssemblyString(
      R"MLAFIX(target datalayout = "e-p:64:64-p1:64:64-p3:32:32"
@global_smem = external addrspace(3) global [0 x i8], align 16
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @__ppukernel_barrier_sync_cnt_void(i32, i32, i32) convergent
declare float @llvm.nvvm.fma.rn.f(float, float, float)
define ptx_kernel void @dense_int8_mla_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, ptr addrspace(1) %5, ptr addrspace(1) %6, ptr addrspace(1) %7, ptr addrspace(1) %8, ptr addrspace(1) %9, ptr addrspace(1) %10, ptr addrspace(1) %11, ptr addrspace(1) %12, ptr addrspace(1) %13, ptr addrspace(1) %14) #0 {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %low = and i32 %tid, 28
  %high = and i32 %tid, 128
  %lo0 = shl i32 %low, 2
  %hi0 = lshr i32 %high, 4
  %offset0 = add i32 %lo0, %hi0
  %lo1 = shl i32 %low, 1
  %hi1 = lshr i32 %high, 1
  %offset1 = add i32 %lo1, %hi1
  %write0 = getelementptr i8, ptr addrspace(3) @global_smem, i32 %offset0
  %write1 = getelementptr i8, ptr addrspace(3) @global_smem, i32 %offset1
  %readbase = getelementptr i8, ptr addrspace(3) @global_smem, i32 81920
  %read = getelementptr i8, ptr addrspace(3) %readbase, i32 %offset0
  %enter = icmp ugt i32 2, 0
  br i1 %enter, label %preheader, label %skip
preheader:
  br label %loop
loop:
  %iteration = phi i32 [ 0, %preheader ], [ %next, %latch ]
  %state = phi <2 x float> [ zeroinitializer, %preheader ], [ %update, %latch ]
  call void @__ppukernel_barrier_sync_cnt_void(i32 0, i32 256, i32 1)
  %sum = load <2 x float>, ptr addrspace(3) %read, align 8
  %scaled = fmul <2 x float> %state, splat (float 5.000000e-01)
  %update = fadd <2 x float> %scaled, %sum
  br label %latch
latch:
  %next = add i32 %iteration, 1
  %more = icmp ult i32 %next, 2
  br i1 %more, label %loop, label %loopexit
loopexit:
  %maximum = fmul <2 x float> %sum, splat (float 1.000000e+00)
  br label %finish
skip:
  br label %finish
finish:
  %denominator = phi <2 x float> [ zeroinitializer, %skip ], [ %update, %loopexit ]
  %rowmax = phi <2 x float> [ splat (float 0xFFF0000000000000), %skip ], [ %maximum, %loopexit ]
  call void @__ppukernel_barrier_sync_cnt_void(i32 0, i32 256, i32 1)
  store <2 x float> %denominator, ptr addrspace(3) %write0, align 8
  call void @__ppukernel_barrier_sync_cnt_void(i32 0, i32 256, i32 1)
  %a = extractelement <2 x float> %denominator, i64 0
  %b = extractelement <2 x float> %denominator, i64 1
  %condition0 = fcmp ogt float %a, 0.000000e+00
  br i1 %condition0, label %fmaf0, label %join0
fmaf0:
  %adjust0 = call float @llvm.nvvm.fma.rn.f(float %a, float 1.000000e+00, float 0.000000e+00)
  br label %join0
join0:
  %log0 = phi float [ %a, %finish ], [ %adjust0, %fmaf0 ]
  %condition1 = fcmp ogt float %b, 0.000000e+00
  br i1 %condition1, label %fmaf1, label %join1
fmaf1:
  %adjust1 = call float @llvm.nvvm.fma.rn.f(float %b, float 1.000000e+00, float 0.000000e+00)
  br label %join1
join1:
  %log1 = phi float [ %b, %join0 ], [ %adjust1, %fmaf1 ]
  %pair0 = insertelement <2 x float> poison, float %log0, i64 0
  %pair1 = insertelement <2 x float> %pair0, float %log1, i64 1
  %lse = fadd <2 x float> %pair1, %rowmax
  call void @__ppukernel_barrier_sync_cnt_void(i32 0, i32 256, i32 1)
  store <2 x float> %lse, ptr addrspace(3) %write1, align 8
  call void @__ppukernel_barrier_sync_cnt_void(i32 0, i32 256, i32 1)
  ret void
}
attributes #0 = { "nvvm.reqntid"="256" "ppu.mla.kv.pipeline"="1" "ppu.mla.pv.batch"="4" "ppu.mla.rope.prefetch"="1" }
)MLAFIX",
      error, context);
}
int main(int argc, char **argv) {
  if (argc != 1 && argc != 2)
    return 2;
  unsigned passed = 0;
  auto test = [&](StringRef name, std::function<void(Module &)> mutate,
                  unsigned expected, unsigned threads = 256,
                  unsigned shared = 94464, bool uniform = true) {
    LLVMContext context;
    SMDiagnostic diagnostic;
    auto module = argc == 2 ? parseIRFile(argv[1], diagnostic, context)
                            : fixture(context);
    if (!module)
      report_fatal_error("parse failed");
    mutate(*module);
    if (verifyModule(*module, &errs()))
      report_fatal_error("invalid fixture");
    auto before = dump(*module);
    SmallVector<std::pair<Instruction *, SmallVector<Value *>>> originalNodes;
    auto matched = expected
                       ? std::optional<mla_final_rows::Match>(inspect(*module))
                       : std::nullopt;
    if (expected)
      for (Function &function : *module)
        for (BasicBlock &block : function)
          for (Instruction &instruction : block)
            if (&instruction != matched->final[0] &&
                &instruction != matched->final[1])
              originalNodes.emplace_back(
                  &instruction,
                  SmallVector<Value *>(instruction.operands().begin(),
                                       instruction.operands().end()));
    unsigned changed = uniqueMlaFinalRows(*module, threads, shared, uniform);
    if (changed != expected) {
      errs() << "RESULT " << changed << "\n";
      report_fatal_error(Twine("wrong result: ") + name);
    }
    if (verifyModule(*module, &errs()))
      report_fatal_error("invalid output");
    if (!changed && dump(*module) != before)
      report_fatal_error("rejection mutated module");
    if (changed) {
      SmallPtrSet<Instruction *, 32> live;
      for (Function &function : *module)
        for (BasicBlock &block : function)
          for (Instruction &instruction : block)
            live.insert(&instruction);
      for (auto &[instruction, operands] : originalNodes)
        if (!live.contains(instruction) ||
            !llvm::equal(instruction->operands(), operands))
          report_fatal_error("non-store graph changed");

      unsigned maskedFinal = 0;
      for (Function &function : *module)
        for (BasicBlock &block : function)
          for (Instruction &instruction : block) {
            auto *call = dyn_cast<CallInst>(&instruction);
            auto *assembly =
                call ? dyn_cast<InlineAsm>(call->getCalledOperand()) : nullptr;
            if (!assembly ||
                !assembly->getAsmString().contains("ppu.st.shared.v2.b32"))
              continue;
            auto base = mla_final_rows::offset(call->getArgOperand(0), 0,
                                               module->getDataLayout());
            if (!base || *base >= 128)
              continue;
            ++maskedFinal;
            BitVector bytes(128);
            unsigned owners = 0;
            for (unsigned tid = 0; tid < 256; ++tid) {
              auto address = mla_final_rows::offset(call->getArgOperand(0), tid,
                                                    module->getDataLayout());
              auto predicate = mla_row_groups::predicate(
                  call->getArgOperand(3), tid, module->getDataLayout());
              if (!address || !predicate || *predicate != ((tid & 99) == 0))
                report_fatal_error("wrong final predicate");
              if (*predicate) {
                ++owners;
                for (unsigned byte = 0; byte < 8; ++byte) {
                  unsigned index = *address + byte;
                  if (index >= 128 || bytes[index])
                    report_fatal_error("wrong packet coverage");
                  bytes.set(index);
                }
              }
            }
            if (owners != 16 || !bytes.all())
              report_fatal_error("missing final owner");
          }
      if (maskedFinal != 2)
        report_fatal_error("wrong effects");
      auto after = dump(*module);
      if (uniqueMlaFinalRows(*module, threads, shared, uniform) ||
          dump(*module) != after)
        report_fatal_error("not idempotent");
    }
    outs() << "PASS " << name << "\n";
    ++passed;
    outs().flush();
  };
  auto unchanged = [](Module &) {};
  test("actual final packets", unchanged, 2);
  test("wrong threads", unchanged, 0, 512);
  test("wrong shared", unchanged, 0, 256, 94208);
  test("no uniform witness", unchanged, 0, 256, 94464, false);
  for (StringRef attr : {"nvvm.reqntid", "ppu.mla.kv.pipeline",
                         "ppu.mla.pv.batch", "ppu.mla.rope.prefetch"})
    test(
        attr,
        [&](Module &module) {
          module.getFunction("dense_int8_mla_kernel")->removeFnAttr(attr);
        },
        0);
  test(
      "volatile final packet",
      [](Module &module) { inspect(module).final[0]->setVolatile(true); }, 0);
  test(
      "misaligned final packet",
      [](Module &module) { inspect(module).final[0]->setAlignment(Align(4)); },
      0);
  test(
      "missing packet publication",
      [](Module &module) {
        inspect(module).final[0]->getNextNode()->eraseFromParent();
      },
      0);
  test(
      "observable loop state",
      [](Module &module) {
        auto x = inspect(module);
        IRBuilder<NoFolder> b(x.update);
        auto *value = b.CreateExtractElement(x.state, b.getInt64(0));
        b.CreateStore(value, x.state->getFunction()->getArg(12));
      },
      0);
  test(
      "nonuniform final payload",
      [](Module &module) {
        auto x = inspect(module);
        IRBuilder<NoFolder> b(x.final[0]);
        auto *value = b.CreateBitCast(x.thread, b.getFloatTy());
        x.final[0]->setOperand(
            0, b.CreateInsertElement(x.final[0]->getValueOperand(), value,
                                     b.getInt64(0)));
      },
      0);
  test(
      "effect under log selector",
      [](Module &module) {
        auto x = inspect(module);
        DominatorTree dt(*x.state->getFunction());
        LoopInfo loops(dt);
        for (BasicBlock &block : *x.state->getFunction())
          for (Instruction &i : block)
            if (auto *branch = dyn_cast<BranchInst>(&i);
                branch && branch->isConditional() &&
                !loops.getLoopFor(&block) &&
                branch->getSuccessor(0)->getName().contains("fmaf")) {
              IRBuilder<NoFolder> b(
                  branch->getSuccessor(0),
                  branch->getSuccessor(0)->getFirstInsertionPt());
              b.CreateStore(b.getInt32(0), x.state->getFunction()->getArg(12));
              return;
            }
        report_fatal_error("missing log control");
      },
      0);
  test(
      "unknown state coefficient",
      [](Module &module) {
        auto x = inspect(module);
        auto *mul = cast<Instruction>(x.update->getOperand(0));
        IRBuilder<NoFolder> b(mul);
        auto *coefficient = mul->getOperand(1);
        auto f = module.getOrInsertFunction(
            "other_coefficient",
            FunctionType::get(coefficient->getType(), {coefficient->getType()},
                              false));
        mul->setOperand(1, b.CreateCall(f, {coefficient}));
      },
      0);
  outs() << passed << " final-packet checks passed\n";
}
