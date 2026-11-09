// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "VersionMlaTiles.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include <cassert>
using namespace llvm;

int main() {
  for (unsigned test = 0; test < 8; ++test) {
    std::string input = R"(
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.smax.i32(i32, i32)
declare void @consume(i1)
define ptx_kernel void @kernel(i32 %length) #0 {
entry:
  %head = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %z = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %biased = add i32 %length, 63
  %pages = sdiv i32 %biased, 64
  %s = mul i32 %pages, %z
  %zp = add i32 %z, 1
  %end = mul i32 %pages, %zp
  %nonempty = icmp slt i32 %s, %end
  br i1 %nonempty, label %pre, label %done
pre:
  br label %loop
loop:
  %page = phi i32 [%s, %pre], [%next, %loop]
  %base = shl i32 %page, 6
)";
    // The out-of-tile offset must retain its check in the fast copy.
    for (unsigned n = 0; n < 17; ++n) {
      auto id = std::to_string(n);
      input += "  %t" + id + " = add i32 %base, " +
               std::to_string(n == 16 ? 128 : n) + "\n";
      input += "  %p" + id + " = icmp slt i32 %t" + id + ", %length\n";
      input += "  call void @consume(i1 %p" + id + ")\n";
    }
    input += R"(  %next = add i32 %page, 2
  %more = icmp slt i32 %next, %end
  br i1 %more, label %loop, label %exit
exit:
  br label %done
done:
  ret void
}
attributes #0 = { "nvvm.reqntid"="256" "ppu.mla.kv.pipeline"="1" "ppu.mla.row.gather"="1" }
)";
    if (test == 1)
      input.replace(input.find("add i32 %page, 2"), 16, "add i32 %page, 1");
    if (test == 2)
      input.replace(input.find("biased, 64"), 10, "biased, 32");
    if (test == 3) {
      // All per-lane uses refer to a valid SSA value, but the request length
      // is no longer uniform and dispatching around barriers would be unsafe.
      auto pos = input.find("%biased =");
      for (; (pos = input.find("%length", pos)) != std::string::npos;)
        input.replace(pos, 7, "%tid");
    }
    if (test == 4)
      input.replace(input.find("\"256\""), 5, "\"512\"");
    if (test >= 5) {
      std::string delta = test == 7 ? "-2" : "-1";
      input.insert(
          input.find("  %biased ="),
          "  %lag = add i32 %length, " + delta +
              "\n  %limit = call i32 @llvm.smax.i32(i32 %lag, i32 0)\n");
      for (unsigned offset : {126u, 127u, 128u}) {
        auto id = std::to_string(offset);
        input.insert(input.find("  %next ="),
                     "  %u" + id + " = add i32 %base, " + id + "\n  %v" + id +
                         " = icmp slt i32 %u" + id +
                         ", %limit\n  call void @consume(i1 %v" + id + ")\n");
      }
    }
    LLVMContext ctx;
    SMDiagnostic diag;
    auto mod = parseAssemblyString(input, diag, ctx);
    assert(mod && !verifyModule(*mod, &errs()));
    auto *f = mod->getFunction("kernel");
    unsigned count =
        triton::ppu::mla_full_tiles::run(*f, test == 5 || test == 7);
    assert(count == (test == 5 ? 17u : (test == 0 || test >= 6 ? 16u : 0u)));
    assert(f->hasFnAttribute("ppu.mla.last.token") == (test == 5));
    assert(!verifyModule(*mod, &errs()));
    unsigned constantUses = 0, guardedUses = 0;
    for (auto &block : *f)
      if (block.getName().ends_with(".full"))
        for (auto &i : block)
          if (auto *call = dyn_cast<CallInst>(&i);
              call && call->getCalledFunction() &&
              call->getCalledFunction()->getName() == "consume") {
            if (isa<ConstantInt>(call->getArgOperand(0)))
              ++constantUses;
            else
              ++guardedUses;
          }
    assert(constantUses == count && guardedUses == (test == 5   ? 3u
                                                    : test >= 6 ? 4u
                                                    : test == 0 ? 1u
                                                                : 0u));
    assert(triton::ppu::mla_full_tiles::run(*f) == 0);
  }
}
