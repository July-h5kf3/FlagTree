// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "GuardMlaScores.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include <cassert>
using namespace llvm;

int main() {
  for (unsigned mode : {1, 2, 3})
    for (unsigned test : {0, 1, 2, 3}) {
      // The final slow block must retain the original page/token conditions.
      std::string ir = R"(
declare void @consume(float)
define ptx_kernel void @dense_int8_mla_kernel(i32 %end, i32 %limit, float %score) #0 {
entry:
  br label %loop
loop:
  %p = phi i32 [0, %entry], [%next,%loop]
  %next = add i32 %p, 2
  %more = icmp slt i32 %next,%end
  %second = add i32 %p, 1
  %okpg = icmp slt i32 %second,%end
  %base = shl i32 %p,6
)";
      unsigned n = (mode == 3 && test == 2) ? 8 : 16;
      for (unsigned k = 0; k < n; ++k) {
        auto i = std::to_string(k);
        unsigned off = (test == 1 && k == 0) ? 128 : k;
        ir += "  %t" + i + " = add i32 %base," + std::to_string(off) + "\n";
        ir += "  %v" + i + " = icmp slt i32 %t" + i + ",%limit\n";
        ir += "  %m" + i + " = and i1 %v" + i + ",%okpg\n";
        if (mode == 3)
          ir += "  %o" + i + " = or i1 " +
                std::string(test == 1 && k == 0 ? "%okpg" : "%more") + ",%m" +
                i + "\n";
      }
      for (unsigned k = 0; k < n; ++k) {
        auto i = std::to_string(k);
        ir += "  %s" + i + " = select i1 %" + (mode == 3 ? "o" : "m") + i +
              ",float %score,float " +
              (test == 3 && k == 1 ? "0.0" : "0xFFF0000000000000") + "\n";
      }
      for (unsigned k = 0; k < n; ++k)
        ir += "  call void @consume(float %s" + std::to_string(k) + ")\n";
      ir += "  br i1 %more,label %loop,label %done\ndone:\n  ret void\n}\n";
      ir += "attributes #0 = { \"nvvm.reqntid\"=\"256\" " +
            std::string(n == 8 ? "\"ppu.mla.h16.kr.copy\"=\"1\""
                               : "\"ppu.mla.row.gather\"=\"1\"") +
            " }\n";
      LLVMContext c;
      SMDiagnostic d;
      auto m = parseAssemblyString(ir, d, c);
      assert(m && !verifyModule(*m, &errs()));
      auto changed = triton::ppu::mla_score_guard::run(*m, mode);
      assert(changed == ((test == 0 || test == 2) ? n : 0));
      assert(!verifyModule(*m, &errs()));
      unsigned merges = 0;
      for (auto &bb : *m->getFunction("dense_int8_mla_kernel"))
        for (auto &phi : bb.phis())
          if (phi.getType()->isFloatTy()) {
            assert(phi.getNumIncomingValues() == 2);
            auto *slow = dyn_cast<SelectInst>(phi.getIncomingValue(1));
            assert(slow && slow->getTrueValue() == phi.getIncomingValue(0));
            if (mode == 3)
              assert(cast<BinaryOperator>(slow->getCondition())->getOpcode() ==
                     Instruction::And);
            ++merges;
          }
      assert(merges == changed);
      assert(triton::ppu::mla_score_guard::run(*m, mode) == 0);
    }
}
