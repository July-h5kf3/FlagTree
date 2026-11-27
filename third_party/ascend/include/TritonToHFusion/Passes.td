/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_HFUSION_CONVERSION_PASSES
#define TRITON_TO_HFUSION_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToHFusion : Pass<"triton-to-hfusion", "mlir::ModuleOp"> {
  let summary = "Convert Triton to HFusion dialect";
  let constructor = "triton::createTritonToHFusionPass()";
  let options = [
    Option<"compileOn910_95", "compile-on-910-95", "bool", /*default=*/"false",
           "Whether compiling for Ascend910_95 targets with native __hmf_rint">
  ];
  let dependentDialects = [
    "hfusion::HFusionDialect",
    "arith::ArithDialect",
    "math::MathDialect"
  ];
}

#endif // TRITON_TO_HFUSION_CONVERSION_PASSES
