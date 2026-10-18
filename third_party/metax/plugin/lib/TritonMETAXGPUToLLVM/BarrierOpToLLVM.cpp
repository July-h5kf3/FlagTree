#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // MetaX global-to-shared loads issue immediately, without a group commit.
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // LDG-to-shared completes on the GVM counter, not the BSM counter. Match
    // arrive_gvmcnt(0) + barrier_inst from the native MACA copy implementation.
    // Waiting for all copies is conservative when num > 0.
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    mlir::LLVM::createBuiltinFunc<triton::gpu::AsyncWaitOp>(
        rewriter, op.getLoc(), op, "llvm.mxc.arrive", getVoidType(),
        {b.i32_val(1 << 6)});
    mlir::LLVM::createBuiltinFunc<triton::gpu::AsyncWaitOp>(
        rewriter, op.getLoc(), op, "llvm.mxc.barrier.inst", getVoidType(), {});
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

} // namespace

void mlir::triton::METAX::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<AsyncCommitGroupOpConversion, AsyncWaitOpConversion>(
      typeConverter, benefit);
}
