/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#include "TritonToHFusion/Passes.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOHFUSION
#include "ascend/include/TritonToHFusion/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hfusion;

namespace {
struct TritonModToHFusionConversion : OpRewritePattern<triton::ascend::ModOp> {
  using OpRewritePattern<triton::ascend::ModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ascend::ModOp op,
                                PatternRewriter &rewriter) const final {
    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    if (!lhsType || !rhsType) {
      return failure();
    }

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), lhsType.getShape(), lhsType.getElementType());
    auto newOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op.getLoc(), hfusion::BinaryFn::mod,
            ValueRange({op.getLhs(), op.getRhs()}),
            ValueRange({emptyTensor.getResult()}));

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct TritonHistogramToHFusionConversion
    : OpRewritePattern<triton::HistogramOp> {
  using OpRewritePattern<triton::HistogramOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::HistogramOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.getSrc();
    auto resultType = op.getResult().getType();

    int64_t numBins = 256; // 256 is default fallback.
    if (auto rankedTy = dyn_cast<RankedTensorType>(resultType))
      if (rankedTy.hasStaticShape() && rankedTy.getNumElements() > 0)
        numBins = rankedTy.getNumElements();

    auto numBinsAttr = rewriter.getI64IntegerAttr(numBins);

    auto newOp = rewriter.create<hfusion::HistogramOp>(loc, resultType, input,
                                                       numBinsAttr, Value());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct TritonFpToFpToHFusionConversion : OpRewritePattern<triton::FpToFpOp> {
  using OpRewritePattern<triton::FpToFpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FpToFpOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.getSrc();
    auto resultType = op.getResult().getType();

    // Only handle float-to-float conversions with non-RTNE rounding modes
    // RTNE (default) rounding is handled by TritonToLinalg pass using
    // arith.truncf/extf
    auto srcType = cast<TensorType>(input.getType());
    auto dstType = cast<TensorType>(resultType);
    if (!srcType.getElementType().isIntOrFloat() ||
        !dstType.getElementType().isIntOrFloat()) {
      return failure();
    }

    // Check if this has a non-RTNE rounding mode
    auto roundingMode = op.getRounding();
    if (!roundingMode.has_value() ||
        roundingMode.value() == triton::RoundingMode::RTNE) {
      // RTNE or no rounding mode specified: let TritonToLinalg handle it
      return failure();
    }

    // Map non-RTNE rounding modes to HFusion rounding mode
    hfusion::RoundMode hfusionRoundMode;
    switch (roundingMode.value()) {
    case triton::RoundingMode::RTZ:
      hfusionRoundMode = hfusion::RoundMode::TRUNC;
      break;
    default:
      return op.emitError("Unsupported rounding mode for HFusion conversion");
    }
    // Note: Only RTZ (and potential future non-RTNE modes) reach here

    // Get or create destination tensor (destination-style)
    SmallVector<Value> dsts;
    if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, dsts)))
      return failure();

    // Create the HFusion cast operation with round_mode attribute
    auto roundModeAttr =
        hfusion::RoundModeAttr::get(rewriter.getContext(), hfusionRoundMode);
    auto modeAttr = rewriter.getNamedAttr("mode", roundModeAttr);

    rewriter.replaceOpWithNewOp<hfusion::CastOp>(
        op, ValueRange{input}, ValueRange{dsts}, ArrayRef{modeAttr});

    return success();
  }
};

struct TritonConv1dToHFusionConversion
    : OpRewritePattern<triton::ascend::Conv1dOp> {
  using OpRewritePattern<triton::ascend::Conv1dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ascend::Conv1dOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Value input = op.getInput();
    Value weight = op.getWeight();
    Value biasValue = op.getBias();
    int64_t stride = op.getStride();
    int64_t padding_size = op.getPaddingSize();
    int64_t dilation = op.getDilation();
    int64_t groups = op.getGroups();

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());
    if (!inputType.hasStaticShape() || !weightType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> weightShape = weightType.getShape();

    bool isBatched = inputShape.size() == 3;
    int64_t N;
    if (isBatched)
      N = inputShape[0];
    int64_t L_in = inputShape[isBatched ? 2 : 1];
    int64_t C_out = weightShape[0];
    int64_t kernel_size = weightShape[2];

    if (stride == 0) {
      return failure();
    }
    int64_t L_out =
        (L_in + 2 * padding_size - dilation * (kernel_size - 1) - 1) / stride +
        1;

    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    Type resultElementType = resultType.getElementType();

    constexpr int64_t dim2 = 2;
    constexpr int64_t dim3 = 3;
    Value initTensor;
    if (isBatched) {
      SmallVector<int64_t, dim3> outputShape{N, C_out, L_out};
      initTensor =
          rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
    } else {
      SmallVector<int64_t, dim2> outputShape{C_out, L_out};
      initTensor =
          rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
    }

    SmallVector<Value, dim3> ins;
    ins.push_back(input);
    ins.push_back(weight);
    if (biasValue) {
      ins.push_back(biasValue);
    }

    auto newOp = rewriter.create<hfusion::Conv1DOp>(
        loc, ins, initTensor, stride, padding_size, dilation, groups);

    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Saturating RINT FP16 -> INT8 fusion to hfusion.cast (lowers to single
// hardware vconv_f162s8r on Ascend910B).
//===----------------------------------------------------------------------===//

static std::optional<double> getFloatSplatConstant(Value v) {
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    v = splat.getSrc();
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(cst.getValue()))
      return floatAttr.getValueAsDouble();
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(cst.getValue())) {
      if (denseAttr.isSplat() && isa<FloatType>(denseAttr.getElementType()))
        return denseAttr.getSplatValue<APFloat>().convertToDouble();
    }
  }
  return std::nullopt;
}

static bool matchMinF(Value v, Value &arg, double &bound) {
  Operation *def = v.getDefiningOp();
  if (!def || !isa<arith::MinNumFOp, arith::MinimumFOp>(def))
    return false;
  Value lhs = def->getOperand(0);
  Value rhs = def->getOperand(1);
  if (auto c = getFloatSplatConstant(rhs)) {
    arg = lhs;
    bound = *c;
    return true;
  }
  if (auto c = getFloatSplatConstant(lhs)) {
    arg = rhs;
    bound = *c;
    return true;
  }
  return false;
}

static bool matchMaxF(Value v, Value &arg, double &bound) {
  Operation *def = v.getDefiningOp();
  if (!def || !isa<arith::MaxNumFOp, arith::MaximumFOp>(def))
    return false;
  Value lhs = def->getOperand(0);
  Value rhs = def->getOperand(1);
  if (auto c = getFloatSplatConstant(rhs)) {
    arg = lhs;
    bound = *c;
    return true;
  }
  if (auto c = getFloatSplatConstant(lhs)) {
    arg = rhs;
    bound = *c;
    return true;
  }
  return false;
}

static Value matchInt8Clamp(Value v) {
  constexpr double kInt8Min = -128.0;
  constexpr double kInt8Max = 127.0;
  if (auto clamp = v.getDefiningOp<triton::ClampFOp>()) {
    auto lo = getFloatSplatConstant(clamp.getMin());
    auto hi = getFloatSplatConstant(clamp.getMax());
    if (lo && hi && *lo == kInt8Min && *hi == kInt8Max)
      return clamp.getX();
    return Value();
  }
  Value inner;
  double boundA = 0.0, boundB = 0.0;
  if (matchMinF(v, inner, boundA) && boundA == kInt8Max) {
    Value orig;
    if (matchMaxF(inner, orig, boundB) && boundB == kInt8Min)
      return orig;
  }
  if (matchMaxF(v, inner, boundA) && boundA == kInt8Min) {
    Value orig;
    if (matchMinF(inner, orig, boundB) && boundB == kInt8Max)
      return orig;
  }
  return Value();
}

static Value matchRintOp(Value v) {
  auto ext = v.getDefiningOp<triton::ExternElementwiseOp>();
  if (!ext || ext.getSymbol() != "__hmf_rint" || ext.getOperands().size() != 1)
    return Value();
  return ext.getOperands().front();
}

static Value unwrapToFP16Source(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty)
    return Value();
  if (ty.getElementType().isF16())
    return v;
  if (auto ext = v.getDefiningOp<arith::ExtFOp>()) {
    auto inTy = dyn_cast<RankedTensorType>(ext.getIn().getType());
    if (inTy && inTy.getElementType().isF16())
      return ext.getIn();
  }
  return Value();
}

struct TritonRintClampFP16ToInt8ToHFusion : OpRewritePattern<arith::FPToSIOp> {
  using OpRewritePattern<arith::FPToSIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::FPToSIOp op,
                                PatternRewriter &rewriter) const final {
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!dstTy || !dstTy.hasStaticShape() ||
        !dstTy.getElementType().isSignlessInteger(8))
      return failure();

    Value in = op.getIn();
    // Match either clamp(rint(x), -128, 127) or rint(clamp(x, -128, 127)),
    // including when clamp promoted rint(x_f16) via arith.extf f16 -> f32.
    Value matched;
    if (Value clamped = matchInt8Clamp(in)) {
      Value maybeRint = clamped;
      if (auto ext = maybeRint.getDefiningOp<arith::ExtFOp>())
        maybeRint = ext.getIn();
      if (Value rintSrc = matchRintOp(maybeRint))
        matched = unwrapToFP16Source(rintSrc);
    }
    if (!matched) {
      if (Value rintSrc = matchRintOp(in)) {
        if (Value clamped = matchInt8Clamp(rintSrc))
          matched = unwrapToFP16Source(clamped);
      }
    }
    if (!matched)
      return failure();

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Value empty = rewriter.create<tensor::EmptyOp>(loc, dstTy.getShape(),
                                                   dstTy.getElementType());
    auto roundMode = hfusion::RoundModeAttr::get(ctx, hfusion::RoundMode::RINT);
    auto enableOverflow = rewriter.getBoolAttr(false);
    auto castMode = hfusion::TypeFnAttr::get(ctx, hfusion::TypeFn::cast_signed);
    rewriter.replaceOpWithNewOp<hfusion::CastOp>(
        op, TypeRange{dstTy}, ValueRange{matched}, ValueRange{empty}, roundMode,
        enableOverflow, castMode);
    return success();
  }
};

// Exact IEEE-754 round-to-nearest-even fallback for standalone __hmf_rint on
// targets without hardware libdevice __hmf_rint (Ascend910B). Uses exact
// binary-FP identities (frac == 0.5 and floor(floor(x)*0.5) == floor(x)*0.5),
// avoiding both FP16 epsilon underflow and int32 conversion overflow.
struct TritonRintFallbackExpansion
    : OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern<triton::ExternElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymbol() != "__hmf_rint" || op.getOperands().size() != 1)
      return failure();

    Value x = op.getOperands().front();
    auto tensorTy = dyn_cast<RankedTensorType>(x.getType());
    if (!tensorTy)
      return failure();
    auto floatTy = dyn_cast<FloatType>(tensorTy.getElementType());
    if (!floatTy)
      return failure();

    Location loc = op.getLoc();
    auto makeConst = [&](double val) -> Value {
      APFloat apf(val);
      bool losesInfo = false;
      apf.convert(floatTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
      return rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(tensorTy, apf));
    };

    Value half = makeConst(0.5);
    Value one = makeConst(1.0);

    Value floorX = rewriter.create<math::FloorOp>(loc, x);
    Value frac = rewriter.create<arith::SubFOp>(loc, x, floorX);
    Value floorPlusOne = rewriter.create<arith::AddFOp>(loc, floorX, one);

    Value isHalf = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, frac, half);
    Value halfFloor = rewriter.create<arith::MulFOp>(loc, floorX, half);
    Value halfFloorInt = rewriter.create<math::FloorOp>(loc, halfFloor);
    Value isEven = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, halfFloorInt, halfFloor);
    Value halfRound =
        rewriter.create<arith::SelectOp>(loc, isEven, floorX, floorPlusOne);

    Value gtHalf = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, frac, half);
    Value normalRound =
        rewriter.create<arith::SelectOp>(loc, gtHalf, floorPlusOne, floorX);

    Value result =
        rewriter.create<arith::SelectOp>(loc, isHalf, halfRound, normalRound);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {
struct TritonToHFusionPass
    : public mlir::triton::impl::TritonToHFusionBase<TritonToHFusionPass> {
  using TritonToHFusionBase::TritonToHFusionBase;
  void runOnOperation() override;
};
} // namespace

void TritonToHFusionPass::runOnOperation() {
  auto module = getOperation();

  // Phase 1: lower Triton ops and fuse saturating RINT FP16->INT8 patterns
  // before any fallback expansion of standalone __hmf_rint.
  RewritePatternSet patterns(&getContext());
  patterns.add<TritonHistogramToHFusionConversion>(patterns.getContext());
  patterns.add<TritonFpToFpToHFusionConversion>(patterns.getContext());
  patterns.add<TritonModToHFusionConversion>(patterns.getContext());
  patterns.add<TritonConv1dToHFusionConversion>(patterns.getContext());
  patterns.add<TritonRintClampFP16ToInt8ToHFusion>(patterns.getContext());

  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Phase 2: expand any surviving standalone __hmf_rint on non-910_95 targets.
  if (!compileOn910_95) {
    RewritePatternSet fallbackPatterns(&getContext());
    fallbackPatterns.add<TritonRintFallbackExpansion>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(fallbackPatterns)))) {
      signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToHFusionPass() {
  return std::make_unique<TritonToHFusionPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToHFusionPass(const TritonToHFusionOptions &options) {
  return std::make_unique<TritonToHFusionPass>(options);
}
