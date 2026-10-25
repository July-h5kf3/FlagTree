#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  using ConvertOpToLLVMPattern<Fp4ToFpOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType = op.getType().getElementType();
    bool toFp16 = elemType.isF16();
    unsigned mantissaBits = toFp16 ? 10 : 7;
    unsigned exponentBias = toFp16 ? 15 : 127;
    unsigned pointFive = toFp16 ? 0x3800 : 0x3f00;
    op.emitWarning(
        "using non-native FP4 conversion on MetaX: decoding e2m1 to ")
        << elemType << " with integer operations";

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto packed = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> results;
    results.reserve(packed.size() * 2);
    for (Value byte : packed) {
      Value bits = b.zext(i16_ty, byte);
      // Fp4ToFpOp's layout places each byte's low nibble before its high
      // nibble.
      for (unsigned shift : {0, 4}) {
        Value nibble = b.lshr(bits, b.i16_val(shift));
        Value sign = b.shl(b.and_(nibble, b.i16_val(8)), b.i16_val(12));
        Value em = b.and_(nibble, b.i16_val(7));
        Value body = b.shl(em, b.i16_val(mantissaBits - 1));
        Value normal = b.icmp_ne(b.and_(em, b.i16_val(6)), b.i16_val(0));
        body = b.select(
            normal, b.add(body, b.i16_val((exponentBias - 1) << mantissaBits)),
            body);
        // E2M1's only nonzero subnormal has magnitude 0.5 in either output
        // type.
        body =
            b.select(b.icmp_eq(em, b.i16_val(1)), b.i16_val(pointFive), body);
        Value decoded = b.or_(sign, body);
        // MetaX's LLVM representation stores BF16 as i16.
        results.push_back(toFp16 ? b.bitcast(decoded, f16_ty) : decoded);
      }
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                          rewriter, op.getType()));
    return success();
  }
};
} // namespace

void mlir::triton::METAX::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
