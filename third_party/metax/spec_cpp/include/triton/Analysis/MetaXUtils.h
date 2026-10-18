#ifndef TRITON_ANALYSIS_METAX_UTILS_H
#define TRITON_ANALYSIS_METAX_UTILS_H

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::METAX {

// Keep allocation/barrier analysis consistent with the packed lowering.
inline bool isPackedInt8MmaToDot(RankedTensorType srcTy,
                                 RankedTensorType dstTy) {
  using namespace mlir::triton::gpu;
  if (srcTy.getRank() != 2 || srcTy.getShape() != dstTy.getShape() ||
      srcTy.getShape()[0] < 16 || srcTy.getShape()[1] != 16 ||
      !srcTy.getElementType().isSignlessInteger(8))
    return false;
  auto src = dyn_cast<MACAMmaEncodingAttr>(srcTy.getEncoding());
  auto dst = dyn_cast<DotOperandEncodingAttr>(dstTy.getEncoding());
  if (!src || !dst || dst.getOpIdx() != 0 || dst.getParent() != src ||
      src.getColMajor() || src.getIsATrans() || src.getIsBTrans())
    return false;
  SmallVector<unsigned> elements{1, 1, 4};
  SmallVector<unsigned> stride{1, 1};
  if (src.getElementsMNK() != ArrayRef<unsigned>(elements) ||
      src.getElementsStride() != ArrayRef<unsigned>(stride))
    return false;
  auto conversion = minimalCvtLayout(srcTy, dstTy);
  auto ctx = srcTy.getContext();
  auto dims = conversion.getOutDimNames();
  return !llvm::is_contained(dims, StringAttr::get(ctx, "warp")) &&
         !llvm::is_contained(dims, StringAttr::get(ctx, "block"));
}

} // namespace mlir::triton::METAX

#endif
