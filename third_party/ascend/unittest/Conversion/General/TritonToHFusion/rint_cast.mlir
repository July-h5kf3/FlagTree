// RUN: triton-opt %s -split-input-file -triton-to-hfusion | FileCheck %s

// CHECK-LABEL: tt.func @fuse_clamp_rint_fp16_to_int8
tt.func @fuse_clamp_rint_fp16_to_int8(%arg0: tensor<256xf16>) -> tensor<256xi8> {
  // CHECK-NOT: tt.extern_elementwise
  // CHECK-NOT: arith.fptosi
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<256xi8>
  // CHECK: %[[CAST:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<256xf16>) outs(%[[EMPTY]] : tensor<256xi8>) -> tensor<256xi8>
  // CHECK: tt.return %[[CAST]]
  %lo = arith.constant dense<-1.280000e+02> : tensor<256xf16>
  %hi = arith.constant dense<1.270000e+02> : tensor<256xf16>
  %r = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__hmf_rint"} : (tensor<256xf16>) -> tensor<256xf16>
  %c = tt.clampf %r, %lo, %hi, propagateNan = none : tensor<256xf16>
  %q = arith.fptosi %c : tensor<256xf16> to tensor<256xi8>
  tt.return %q : tensor<256xi8>
}

// -----

// CHECK-LABEL: tt.func @fuse_minmax_rint_extf_fp16_to_int8
tt.func @fuse_minmax_rint_extf_fp16_to_int8(%arg0: tensor<128xf16>) -> tensor<128xi8> {
  // CHECK-NOT: tt.extern_elementwise
  // CHECK-NOT: arith.fptosi
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<128xi8>
  // CHECK: %[[CAST:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<128xf16>) outs(%[[EMPTY]] : tensor<128xi8>) -> tensor<128xi8>
  // CHECK: tt.return %[[CAST]]
  %lo = arith.constant dense<-1.280000e+02> : tensor<128xf32>
  %hi = arith.constant dense<1.270000e+02> : tensor<128xf32>
  %x32 = arith.extf %arg0 : tensor<128xf16> to tensor<128xf32>
  %r = tt.extern_elementwise %x32 {libname = "", libpath = "", pure = true, symbol = "__hmf_rint"} : (tensor<128xf32>) -> tensor<128xf32>
  %mx = arith.maxnumf %r, %lo : tensor<128xf32>
  %mn = arith.minnumf %mx, %hi : tensor<128xf32>
  %q = arith.fptosi %mn : tensor<128xf32> to tensor<128xi8>
  tt.return %q : tensor<128xi8>
}

// -----

// CHECK-LABEL: tt.func @standalone_rint_fallback_910b
tt.func @standalone_rint_fallback_910b(%arg0: tensor<64xf16>) -> tensor<64xf16> {
  // CHECK-NOT: tt.extern_elementwise
  // CHECK: math.floor
  // CHECK: arith.cmpf oeq
  // CHECK: arith.select
  %r = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__hmf_rint"} : (tensor<64xf16>) -> tensor<64xf16>
  tt.return %r : tensor<64xf16>
}

// -----

// CHECK-LABEL: tt.func @skip_plain_fptosi
tt.func @skip_plain_fptosi(%arg0: tensor<64xf16>) -> tensor<64xi8> {
  // CHECK-NOT: hfusion.cast
  // CHECK: arith.fptosi %arg0 : tensor<64xf16> to tensor<64xi8>
  %q = arith.fptosi %arg0 : tensor<64xf16> to tensor<64xi8>
  tt.return %q : tensor<64xi8>
}

// -----

// CHECK-LABEL: tt.func @fuse_clamp_extf_rint_fp16_to_int8
tt.func @fuse_clamp_extf_rint_fp16_to_int8(%arg0: tensor<256xf16>) -> tensor<256xi8> {
  // CHECK-NOT: tt.extern_elementwise
  // CHECK-NOT: arith.fptosi
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<256xi8>
  // CHECK: %[[CAST:.*]] = hfusion.cast {cast = #hfusion.type_fn<cast_signed>, enable_overflow = false, round_mode = #hfusion.round_mode<rint>} ins(%arg0 : tensor<256xf16>) outs(%[[EMPTY]] : tensor<256xi8>) -> tensor<256xi8>
  // CHECK: tt.return %[[CAST]]
  %lo = arith.constant dense<-1.280000e+02> : tensor<256xf32>
  %hi = arith.constant dense<1.270000e+02> : tensor<256xf32>
  %r = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "__hmf_rint"} : (tensor<256xf16>) -> tensor<256xf16>
  %ext = arith.extf %r : tensor<256xf16> to tensor<256xf32>
  %c = tt.clampf %ext, %lo, %hi, propagateNan = none : tensor<256xf32>
  %q = arith.fptosi %c : tensor<256xf32> to tensor<256xi8>
  tt.return %q : tensor<256xi8>
}
