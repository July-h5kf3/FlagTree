// RUN: triton-opt %s -split-input-file -commonir-to-hivm | FileCheck %s

// 1. P1-1: Fold GM -> staging alloc -> to_tensor -> to_buffer -> hivm.copy(#cbuf)
//    into direct GM -> #cbuf memref.copy.
module {
  // CHECK-LABEL: func.func @fold_staging_hivm_copy_l1
  // CHECK-NOT: hivm.hir.copy
  // CHECK: %[[L1:.*]] = memref.alloc() : memref<64x512xi8, #hivm.address_space<cbuf>>
  // CHECK: %[[GM:.*]] = memref.memory_space_cast %{{.*}} : memref<64x512xi8, strided<[512, 1]>> to memref<64x512xi8, strided<[512, 1]>, #hivm.address_space<gm>>
  // CHECK: memref.copy %[[GM]], %[[L1]]
  func.func @fold_staging_hivm_copy_l1(%ptr: memref<?xi8>) {
    %l1 = memref.alloc() : memref<64x512xi8, #hivm.address_space<cbuf>>
    %gm = memref.reinterpret_cast %ptr to offset: [0], sizes: [64, 512], strides: [512, 1] : memref<?xi8> to memref<64x512xi8, strided<[512, 1]>>
    %stage = memref.alloc() : memref<64x512xi8>
    memref.copy %gm, %stage : memref<64x512xi8, strided<[512, 1]>> to memref<64x512xi8>
    %t = bufferization.to_tensor %stage restrict writable : memref<64x512xi8> to tensor<64x512xi8>
    %b = bufferization.to_buffer %t : tensor<64x512xi8> to memref<64x512xi8>
    hivm.hir.copy ins(%b : memref<64x512xi8>) outs(%l1 : memref<64x512xi8, #hivm.address_space<cbuf>>)
    return
  }
}

// -----

// 2. Bug-1: Preserve #hivm.address_space<ub> across tensor.expand_shape and
//    linalg.broadcast on 1D UB subviews.
module {
  // CHECK-LABEL: func.func @fold_expand_shape_ub_1d
  // CHECK-NOT: tensor.expand_shape
  // CHECK: %[[EXP:.*]] = memref.expand_shape %{{.*}} {{\[\[}}0, 1{{\]\]}} output_shape [1, 64] : memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>> into memref<1x64xf32, strided<[64, 1], offset: 64>, #hivm.address_space<ub>>
  // CHECK: bufferization.to_tensor %[[EXP]]
  func.func @fold_expand_shape_ub_1d() -> tensor<1x64xf32> {
    %ub = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %sv = memref.subview %ub[64] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>>
    %t = bufferization.to_tensor %sv restrict : memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>> to tensor<64xf32>
    %e = tensor.expand_shape %t [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
    return %e : tensor<1x64xf32>
  }

  // CHECK-LABEL: func.func @fold_broadcast_ub_1d
  // CHECK-NOT: linalg.broadcast
  // CHECK: %[[EXP:.*]] = memref.expand_shape %{{.*}} {{\[\[}}0, 1{{\]\]}} output_shape [1, 64] : memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>> into memref<1x64xf32, strided<[64, 1], offset: 64>, #hivm.address_space<ub>>
  // CHECK: %[[EXPT:.*]] = bufferization.to_tensor %[[EXP]] restrict
  // CHECK: hivm.hir.vbrc ins(%[[EXPT]] : tensor<1x64xf32>) outs(%{{.*}} : tensor<16x64xf32>) broadcast_dims = [0] -> tensor<16x64xf32>
  func.func @fold_broadcast_ub_1d() -> tensor<16x64xf32> {
    %ub = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    %sv = memref.subview %ub[64] [64] [1] : memref<128xf32, #hivm.address_space<ub>> to memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>>
    %t = bufferization.to_tensor %sv restrict : memref<64xf32, strided<[1], offset: 64>, #hivm.address_space<ub>> to tensor<64xf32>
    %e = tensor.empty() : tensor<16x64xf32>
    %b = linalg.broadcast ins(%t : tensor<64xf32>) outs(%e : tensor<16x64xf32>) dimensions = [0]
    return %b : tensor<16x64xf32>
  }
}

// -----

// 3. Bug-2: Split 3D GM load + reshape -> 2D matmul into D0 independent
//    contiguous 2D loads and D0 matmuls.
module {
  // CHECK-LABEL: func.func @decompose_cube_3d_load_matmul
  // CHECK-NOT: tensor.reshape
  // CHECK-NOT: memref.alloc() : memref<2x16x64xi8>
  // CHECK: memref.subview %{{.*}}[0, 0, 0] [1, 16, 64] [1, 1, 1]
  // CHECK: %[[A0:.*]] = memref.alloc() : memref<16x64xi8>
  // CHECK: memref.copy %{{.*}}, %[[A0]]
  // CHECK: %[[Q0:.*]] = bufferization.to_tensor %[[A0]] restrict writable
  // CHECK: memref.subview %{{.*}}[1, 0, 0] [1, 16, 64] [1, 1, 1]
  // CHECK: %[[A1:.*]] = memref.alloc() : memref<16x64xi8>
  // CHECK: memref.copy %{{.*}}, %[[A1]]
  // CHECK: %[[Q1:.*]] = bufferization.to_tensor %[[A1]] restrict writable
  // CHECK: linalg.matmul ins(%[[Q0]], %arg1 : tensor<16x64xi8>, tensor<64x64xi8>)
  // CHECK: bufferization.materialize_in_destination
  // CHECK: linalg.matmul ins(%[[Q1]], %arg1 : tensor<16x64xi8>, tensor<64x64xi8>)
  // CHECK: bufferization.materialize_in_destination
  func.func @decompose_cube_3d_load_matmul(%q_ptr: memref<?xi8>, %k_t: tensor<64x64xi8>, %c_ptr: memref<?xi32>) {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<32x64xi32>
    %init = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x64xi32>) -> tensor<32x64xi32>
    %q_gm = memref.reinterpret_cast %q_ptr to offset: [0], sizes: [2, 16, 64], strides: [4096, 64, 1] : memref<?xi8> to memref<2x16x64xi8, strided<[4096, 64, 1]>>
    %q_alloc = memref.alloc() : memref<2x16x64xi8>
    memref.copy %q_gm, %q_alloc : memref<2x16x64xi8, strided<[4096, 64, 1]>> to memref<2x16x64xi8>
    %q_t = bufferization.to_tensor %q_alloc restrict writable : memref<2x16x64xi8> to tensor<2x16x64xi8>
    %shape = arith.constant dense<[32, 64]> : tensor<2xi64>
    %q_2d = tensor.reshape %q_t(%shape) : (tensor<2x16x64xi8>, tensor<2xi64>) -> tensor<32x64xi8>
    %mm = linalg.matmul ins(%q_2d, %k_t : tensor<32x64xi8>, tensor<64x64xi8>) outs(%init : tensor<32x64xi32>) -> tensor<32x64xi32>
    %c_gm = memref.reinterpret_cast %c_ptr to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<?xi32> to memref<32x64xi32, strided<[64, 1]>>
    bufferization.materialize_in_destination %mm in writable %c_gm : (tensor<32x64xi32>, memref<32x64xi32, strided<[64, 1]>>) -> ()
    return
  }
}

// -----

// 4. P0-2: Fold splat fill and in-place arith/math UB updates directly into
//    the destination UB buffer via hivm.hir.vbrc / hivm.hir.v*.
module {
  // CHECK-LABEL: func.func @fold_splat_and_inplace_ub_arith
  // CHECK-NOT: hivm.hir.copy
  // CHECK: %[[ACC:.*]] = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ACC]] : memref<32x256xf32, #hivm.address_space<ub>>)
  // CHECK: %[[SV:.*]] = memref.subview %[[ACC]][0, 0] [16, 256] [1, 1]
  // CHECK: %[[T0:.*]] = bufferization.to_tensor %[[SV]] restrict writable
  // CHECK: %[[MUL:.*]] = hivm.hir.vmul ins(%[[T0]], %{{.*}} : tensor<16x256xf32>, f32) outs(%[[T0]] : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: %[[ADD:.*]] = hivm.hir.vadd ins(%[[MUL]], %arg0 : tensor<16x256xf32>, tensor<16x256xf32>) outs(%[[MUL]] : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: bufferization.materialize_in_destination %[[ADD]] in restrict writable %[[SV]]
  func.func @fold_splat_and_inplace_ub_arith(%x: tensor<16x256xf32>) {
    %acc = memref.alloc() : memref<32x256xf32, #hivm.address_space<ub>>
    %zero = arith.constant dense<0.0> : tensor<32x256xf32>
    %zm = bufferization.to_buffer %zero : tensor<32x256xf32> to memref<32x256xf32>
    hivm.hir.copy ins(%zm : memref<32x256xf32>) outs(%acc : memref<32x256xf32, #hivm.address_space<ub>>)
    %sv = memref.subview %acc[0, 0] [16, 256] [1, 1] : memref<32x256xf32, #hivm.address_space<ub>> to memref<16x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>
    %t0 = bufferization.to_tensor %sv restrict : memref<16x256xf32, strided<[256, 1]>, #hivm.address_space<ub>> to tensor<16x256xf32>
    %half = arith.constant dense<0.5> : tensor<16x256xf32>
    %mul = arith.mulf %t0, %half : tensor<16x256xf32>
    %add = arith.addf %mul, %x : tensor<16x256xf32>
    %bm = bufferization.to_buffer %add : tensor<16x256xf32> to memref<16x256xf32>
    hivm.hir.copy ins(%bm : memref<16x256xf32>) outs(%sv : memref<16x256xf32, strided<[256, 1]>, #hivm.address_space<ub>>)
    return
  }
}

// -----

// 5. P0-1b: Scalarize small non-unit-stride 1D GM stores while preserving
//    unit-stride GM stores.
module {
  // CHECK-LABEL: func.func @isolate_small_strided_1d_gm_store
  // CHECK-NOT: bufferization.materialize_in_destination
  // CHECK: scf.for %[[IV:.*]] = %c0 to %c8 step %c1
  // CHECK:   %[[VAL:.*]] = tensor.extract %arg0[%[[IV]]] : tensor<8xf32>
// CHECK: memref.store %[[VAL]], %{{.*}}[%c0] : memref<1xf32, #hivm.address_space<ub>>
// CHECK: memref.reinterpret_cast {{.*}} sizes: [1], strides: [1]
// CHECK: hivm.hir.store
  func.func @isolate_small_strided_1d_gm_store(%t: tensor<8xf32>, %out: memref<?xf32>) {
    %gm = memref.reinterpret_cast %out to offset: [0], sizes: [8], strides: [2] : memref<?xf32> to memref<8xf32, strided<[2]>>
    bufferization.materialize_in_destination %t in writable %gm : (tensor<8xf32>, memref<8xf32, strided<[2]>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @keep_unit_stride_1d_gm_store
  // CHECK-NOT: scf.for
  // CHECK: bufferization.materialize_in_destination
  func.func @keep_unit_stride_1d_gm_store(%t: tensor<8xf32>, %out: memref<?xf32>) {
    %gm = memref.reinterpret_cast %out to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
    bufferization.materialize_in_destination %t in writable %gm : (tensor<8xf32>, memref<8xf32, strided<[1]>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_p01b_skip_autosplit_mix
  // CHECK-NOT: memref.store
  // CHECK: bufferization.materialize_in_destination
  func.func @test_p01b_skip_autosplit_mix(%src: tensor<32xf32>, %gm: memref<?xf32>, %off: index) attributes {mix_mode = "mix"} {
    %dst = memref.reinterpret_cast %gm to offset: [%off], sizes: [32], strides: [4] : memref<?xf32> to memref<32xf32, strided<[4], offset: ?>>
    bufferization.materialize_in_destination %src in writable %dst : (tensor<32xf32>, memref<32xf32, strided<[4], offset: ?>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_p01b_explicit_vector_scope_in_mix
  // CHECK: memref.alloc() : memref<1xf32, #hivm.address_space<ub>>
  // CHECK: scf.for
  // CHECK: tensor.extract
  // CHECK: memref.store
  // CHECK: memref.reinterpret_cast {{.*}} sizes: [1], strides: [1]
  // CHECK: hivm.hir.store
  // CHECK-NOT: bufferization.materialize_in_destination
  func.func @test_p01b_explicit_vector_scope_in_mix(%src: tensor<8xf32>, %gm: memref<?xf32>, %off: index) attributes {mix_mode = "mix"} {
    scope.scope : () -> () {
      %dst = memref.reinterpret_cast %gm to offset: [%off], sizes: [8], strides: [2] : memref<?xf32> to memref<8xf32, strided<[2], offset: ?>>
      bufferization.materialize_in_destination %src in writable %dst : (tensor<8xf32>, memref<8xf32, strided<[2], offset: ?>>) -> ()
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // CHECK-LABEL: func.func @test_infer_cube_mix_mode_from_custom
  // CHECK-SAME: mix_mode = "aic"
  func.func @test_infer_cube_mix_mode_from_custom(%src: i64) attributes {mix_mode = "aiv"} {
    %buf = memref.alloc() : memref<16x64xi8, #hivm.address_space<cbuf>>
    %t = bufferization.to_tensor %buf restrict writable : memref<16x64xi8, #hivm.address_space<cbuf>> to tensor<16x64xi8>
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %res = hivm.hir.custom {
      arg_attrs = [{}, {}, {}, {}, {}],
      bitcode = "/tmp/dummy.bc",
      hivm.pipe = #hivm.pipe<PIPE_MTE1>,
      hivm.tcore_type = #hivm.tcore_type<CUBE>,
      hivm.vf_mode = #hivm.vf_mode<SIMD>,
      symbol = "custom_nd2nz_gm_to_l1_s8"
    } "nd2nz_gm_to_l1" ins(%src, %c16, %c64, %c64 : i64, i32, i32, i32) outs(%t : tensor<16x64xi8>) -> tensor<16x64xi8>
    %res_buf = bufferization.to_buffer %res : tensor<16x64xi8> to memref<16x64xi8>
    hivm.hir.copy ins(%res_buf : memref<16x64xi8>) outs(%buf : memref<16x64xi8, #hivm.address_space<cbuf>>)
    return
  }

  // CHECK-LABEL: func.func @test_infer_mix_mode_from_cube_and_ub
  // CHECK-SAME: mix_mode = "mix"
  func.func @test_infer_mix_mode_from_cube_and_ub(%src: i64, %val: tensor<16xf32>) attributes {mix_mode = "aiv"} {
    %l1 = memref.alloc() : memref<16x64xi8, #hivm.address_space<cbuf>>
    %t = bufferization.to_tensor %l1 restrict writable : memref<16x64xi8, #hivm.address_space<cbuf>> to tensor<16x64xi8>
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %res = hivm.hir.custom {
      arg_attrs = [{}, {}, {}, {}, {}],
      bitcode = "/tmp/dummy.bc",
      hivm.pipe = #hivm.pipe<PIPE_MTE1>,
      hivm.tcore_type = #hivm.tcore_type<CUBE>,
      hivm.vf_mode = #hivm.vf_mode<SIMD>,
      symbol = "custom_nd2nz_gm_to_l1_s8"
    } "nd2nz_gm_to_l1" ins(%src, %c16, %c64, %c64 : i64, i32, i32, i32) outs(%t : tensor<16x64xi8>) -> tensor<16x64xi8>
    bufferization.materialize_in_destination %res in restrict writable %l1 : (tensor<16x64xi8>, memref<16x64xi8, #hivm.address_space<cbuf>>) -> ()
    %ub = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    bufferization.materialize_in_destination %val in restrict writable %ub : (tensor<16xf32>, memref<16xf32, #hivm.address_space<ub>>) -> ()
    return
  }

  // CHECK-LABEL: func.func @test_fold_inplace_self_copy_cbuf
  // CHECK: %[[BUF:.*]] = memref.alloc() : memref<16x64xi8, #hivm.address_space<cbuf>>
  // CHECK: %[[RES:.*]] = hivm.hir.custom
  // CHECK: bufferization.materialize_in_destination %[[RES]] in restrict writable %[[BUF]]
  // CHECK-NOT: hivm.hir.copy
  func.func @test_fold_inplace_self_copy_cbuf(%src: i64) attributes {mix_mode = "mix"} {
    %buf = memref.alloc() : memref<16x64xi8, #hivm.address_space<cbuf>>
    %t = bufferization.to_tensor %buf restrict writable : memref<16x64xi8, #hivm.address_space<cbuf>> to tensor<16x64xi8>
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %res = hivm.hir.custom {
      arg_attrs = [{}, {}, {}, {}, {}],
      bitcode = "/tmp/dummy.bc",
      hivm.pipe = #hivm.pipe<PIPE_MTE1>,
      hivm.tcore_type = #hivm.tcore_type<CUBE>,
      hivm.vf_mode = #hivm.vf_mode<SIMD>,
      symbol = "custom_nd2nz_gm_to_l1_s8"
    } "nd2nz_gm_to_l1" ins(%src, %c16, %c64, %c64 : i64, i32, i32, i32) outs(%t : tensor<16x64xi8>) -> tensor<16x64xi8>
    %res_buf = bufferization.to_buffer %res : tensor<16x64xi8> to memref<16x64xi8>
    hivm.hir.copy ins(%res_buf : memref<16x64xi8>) outs(%buf : memref<16x64xi8, #hivm.address_space<cbuf>>)
    return
  }

  // CHECK-LABEL: func.func @test_fix_offset_extract_aligned_pointer
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?xi32> -> index
  // CHECK: %[[BYTES:.*]] = arith.muli %arg1, %[[C4]] : index
  // CHECK: %[[ADDR:.*]] = arith.addi %[[BASE]], %[[BYTES]] : index
  // CHECK: return %[[ADDR]]
  func.func @test_fix_offset_extract_aligned_pointer(%gm: memref<?xi32>, %off: index) -> index {
    %view = memref.reinterpret_cast %gm to offset: [%off], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
    %ptr = memref.extract_aligned_pointer_as_index %view : memref<1xi32, strided<[1], offset: ?>> -> index
    return %ptr : index
  }

  // CHECK-LABEL: func.func @test_fix_offset_extract_aligned_pointer_i8
  // CHECK: %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<?xi8> -> index
  // CHECK: %[[ADDR:.*]] = arith.addi %[[BASE]], %arg1 : index
  // CHECK-NOT: arith.muli
  // CHECK: return %[[ADDR]]
  func.func @test_fix_offset_extract_aligned_pointer_i8(%gm: memref<?xi8>, %off: index) -> index {
    %view = memref.reinterpret_cast %gm to offset: [%off], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %ptr = memref.extract_aligned_pointer_as_index %view : memref<1xi8, strided<[1], offset: ?>> -> index
    return %ptr : index
  }
}
