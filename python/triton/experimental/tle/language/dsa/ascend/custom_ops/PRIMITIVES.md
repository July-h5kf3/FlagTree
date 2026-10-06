# INT8 Cube and Vector Cast primitives

These nine Ascend primitives expose CANN 9.1 `dav_c220` operations used by TLE
attention kernels. The caller owns tiling, local-memory placement, bounds,
padding, and synchronization. No attention loop or softmax is implemented here.

## Cube

| Python operation | CANN operation | Address spaces |
| --- | --- | --- |
| `cube_nd2nz_i8` | `DataCopyGM2L1ND2NZImplBase` | GM to L1 |
| `cube_load3d_a_into` | `LoadData3DV2L12L0ACal` | L1 to L0A |
| `cube_load2d_b_into` | `LoadData2DL12L0BCal` | L1 to L0B |
| `cube_load_transpose_b_into` | `LoadData2DL12L0BTransposeCal` | L1 to L0B |
| `cube_mmad_into` | `asc_mmad_impl`, INT8 to INT32 | L0A/L0B to L0C |
| `cube_set_l0c_copy_params` | `SetFixpipeNz2ndFlagImpl` | Fixpipe configuration |
| `cube_copy_l0c2gm_i32` | `asc_copy_l0c2gm_impl`, INT32 to INT32 | L0C to GM |

GM addresses are unsigned 64-bit byte addresses. Local addresses are unsigned
32-bit byte offsets in the named hardware memory space. Keep byte units when
adding offsets to addresses. The scalar parameters preserve the corresponding
CANN overload, including fields ignored by `dav_c220`.

Use `al.scope(core_mode="cube")` for Cube work. Configure L0C writeback before
copying results. Issue TLE pipeline events between producer and consumer
operations; the primitives do not insert barriers or allocate local buffers.

## Cast

`cast_fp32_to_int16` and `cast_fp16_to_int8` expose the CANN Cast count overload:
`(source, roundMode, count, out=destination)`.

| Mode | Rounding |
| ---: | --- |
| 1 | Nearest, ties to even |
| 2 | Floor |
| 3 | Ceiling |
| 4 | Nearest, ties away from zero |
| 5 | Truncate toward zero |

Source and destination must be disjoint, contiguous, 32-byte-aligned 1D UB
tensors with equal power-of-two capacity of at least 32 elements. `count` is a
compile-time integer in `[1, capacity]`; elements beyond it retain their previous
destination values. Supply finite inputs representable after rounding. UB
capacity and synchronization remain the caller's responsibility. The operations
restore normal vector-mask mode with all lanes enabled.

## Build

The existing `AscendCustomOpsBitcode` CMake target builds both normal and mixed
entries into `custom_ops.bc`. The existing `build_custom_ops.sh` provides the
manual rebuild path. `CUBE_PRIMITIVES_BITCODE` and `CAST_PRIMITIVES_BITCODE` remain
available as aliases for this shared bundle; separate build scripts or binary
bundles are not needed.

With an Ascend build and CANN environment:

```sh
bash python/triton/experimental/tle/language/dsa/ascend/custom_ops/build_custom_ops.sh
```

## Provenance and scope

The C++ implementations copy the applicable dtype branches from the CANN 9.1
SDK sources listed below. CANN tensor descriptors are flattened to the custom-op
ABI, and dtype dispatch is resolved by each fixed-type entry. Cube entries are
compiled for `dav-c220-cube`, and Cast entries for `dav-c220-vec`, so the SDK core
guards are resolved at build time. SDK debug/overflow checks and TSCM dispatch
are outside this explicit-address interface. The caller supplies valid addresses,
strides and buffer capacities.

| C++ entry | Copied CANN source under `aarch64-linux/asc/impl/` | Leaf intrinsic (compiler declaration line) |
| --- | --- | --- |
| `data_copy_nd2nz_i8` | `basic_api/dav_c220/kernel_operator_data_copy_impl.h:245-258`: INT8 branch of `DataCopyGM2L1ND2NZImplBase` | `copy_gm_to_cbuf_multi_nd2nz_b8` (981) |
| `load_data_2d_int8_b` | `basic_api/dav_c220/kernel_operator_mm_impl.h:50-68`: INT8 branch of `LoadData2DL12L0BCal` | `load_cbuf_to_cb` (1385) |
| `load_data_transpose_int8_b` | `basic_api/dav_c220/kernel_operator_mm_impl.h:163-176`: INT8 branch of `LoadData2DL12L0BTransposeCal` | `load_cbuf_to_cb_transpose` (1395) |
| `load_data_3d_int8_a` | `basic_api/kernel_operator_mm_base_impl.h:173-178` and `basic_api/dav_c220/kernel_operator_mm_impl.h:191-209,424-441,462-475`: FMatrix/padding setup and INT8 Load3D | `set_fmatrix` (2079), `set_padding` (2279), `img2colv2_cbuf_to_ca` (1349) |
| `mmad_int8` | `c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_mmad_impl.h:146-154`: INT8 no-offset overload | `mad` (1451), called as `__builtin_cce_mad` |
| `set_l0c_copy_params` | `basic_api/dav_c220/kernel_operator_fixpipe_impl.h:65-75`: `SetFixpipeNz2ndFlagImpl` register packing | `set_nd_para` (2269) |
| `copy_l0c2gm_i32` | `c_api/instr_impl/npu_arch_2201/cube_datamove_impl/asc_copy_l0c2gm_impl.h:174-182`: INT32-to-INT32 overload | `copy_matrix_cc_to_gm` (1013) |
| `cast_fp32_to_int16` | `basic_api/dav_c220/kernel_operator_vec_vconv_impl.h:801-828,585-609`: count masking/strides and rounding modes 1-5 | `vconv_f322s16{a,c,f,r,z}` (2575-2583), mask intrinsics below |
| `cast_fp16_to_int8` | `basic_api/dav_c220/kernel_operator_vec_vconv_impl.h:801-828,163-187`: count masking/strides and rounding modes 1-5 | `vconv_f162s8{a,c,f,r,z}` (2519-2527), mask intrinsics below |

Compiler declaration lines refer to CANN 9.1.0
`tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h`.
Each listed intrinsic is declared as `clang_builtin_alias(__builtin_cce_<name>)`.
The Cast mask intrinsics are `set_mask_count` (2243), `set_mask_norm` (2245),
and `set_vector_mask` (2317); they use the same declaration mechanism.
The existing MrgSort custom op's `vmrgsort4` is also such an alias (2699).
These declarations do not forward to a CANN C++ function body. Calling the
alias or its `__builtin_cce_` name selects the same compiler builtin.

In particular, CANN's INT8 `DataCopyGM2L1ND2NZImplBase` branch directly invokes
`copy_gm_to_cbuf_multi_nd2nz_b8` with the fields of `Nd2NzParams`. Its INT32
`asc_copy_l0c2gm_impl` overload directly invokes `copy_matrix_cc_to_gm` with
`sid=0` and the `QuantMode_t` cast retained here. The compiled custom entries
contain `llvm.hivm.MOV.OUT.TO.L1.MULTI.ND2NZ.s8` and
`llvm.hivm.FIX.L0C.TO.OUT.s32`, respectively. All nine entries were checked in
the linked bitcode: their calls target `llvm.hivm.*` instructions and LLVM
lifetime markers, with no calls to AscendC API functions.

`copy_l0c2gm_i32` covers the low-level C API overload above. It does not implement
the full AscendC `Fixpipe` contract. `FixpipeInfoParams` stride conversion,
`GenFixpipeTiling`, quantization-table transfers and internal pipeline barriers
are outside this interface; ND configuration is exposed separately through
`set_l0c_copy_params`. Callers provide instruction-level sizes/strides and perform
the required setup and synchronization.

For MMA, `basic_api/dav_c220/kernel_operator_mm_impl.h:341-360` contains
`MmadCal`. Its INT8 branch has the same leaf call as `asc_mmad_impl` when
`isBias=false`. There is no C++ matrix-multiply loop behind that overload.

Source hashes are recorded in `primitives_sources.json`; paths are relative to
the CANN installation root. The C++ files retain Huawei's copyright notices and are
covered by `LICENSE.CANN` (CANN Open Software License Agreement Version 2.0).

This is the currently validated primitive set for the explicit-address path,
not a proof that all nine are irreducible. Native Cast replacements and
compiler-managed `tl.dot` can express the computations, but the current full
attention replacements still encounter UB or lowering limits. Reassess those
constraints before removing a primitive.
