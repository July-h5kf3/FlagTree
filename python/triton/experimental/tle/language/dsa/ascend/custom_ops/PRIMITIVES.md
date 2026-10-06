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

| Source under `aarch64-linux/asc/impl/` | Copied implementation |
| --- | --- |
| `basic_api/dav_c220/kernel_operator_data_copy_impl.h:245-258` | `DataCopyGM2L1ND2NZImplBase`, INT8 branch |
| `basic_api/dav_c220/kernel_operator_mm_impl.h:50-68,163-176` | `LoadData2DL12L0BCal`, `LoadData2DL12L0BTransposeCal`, INT8 branches |
| `basic_api/kernel_operator_mm_base_impl.h:173-178` and `basic_api/dav_c220/kernel_operator_mm_impl.h:191-209,424-441,462-475` | Load3D FMatrix/padding setup and INT8 L1-to-L0A instruction |
| `c_api/instr_impl/npu_arch_2201/cube_compute_impl/asc_mmad_impl.h:146-154` | INT8 no-offset `asc_mmad_impl` overload |
| `basic_api/dav_c220/kernel_operator_fixpipe_impl.h:65-75` | `SetFixpipeNz2ndFlagImpl` register packing |
| `c_api/instr_impl/npu_arch_2201/cube_datamove_impl/asc_copy_l0c2gm_impl.h:174-182` | INT32-to-INT32 `asc_copy_l0c2gm_impl` overload |
| `basic_api/dav_c220/kernel_operator_vec_vconv_impl.h:801-828,163-187,585-609` | `CastImpl` count masking/strides and `CastIntrinsicsImpl` modes 1-5 |

For MMA, the corresponding AscendC `MmadCal` implementation is at
`basic_api/dav_c220/kernel_operator_mm_impl.h:341-360`. Its INT8 path ends in
`mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource, cmatrixInitVal)`;
for this no-bias interface, `isBias=false`, so `cmatrixInitVal` is unchanged.
CANN's compiler header `tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h:1451`
declares `mad` as a `clang_builtin_alias` of `__builtin_cce_mad`.
`mmad_int8.cpp` calls that builtin directly. There is no C++ matrix-multiply
loop behind this overload to copy; the builtin emits the Cube instruction.
The other entries likewise retain the leaf instructions from their CANN source
bodies, as the existing MrgSort custom op retains `vmrgsort4`.

Source hashes are recorded in `primitives_sources.json`; paths are relative to
the CANN installation root. The C++ files retain Huawei's copyright notices and are
covered by `LICENSE.CANN` (CANN Open Software License Agreement Version 2.0).

This is the currently validated primitive set for the explicit-address path,
not a proof that all nine are irreducible. Native Cast replacements and
compiler-managed `tl.dot` can express the computations, but the current full
attention replacements still encounter UB or lowering limits. Reassess those
constraints before removing a primitive.
