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
The tests show QK and transposed PV layouts, including strided GM input.

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

## Build and validation

The existing `AscendCustomOpsBitcode` CMake target builds both normal and mixed
entries into `custom_ops.bc`. The existing `build_custom_ops.sh` provides the
manual rebuild path. `CUBE_PRIMITIVES_BITCODE` and `CAST_PRIMITIVES_BITCODE` remain
available as aliases for this shared bundle; separate build scripts or binary
bundles are not needed.

With an Ascend build and CANN environment, on a verified idle NPU:

```sh
bash python/triton/experimental/tle/language/dsa/ascend/custom_ops/build_custom_ops.sh
pytest -q python/test/unit/language/test_ascend_cube_primitives.py \
  python/test/unit/language/test_ascend_cast_primitives.py
```

The tests cover exact INT32 QK/PV results, rounding boundaries, partial Cast
counts, mixed Cube/Vector execution, and input updates under NPUGraph replay.

## Provenance and scope

The C++ implementations are derived from CANN 9.1 SDK headers. Their hashes are
recorded in `primitives_sources.json`; SDK paths are relative to
`aarch64-linux/asc`. The C++ files retain Huawei's copyright notices and are
covered by `LICENSE.CANN` (CANN Open Software License Agreement Version 2.0).

This is the currently validated primitive set for the explicit-address path,
not a proof that all nine are irreducible. Native Cast replacements and
compiler-managed `tl.dot` can express the computations, but the current full
attention replacements still encounter UB or lowering limits. Reassess those
constraints before removing a primitive.
