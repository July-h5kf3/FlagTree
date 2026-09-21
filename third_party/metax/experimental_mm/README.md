# Experimental C550 MM LLVM optimization

This opt-in LLVM transformation runs after MetaX external-library linking and before MACA machine-code generation. It recognizes the lowered 128x128x256 INT8 MM tile by CFG, def-use chains, byte ownership, and affine addresses. It generates the copy/MMA pipeline, flattens input addresses, permutes input/output ownership, removes the shared output exchange, and packs output stores. It never loads a saved candidate IR file.

The pass is specialized, not a general LLVM scheduler. The caller must provide the exact M/N/K shape, 256 threads, a canonical ceil-div grid, contiguous INT8 A and transposed-contiguous INT8 B, contiguous FP32 row/column scales, contiguous BF16 output, and 16-byte aligned bases. N must be divisible by 128. Unsupported IR is rejected when the optimization is requested; it is never silently partially transformed.

## Build

The ordinary backend does not require the optional executable. With an existing compatible LLVM SDK:

```sh
cmake -S third_party/metax/experimental_mm -B build/metax-mm \
  -DLLVM_DIR="$LLVM_SYSPATH/lib/cmake/llvm" -DCMAKE_BUILD_TYPE=Release
cmake --build build/metax-mm --parallel 2
cp build/metax-mm/shared-load-address third_party/metax/backend/
```

A full FlagTree build can enable the same target with `TRITON_METAX_EXPERIMENTAL_MM=ON`. When enabled, the build selects the validated MetaX plugin 0.6.2; the default build retains plugin 0.6.1. The backend package includes the executable. Editable builds place it beside `compiler.py`; normal builds place it in the backend build package. The binary and Python adapter are included in the backend cache key. No optimization option is enabled by default.

The validated environment uses the SDK reporting LLVM 22.0.0git (distributed under the name `metax-llvm19`) and the MACA 3.7.2 consumer. `address_pass.py` transports the vendor calling convention and capture-attribute syntax between those versions. The C++ code performs the instruction transformation. Other SDK/consumer combinations have not been validated.

## Backend options

```python
options = {
    "num_warps": 4,
    "num_stages": 1,
    "pipeline": "cpasync",
    "pipeline_load_num": 2,
    "enable_fp_fusion": False,
    "experimental_mm_pipeline": True,
    "experimental_mm_shape": (M, N, K),
    "experimental_mm_group_rows": 1,
}
```

Non-split K must be divisible by 256 and the INT32 accumulation must be safe. The pass proves the row masks, including scalar/v2/v4 row-scale loads. Invalid M rows may be clamped only because their output stores remain masked. CTA groups of 1..64 include incomplete final groups. Large outputs use explicitly zero-extended element indices; padded M*N must fit a positive signed i32 element index, and A/B byte ranges must fit signed i32.

`experimental_mm_rematerialize_indices=True` recomputes thread-derived indices at the peeled tail instead of keeping all of them live across the steady-state loop. A tied empty inline asm preserves the thread value and prevents common-subexpression elimination from restoring those long live ranges; it emits no hardware instruction or barrier. Only pure entry-defined arithmetic is cloned. The MMA/copy schedule, memory operations, output layout, and arithmetic order stay the same. This option requires both the pipeline and an explicit shape, defaults to false, and participates in the backend cache key. Measure it per configuration: removing spills changes physical register allocation and does not guarantee a speedup. The recorded plan enables it on 47 compiler routes and keeps it off for `(1848,1536,128256)` because that route regresses when enabled.

`experimental_mm_split_partial_i32=S` is an explicit **ABI-changing lowering contract**, intended for an MM plan that also owns scratch storage and final reduction. It must not be passed to an ordinary BF16-output launcher. The fifth pointer then addresses INT32 scratch `[S,M,N]`, grid.y is S, and the compiler emits a chunk of `ceil(K/256)/S` iterations. The input row stride remains K. Tiles must divide evenly across splits; each chunk contains 2..511 tiles, and K must be divisible by 16. K-tail masks are proved over the complete reduction interval and regenerated for each initial/future copy.

The plan must run a second reduction over scratch, use INT64 when K>131071, then convert once to FP32 and evaluate `(sum*SA)*SB` before BF16 conversion. This explicitly requests exact large-K MM semantics; it is not a claim that overflowing INT32 source arithmetic is equivalent to INT64 accumulation. The example `test/split_runtime.py` implements this caller contract. Allocation and all component launches belong to the plan; timings include both kernels.

## Validation

```sh
python -m pytest third_party/metax/experimental_mm/test -q
```

These tests compile ordinary source fixtures with the default backend, then test the optional structural transform, including incomplete row tiles, CTA grouping, >2 GiB output, split-K, wrong shape, altered tail masks, and accumulation bounds. They need the MetaX runtime/compiler and optional executable, but do not launch GPU work. Recorded GPU validation ([details](VALIDATION.md)) covers: 48 former manual routes replaced, 72 retained source/config routes, and all 120 shapes validated. Native `(4096,1,152064)` is excluded from timing after an earlier native memory violation.
