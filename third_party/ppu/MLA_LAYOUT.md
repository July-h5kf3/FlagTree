# Optional INT8 MLA operand reuse

These PPU backend options preserve the INT8 NoPE/P14 products and FP32 state.
They are disabled by default. They do not require warp specialization.

| Option | Default | Purpose |
| --- | --- | --- |
| `ppu_vectorize_i8_shared_store` | `false` | Pack supported shared INT8 stores without changing their bytes or addresses. |
| `ppu_reuse_mla_kv` | `false` | Let supported QK and PV operands read views of one immutable shared KV allocation. |
| `ppu_hoist_readonly_shared_matrices` | `0` | Move a bounded number of proven immutable query matrix reads out of the KV loop. |
| `ppu_async_mla_kv` | `false` | Promote a proven aligned zero-padded KV initializer to async copy, commit and wait. |
| `ppu_publish_shared_float_scales_early` | `false` | Publish supported FP32 row scales at the existing pre-PV synchronization. |
| `ppu_prefetch_mla_nope` | `false` | Schedule supported NoPE operand loads with their existing MMA consumers. |
| `ppu_scalarize_mla_pv` | `false` | Schedule supported PV seed and FP32 update lanes without changing their expressions. |

The initial KV-view matcher targets the paged decode content tile
`[256, 512]`, signed INT8 dot operands with `kWidth=4`, 16 warps and one CTA.
The backend's internal capability check is a code-generation condition, not a
claim that the PPU implements NVIDIA warp-group instructions.

## Safety conditions

- KV reuse requires one dominating immutable shared allocation, the supported
  swizzle, sufficient alignment and only supported readers. Ambiguous roots,
  escaping users and explicit deallocation are rejected before mutation.
- Read hoisting enumerates the shared address footprint and checks loop writers,
  calls and clobbers. Unknown accesses or overlapping writes retain the original
  code. The hoist count is bounded to control register pressure.
- Async promotion requires a supported contiguous initializer, alignment,
  zero padding and non-escaping initialization. It retains completion ordering.
- Early scale publication checks address bounds, types, masks, lifetime and
  side effects. Shared size and kernel metadata are updated together.
- NoPE prefetch and PV scalarization require the expected earlier transforms.
  They preserve original MMA operations, operands and side effects. The FP32
  update graph is checked rather than reassociated for an estimated speedup.

Unmatched kernels retain their original IR. Enabling an option is not a
guarantee that its matcher applies or that a configuration is faster.

The backend records application counters in compiled metadata, including
`mla_shared_kv_views_reused`, `readonly_shared_matrices_hoisted`,
`mla_async_kv_copies`, `shared_float_scales_published_early`,
`mla_nope_fragments_prefetched` and `mla_pv_scalarized`.

## Validation

`PPUMlaNopePrefetchTest` and `PPUMlaPvScalarizationTest` are registered with
CTest when `TRITON_BUILD_UT=ON`. Additional LLVM tests accept real lowered MLA
modules to check immutable query reads, scale publication and copy ordering.

`python/test/unit/language/test_ppu_int8_{transpose,subview}.py` checks packed
load correctness. PPU unit tests under `third_party/ppu/python/test/` check
shared stores, alignment/stride fallbacks, immediate offsets, KV-view guards,
async-copy guards and idempotence. Operator integration must additionally
validate its original output/LSE oracles and complete decode plus merge time.

Consumers can inspect the backend option schema before adding these options
to their tuning space. Older compiler packages should retain their existing
configuration space without forwarding unsupported option keywords.
