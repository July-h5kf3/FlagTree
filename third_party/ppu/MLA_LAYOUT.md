# Optional INT8 MLA operand reuse

These PPU backend options reuse INT8 NoPE/PV operands and retain the FP32 update graph.
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
| `ppu_scalarize_mla_p15` | `false` | Admit a proven radix256 shift seed when PV scalarization is requested. |
| `ppu_pipeline_mla_kv` | `false` | Start the next KV copy after the current iteration's final KV read. |
| `ppu_pipeline_mla_kv_pages` | `false` | Request page-result reuse on legacy paths; the H32/K128 bundle tries it automatically. |

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

## Optional P15 probability seeds

P14 uses an integer high product multiplied by127 before the residual product.
An operator may instead encode normalized probabilities on a finer radix256
grid and request `ppu_scalarize_mla_p15` together with `ppu_scalarize_mla_pv`.
The compiler then admits the existing scalar or vector shift-by8 seed. It does
not change the probability encoding, MMA inputs or floating expressions itself.
The original multiplication-by127 path remains supported with either flag value.

The P15 path retains the existing geometry, earlier-transform counters,
shared-read and FP32 dataflow checks. Unsupported shifts or unproved expressions
are rejected before mutation. `PPUMlaPvScalarizationTest` exercises scalar/vector
seeds with the option off and on, including rejection and idempotence.
The Python binding retains its three-argument form; its optional fourth argument
is `allow_p15=False`.

A consumer must retain older candidate sets when this option is absent. For
persistent tuning caches, normalize all candidate fields to the same key set,
including any probability-radix constexpr, before adding the P15 candidates.

## KV copy pipeline

The pipeline targets the existing MLA loop with 512 threads with a 128 KiB shared KV
region and the earlier reuse, async-copy, prefetch and scalarization transforms.
Before LLVM lowering, the backend proves that the supported i32 loop bounds are
equal for every thread in the CTA. It accepts public entry arguments, program IDs, supported scalar
arithmetic/loads and whole-program scalar reductions. Opaque assembly, volatile
loads, arguments of private functions or blocks other than the entry block, unknown producers, nested
loops and unsupported steps retain the original schedule.

The LLVM transform enumerates all 512 threads to verify the complete copy/read
footprints. It checks all loop writers, calls, source dependencies and accesses
after the last KV read before changing IR. Later probability/scale reads must be
disjoint from KV. The first copy stays in the loop preheader; a following copy
runs only when the original next/end condition permits another iteration. The
existing CTA barrier after all KV reads precedes the overwrite. The next header
retains its wait and publication barrier. The final iteration issues no extra
copy. MMA inputs and FP32 expressions are unchanged.

After this rewrite, the supported 35-MMA PV continuation schedules its existing
scalar FP32 output lanes through their SSA dependencies. This shortens the delay
between independent integer products and their output updates without moving
instructions across the copy or synchronization boundary. The continuation must
contain the expected 32-lane output pack and no writers, synchronization or
unknown effects. Rejected continuations retain their instruction order. This is
part of the existing pipeline option and adds no operator configuration.

Page reuse adds one current/next result PHI pair per page: four for the 128 KiB
KV tile and two for 64 KiB tiles, including H32/K128. It requires matching the
canonical table loads and their exact pointer/predicate aliases. The skip
edge cannot feed another loop iteration. Reads and copies remain inside each
operator call; no input conversion or page lookup is cached across calls.

Inspect `mla_kv_loop_uniform`, `mla_kv_pipelined` and
`mla_kv_page_results_reused` in compiled metadata. A requested option can still
have an application count of zero. Consumers should retain their earlier
candidates and normalize supported option defaults before SQL caching; older
packages must not receive the new keywords.

## Validation

`PPUMlaNopePrefetchTest`, `PPUMlaPvScalarizationTest`,
`PPUMlaKvPipelineTest` and `PPUMlaPvTailScheduleTest` are registered with
CTest when `TRITON_BUILD_UT=ON`. Additional LLVM tests accept real lowered MLA
modules to check immutable query reads, scale publication and copy ordering.
`test_uniform_mla_loop.py` checks supported and rejected scalar loop bounds
without executing a GPU kernel.

`python/test/unit/language/test_ppu_int8_{transpose,subview}.py` checks packed
load correctness. PPU unit tests under `third_party/ppu/python/test/` check
shared stores, alignment/stride fallbacks, immediate offsets, KV-view guards,
async-copy guards and idempotence. Operator integration must additionally
validate its original output/LSE oracles and complete decode plus merge time.

Consumers can inspect the backend option schema before adding these options
to their tuning space. Older compiler packages should retain their existing
configuration space without forwarding unsupported option keywords.

`ppu_mla_h64_k128` opts the dense INT8 MLA kernel into the 64-head,
128-token path on a 16-warp PPU CTA. It selects a `[1, 16]` PV warp layout,
reuses the 64 KiB KV buffer, hoists eight query fragments, publishes FP32
scales, streams NoPE/PV operands, and pipelines the next KV tile before the
remaining PV instructions. The same address, writer, SSA and uniform-loop
proofs apply to these supported geometries. Encoded shared storage is 123392
bytes. Two signed INT8 PV products and FP32 recurrence state are preserved.
The option defaults to false; unsupported shapes keep their existing lowering.

`ppu_mla_h32_k128` enables the 32-head, 128-token path on an eight-warp
PPU CTA. It uses a `[1, 8]` PV warp layout and a 64 KiB shared KV buffer.
Query fragments remain in shared memory. FP32 scale publication, NoPE/PV
streaming and KV prefetch use the same effect and SSA checks with the actual
256-thread CTA: 16 copies, 64 unique PV word readers and 49 remaining PV MMA
instructions. Loop indices retain their original 32-bit or 64-bit type.
The two FP32 scale planes raise encoded shared storage from 94208 to 94464
bytes. Two signed INT8 PV products and FP32 recurrence state are preserved.
The bundle first tries to reuse the two page-table results. If the attempt is
rejected without mutation, the compiler retries the existing KV pipeline without
page-result reuse; if both attempts are rejected, the original lowering remains. Successful reuse
records `mla_kv_pipelined=1` and `mla_kv_page_results_reused=2`, retaining the
existing four KR prefetches. The `ppu_pipeline_mla_kv_pages` option continues to
control the legacy paths. The H32 bundle option still defaults to false;
configurations that do not enable it retain their prior lowering.

The 64-head and 32-head paths schedule PV in batches of four independent output tiles.
The dependency graph retains every original MMA input and FP32 expression,
while completing a batch before starting more output state. The same KV-reader
proof permits the next copy with 51 or 49 PV MMA instructions remaining,
respectively. This whole PV schedule replaces the separate continuation-only
tail schedule on both paths. Scalarization callers that omit the batch argument
retain their existing schedule.

## Optional H32 row-partial gather

`ppu_gather_mla_row_reductions` defaults to false. With the existing H32/K128
bundle it can replace two supported row-reduction publication phases by six
aligned vector reads of the original partial tables. Each reader combines
partials in the original XOR2/XOR1 order, keeping FP32 arithmetic and flags.
The NoPE/RoPE/PV operands, query/key layouts and shared capacity stay unchanged.

The proof requires the exact two table regions, complete per-thread writer/read
coverage, a closed intermediate reduction graph, and a subsequent overwrite
before any other read of the removed writeback values. Unknown effects or
unmatched geometry retain the original lowering. The initial publication and
surrounding lifetime barriers remain; only the two intermediate completion
barriers are removed after the proof succeeds. Inspect
`mla_row_reductions_gathered` (0 or 6) in metadata.

Keep retained operator configurations available. This transformation is an
optional tuning candidate because representative head counts have different
performance outcomes. The operator can discover the option schema and append
enabled H32 candidates while preserving normalized disabled configurations.

## Optional unique scale writers

`ppu_unique_mla_scale_writers` defaults to false. It selects one writer for
each eight-byte packet in the two early FP32 scale tables. The recognized
256-thread H32 layout has sixteen distinct packets per table, with sixteen
threads storing each packet. The pass proves complete address coverage and
equal payloads for each address class before replacing the two stores with
predicated vector stores. Metadata `mla_scale_stores_deduplicated` is 0 or 2.

Payload equivalence follows published row-table reads, pure operations, the
branch condition of each exp2 diamond, and the initialized loop row state.
Unknown values, effects, control or geometry reject before mutation. Original
arithmetic, memory addresses, barriers and non-store instructions are retained.
The SDK may lower the predicate through execution masks and scalar branches;
fewer active writers do not imply fewer issued waits or improved latency.
Retain disabled configurations and measure each tuning key before selection.
