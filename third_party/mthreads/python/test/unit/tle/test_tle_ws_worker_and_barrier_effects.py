"""Regression tests for heterogeneous WS worker layouts and barrier memory effects."""

import re

import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from test_tle_utils import compile_musa, require_mthreads_libtriton
from triton.experimental.tle.language.gpu.mthreads.types import MusaSqmmaEncoding

require_mthreads_libtriton()


@triton.jit
def heterogeneous_default_partition(out_ptr):
    offs = tl.arange(0, 256)
    tl.store(out_ptr + offs, offs)


@triton.jit
def heterogeneous_worker_partition(out_ptr, smem):
    offs = tl.arange(0, 128)
    local = tle.gpu.local_ptr(smem, (offs, ))
    tl.store(local, offs)
    loaded = tl.load(local)
    tl.store(out_ptr + 256 + offs, loaded)


@triton.jit
def heterogeneous_ws_kernel(out_ptr):
    smem = tle.gpu.alloc((128, ), dtype=tl.int32, nv_mma_shared_layout=False)
    tle.gpu.warp_specialize(
        [
            (heterogeneous_default_partition, (out_ptr, )),
            (heterogeneous_worker_partition, (out_ptr, smem)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[32],
    )


@triton.jit
def barrier_ordered_producer(src_ptr, smem, ready, empty, iters):
    for step in range(0, iters):
        phase = step & 1
        tle.gpu.barrier_wait(empty, phaseIdx=phase)
        val = tl.load(src_ptr + step)
        tl.store(tle.gpu.local_ptr(smem, (0, )), val)
        tle.gpu.barrier_arrive(ready, phaseIdx=phase)


@triton.jit
def barrier_ordered_consumer(out_ptr, smem, ready, empty, iters):
    acc = tl.full((), 0, tl.int32)
    for step in range(0, iters):
        phase = step & 1
        tle.gpu.barrier_wait(ready, phaseIdx=phase)
        acc += tl.load(tle.gpu.local_ptr(smem, (0, )))
        tle.gpu.barrier_arrive(empty, phaseIdx=phase)
    tl.store(out_ptr, acc)


@triton.jit
def barrier_ordered_loop_kernel(src_ptr, out_ptr, iters):
    smem = tle.gpu.alloc((1, ), dtype=tl.int32, nv_mma_shared_layout=False)
    ready = tle.gpu.alloc_barrier(arrive_count=4, init=tle.gpu.PENDING)
    empty = tle.gpu.alloc_barrier(arrive_count=4, init=tle.gpu.READY)
    tle.gpu.warp_specialize(
        [
            (barrier_ordered_producer, (src_ptr, smem, ready, empty, iters)),
            (barrier_ordered_consumer, (out_ptr, smem, ready, empty, iters)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[32],
    )


def test_mthreads_tle_heterogeneous_ws_worker_warps_ttgir():
    compiled = compile_musa(
        heterogeneous_ws_kernel,
        signature={"out_ptr": "*i32"},
        options={"num_warps": 8, "num_stages": 1},
    )
    ttgir = compiled.asm["ttgir"]
    assert "warpsPerCTA = [4]" in ttgir
    assert "warpsPerCTA = [8]" in ttgir
    assert compiled.asm["llir"]


def test_mthreads_tle_barrier_blocks_smem_load_hoisting():
    compiled = compile_musa(
        barrier_ordered_loop_kernel,
        signature={"src_ptr": "*i32", "out_ptr": "*i32", "iters": "i32"},
        options={"num_warps": 4, "num_stages": 1},
    )
    llir = compiled.asm["llir"]
    # The consumer loop must reload from addrspace(3) after each barrier wait
    # rather than hoisting the loop-invariant shared-memory pointer load.
    wait_positions = [match.start() for match in re.finditer(r"llvm\.musa\.async\.wait", llir)]
    smem_loads = [match.start() for match in re.finditer(r"load i32, ptr addrspace\(3\)", llir)]
    assert wait_positions and smem_loads
    assert any(load_pos > wait_positions[0] for load_pos in smem_loads)


@triton.jit
def waited_sqmma_chain_kernel(a_ptr, b_ptr, out_ptr):
    m = tl.arange(0, 64)[:, None]
    k = tl.arange(0, 128)[None, :]
    n = tl.arange(0, 128)[None, :]
    a_smem = tle.gpu.alloc(
        (64, 128),
        dtype=tl.float8e4nv,
        init_value=tl.load(a_ptr + m * 128 + k),
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (128, 128),
        dtype=tl.float8e4nv,
        init_value=tl.load(b_ptr + tl.arange(0, 128)[:, None] * 128 + n),
        nv_mma_shared_layout=True,
    )
    acc = tl.zeros((64, 128), dtype=tl.float32)
    acc = tle.gpu.wgmma(a_smem, b_smem, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    acc = tle.gpu.wgmma(a_smem, b_smem, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    tl.store(out_ptr + m * 128 + n, acc)


EXPLICIT_LOOP_SQMMA = tl.constexpr(MusaSqmmaEncoding([3, 1], [4, 4], [64, 128, 128]))


@triton.jit
def loop_sqmma_explicit_layout_kernel(a_ptr, b_ptr, out_ptr, steps):
    m = tl.arange(0, 64)[:, None]
    k = tl.arange(0, 128)[None, :]
    n = tl.arange(0, 512)[None, :]
    a_smem = tle.gpu.alloc(
        (64, 128),
        dtype=tl.float8e4nv,
        init_value=tl.load(a_ptr + m * 128 + k),
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (128, 512),
        dtype=tl.float8e4nv,
        init_value=tl.load(b_ptr + tl.arange(0, 128)[:, None] * 512 + n),
        nv_mma_shared_layout=True,
    )
    acc = tle.gpu.set_layout(tl.zeros((64, 512), dtype=tl.float32), EXPLICIT_LOOP_SQMMA)
    for _ in range(0, steps):
        acc = tle.gpu.wgmma(a_smem, b_smem, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    acc = tle.gpu.set_layout(acc, EXPLICIT_LOOP_SQMMA)
    tl.store(out_ptr + m * 512 + n, acc)


def test_mthreads_tle_waited_sqmma_chain_preserves_warp_split():
    compiled = compile_musa(
        waited_sqmma_chain_kernel,
        signature={"a_ptr": "*fp8e4nv", "b_ptr": "*fp8e4nv", "out_ptr": "*fp32"},
        options={"num_warps": 16, "num_stages": 1},
    )
    ttgir = compiled.asm["ttgir"]
    sqmma_encodings = re.findall(r"#ttg\.musa_sqmma<[^>]+>", ttgir)
    assert len(sqmma_encodings) == 1
    assert "warpsPerCTA = [16, 1]" in sqmma_encodings[0]
    assert compiled.asm["llir"]


def test_mthreads_tle_loop_sqmma_explicit_layout_unpack():
    compiled = compile_musa(
        loop_sqmma_explicit_layout_kernel,
        signature={"a_ptr": "*fp8e4nv", "b_ptr": "*fp8e4nv", "out_ptr": "*fp32", "steps": "i32"},
        options={"num_warps": 16, "num_stages": 1},
    )
    ttgir = compiled.asm["ttgir"]
    assert "mtgpu.sqmma_wait" in ttgir
    assert "mtgpu.unpack_sqmma_accumulator" in ttgir
    assert compiled.asm["llir"]
