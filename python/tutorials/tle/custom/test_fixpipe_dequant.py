# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Caller-owned L0C writeback; run directly to include a Triton comparison."""

import struct
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle.language.raw as tle_raw
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.raw import dialect


@dialect(name="cann", file=str(Path(__file__).with_name("fixpipe_test_setup.cpp")),
         extern_func_name="fixpipe_test_setup", pipeline={"core": "cube", "pipe": "PIPE_M"})
def fixpipe_test_setup(*args, **kwargs):
    ...


@triton.jit
def fixpipe_kernel(A, B, Out, ROWS: tl.constexpr, COLUMNS: tl.constexpr, STRIDE: tl.constexpr,
                   L0C_ADDRESS: tl.constexpr, SCALE_BITS: tl.constexpr):
    with al.scope(core_mode="cube"):
        tile = tl.program_id(0)
        tle_raw.call(fixpipe_test_setup, [(A + tile * 16384).to(tl.uint64), (B + tile * 16384).to(tl.uint64),
                                          tl.full((), L0C_ADDRESS, tl.uint32)])
        tle_raw.call("ascend_cube_copy_l0c2gm_deqf16", (Out + tile * ROWS * STRIDE).to(tl.uint64), L0C_ADDRESS, COLUMNS,
                     ROWS, STRIDE, 128, 0, SCALE_BITS)
        tl.debug_barrier()


@triton.jit
def triton_kernel(A, B, Out, SCALE: tl.constexpr):
    tile = tl.program_id(0)
    rows = tl.arange(0, 128)
    columns = tl.arange(0, 128)
    a = tl.load(A + tile * 16384 + rows[:, None] * 128 + columns[None, :])
    b = tl.load(B + tile * 16384 + rows[:, None] * 128 + columns[None, :])
    accumulator = tl.dot(a, tl.trans(b))
    values = (accumulator.to(tl.float32) * SCALE).to(tl.float16)
    tl.store(Out + tile * 16384 + rows[:, None] * 128 + columns[None, :], values)


def encoded_scale(scale):
    bits = struct.unpack("I", struct.pack("f", scale))[0] & 0xFFFFE000
    return bits, struct.unpack("f", struct.pack("I", bits))[0]


def inputs(tiles=1):
    generator = torch.Generator().manual_seed(110231)
    a = torch.randint(-127, 128, (tiles, 128, 128), dtype=torch.int8, generator=generator)
    b = torch.randint(-127, 128, (tiles, 128, 128), dtype=torch.int8, generator=generator)
    return a, b


@pytest.mark.parametrize("rows,columns,stride,l0c_address", [(128, 128, 128, 0), (64, 128, 144, 65536),
                                                             (16, 16, 32, 0)])
@pytest.mark.parametrize("scale", [0.0, 1 / 256, -1 / 256, 0.0001])
def test_fixpipe_dequant(rows, columns, stride, l0c_address, scale):
    a, b = inputs()
    bits, effective_scale = encoded_scale(scale)
    storage = torch.full((32 + rows * stride, ), 123, dtype=torch.float16, device="npu")
    output = storage[16:-16].view(rows, stride)
    fixpipe_kernel[(1, )](a.to("npu"), b.to("npu"), output, rows, columns, stride, l0c_address, bits, num_warps=1,
                          multibuffer=False, disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=True)
    accumulator = a[0].numpy().astype(np.int64) @ b[0].numpy().astype(np.int64).T
    expected = (accumulator[:rows, :columns].astype(np.float32) * effective_scale).astype(np.float16)
    actual = output.cpu().numpy()
    np.testing.assert_array_equal(actual[:, :columns], expected)
    np.testing.assert_array_equal(actual[:, columns:], 123)
    np.testing.assert_array_equal(storage[:16].cpu().numpy(), 123)
    np.testing.assert_array_equal(storage[-16:].cpu().numpy(), 123)


def bench_fixpipe_dequant():
    tiles = 1
    a, b = inputs(tiles)
    bits, scale = encoded_scale(1 / 256)
    a, b = a.to("npu"), b.to("npu")
    custom = torch.empty((tiles, 128, 128), dtype=torch.float16, device="npu")
    ordinary = torch.empty_like(custom)

    def launch_custom():
        return fixpipe_kernel[(tiles, )](a, b, custom, 128, 128, 128, 0, bits, num_warps=1, multibuffer=False,
                                         disable_auto_inject_block_sync=True, enable_auto_bind_sub_block=True)

    def launch_triton():
        return triton_kernel[(tiles, )](a, b, ordinary, scale, num_warps=1, multibuffer=False)

    launch_custom()
    launch_triton()
    np.testing.assert_array_equal(custom.cpu().numpy(), ordinary.cpu().numpy())
    measured = {"custom": [], "triton": []}
    for order in (("triton", "custom"), ("custom", "triton")):
        for name in order:
            launch = launch_custom if name == "custom" else launch_triton
            measured[name].append(triton.testing.do_bench(launch, return_mode="median"))
    print(f"[BENCH] {tiles} tiles of INT8 128x128x128: rounds_ms={measured}; "
          f"triton/custom={np.mean(measured['triton']) / np.mean(measured['custom']):.3f}x")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]) or bench_fixpipe_dequant())
