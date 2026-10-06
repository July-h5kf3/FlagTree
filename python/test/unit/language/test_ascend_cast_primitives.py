# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend NPU required", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.language.extra.cann.extension as al  # noqa: E402
from triton.experimental.tle.language.dsa.ascend.custom_ops.cast_primitives import CAST_PRIMITIVES_BITCODE  # noqa: E402


def reference_cast(values, mode, dtype):
    values = values.double()
    if mode == 1:
        rounded = values.round()
    elif mode == 2:
        rounded = values.floor()
    elif mode == 3:
        rounded = values.ceil()
    elif mode == 4:
        rounded = values.sign() * (values.abs() + .5).floor()
    else:
        rounded = values.trunc()
    return rounded.to(dtype)


@triton.jit
def convert(Source, Destination, COUNT: tl.constexpr, CAPACITY: tl.constexpr, MODE: tl.constexpr,
            FP32_SOURCE: tl.constexpr, KEY: tl.constexpr):
    offsets = tl.arange(0, CAPACITY)
    values = tl.load(Source + offsets)
    original = tl.load(Destination + offsets)
    if FP32_SOURCE:
        converted = al.custom("cast_fp32_to_int16", values, MODE, COUNT, out=original)
    else:
        converted = al.custom("cast_fp16_to_int8", values, MODE, COUNT, out=original)
    tl.store(Destination + offsets, converted)


@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("count", [1, 137, 8192, 16384])
@pytest.mark.parametrize("fp32_source", [True, False])
def test_cast(mode, count, fp32_source):
    capacity = max(32, triton.next_power_of_2(count))
    dtype = torch.float32 if fp32_source else torch.float16
    output_dtype = torch.int16 if fp32_source else torch.int8
    values = (((torch.arange(capacity, dtype=torch.float32) % 499) - 249) / 2).to(dtype)
    reference = reference_cast(values, mode, output_dtype)
    reference[count:] = 37
    output = torch.full((capacity, ), 37, dtype=output_dtype, device="npu")
    key = hashlib.sha256(Path(CAST_PRIMITIVES_BITCODE).read_bytes()).hexdigest()
    convert[(1, )](values.npu(), output, count, capacity, mode, fp32_source, key, multibuffer=False)
    torch.testing.assert_close(output.cpu(), reference, atol=0, rtol=0)


@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("fp32_source", [True, False])
def test_cast_rounding_boundaries(mode, fp32_source):
    dtype = torch.float32 if fp32_source else torch.float16
    output_dtype = torch.int16 if fp32_source else torch.int8
    points = torch.tensor([-126.5, -2.5, -1.5, -.5, .5, 1.5, 2.5, 126.5], dtype=dtype)
    below = torch.nextafter(points, torch.full_like(points, -float("inf")))
    above = torch.nextafter(points, torch.full_like(points, float("inf")))
    extremes = [-32768., -32512., 32512., 32767.] if fp32_source else [-128., -127., 126., 127.]
    values = torch.cat((below, points, above, torch.tensor(extremes + [-0., 0., -1., 1.], dtype=dtype)))
    output = torch.empty((32, ), dtype=output_dtype, device="npu")
    key = hashlib.sha256(Path(CAST_PRIMITIVES_BITCODE).read_bytes()).hexdigest()
    convert[(1, )](values.npu(), output, 32, 32, mode, fp32_source, key, multibuffer=False)
    torch.testing.assert_close(output.cpu(), reference_cast(values, mode, output_dtype), atol=0, rtol=0)


@pytest.mark.parametrize("fp32_source", [True, False])
def test_cast_graph_updates(fp32_source):
    dtype = torch.float32 if fp32_source else torch.float16
    output_dtype = torch.int16 if fp32_source else torch.int8
    source = torch.zeros((256, ), dtype=dtype, device="npu")
    destination = torch.full((256, ), 37, dtype=output_dtype, device="npu")
    key = hashlib.sha256(Path(CAST_PRIMITIVES_BITCODE).read_bytes()).hexdigest()

    def launch():
        convert[(1, )](source, destination, 137, 256, 4, fp32_source, key, multibuffer=False)

    launch()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        launch()
    for shift in [-1.25, .5, 1.75]:
        values = (((torch.arange(256, dtype=torch.float32) % 97) - 48) / 2 + shift).to(dtype)
        source.copy_(values)
        graph.replay()
        expected = reference_cast(values, 4, output_dtype)
        expected[137:] = 37
        torch.testing.assert_close(destination.cpu(), expected, atol=0, rtol=0)


@triton.jit
def convert_mixed(Source, Destination, MODE: tl.constexpr, FP32_SOURCE: tl.constexpr, KEY: tl.constexpr):
    with al.scope(core_mode="cube"):
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tl.debug_barrier()
    with al.scope(core_mode="vector"):
        offsets = al.sub_vec_id().to(tl.int32) * 256 + tl.arange(0, 256)
        values = tl.load(Source + offsets)
        original = tl.load(Destination + offsets)
        if FP32_SOURCE:
            converted = al.custom("cast_fp32_to_int16", values, MODE, 137, out=original)
        else:
            converted = al.custom("cast_fp16_to_int8", values, MODE, 137, out=original)
        tl.store(Destination + offsets, converted)


@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("fp32_source", [True, False])
def test_cast_mixed(mode, fp32_source):
    dtype = torch.float32 if fp32_source else torch.float16
    output_dtype = torch.int16 if fp32_source else torch.int8
    values = (((torch.arange(512, dtype=torch.float32) % 499) - 249) / 2).to(dtype)
    expected = reference_cast(values, mode, output_dtype).reshape(2, 256)
    expected[:, 137:] = 37
    destination = torch.full((512, ), 37, dtype=output_dtype, device="npu")
    key = hashlib.sha256(Path(CAST_PRIMITIVES_BITCODE).read_bytes()).hexdigest()
    convert_mixed[(1, )](values.npu(), destination, mode, fp32_source, key, multibuffer=False,
                         disable_auto_inject_block_sync=True, enable_legacy_insert_load_store_for_mix_cv=False)
    torch.testing.assert_close(destination.cpu().view(2, 256), expected, atol=0, rtol=0)
