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
import triton.experimental.tle as tle  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.language.extra.cann.extension as al  # noqa: E402
from triton.experimental.tle.language.dsa.ascend.custom_ops.cube_primitives import CUBE_PRIMITIVES_BITCODE  # noqa: E402

# CommonIR TilePipe numbering.
MTE2 = tl.constexpr(3)
MTE1 = tl.constexpr(2)
M = tl.constexpr(0)
FIX = tl.constexpr(5)


@triton.jit
def sync_pair(PRODUCER: tl.constexpr, CONSUMER: tl.constexpr):
    tle.dsa.tile_set_flag(PRODUCER, CONSUMER, 1)
    tle.dsa.tile_wait_flag(PRODUCER, CONSUMER, 1)


@triton.jit
def qk_product(Query, Key, Output, WIDTH: tl.constexpr, BITCODE_KEY: tl.constexpr, SIGNAL: tl.constexpr = False):
    with al.scope(core_mode="cube"):
        al.custom("cube_nd2nz_i8", 0, Query.to(tl.uint64), 1, 32, 128, 0, 128, 32, 1, 0)
        al.custom("cube_nd2nz_i8", 4096, Key.to(tl.uint64), 1, WIDTH, 128, 0, 384, WIDTH, 1, 0)
        sync_pair(MTE2, MTE1)
        al.custom("cube_load3d_a_into", 0, [0, 0, 0, 255], 1, 32, 128, 128, 32, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                  0, 1, 1, 0)
        al.custom("cube_load2d_b_into", 4096, 0, WIDTH * 128 // 512, 1, 0, 0, 0, 0, 0)
        sync_pair(MTE1, M)
        al.custom("cube_mmad_into", 0, 0, 32, 128, WIDTH, 0, 0, 0, 1, 0)
        sync_pair(M, FIX)
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        al.custom("cube_copy_l0c2gm_i32", Output.to(tl.uint64), 0, WIDTH, 32, WIDTH, 32, 0, 0, 0, 0, 1)
        tl.debug_barrier()
        if SIGNAL:
            al.sync_block_set("cube", "vector", 2)
        else:
            pass


@triton.jit
def pv_product(Probability, Value, Output, WIDTH: tl.constexpr, BITCODE_KEY: tl.constexpr,
               SIGNAL: tl.constexpr = False):
    with al.scope(core_mode="cube"):
        al.custom("cube_nd2nz_i8", 0, Probability.to(tl.uint64), 1, 32, WIDTH, 0, WIDTH, 32, 1, 0)
        al.custom("cube_nd2nz_i8", 32 * WIDTH, Value.to(tl.uint64), 1, WIDTH, 128, 0, 128, WIDTH, 1, 0)
        sync_pair(MTE2, MTE1)
        al.custom("cube_load3d_a_into", 0, [0, 0, 0, 255], 1, 32, WIDTH, WIDTH, 32, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                  0, 0, 1, 1, 0)
        for section in tl.static_range(WIDTH // 32):
            al.custom("cube_load_transpose_b_into", 32 * WIDTH + section * 32 * 32, 0, 4, WIDTH // 32, 1, 0, 0,
                      section * 32 * 128)
        sync_pair(MTE1, M)
        al.custom("cube_mmad_into", 0, 0, 32, WIDTH, 128, 0, 0, 0, 1, 0)
        sync_pair(M, FIX)
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        al.custom("cube_copy_l0c2gm_i32", Output.to(tl.uint64), 0, 128, 32, 128, 32, 0, 0, 0, 0, 1)
        tl.debug_barrier()
        if SIGNAL:
            al.sync_block_set("cube", "vector", 2)
        else:
            pass


@pytest.mark.parametrize("width", [16, 64, 128, 256, 512])
def test_qk_primitives(width):
    torch.manual_seed(20260930 + width)
    query = torch.randint(-128, 128, (32, 128), dtype=torch.int8)
    key = torch.randint(-128, 128, (width, 384), dtype=torch.int8)
    expected = query.int() @ key[:, :128].int().T
    output = torch.empty((32, width), dtype=torch.int32, device="npu")
    bitcode_key = hashlib.sha256(Path(CUBE_PRIMITIVES_BITCODE).read_bytes()).hexdigest()
    qk_product[(1, )](query.npu(), key.npu(), output, width, bitcode_key, multibuffer=False,
                      enable_legacy_insert_load_store_for_mix_cv=True)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("width", [128, 256, 512])
def test_pv_primitives(width):
    torch.manual_seed(20261000 + width)
    probability = torch.randint(-128, 128, (32, width), dtype=torch.int8)
    value = torch.randint(-128, 128, (width, 128), dtype=torch.int8)
    expected = probability.int() @ value.int()
    output = torch.empty((32, 128), dtype=torch.int32, device="npu")
    bitcode_key = hashlib.sha256(Path(CUBE_PRIMITIVES_BITCODE).read_bytes()).hexdigest()
    pv_product[(1, )](probability.npu(), value.npu(), output, width, bitcode_key, multibuffer=False,
                      enable_legacy_insert_load_store_for_mix_cv=True)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@triton.jit
def mixed_product(Query, Key, Workspace, Output, PV: tl.constexpr, KEY: tl.constexpr):
    if PV:
        pv_product(Query, Key, Workspace, 128, KEY, SIGNAL=True)
    else:
        qk_product(Query, Key, Workspace, 128, KEY, SIGNAL=True)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 2)
        offsets = al.sub_vec_id().to(tl.int32) * 2048 + tl.arange(0, 2048)
        tl.store(Output + offsets, tl.load(Workspace + offsets))


@pytest.mark.parametrize("pv", [False, True])
def test_cube_vector_handoff(pv):
    torch.manual_seed(20261006)
    query = torch.randint(-128, 128, (32, 128), dtype=torch.int8)
    key = torch.randint(-128, 128, (128, 384 if not pv else 128), dtype=torch.int8)
    expected = query.int() @ (key.int() if pv else key[:, :128].int().T)
    workspace = torch.empty((32, 128), dtype=torch.int32, device="npu")
    output = torch.empty_like(workspace)
    bitcode_key = hashlib.sha256(Path(CUBE_PRIMITIVES_BITCODE).read_bytes()).hexdigest()
    mixed_product[(1, )](query.npu(), key.npu(), workspace, output, pv, bitcode_key, multibuffer=False,
                         disable_auto_inject_block_sync=True, enable_legacy_insert_load_store_for_mix_cv=False)
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
