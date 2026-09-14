# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone tests for packed comparisons, compaction and sorting primitives."""
import numpy as np
import torch
import triton
import triton.language as tl
from test_mask_ops import _init, _pack, _tensor
from triton.experimental import tle
from triton.experimental.tle.language.dsa.ascend.custom_ops import compare_scalar, merge_sort4, sort32


@triton.jit
def compare(X, Y, scalar, N: tl.constexpr, MODE: tl.constexpr):
    x = tl.load(X + tl.arange(0, N))
    y = tle.dsa.ascend.raw("compare_scalar", x, scalar, MODE, out=tl.full((N // 16, ), 0, tl.uint16))
    tl.store(Y + tl.arange(0, N // 16), y)


@triton.jit
def gather(X, M, Y, C, N: tl.constexpr, CNT: tl.constexpr):
    x = tl.load(X + tl.arange(0, N))
    m = tl.load(M + tl.arange(0, N // 16))
    y, c = tle.dsa.ascend.raw(
        "gather_mask", x, m, out=[
            tle.dsa.to_tensor(tle.dsa.alloc((N, ), tl.float16, tle.dsa.ascend.UB)),
            tle.dsa.to_tensor(tle.dsa.alloc((CNT, ), tl.int32, tle.dsa.ascend.UB))
        ])
    # Avoid a full int32[N] store-index tensor exhausting UB at N=32768.
    B: tl.constexpr = 4096 if N >= 4096 else N  # noqa: FURB136
    for start in tl.static_range(0, N, B):
        chunk = tle.dsa.extract_slice(y, [start], [B], [1])
        offsets = start + tl.arange(0, B)
        tl.store(Y + offsets, chunk, offsets < tl.max(c, 0))
    tl.store(C + tl.arange(0, CNT), c)


@triton.jit
def sort(X, I, Y, N: tl.constexpr):
    x = tl.load(X + tl.arange(0, N))
    i = tl.load(I + tl.arange(0, N))
    y = tle.dsa.ascend.raw("sort32", x, i, out=tl.full((2 * N, ), 0, tl.float32))
    tl.store(Y + tl.arange(0, 2 * N), y)


@triton.jit
def merge(X, Y, N: tl.constexpr, L: tl.constexpr, W: tl.constexpr):
    x = tl.load(X + tl.arange(0, N))
    y = tle.dsa.ascend.raw("merge_sort4", x, L, W, out=tl.full((N, ), 0, tl.float32))
    tl.store(Y + tl.arange(0, N), y)


def test_half_masks():
    rng = np.random.default_rng(11)
    for n in (256, 4096, 8192, 32768):
        values = rng.integers(-10, 11, n).astype(np.float16)
        values[:7] = [0, -0., np.inf, -np.inf, np.nan, 1., 1.001]
        x = torch.from_numpy(values).npu()
        out = torch.empty(n // 16, dtype=torch.uint16, device="npu")
        for scalar in (0., 1.0003, np.inf, -np.inf, np.nan):
            for mode, op in enumerate((np.equal, np.greater, np.greater_equal)):
                compare[(1, )](x, out, float(scalar), n, mode)
                np.testing.assert_array_equal(out.cpu().numpy(), _pack(op(values, np.float16(scalar))))
        # Gather must preserve arbitrary 16-bit payloads, including NaN indices.
        bits = rng.integers(0, 65536, n, dtype=np.uint16)
        bits[:6] = [0, 32768, 0x7c00, 0xfc00, 0x7e01, 0xffff]
        x = torch.from_numpy(bits.view(np.float16)).npu()
        boundary = np.zeros(n, bool)
        boundary[[0, 15, 16, 31, 32, 255, n - 1]] = True
        for predicate in (np.zeros(n, bool), np.ones(n, bool), boundary, rng.random(n) < .5):
            mask = torch.from_numpy(_pack(predicate)).npu()
            for cnt in (1, 8):
                y = torch.empty_like(x)
                c = torch.empty(cnt, dtype=torch.int32, device="npu")
                gather[(1, )](x, mask, y, c, n, cnt)
                found = int(predicate.sum())
                np.testing.assert_array_equal(c.cpu().numpy(), np.full(cnt, found))
                np.testing.assert_array_equal(y.cpu().numpy()[:found].view(np.uint16), bits[predicate])


def check_pairs(result, keys, ids, width):
    for start in range(0, len(keys), width):
        actual = result[2 * start:2 * (start + width)].reshape(-1, 2)
        np.testing.assert_array_equal(actual[:, 0], np.sort(keys[start:start + width])[::-1])
        mapping = dict(zip(ids[start:start + width].tolist(), keys[start:start + width].tolist()))
        payload = actual[:, 1].copy().view(np.uint32)
        assert set(payload.tolist()) == set(mapping)
        np.testing.assert_array_equal(actual[:, 0], np.array([mapping[int(i)] for i in payload]))


def test_sorting():
    rng = np.random.default_rng(19)
    for n in (32, 64, 256, 4096):
        keys = rng.integers(-20, 21, n).astype(np.float32)
        keys[:2] = [-np.inf, np.inf]
        ids = np.arange(n, dtype=np.uint32) + np.uint32(0x7fc00000)
        y = torch.empty(2 * n, dtype=torch.float32, device="npu")
        sort[(1, )](torch.from_numpy(keys).npu(), torch.from_numpy(ids).npu(), y, n)
        check_pairs(y.cpu().numpy(), keys, ids, 32)
    for length, ways, groups in ((8, 2, 1), (8, 4, 128), (16, 4, 4), (32, 2, 1), (128, 4, 2), (1024, 4, 1), (2048, 2,
                                                                                                             1)):
        print("MERGE", length, ways, groups, flush=True)
        n = length * ways * groups
        keys = rng.integers(-50, 51, n).astype(np.float32).reshape(-1, length)
        keys[:, 0] = np.inf
        keys[:, -1] = -np.inf
        keys = np.sort(keys, axis=1)[:, ::-1].copy().ravel()
        ids = np.arange(n, dtype=np.uint32) + np.uint32(0xffc00000)
        pairs = np.empty((n, 2), dtype=np.float32)
        pairs[:, 0] = keys
        pairs[:, 1] = ids.view(np.float32)
        x = torch.from_numpy(pairs.ravel()).npu()
        y = torch.empty_like(x)
        merge[(1, )](x, y, 2 * n, length, ways)
        check_pairs(y.cpu().numpy(), keys, ids, length * ways)


def test_signatures():
    src = _tensor(tl.float32, [256])
    idx = _tensor(tl.uint32, [256])
    dst = _tensor(tl.float32, [512])
    invalid = [(compare_scalar, (src, 0., 3), _tensor(tl.uint16, [16])), (sort32, (src, _tensor(tl.int32, [256])), dst),
               (sort32, (src, idx), src), (merge_sort4, (dst, 7, 4), dst), (merge_sort4, (dst, 32, 3), dst),
               (merge_sort4, (dst, 32, 2), dst), (merge_sort4, (dst, 32, 4), src)]
    for cls, args, out in invalid:
        try:
            _init(cls, *args, out=out)
        except AssertionError:
            continue
        raise AssertionError(f"{cls.__name__} accepted invalid signature")


def main():
    test_signatures()
    test_half_masks()
    test_sorting()
    print("All TopK primitive tests passed.")


if __name__ == "__main__":
    main()
