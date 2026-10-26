import numpy as np
import pytest
import torch

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError


@triton.jit
def _encode(X, Y, N: tl.constexpr, ROUND: tl.constexpr, AXIS: tl.constexpr = -1, BLOCK: tl.constexpr = 512):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    if ROUND == "default":
        packed = tl.fp_to_fp4(x, axis=AXIS)
    else:
        packed = tl.fp_to_fp4(x, axis=AXIS, fp_downcast_rounding=ROUND)
    j = tl.program_id(0) * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    tl.store(Y + j, packed, j < (N + 1) // 2)


@triton.jit
def _encode_2d(X, Y, AXIS: tl.constexpr, ROUND: tl.constexpr):
    i, j = tl.arange(0, 16), tl.arange(0, 32)
    x = tl.load(X + i[:, None] * 32 + j[None, :])
    y = tl.fp_to_fp4(x, axis=AXIS, fp_downcast_rounding=ROUND)
    r, c = tl.arange(0, 8 if AXIS == 0 else 16), tl.arange(0, 32 if AXIS == 0 else 16)
    tl.store(Y + r[:, None] * (32 if AXIS == 0 else 16) + c[None, :], y)


def _reference_codes(values, rounding):
    x = values.to(torch.float64).numpy()
    a = np.abs(x)
    boundaries = np.array([.5, 1., 1.5, 2., 3., 4., 6.] if rounding == "rtz" else [.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    indices = np.searchsorted(boundaries, a, side="right" if rounding == "rtz" else "left")
    if rounding != "rtz":
        exact_ties = (indices < 7) & (a == boundaries[np.minimum(indices, 6)])
        indices += exact_ties & (indices % 2 == 1)
    indices = np.where(np.isnan(a), 7, indices)
    return torch.from_numpy((indices | (np.signbit(x).astype(np.uint8) << 3)).astype(np.uint8))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "fp64"])
@pytest.mark.parametrize("rounding", ["default", "rtne", "rtz"])
def test_metax_fp4_encode(dtype, rounding, device):
    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}[dtype]
    if dtype in ("fp16", "bf16"):
        values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch_dtype)
    else:
        boundaries = torch.tensor([.25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 5., 6.], dtype=torch_dtype)
        boundaries = torch.cat((boundaries, -boundaries))
        lo = torch.nextafter(boundaries, torch.full_like(boundaries, -float("inf")))
        hi = torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))
        extra = torch.tensor([
            0., -0.,
            float("inf"), -float("inf"),
            float("nan"), -float("nan"),
            torch.finfo(torch_dtype).max, -torch.finfo(torch_dtype).max
        ], dtype=torch_dtype)
        rng = np.random.default_rng(20261026)
        raw = rng.integers(0, 2**32, size=4096, dtype=np.uint32)
        random = raw.view(np.float32) if dtype == "fp32" else rng.integers(0, 2**64, size=4096, dtype=np.uint64).view(
            np.float64)
        values = torch.cat((lo, boundaries, hi, extra, torch.from_numpy(random)))
    codes = _reference_codes(values, rounding)
    if codes.numel() % 2:
        codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8)))
    expected = codes[::2] | (codes[1::2] << 4)
    output = torch.full((expected.numel() + 64, ), 165, dtype=torch.uint8, device=device)
    _encode[(triton.cdiv(values.numel(), 512), )](values.to(device), output[32:-32], values.numel(), rounding)
    reference = torch.full((expected.numel() + 64, ), 165, dtype=torch.uint8)
    reference[32:-32] = expected
    assert torch.equal(output.cpu(), reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("rounding", ["rtne", "rtz"])
def test_metax_fp4_encode_axis(dtype, axis, rounding, metax_device):
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=dtype)
    codes = torch.arange(512).remainder(16).reshape(16, 32)
    values = table[codes]
    shape = (8, 32) if axis == 0 else (16, 16)
    expected = torch.empty(shape, dtype=torch.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            a, b = ((2 * i, j), (2 * i + 1, j)) if axis == 0 else ((i, 2 * j), (i, 2 * j + 1))
            expected[i, j] = int(codes[a]) | (int(codes[b]) << 4)
    output = torch.full((320, ), 165, dtype=torch.uint8, device=metax_device)
    _encode_2d[(1, )](values.to(metax_device), output[32:288], axis, rounding)
    reference = torch.full((320, ), 165, dtype=torch.uint8)
    reference[32:288] = expected.flatten()
    assert torch.equal(output.cpu(), reference)


@triton.jit
def _scaled(X, S, Y, M: tl.constexpr, N: tl.constexpr, AXIS: tl.constexpr, GROUPED: tl.constexpr, DTYPE: tl.constexpr):
    BM: tl.constexpr = 16 if AXIS == 1 else 32
    BN: tl.constexpr = 32 if AXIS == 1 else 16
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    x = tl.load(X + r[:, None] * N + c[None, :], (r[:, None] < M) & (c[None, :] < N), other=0)
    if GROUPED:
        if AXIS == 1:
            g = tl.program_id(1) * 2 + tl.arange(0, 2)
            s = tl.load(S + r[:, None] * tl.cdiv(N, 16) + g[None, :], (r[:, None] < M) & (g[None, :] < tl.cdiv(N, 16)),
                        other=127)
        else:
            g = tl.program_id(0) * 2 + tl.arange(0, 2)
            s = tl.load(S + g[:, None] * N + c[None, :], (g[:, None] < tl.cdiv(M, 16)) & (c[None, :] < N), other=127)
    else:
        s = tl.load(S + r[:, None] * N + c[None, :], (r[:, None] < M) & (c[None, :] < N), other=127)
    y = tl.fp4_to_fp_scaled(x, s, DTYPE, AXIS - 2)
    out_r = tl.program_id(0) * (BM * 2 if AXIS == 0 else BM) + tl.arange(0, BM * 2 if AXIS == 0 else BM)
    out_c = tl.program_id(1) * (BN * 2 if AXIS == 1 else BN) + tl.arange(0, BN * 2 if AXIS == 1 else BN)
    out_m: tl.constexpr = M * 2 if AXIS == 0 else M
    out_n: tl.constexpr = N * 2 if AXIS == 1 else N
    tl.store(Y + out_r[:, None] * out_n + out_c[None, :], y, (out_r[:, None] < out_m) & (out_c[None, :] < out_n))


@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_metax_fp4_scaled(dtype, axis, grouped, signed, metax_device):
    raw = torch.arange(259, dtype=torch.int64).remainder(256).to(torch.uint8).repeat(256, 1)
    width = 17 if grouped else 259
    scales = (torch.arange(256)[:, None] + 7 * torch.arange(width)[None, :]).remainder(256).to(torch.uint8)
    expanded = scales.repeat_interleave(16, 1)[:, :259] if grouped else scales
    codes = torch.stack((raw & 15, raw >> 4), dim=2).flatten(1, 2).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         dtype=torch.float64)
    factors = torch.from_numpy(np.ldexp(np.ones(expanded.shape), expanded.numpy().astype(np.int16) - 127))
    factors[expanded == 255] = float("nan")
    reference = table[codes] * factors.repeat_interleave(2, 1)
    if axis == 0:
        raw, scales, reference = raw.T.contiguous(), scales.T.contiguous(), reference.T.contiguous()
    out_dtype = torch.float16 if dtype == tl.float16 else torch.bfloat16
    expected = reference.to(out_dtype).contiguous().flatten()
    x, s = raw.to(metax_device), scales.to(metax_device)
    if signed:
        x, s = x.view(torch.int8), s.view(torch.int8)
    output = torch.full((expected.numel() + 64, ), 11., dtype=out_dtype, device=metax_device)
    m, n = raw.shape
    grid = (triton.cdiv(m, 16 if axis == 1 else 32), triton.cdiv(n, 32 if axis == 1 else 16))
    kernel = _scaled[grid](x, s, output[32:-32], m, n, axis, grouped, dtype)
    actual = output.cpu()
    guard = torch.full((32, ), 11., dtype=out_dtype).view(torch.uint8)
    assert torch.equal(actual[:32].view(torch.uint8), guard) and torch.equal(actual[-32:].view(torch.uint8), guard)
    actual = actual[32:-32]
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(torch.uint8), expected[~nan].view(torch.uint8))
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


def test_metax_fp4_public_exports():
    assert tl.fp_to_fp4 is tl.core.fp_to_fp4
    assert tl.fp4_to_fp_scaled is tl.core.fp4_to_fp_scaled


@triton.jit
def _scaled_documented(X, S, Y, DTYPE: tl.constexpr, RAW: tl.constexpr, WIDTH: tl.constexpr = 1,
                       AXIS: tl.constexpr = 1):
    r, c = tl.arange(0, 32), tl.arange(0, 16)
    x = tl.load(X + r[:, None] * 16 + c[None, :])
    s = tl.load(S + r[:, None] * WIDTH + tl.arange(0, WIDTH)[None, :])
    if RAW:
        y = tl.core.fp4_to_fp_scaled(x, s, DTYPE, AXIS)
    else:
        y = tl.fp4_to_fp_scaled(x, s, DTYPE, AXIS)
    j = tl.arange(0, 32)
    tl.store(Y + r[:, None] * 32 + j[None, :], y)


@pytest.mark.parametrize("dtype", [tl.float16, tl.bfloat16])
@pytest.mark.parametrize("raw_api", [False, True])
def test_metax_fp4_scaled_documented(dtype, raw_api, metax_device):
    x = torch.arange(512, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(32, 16)
    s = torch.arange(96, 128, dtype=torch.uint8)
    codes = torch.stack((x & 15, x >> 4), 2).flatten(1, 2).long()
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         dtype=torch.float64)
    values = table[codes] * torch.from_numpy(np.ldexp(np.ones((32, 1)), s.numpy().astype(np.int16)[:, None] - 127))
    out_dtype = torch.float16 if dtype == tl.float16 else torch.bfloat16
    output = torch.full((1088, ), 11., dtype=out_dtype, device=metax_device)
    kernel = _scaled_documented[(1, )](x.to(metax_device), s.to(metax_device), output[32:1056], dtype, raw_api)
    expected = torch.full((1088, ), 11., dtype=out_dtype)
    expected[32:1056] = values.to(out_dtype).flatten()
    assert torch.equal(output.cpu().view(torch.uint8), expected.view(torch.uint8))
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


@pytest.mark.parametrize("src,scale,dtype,width,axis,error", [
    ("fp16", "u8", tl.bfloat16, 1, 1, "src must have int8 or uint8 dtype"),
    ("u8", "fp32", tl.bfloat16, 1, 1, "scale must have int8 or uint8 dtype"),
    ("u8", "u8", tl.float32, 1, 1, "elem_type must be tl.float16 or tl.bfloat16"),
    ("u8", "u8", tl.bfloat16, 2, 1, "scale packing dimension must be"),
    ("u8", "u8", tl.bfloat16, 1, 2, "axis 2 must be"),
])
def test_metax_fp4_scaled_invalid(src, scale, dtype, width, axis, error):
    source = ASTSource(_scaled_documented, {"X": "*" + src, "S": "*" + scale, "Y": "*bf16"},
                       {"DTYPE": dtype, "RAW": False, "WIDTH": width, "AXIS": axis})
    with pytest.raises(CompilationError, match=error):
        triton.compile(source, target=GPUTarget("maca", 80, 64))


@pytest.mark.parametrize("dtype,rounding,axis,block,error", [
    ("i32", "rtne", -1, 512, "requires FP16/BF16/FP32/FP64"),
    ("fp32", "bad", -1, 512, "rounding must be rtne or rtz"),
    ("fp32", "rtne", 1, 512, "axis 1 must be"),
    ("fp32", "rtne", 0, 1, "packing dimension must be even"),
])
def test_metax_fp4_encode_invalid(dtype, rounding, axis, block, error):
    source = ASTSource(_encode, {"X": "*" + dtype, "Y": "*u8"},
                       {"N": 512, "ROUND": rounding, "AXIS": axis, "BLOCK": block})
    with pytest.raises(CompilationError, match=error):
        triton.compile(source, target=GPUTarget("maca", 80, 64))


@pytest.mark.parametrize("arch", [80, 86])
def test_metax_fp4_scaled_diagnostic(arch, fresh_triton_cache, monkeypatch, capfd):
    monkeypatch.setenv("MLIR_ENABLE_DIAGNOSTICS", "warnings")
    source = ASTSource(_scaled, {"X": "*u8", "S": "*u8", "Y": "*bf16"},
                       {"M": 32, "N": 16, "AXIS": 1, "GROUPED": True, "DTYPE": tl.bfloat16})
    kernel = triton.compile(source, target=GPUTarget("maca", arch, 64))
    assert "non-native scaled FP4 conversion on MetaX" in capfd.readouterr().err
    assert "llvm.mxc.cvt.pk4.f4" not in kernel.asm["llir"]


def test_metax_fp4_encode_diagnostic(fresh_triton_cache):
    source = ASTSource(_encode, {"X": "*fp32", "Y": "*u8"}, {"N": 512, "ROUND": "rtne", "AXIS": -1, "BLOCK": 512})
    with pytest.warns(UserWarning, match="non-native GPU software encoding"):
        kernel = triton.compile(source, target=GPUTarget("maca", 80, 64))
    assert "llvm.mxc.cvt.pk4.f32tof4" not in kernel.asm["llir"]


@pytest.mark.parametrize("rounding", ["rtne", "rtz"])
def test_metax_fp4_encode_odd_tail(rounding, metax_device):
    values = torch.arange(259, dtype=torch.float32).remainder(17) - 8
    values[-1] = -0.
    codes = _reference_codes(values, rounding)
    codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8)))
    reference = torch.full((194, ), 165, dtype=torch.uint8)
    reference[32:162] = codes[::2] | (codes[1::2] << 4)
    output = torch.full_like(reference, 165, device=metax_device)
    _encode[(1, )](values.to(metax_device), output[32:162], 259, rounding)
    assert torch.equal(output.cpu(), reference)
    # The final low nibble retains -0; the padded high nibble is +0.
    assert int(output.cpu()[161]) == 8
