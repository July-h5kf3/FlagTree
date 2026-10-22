"""FP32 accumulator reuse across SQMMA operand dtypes."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from test_tle_utils import require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _alloc_sqmma(M: tl.constexpr, N: tl.constexpr, DTYPE: tl.constexpr):
    if DTYPE == 0:
        value = tle.gpu.alloc((M, N), tl.float8e4nv, layout=None, nv_mma_shared_layout=True)
    elif DTYPE == 1:
        value = tle.gpu.alloc((M, N), tl.float16, layout=None, nv_mma_shared_layout=True)
    else:
        value = tle.gpu.alloc((M, N), tl.bfloat16, layout=None, nv_mma_shared_layout=True)
    return value


@triton.jit
def _cross_dtype_kernel(A0, B0, A1, B1, Out, T0: tl.constexpr, T1: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
                        REPEATS: tl.constexpr):
    a0 = _alloc_sqmma(64, K0, T0)
    b0 = _alloc_sqmma(K0, 64, T0)
    a1 = _alloc_sqmma(64, K1, T1)
    b1 = _alloc_sqmma(K1, 64, T1)
    a_ready0 = tle.gpu.alloc_barrier(expect_bytes=64 * K0 * (1 if T0 == 0 else 2))
    b_ready0 = tle.gpu.alloc_barrier(expect_bytes=64 * K0 * (1 if T0 == 0 else 2))
    a_ready1 = tle.gpu.alloc_barrier(expect_bytes=64 * K1 * (1 if T1 == 0 else 2))
    b_ready1 = tle.gpu.alloc_barrier(expect_bytes=64 * K1 * (1 if T1 == 0 else 2))
    tle.gpu.copy(A0, a0, (64, K0), (0, 0), barrier=a_ready0)
    tle.gpu.copy(B0, b0, (K0, 64), (0, 0), barrier=b_ready0)
    tle.gpu.copy(A1, a1, (64, K1), (0, 0), barrier=a_ready1)
    tle.gpu.copy(B1, b1, (K1, 64), (0, 0), barrier=b_ready1)
    tle.gpu.barrier_wait(a_ready0, phaseIdx=0)
    tle.gpu.barrier_wait(b_ready0, phaseIdx=0)
    tle.gpu.barrier_wait(a_ready1, phaseIdx=0)
    tle.gpu.barrier_wait(b_ready1, phaseIdx=0)
    acc = tl.full((64, 64), 0.5, tl.float32)
    for _ in range(REPEATS):
        acc = tle.gpu.wgmma(a0, b0, acc)
        acc = tle.gpu.wgmma(a1, b1, acc)
    acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(Out + offsets, acc)


@pytest.mark.parametrize("other", ["float16", "bfloat16"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("repeats", [1, 2])
def test_cross_dtype_accumulator_runtime(other, reverse, repeats):
    from triton.tools.tensor_descriptor import TensorDescriptor

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    torch.manual_seed(917)
    dtypes = [(torch.float8_e4m3fn, 0, "e4m3", 32),
              (getattr(torch, other), 1 if other == "float16" else 2, "fmma" if other == "float16" else "bfmma", 16)]
    if reverse:
        dtypes.reverse()
    inputs, descriptors = [], []
    for torch_type, _, _, k in dtypes:
        a_shape, b_shape = (64, k), (k, 64)
        a = (torch.randn(a_shape) * 0.25).to(torch_type)
        b = (torch.randn(b_shape) * 0.25).to(torch_type)
        inputs.append((a, b))
        descriptors.extend([
            TensorDescriptor.from_tensor(a.to("musa"), list(a_shape)),
            TensorDescriptor.from_tensor(b.to("musa"), list(b_shape))
        ])
    shape = (64, 64)
    out = torch.empty(shape, dtype=torch.float32, device="musa")
    kernel = _cross_dtype_kernel[(1, )](
        *descriptors,
        out,
        dtypes[0][1],
        dtypes[1][1],
        dtypes[0][3],
        dtypes[1][3],
        repeats,
        num_warps=4,
        num_stages=1,
    )
    expected = torch.full(shape, 0.5, dtype=torch.float64)
    for _ in range(repeats):
        for a, b in inputs:
            expected += a.double() @ b.double()
    torch.testing.assert_close(out.cpu(), expected.float(), rtol=1e-3, atol=2e-3)
    for _, _, tag, k in dtypes:
        assert f"llvm.musa.sqmma.{tag}.m64n64k{k}.mma" in kernel.asm["llir"]
    lines = kernel.asm["ttgir"].splitlines()
    dots = [i for i, line in enumerate(lines) if re.search(r" = (ttmg.squad_dot|mtgpu.sqmma) ", line)]
    assert len(dots) >= 2
    assert "ttg.convert_layout" not in "\n".join(lines[dots[0]:dots[-1] + 1])
