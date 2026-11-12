"""Cross-precision accumulators carried through runtime loop boundaries."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._flagtree_backend import FLAGTREE_BACKEND
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.tools.tensor_descriptor import TensorDescriptor

pytestmark = pytest.mark.skipif(FLAGTREE_BACKEND not in ("", "nvidia", "mthreads"), reason="Requires WGMMA or SQMMA")

FORMATS = {
    "fp8e4nv": (tl.float8e4nv, torch.float8_e4m3fn, 32),
    "fp16": (tl.float16, torch.float16, 16),
    "bf16": (tl.bfloat16, torch.bfloat16, 16),
    "i8": (tl.int8, torch.int8, 128),
}
PAIRS = [(a, b) for a in FORMATS for b in FORMATS if a != b and "i8" not in (a, b)]
SHAPES = [(64, 64, 4), (128, 64, 8), (64, 128, 8)]


@triton.jit(do_not_specialize=["outer", "inner"])
def accumulator_kernel(A0, B0, A1, B1, Out, outer, inner, D0: tl.constexpr, D1: tl.constexpr, K0: tl.constexpr,
                       K1: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MODE: tl.constexpr,
                       WAIT: tl.constexpr = True, MATERIALIZE: tl.constexpr = False):
    a0 = tle.gpu.alloc((M, K0), D0, layout=None, nv_mma_shared_layout=True)
    b0 = tle.gpu.alloc((N, K0), D0, layout=None, nv_mma_shared_layout=True)
    a1 = tle.gpu.alloc((M, K1), D1, layout=None, nv_mma_shared_layout=True)
    b1 = tle.gpu.alloc((N, K1), D1, layout=None, nv_mma_shared_layout=True)
    ready_a0 = tle.gpu.alloc_barrier(expect_bytes=M * K0 * (1 if D0 == tl.float8e4nv or D0 == tl.int8 else 2))
    ready_b0 = tle.gpu.alloc_barrier(expect_bytes=N * K0 * (1 if D0 == tl.float8e4nv or D0 == tl.int8 else 2))
    ready_a1 = tle.gpu.alloc_barrier(expect_bytes=M * K1 * (1 if D1 == tl.float8e4nv or D1 == tl.int8 else 2))
    ready_b1 = tle.gpu.alloc_barrier(expect_bytes=N * K1 * (1 if D1 == tl.float8e4nv or D1 == tl.int8 else 2))
    tle.gpu.copy(A0, a0, (M, K0), (0, 0), barrier=ready_a0)
    tle.gpu.copy(B0, b0, (N, K0), (0, 0), barrier=ready_b0)
    tle.gpu.copy(A1, a1, (M, K1), (0, 0), barrier=ready_a1)
    tle.gpu.copy(B1, b1, (N, K1), (0, 0), barrier=ready_b1)
    tle.gpu.barrier_wait(ready_a0, phaseIdx=0)
    tle.gpu.barrier_wait(ready_b0, phaseIdx=0)
    tle.gpu.barrier_wait(ready_a1, phaseIdx=0)
    tle.gpu.barrier_wait(ready_b1, phaseIdx=0)
    ACC_DTYPE: tl.constexpr = tl.int32 if D0 == tl.int8 else tl.float32
    ACC_INITIAL: tl.constexpr = 17 if D0 == tl.int8 else 0.5
    acc = tl.full((M, N), ACC_INITIAL, ACC_DTYPE)
    if MODE == 0:
        acc = tle.gpu.wgmma(a0, b0, acc, out_dtype=ACC_DTYPE, trans_b=True)
        if MATERIALIZE:
            acc = tle.gpu.wgmma_wait(0, acc)
        acc = tle.gpu.wgmma(a1, b1, acc, out_dtype=ACC_DTYPE, trans_b=True)
    elif MODE == 1:
        for _ in range(inner):
            acc = tle.gpu.wgmma(a0, b0, acc, out_dtype=ACC_DTYPE, trans_b=True)
            acc = tle.gpu.wgmma(a1, b1, acc, out_dtype=ACC_DTYPE, trans_b=True)
    else:
        for _ in range(outer):
            for _ in range(inner):
                acc = tle.gpu.wgmma(a0, b0, acc, out_dtype=ACC_DTYPE, trans_b=True)
                acc = tle.gpu.wgmma(a1, b1, acc, out_dtype=ACC_DTYPE, trans_b=True)
    if WAIT:
        acc = tle.gpu.wgmma_wait(0, acc)
    offsets = tl.program_id(0) * M * N + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    tl.store(Out + offsets, acc)


def compile_accumulator(first, second, m, n, warps, mode, wait=True, materialize=False):
    d0, _, k0 = FORMATS[first]
    d1, _, k1 = FORMATS[second]
    signature = dict(A0=f"tensordesc<{first}[{m},{k0}]>", B0=f"tensordesc<{first}[{n},{k0}]>",
                     A1=f"tensordesc<{second}[{m},{k1}]>", B1=f"tensordesc<{second}[{n},{k1}]>",
                     Out="*i32" if first == "i8" else "*fp32", outer="i32", inner="i32", D0="constexpr", D1="constexpr",
                     K0="constexpr", K1="constexpr", M="constexpr", N="constexpr", MODE="constexpr", WAIT="constexpr",
                     MATERIALIZE="constexpr")
    constants = dict(D0=d0, D1=d1, K0=k0, K1=k1, M=m, N=n, MODE=mode, WAIT=wait, MATERIALIZE=materialize)
    musa = FLAGTREE_BACKEND == "mthreads"
    attrs = {(i, ): [["musa.tme_tail_divisibility", 4]] for i in range(4)} if musa else {}
    target = GPUTarget("musa", "ph1", 32) if musa else GPUTarget("cuda", 90, 32)
    return triton.compile(ASTSource(accumulator_kernel, signature, constexprs=constants, attrs=attrs), target=target,
                          options=dict(num_warps=warps, num_stages=1))


@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("m,n,warps", SHAPES)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_accumulator_loop_compilation(first, second, m, n, warps, mode, capfd):
    if FLAGTREE_BACKEND != "mthreads" and mode:
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            compile_accumulator(first, second, m, n, warps, mode)
        assert "unwaited async accumulators across scf.for yields" in capfd.readouterr().err
        return
    kernel = compile_accumulator(first, second, m, n, warps, mode)
    assert "musa_tle.sqmma" not in kernel.asm["ttgir"]
    assert "tle.wgmma " not in kernel.asm["ttgir"]
    if mode:
        assert kernel.asm["ttgir"].count("scf.for") == mode


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_accumulator_requires_wait(mode, capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_accumulator("fp8e4nv", "fp16", 64, 64, 4, mode, wait=False)
    diagnostic = capfd.readouterr().err
    if FLAGTREE_BACKEND != "mthreads" and mode:
        assert "unwaited async accumulators across scf.for yields" in diagnostic
    else:
        assert "async result must be consumed" in diagnostic


@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("m,n,warps", SHAPES)
@pytest.mark.parametrize("mode,outer,inner", [(0, 1, 1), (2, 0, 2), (2, 2, 0), (2, 1, 1), (2, 2, 3)])
def test_accumulator_numerics(first, second, m, n, warps, mode, outer, inner):
    if FLAGTREE_BACKEND != "mthreads" and mode:
        pytest.skip("NVIDIA explicitly rejects unwaited async loop yields")
    if FLAGTREE_BACKEND == "mthreads":
        if not torch.musa.is_available():
            pytest.skip("Requires a MUSA device")
        device = "musa"
    else:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("Requires Hopper")
        device = "cuda"
    torch.manual_seed(1401)
    descriptors, products = [], []
    for name in (first, second):
        _, dtype, k = FORMATS[name]
        a = (torch.randn((m, k)) * 0.25).to(dtype)
        b = (torch.randn((n, k)) * 0.25).to(dtype)
        products.append(a.double() @ b.double().T)
        descriptors.extend(
            [TensorDescriptor.from_tensor(a.to(device), [m, k]),
             TensorDescriptor.from_tensor(b.to(device), [n, k])])
    out = torch.empty((m, n), dtype=torch.float32, device=device)
    accumulator_kernel[(1, )](*descriptors, out, outer, inner, FORMATS[first][0], FORMATS[second][0], FORMATS[first][2],
                              FORMATS[second][2], m, n, mode, num_warps=warps, num_stages=1)
    expected = (0.5 + outer * inner * (products[0] + products[1])).float()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-3, atol=2e-3)


@pytest.mark.skipif(FLAGTREE_BACKEND != "mthreads", reason="Requires MUSA INT8 SQMMA")
@pytest.mark.parametrize("m,n,warps", SHAPES)
@pytest.mark.parametrize("mode,materialize", [(0, False), (0, True), (1, False), (2, False)])
def test_integer_accumulator_loop_compilation(m, n, warps, mode, materialize):
    kernel = compile_accumulator("i8", "i8", m, n, warps, mode, materialize=materialize)
    assert "musa_tle.sqmma" not in kernel.asm["ttgir"]
    assert re.search(r"\bcall\s+<\d+\s+x\s+i32>\s+@llvm\.musa\.sqmma\.smma\.", kernel.asm["llir"])
    if mode:
        assert kernel.asm["ttgir"].count("scf.for") == mode


@pytest.mark.skipif(FLAGTREE_BACKEND != "mthreads", reason="Requires MUSA INT8 SQMMA")
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_integer_accumulator_requires_wait(mode, capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compile_accumulator("i8", "i8", 64, 64, 4, mode, wait=False)
    assert "async result must be consumed" in capfd.readouterr().err


@pytest.mark.skipif(FLAGTREE_BACKEND != "mthreads", reason="Requires MUSA INT8 SQMMA")
@pytest.mark.parametrize("m,n,warps", SHAPES)
@pytest.mark.parametrize("mode,outer,inner,materialize", [
    (0, 1, 1, False),
    (0, 1, 1, True),
    (1, 1, 3, False),
    (2, 0, 2, False),
    (2, 2, 0, False),
    (2, 2, 3, False),
])
@pytest.mark.parametrize("case", ["random", "extreme", "cancellation"])
def test_integer_accumulator_numerics(m, n, warps, mode, outer, inner, materialize, case):
    if not torch.musa.is_available():
        pytest.skip("Requires a MUSA device")
    generator = torch.Generator(device="cpu").manual_seed(1401)
    k = FORMATS["i8"][2]
    descriptors, products = [], []
    for index in range(2):
        if case == "random":
            a = torch.randint(-128, 128, (m, k), dtype=torch.int8, generator=generator)
            b = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=generator)
        elif case == "extreme":
            value = -128 if index == 0 else 127
            a = torch.full((m, k), value, dtype=torch.int8)
            b = torch.full((n, k), value, dtype=torch.int8)
        else:
            a = torch.full((m, k), 127, dtype=torch.int8)
            b = torch.full((n, k), 127 if index == 0 else -127, dtype=torch.int8)
        products.append(a.to(torch.int64) @ b.to(torch.int64).T)
        descriptors.extend([
            TensorDescriptor.from_tensor(a.to("musa"), [m, k]),
            TensorDescriptor.from_tensor(b.to("musa"), [n, k]),
        ])
    out = torch.empty((m, n), dtype=torch.int32, device="musa")
    accumulator_kernel[(1, )](*descriptors, out, outer, inner, tl.int8, tl.int8, k, k, m, n, mode,
                              MATERIALIZE=materialize, num_warps=warps, num_stages=1)
    repeats = (1, inner, outer * inner)[mode]
    expected = (17 + repeats * (products[0] + products[1])).to(torch.int32)
    assert torch.equal(out.cpu(), expected)
