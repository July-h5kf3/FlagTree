# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end Ascend NPU verification for CommonIRToHIVM and TritonToHFusion
compiler fixes and optimizations:
  1. Bug-1: 1D UB subview broadcast/expand preserving #hivm.address_space<ub>.
  2. Bug-2: Cube 3D strided GM load + reshape -> tl.dot matmul decomposition.
  3. P1-1: L1 buffer residency across Cube/Vector sync (zero GM reload in PV).
  4. P0-2: Splat fill and in-place UB update folding (zero hivm.hir.copy).
  5. P0-1b: Small non-unit-stride 1D GM store scalarization (zero stride-8 UB blowup).
  6. P2-1: Saturating RINT FP16->INT8 fusion to single hivm.hir.vcast (vconv_f162s8r)
           and exact FP16/FP32 standalone libdevice.rint fallback.
"""

import glob
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
import triton.language.extra.cann.libdevice as libdevice
from triton.backends.ascend.compiler import _get_npucompiler_path

CV_OPTS = dict(
    enable_auto_bind_sub_block=False,
    disable_auto_inject_block_sync=True,
    multibuffer=False,
)


def _fresh_cache():
    cache_dir = tempfile.mkdtemp(prefix="ascend_opt_test_")
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    triton.knobs.cache.dir = cache_dir
    return cache_dir


def _dump_hivm_ir(cache_dir, kernel_name):
    bc_files = glob.glob(f"{cache_dir}/**/*{kernel_name}*.bcmlir", recursive=True)
    assert bc_files, f"No .bcmlir found for {kernel_name} in {cache_dir}"
    compiler, env = _get_npucompiler_path()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_obj = os.path.join(tmpdir, f"{kernel_name}.o")
        proc = subprocess.run(
            [
                compiler,
                bc_files[0],
                "--enable-hivm-compile=true",
                "--enable-auto-multi-buffer=true",
                "--enable-hfusion-compile=true",
                "--enable-triton-kernel-compile=true",
                "--target=Ascend910B4",
                "--bishengir-print-ir-after=hivm-graph-sync-solver",
                "-o",
                out_obj,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"bishengir-compile failed for {kernel_name}:\n{proc.stderr[-800:]}"
        return proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# 1. Bug-1: 1D UB subview + [None, :] broadcast
# ---------------------------------------------------------------------------
@triton.jit
def _bug1_1d_ub_expand_kernel(X, Y, Out, N: tl.constexpr, M: tl.constexpr):
    buf = tle.dsa.alloc([2 * N], tl.float32, tle.dsa.ascend.UB)
    sv = tle.dsa.subview(buf, [N], [N], [1])
    offs_n = tl.arange(0, N)
    offs_m = tl.arange(0, M)
    x = tl.load(X + offs_n)
    tle.dsa.to_buffer(x, bind_buffer=sv)
    x_row = tle.dsa.to_tensor(sv)[None, :]
    y = tl.load(Y + offs_m[:, None] * N + offs_n[None, :])
    tl.store(Out + offs_m[:, None] * N + offs_n[None, :], y + x_row)


def test_bug1_ub_1d_broadcast():
    cache_dir = _fresh_cache()
    try:
        torch.manual_seed(0)
        M, N = 16, 64
        x = torch.randn((N, ), dtype=torch.float32, device="npu")
        y = torch.randn((M, N), dtype=torch.float32, device="npu")
        out = torch.empty((M, N), dtype=torch.float32, device="npu")
        _bug1_1d_ub_expand_kernel[(1, )](x, y, out, N=N, M=M)
        torch.npu.synchronize()
        ref = y.cpu() + x.cpu()[None, :]
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)
        print("[PASS] Bug-1: 1D UB subview + [None, :] compiles and matches PyTorch")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Bug-2: Cube 3D strided GM load + reshape -> tl.dot
# ---------------------------------------------------------------------------
@triton.jit
def _bug2_3d_load_dot_kernel(Q, K, C, H: tl.constexpr, D: tl.constexpr):
    s = tl.arange(0, 2)
    h = tl.arange(0, 16)
    d = tl.arange(0, D)
    q3 = tl.load(Q + s[:, None, None] * (H * D) + h[None, :, None] * D + d[None, None, :])
    q = q3.reshape(32, D)
    k = tl.load(K + tl.arange(0, 64)[:, None] * D + d[None, :])
    c = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
    tl.store(C + tl.arange(0, 32)[:, None] * 64 + tl.arange(0, 64)[None, :], c)


@triton.jit
def _bug2_nonlinear_dot_kernel(Q, K, C, H: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, 32)
    rows = (offs // 16) * H + (offs % 16)
    d = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * D + d[None, :])
    k = tl.load(K + tl.arange(0, 64)[:, None] * D + d[None, :])
    c = tl.dot(q, tl.trans(k), out_dtype=tl.int32)
    tl.store(C + offs[:, None] * 64 + tl.arange(0, 64)[None, :], c)


def test_bug2_cube_3d_load_dot():
    cache_dir = _fresh_cache()
    try:
        torch.manual_seed(1)
        H, D = 64, 64
        Q = torch.randint(-5, 6, (2, H, D), dtype=torch.int8, device="npu")
        K = torch.randint(-5, 6, (64, D), dtype=torch.int8, device="npu")
        q_ref = torch.cat([Q[0, :16], Q[1, :16]], dim=0).cpu().to(torch.int32)
        ref = q_ref @ K.cpu().to(torch.int32).T
        for name, kern in [("3D.reshape", _bug2_3d_load_dot_kernel), ("nonlinear_rows", _bug2_nonlinear_dot_kernel)]:
            C = torch.empty((32, 64), dtype=torch.int32, device="npu")
            kern[(1, )](Q, K, C, H=H, D=D)
            torch.npu.synchronize()
            c_cpu = C.cpu()
            err_q0 = (c_cpu[:16] - ref[:16]).abs().max().item()
            err_q1 = (c_cpu[16:] - ref[16:]).abs().max().item()
            assert err_q0 == 0 and err_q1 == 0, f"Bug-2 {name} mismatch: err_q0={err_q0}, err_q1={err_q1}"
            print(f"[PASS] Bug-2 ({name}): err_q0={err_q0}, err_q1={err_q1}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. P1-1: L1 residency across CV sync (zero GM reload in PV)
# ---------------------------------------------------------------------------
@triton.jit
def _p11_l1_cv_residency_kernel(Q, K, C_out, P_buf, PV_out):
    with al.scope(core_mode="cube"):
        k_l1 = tle.dsa.alloc([64, 512], tl.int8, tle.dsa.ascend.L1)
        q = tl.load(Q + tl.arange(0, 16)[:, None] * 512 + tl.arange(0, 512)[None, :])
        k_gm = tl.load(K + tl.arange(0, 64)[:, None] * 512 + tl.arange(0, 512)[None, :])
        tle.dsa.to_buffer(k_gm, bind_buffer=k_l1)
        c_p = tl.dot(q, tl.trans(tle.dsa.to_tensor(k_l1)), out_dtype=tl.int32)
        tl.store(C_out + tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :], c_p)
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)
        p_blk = tl.load(P_buf + tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :])
        pv = tl.dot(p_blk, tle.dsa.to_tensor(k_l1), out_dtype=tl.int32)
        tl.store(PV_out + tl.arange(0, 16)[:, None] * 512 + tl.arange(0, 512)[None, :], pv)
    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2)
        if al.sub_vec_id() == 0:
            c_v = tl.load(C_out + tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :])
            p_v = tl.clamp(c_v.to(tl.float32) * 0.01, -10.0, 10.0).to(tl.int8)
            tl.store(P_buf + tl.arange(0, 16)[:, None] * 64 + tl.arange(0, 64)[None, :], p_v)
        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2)


def test_p11_l1_cv_residency():
    cache_dir = _fresh_cache()
    try:
        torch.manual_seed(2)
        Q = torch.randint(-4, 5, (16, 512), dtype=torch.int8, device="npu")
        K = torch.randint(-4, 5, (64, 512), dtype=torch.int8, device="npu")
        C_out = torch.empty((16, 64), dtype=torch.int32, device="npu")
        P_buf = torch.empty((16, 64), dtype=torch.int8, device="npu")
        PV_out = torch.empty((16, 512), dtype=torch.int32, device="npu")
        _p11_l1_cv_residency_kernel[(1, )](Q, K, C_out, P_buf, PV_out, **CV_OPTS)
        torch.npu.synchronize()
        ref_c = Q.cpu().to(torch.int32) @ K.cpu().to(torch.int32).T
        ref_p = (ref_c.float() * 0.01).clamp(-10.0, 10.0).to(torch.int8)
        ref_pv = ref_p.to(torch.int32) @ K.cpu().to(torch.int32)
        assert (C_out.cpu() - ref_c).abs().max().item() == 0
        assert (P_buf.cpu() - ref_p).abs().max().item() == 0
        assert (PV_out.cpu() - ref_pv).abs().max().item() == 0
        ir = _dump_hivm_ir(cache_dir, "_p11_l1_cv_residency_kernel")
        nd2nz_lines = [ln.strip() for ln in ir.splitlines() if "hivm.hir.nd2nz" in ln]
        # Exactly 3 nd2nz ops: Q (16x512), K (64x512, reused in PV!), P (16x64).
        assert len(nd2nz_lines) == 3, f"Expected 3 nd2nz ops, got {len(nd2nz_lines)}:\n" + "\n".join(nd2nz_lines)
        print(f"[PASS] P1-1: L1 residency across CV sync exact (err_c=0, err_p=0, err_pv=0), nd2nz={len(nd2nz_lines)}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. P0-2: Splat fill + in-place UB update folding (zero hivm.hir.copy)
# ---------------------------------------------------------------------------
@triton.jit
def _p02_ub_inplace_kernel(X, Out):
    acc = tle.dsa.alloc([32, 256], tl.float32, tle.dsa.ascend.UB)
    tle.dsa.to_buffer(tl.zeros((32, 256), dtype=tl.float32), bind_buffer=acc)
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 256)
    for it in range(4):
        for half in range(2):
            sv = tle.dsa.subview(acc, [half * 16, 0], [16, 256], [1, 1])
            x = tl.load(X + ((it * 32 + half * 16 + offs_m)[:, None] * 256 + offs_n[None, :]))
            acc_new = tle.dsa.to_tensor(sv) * 0.5 + x
            tle.dsa.to_buffer(acc_new, bind_buffer=sv)
    tl.store(Out + tl.arange(0, 32)[:, None] * 256 + offs_n[None, :], tle.dsa.to_tensor(acc))


def test_p02_ub_inplace_reuse():
    cache_dir = _fresh_cache()
    try:
        torch.manual_seed(3)
        X = torch.randn((4, 32, 256), dtype=torch.float32, device="npu")
        Out = torch.empty((32, 256), dtype=torch.float32, device="npu")
        _p02_ub_inplace_kernel[(1, )](X, Out)
        torch.npu.synchronize()
        ref = torch.zeros((32, 256), dtype=torch.float32)
        for it in range(4):
            ref = ref * 0.5 + X[it].cpu()
        torch.testing.assert_close(Out.cpu(), ref, rtol=1e-5, atol=1e-5)
        ir = _dump_hivm_ir(cache_dir, "_p02_ub_inplace_kernel")
        copies = [ln.strip() for ln in ir.splitlines() if "hivm.hir.copy" in ln]
        ptrs = [ln.strip() for ln in ir.splitlines() if "hivm.hir.pointer_cast" in ln]
        assert len(copies) == 0, "Expected 0 hivm.hir.copy, got:\n" + "\n".join(copies)
        print(f"[PASS] P0-2: exact accuracy, hivm.hir.copy=0, pointer_casts={len(ptrs)}")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. P0-1b: Small strided 1D GM store isolated from UB reduction buffers
# ---------------------------------------------------------------------------
@triton.jit
def _p01b_strided_lse_kernel(X, Lse):
    m_buf = tle.dsa.alloc([16, 1], tl.float32, tle.dsa.ascend.UB)
    offs_r = tl.arange(0, 8)
    offs_c = tl.arange(0, 64)
    x = tl.load(X + offs_r[:, None] * 64 + offs_c[None, :])
    m_val = tl.max(x, axis=1, keep_dims=True)
    sv = tle.dsa.subview(m_buf, [0, 0], [8, 1], [1, 1])
    tle.dsa.to_buffer(m_val, bind_buffer=sv)
    m_out = tle.dsa.to_tensor(sv).reshape(8)
    tl.store(Lse + offs_r * 2 + 1, m_out)


def test_p01b_strided_1d_store_isolation():
    cache_dir = _fresh_cache()
    try:
        torch.manual_seed(4)
        X = torch.randn((8, 64), dtype=torch.float32, device="npu")
        Lse = torch.full((16, ), -999.0, dtype=torch.float32, device="npu")
        _p01b_strided_lse_kernel[(1, )](X, Lse)
        torch.npu.synchronize()
        ref = X.cpu().max(dim=1).values
        lse_cpu = Lse.cpu()
        torch.testing.assert_close(lse_cpu[1::2], ref, rtol=1e-5, atol=1e-5)
        assert torch.all(lse_cpu[0::2] == -999.0)
        ir = _dump_hivm_ir(cache_dir, "_p01b_strided_lse_kernel")
        bad = [ln.strip() for ln in ir.splitlines() if "strided<[8" in ln]
        assert len(bad) == 0, "Stride-8 contagion still present:\n" + "\n".join(bad)
        print("[PASS] P0-1b: strided 1D GM store exact with ZERO strided<[8,*] contagion")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. P2-1: Saturating RINT FP16->INT8 fusion + exact standalone rint fallback
# ---------------------------------------------------------------------------
@triton.jit
def _p21_fused_quant_kernel(X, Y, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    q = tl.clamp(libdevice.rint(x), -128.0, 127.0).to(tl.int8)
    tl.store(Y + offs, q)


@triton.jit
def _p21_standalone_rint_kernel(X, Y, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    tl.store(Y + offs, libdevice.rint(x))


@triton.jit
def _p21_bench_kernel(X, Y, ELEMS: tl.constexpr, TILES: tl.constexpr, MODE: tl.constexpr):
    for tile in range(tl.program_id(0), TILES, tl.num_programs(0)):
        vals = tl.load(X + tile * ELEMS + tl.arange(0, ELEMS))
        if MODE == 0:
            out_vals = tl.clamp(libdevice.rint(vals), -128.0, 127.0).to(tl.int8)
        elif MODE == 1:
            out_vals = tle.dsa.ascend.raw("cast_fp16_to_int8_rint", vals, ELEMS, out=tl.full((ELEMS, ), 0, tl.int8))
        else:
            fp32 = vals.to(tl.float32)
            out_vals = tl.maximum(-128.0, tl.minimum(127.0, tl.floor(fp32 + 0.5))).to(tl.int8)
        tl.store(Y + tile * ELEMS + tl.arange(0, ELEMS), out_vals)


def test_p21_rint_saturating_cast():
    cache_dir = _fresh_cache()
    try:
        rng = np.random.default_rng(42)
        raw = rng.uniform(-160.0, 160.0, size=1024).astype(np.float16)
        special = np.array(
            [-300.0, -128.5, -128.0, -3.5, -2.5, -1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5, 2.5, 3.5, 127.0, 127.5, 300.0],
            dtype=np.float16,
        )
        raw[:len(special)] = special
        x = torch.from_numpy(raw).to("npu")

        y_q = torch.empty(1024, dtype=torch.int8, device="npu")
        _p21_fused_quant_kernel[(1, )](x, y_q, N=1024)
        torch.npu.synchronize()
        ref_q = np.clip(np.rint(raw.astype(np.float32)), -128.0, 127.0).astype(np.int8)
        np.testing.assert_array_equal(y_q.cpu().numpy(), ref_q)
        ir = _dump_hivm_ir(cache_dir, "_p21_fused_quant_kernel")
        vcasts = [ln.strip() for ln in ir.splitlines() if "hivm.hir.vcast" in ln]
        assert len(vcasts) == 1, f"Expected 1 vcast op, got {len(vcasts)}:\n" + "\n".join(vcasts)
        print(f"[PASS] P2-1 fused quant: bit-exact, single vcast ({vcasts[0]})")

        y_f16 = torch.empty(1024, dtype=torch.float16, device="npu")
        _p21_standalone_rint_kernel[(1, )](x, y_f16, N=1024)
        torch.npu.synchronize()
        np.testing.assert_array_equal(y_f16.cpu().numpy(), np.rint(raw))

        x_f32 = torch.from_numpy(raw.astype(np.float32)).to("npu")
        y_f32 = torch.empty(1024, dtype=torch.float32, device="npu")
        _p21_standalone_rint_kernel[(1, )](x_f32, y_f32, N=1024)
        torch.npu.synchronize()
        np.testing.assert_array_equal(y_f32.cpu().numpy(), np.rint(raw.astype(np.float32)))
        print("[PASS] P2-1 standalone libdevice.rint: bit-exact on FP16 and FP32 ties")

        elems, tiles = 8192, 2560
        bx = torch.from_numpy(np.linspace(-140.0, 140.0, elems * tiles, dtype=np.float16)).to("npu")
        by = torch.empty(elems * tiles, dtype=torch.int8, device="npu")

        def launch(mode):
            return _p21_bench_kernel[(40, )](bx, by, elems, tiles, mode, num_warps=1, multibuffer=False,
                                             enable_fp_fusion=False)

        for mode in (0, 1, 2):
            launch(mode)
            torch.npu.synchronize()
        timings_us = {}
        for name, mode in [("compiler_fused_vcast", 0), ("custom_op_vconv", 1), ("old_floor_fp32", 2)]:
            timings_us[name] = triton.testing.do_bench(lambda m=mode: launch(m), return_mode="median") * 1000.0
        print(f"[BENCH] P2-1 (8192x2560): compiler_fused={timings_us['compiler_fused_vcast']:.1f} us, "
              f"custom_op={timings_us['custom_op_vconv']:.1f} us, "
              f"old_floor_fp32={timings_us['old_floor_fp32']:.1f} us "
              f"(speedup vs old={timings_us['old_floor_fp32'] / timings_us['compiler_fused_vcast']:.2f}x)")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    test_bug1_ub_1d_broadcast()
    test_bug2_cube_3d_load_dot()
    test_p11_l1_cv_residency()
    test_p02_ub_inplace_reuse()
    test_p01b_strided_1d_store_isolation()
    test_p21_rint_saturating_cast()
    print("ALL 6 ASCEND COMPILER OPTIMIZATION TESTS PASSED!")


if __name__ == "__main__":
    main()
