import triton
import triton.language as tl


@triton.jit
def mm_masked(A, B, SA, SB, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr = 128,
              BN: tl.constexpr = 128, BK: tl.constexpr = 256):
    pid = tl.program_id(0)
    pm = pid % tl.cdiv(M, BM)
    pn = pid // tl.cdiv(M, BM)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    pa = A + rm[:, None] * K + rk[None, :]
    pb = B + rn[None, :] * K + rk[:, None]
    acc = tl.full((BM, BN), 0, tl.int32)
    for i in range(tl.cdiv(K, BK)):
        if K % BK:
            a = tl.load(pa, (rm[:, None] < M) & (rk[None, :] + i * BK < K), other=0)
            b = tl.load(pb, (rn[None, :] < N) & (rk[:, None] + i * BK < K), other=0)
        else:
            a = tl.load(pa, rm[:, None] < M, other=0)
            b = tl.load(pb, rn[None, :] < N, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        pa += BK
        pb += BK
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    value = (acc.to(tl.float32) * sa[:, None]) * sb[None, :]
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], value, (rm[:, None] < M) & (rn[None, :] < N))
