import torch
import triton
import triton.language as tl


@triton.jit
def finish_mm(Partial, SA, SB, Out, M: tl.constexpr, N: tl.constexpr, SPLITS: tl.constexpr, WIDE: tl.constexpr,
              BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WIDE:
        accumulator = tl.full((BLOCK, ), 0, tl.int64)
    else:
        accumulator = tl.full((BLOCK, ), 0, tl.int32)
    for part in tl.static_range(SPLITS):
        accumulator += tl.load(Partial + part * M * N + offsets, offsets < M * N, other=0)
    scale_a = tl.load(SA + offsets // N, offsets < M * N, other=0)
    scale_b = tl.load(SB + offsets % N, offsets < M * N, other=0)
    tl.store(Out + offsets, (accumulator.to(tl.float32) * scale_a) * scale_b, offsets < M * N)


class SplitMatmul:
    """Execute a compiler-generated INT32 partial kernel and exact final reduction."""

    def __init__(self, kernel, shape, splits):
        self.kernel = kernel
        self.shape = tuple(shape)
        self.splits = splits
        m, n, k = self.shape
        tiles = triton.cdiv(k, 256)
        if splits < 2 or tiles % splits or not 2 <= tiles // splits < 512:
            raise ValueError("Invalid split geometry or unsafe INT32 partial accumulation")
        self.grid = (triton.cdiv(m, 128) * triton.cdiv(n, 128), splits, 1)
        self.invoke = kernel[self.grid]
        self.workspace = torch.empty((splits, m, n), device="cuda", dtype=torch.int32)

    def __call__(self, a, b, sa, sb, out):
        m, n, k = self.shape
        if a.shape != (m, k) or b.shape != (k, n) or out.shape != (m, n):
            raise ValueError("Tensor dimensions do not match the compiled specialization")
        if a.dtype != torch.int8 or b.dtype != torch.int8 or out.dtype != torch.bfloat16:
            raise ValueError("Split MM requires INT8 operands and BF16 output")
        if a.stride() != (k, 1) or b.stride() != (1, k) or not out.is_contiguous():
            raise ValueError("Split MM requires contiguous A, transposed-contiguous B, and contiguous output")
        if sa.numel() != m or sb.numel() != n or sa.dtype != torch.float32 or sb.dtype != torch.float32:
            raise ValueError("Split MM requires one FP32 scale per row and column")
        if not sa.is_contiguous() or not sb.is_contiguous():
            raise ValueError("Scales must be contiguous")
        for tensor in (a, b, sa, sb, out):
            if tensor.device != self.workspace.device or tensor.data_ptr() % 16:
                raise ValueError("Tensors must share the workspace device and 16-byte alignment")
        self.invoke(a, b, sa, sb, self.workspace)
        return finish_mm[(triton.cdiv(m * n, 256), )](self.workspace, sa, sb, out, m, n, self.splits, k > 131071, 256,
                                                      num_warps=4, num_stages=1, enable_fp_fusion=False)
