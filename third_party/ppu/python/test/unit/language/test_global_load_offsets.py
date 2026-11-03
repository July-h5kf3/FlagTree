import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def paged_copy_with_offsets(Source, Output, Table, Length, RuntimeStride, ROW_STRIDE: tl.constexpr,
                            DYNAMIC: tl.constexpr, CACHE: tl.constexpr):
    rows = tl.arange(0, 128)
    columns = tl.arange(0, 512)
    page = tl.load(Table + rows // 64).to(tl.int64)
    stride = tl.load(RuntimeStride) if DYNAMIC else ROW_STRIDE
    pointer = (Source + page[:, None] * (64 * ROW_STRIDE) + (rows % 64)[:, None] * stride + columns[None, :])
    values = tl.load(pointer, rows[:, None] < tl.load(Length), 0, cache_modifier=CACHE)
    tl.store(Output + rows[:, None] * 512 + columns[None, :], values)


@pytest.mark.parametrize('dtype,stride,offset,dynamic,cache,active,should_fold', [
    (torch.int8, 512, 0, False, '.cg', 97, True),
    (torch.int8, 528, 0, False, '.cg', 127, True),
    (torch.int8, 512, 0, True, '.cg', 129, False),
    (torch.int8, 512, 1, False, '.cg', 63, False),
    (torch.int32, 512, 0, False, '.cg', 97, False),
    (torch.int8, 512, 0, False, '.ca', 97, False),
])
def test_global_load_offsets(dtype, stride, offset, dynamic, cache, active, should_fold, device):
    generator = torch.Generator().manual_seed(7531)
    storage = torch.randint(-128, 128, (3 * 64 * stride + offset, ), dtype=dtype, generator=generator)
    source = storage.to(device)[offset:]
    table_cpu = torch.tensor([2, 0], dtype=torch.int32)
    table = table_cpu.to(device)
    length = torch.tensor([active], dtype=torch.int32, device=device)
    runtime_stride = torch.tensor([stride], dtype=torch.int32, device=device)
    expected = storage[offset:].view(3, 64, stride)[table_cpu.long(), :, :512]
    expected = expected.reshape(128, 512).clone()
    expected[min(active, 128):] = 0
    output = torch.empty((128, 512), dtype=dtype, device=device)
    for enabled in (False, True):
        output.fill_(51)
        compiled = paged_copy_with_offsets[(1, )](source, output, table, length, runtime_stride, stride, dynamic, cache,
                                                  num_warps=8, num_stages=1, ppu_fold_global_load_offsets=enabled)
        assert torch.equal(output.cpu(), expected)
        folded = compiled.metadata.global_load_offsets_folded if enabled else 0
        assert bool(folded) == bool(enabled and should_fold)
