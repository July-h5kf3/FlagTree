from pathlib import Path

import pytest
import torch

import triton
from triton._internal_testing import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires a PPU")


@pytest.mark.parametrize("offset", [0, 64])
def test_int8_transposed_rhs_subview(offset, tmp_path, device):
    source = Path(__file__).with_name("ppu_int8_subview.ttgir").read_text()
    path = tmp_path / "subview.ttgir"
    path.write_text(source.replace("@OFFSET@", str(offset)))
    generator = torch.Generator().manual_seed(17)
    a_cpu = torch.randint(-128, 128, (16, 32), dtype=torch.int8, generator=generator)
    b_cpu = torch.randint(-128, 128, (32, 128), dtype=torch.int8, generator=generator)
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    output = torch.empty((16, 64), dtype=torch.int32, device=device)
    kernel = triton.compile(str(path))
    kernel[(1, 1, 1)](a, b, output)
    expected = a_cpu.int() @ b_cpu[:, offset:offset + 64].int()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
