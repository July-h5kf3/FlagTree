"""Keep MACA dot operand permutations in registers when lane/warp are unchanged."""

import pytest

from triton._C.libtriton import ir, passes
from triton.backends.metax.compiler import metax

MODULE = """
#mma = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], elementsMNK = [1, 1, 4], colMajor = 0, isATrans = false, isBTrans = false, elementsStride = [1, 1]}>
#src = #ttg.linear<{register = [[0, 1], [0, 2], [64, 0], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
#trans = #ttg.linear<{register = [[1, 0], [2, 0], [0, 64], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]], warp = [[0, 16], [0, 32]], block = []}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 64 : i32, use.opt.maca.mma = 1 : i32} {
  tt.func public @probability_permutation(%p: tensor<128x64xf16, #src>, %v: tensor<128x64xf16, #a>, %c: tensor<128x128xf32, #mma>) -> tensor<128x128xf32, #mma> {
    %t = tt.trans %p {order = array<i32: 1, 0>} : tensor<128x64xf16, #src> -> tensor<64x128xf16, #trans>
    %b = ttg.convert_layout %t : tensor<64x128xf16, #trans> -> tensor<64x128xf16, #b>
    %d = tt.dot %v, %b, %c : tensor<128x64xf16, #a> * tensor<64x128xf16, #b> -> tensor<128x128xf32, #mma>
    tt.return %d : tensor<128x128xf32, #mma>
  }
}
"""


@pytest.mark.parametrize("cross_lane", [False, True])
def test_maca_register_only_dot_operand(cross_lane, tmp_path):
    source = MODULE
    if cross_lane:
        # Exchange K bit 16 in registers with K bit 4 in lanes. This requires
        # communication, so the existing shared rewrite must remain enabled.
        source = source.replace("[64, 0], [0, 16], [0, 32]]", "[64, 0], [0, 4], [0, 32]]")
        source = source.replace("[0, 4], [0, 8]], warp", "[0, 16], [0, 8]], warp")
        source = source.replace("[0, 64], [16, 0], [32, 0]]", "[0, 64], [4, 0], [32, 0]]")
        source = source.replace("[4, 0], [8, 0]], warp", "[16, 0], [8, 0]], warp")
    path = tmp_path / "probability.mlir"
    path.write_text(source)
    context = ir.context()
    ir.load_dialects(context)
    metax.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    passes.ttgpuir.add_optimize_dot_operands(pm, True)
    pm.run(module, "register-only-dot-operand")
    result = str(module)
    assert result.count("ttg.local_alloc") == int(cross_lane)
    assert result.count("ttg.local_load") == int(cross_lane)
    assert "tt.dot" in result
    if not cross_lane:
        assert "ttg.convert_layout" in result
