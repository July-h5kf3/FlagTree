from dataclasses import replace

import pytest
import triton
from triton._C.libtriton import gluon_ir, ir
from triton.compiler.compiler import make_backend


@pytest.mark.parametrize("operand", ["content", "rope"])
@pytest.mark.parametrize("mode", ["enabled", "disabled", "other_arch", "other_warps", "other_ctas", "other_dtype"])
def test_gfx936_mmac_shared_conversion(tmp_path, mode, operand):
    target = triton.runtime.driver.active.get_current_target()
    if target.arch != "gfx936":
        pytest.skip("Requires gfx936 MMAC layouts")
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 1})
    assert options.enable_mmac_i8_shared is False
    enabled = replace(options, enable_mmac_i8_shared=True)
    assert enabled.hash() != options.hash()
    variants = {
        "enabled": enabled,
        "disabled": options,
        "other_arch": replace(enabled, arch="gfx942"),
        "other_warps": replace(enabled, num_warps=8),
        "other_ctas": replace(enabled, num_ctas=2),
        "other_dtype": enabled,
    }
    dtype = "i8" if operand == "content" else "bf16"
    if mode == "other_dtype":
        dtype = "bf16" if operand == "content" else "f16"
    text = """
#src = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dst = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32], [0, 64], [16, 0], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx936", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @convert(%input: tensor<64x128xDTYPE, #src>) -> tensor<64x128xDTYPE, #dst> {
    %output = ttg.convert_layout %input : tensor<64x128xDTYPE, #src> -> tensor<64x128xDTYPE, #dst>
    tt.return %output : tensor<64x128xDTYPE, #dst>
  }
}
""".replace("DTYPE", dtype)
    if operand == "rope":
        text = text.replace("64x128", "64x64").replace("[1, 16]", "[1, 8]")
        text = text.replace("[[0, 1], [0, 2], [0, 4], [0, 32], [0, 64], [16, 0], [32, 0]]",
                            "[[0, 1], [0, 2], [0, 16], [0, 32], [16, 0], [32, 0]]")
        text = text.replace("[0, 8], [0, 16]", "[0, 4], [0, 8]")
    filename = tmp_path / "convert.ttgir"
    filename.write_text(text)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    mod = ir.parse_mlir_module(str(filename), context)
    mod.context = context
    before = str(mod)
    count = backend.swizzle_mmac_inputs(mod, variants[mode])
    returned = []
    mod.walk(lambda op: returned.append(op.get_operand(0)) if op.get_name() == "tt.return" else None)
    assert len(returned) == 1
    layout = gluon_ir.GluonOpBuilder(context).get_gluon_layout_from_tensor(returned[0])
    assert layout.shape == [64, 128 if operand == "content" else 64]
    if mode != "enabled":
        assert count == 0 and str(mod) == before
        return
    assert count == 1
    rewritten = str(mod)
    assert "ttg.local_alloc" in rewritten and "ttg.local_load" in rewritten
    assert "ttg.convert_layout" not in rewritten
    vector = 8 if operand == "content" else 4
    assert f"vec = {vector}, perPhase = 1, maxPhase = 16" in rewritten
    assert backend.swizzle_mmac_inputs(mod, enabled) == 0
    assert str(mod) == rewritten
