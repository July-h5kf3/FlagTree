import pytest

from triton._C.libtriton import gluon_ir, ir, passes
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import make_backend

CASES = [("i8", 32, 64, 2), ("i8", 64, 128, 4), ("i8", 128, 256, 8), ("bf16", 32, 32, 2), ("bf16", 64, 64, 4),
         ("f16", 128, 128, 8)]


def conversion_module(tmp_path, dtype, rows, width, warps, operand, mode):
    context = ir.context()
    ir.load_dialects(context)
    make_backend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    vector = 8 if dtype == "i8" else 4
    warp_layout = [warps, 1] if operand == 1 else [1, warps]
    mma = (f"#ttg.amd_mfma<{{version = 3, warpsPerCTA = {warp_layout}, "
           f"instrShape = [16, 16, {vector * 4}], isTransposed = false, mmacLayout = 1}}>")
    dot = f"#ttg.dot_op<{{opIdx = {operand}, parent = {mma}, kWidth = {vector}}}>"
    shape = [rows, width] if operand == 1 else [width, rows]
    fragment_type = f"tensor<{shape[1]}x{shape[0]}x{dtype}, {dot}>"
    attrs = (f'"ttg.num-warps" = {warps} : i32, "ttg.num-ctas" = 1 : i32, '
             '"ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"')
    filename = tmp_path / "convert.ttgir"
    filename.write_text(f"module attributes {{{attrs}}} {{\n"
                        f"tt.func @fragment(%x: {fragment_type}) -> {fragment_type} {{\n"
                        f"tt.return %x : {fragment_type}\n}}\n}}")
    seed = ir.parse_mlir_module(str(filename), context)
    returns = []
    seed.walk(lambda op: returns.append(op) if op.get_name() == "tt.return" else None)
    builder = gluon_ir.GluonOpBuilder(context)
    builder.set_insertion_point_to_start(returns[0].get_block())
    fragment = returns[0].get_operand(0)
    linear_type = str(builder.create_trans(fragment, [1, 0]).get_type())
    spt = [1, vector * 2] if operand == 1 else [vector * 2, 1]
    order = [1, 0] if operand == 1 else [0, 1]
    if mode == "non_k_contiguous":
        order = order[::-1]
        spt = spt[::-1]
    blocked = (f"#ttg.blocked<{{sizePerThread = {spt}, threadsPerWarp = [8, 8], "
               f"warpsPerCTA = {warp_layout}, order = {order}}}>")
    source_type = f"tensor<{shape[0]}x{shape[1]}x{dtype}, {blocked}>"
    linear_encoding = linear_type[linear_type.index("#"):-1]
    explicit = f' {{"tle.explicit_encoding.0" = {linear_encoding}}}' if mode == "explicit" else ""
    explicit_transpose = f', "tle.explicit_encoding.0" = {dot}' if mode == "explicit_transpose" else ""
    result_types = f"({fragment_type}, {linear_type})" if mode == "multiple_users" else fragment_type
    returned = f"%result, %converted : {fragment_type}, {linear_type}" if mode == "multiple_users" else f"%result : {fragment_type}"
    filename.write_text(
        f"module attributes {{{attrs}}} {{\n"
        f"tt.func @convert(%input: {source_type}) -> {result_types} {{\n"
        f"%converted = ttg.convert_layout %input{explicit} : {source_type} -> {linear_type}\n"
        f"%result = tt.trans %converted {{order = array<i32: 1, 0>{explicit_transpose}}} : {linear_type} -> {fragment_type}\n"
        f"tt.return {returned}\n}}\n}}")
    mod = ir.parse_mlir_module(str(filename), context)
    mod.context = context
    return mod, vector


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("dtype,rows,width,warps", CASES)
def test_transposed_dot_shared_layout(tmp_path, dtype, rows, width, warps, operand):
    mod, vector = conversion_module(tmp_path, dtype, rows, width, warps, operand, "normal")
    pm = ir.pass_manager(mod.context)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    pm.run(mod, "transposed_dot")
    rewritten = str(mod)
    assert "ttg.convert_layout" not in rewritten
    assert "ttg.local_alloc" in rewritten and "ttg.local_load" in rewritten
    assert "tt.trans" in rewritten
    bitwidth = 8 if dtype == "i8" else 16
    per_phase = max(1, 1024 // bitwidth // width)
    max_phase = max(1, min(16 // per_phase, width // vector))
    assert f"vec = {vector}, perPhase = {per_phase}, maxPhase = {max_phase}" in rewritten
    pm = ir.pass_manager(mod.context)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    pm.run(mod, "transposed_dot_again")
    assert str(mod) == rewritten


@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("mode", ["multiple_users", "explicit", "explicit_transpose", "non_k_contiguous"])
def test_transposed_dot_preserves_layout_constraints(tmp_path, operand, mode):
    mod, _ = conversion_module(tmp_path, "i8", 64, 128, 4, operand, mode)
    before = str(mod)
    pm = ir.pass_manager(mod.context)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    pm.run(mod, "transposed_dot_constraints")
    assert str(mod) == before
