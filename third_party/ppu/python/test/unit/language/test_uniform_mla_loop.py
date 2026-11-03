import pytest
from triton._C.libtriton import ir, ppu


@pytest.mark.parametrize('case', [
    'arg', 'program', 'load', 'addptr', 'reduce', 'opaque_end', 'opaque_start', 'volatile', 'private', 'block_argument',
    'step', 'nested', 'none'
])
def test_uniform(case, tmp_path):
    source = '  %end = arith.addi %arg, %c4 : i32'
    start = '%c0'
    inner = ''
    if case == 'program': source = '  %end = tt.get_program_id x : i32'
    elif case == 'volatile': source = '  %end = tt.load %ptr {isVolatile = true} : !tt.ptr<i32>'
    elif case == 'load': source = '  %end = tt.load %ptr : !tt.ptr<i32>'
    elif case == 'addptr':
        source = '  %pointer = tt.addptr %ptr, %arg : !tt.ptr<i32>, i32\n  %end = tt.load %pointer : !tt.ptr<i32>'
    elif case == 'reduce':
        source = '''  %end = "tt.reduce"(%values) ({
  ^bb0(%a: i32, %b: i32):
    %sum = arith.addi %a, %b : i32
    tt.reduce.return %sum : i32
  }) {axis = 0 : i32} : (tensor<128xi32,#blocked>) -> i32'''
    elif case in ('opaque_end', 'opaque_start'):
        source = '  %end = tt.elementwise_inline_asm "mov.u32 $0, %tid.x;" {constraints = "=r", packed_element = 1 : i32, pure = true} -> i32'
        if case == 'opaque_start':
            start = '%end'
            source += '\n  %upper = arith.constant 16 : i32'
    elif case == 'nested':
        inner = '    scf.for %j = %c0 to %arg step %c4 : i32 { }'
    upper = '%upper' if case == 'opaque_start' else '%end'
    text = f'''#blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}}>
module attributes {{"ttg.num-warps" = 16 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32}} {{
 tt.func @uniform(%arg: i32, %ptr: !tt.ptr<i32>, %values: tensor<128xi32,#blocked>) {{
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant {2 if case=='step' else 4} : i32
{source}
  scf.for %i = {start} to {upper} step %c4 : i32 {{
{inner}
  }}
  tt.return
 }}
}}'''
    if case == 'private':
        text = text.replace('tt.func @uniform', 'tt.func private @uniform')
    if case == 'block_argument':
        text = text.replace(
            source,
            '''  %opaque = tt.elementwise_inline_asm "mov.u32 $0, %tid.x;" {constraints = "=r", packed_element = 1 : i32, pure = true} -> i32
  cf.br ^next(%opaque : i32)
 ^next(%end: i32):''')
    if case == 'none': text = text.replace(f'  scf.for %i = {start} to {upper} step %c4 : i32 {{\n{inner}\n  }}', '')
    context = ir.context()
    ir.load_dialects(context)
    ppu.load_dialects(context)
    path = tmp_path / 'loop.ttgir'
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    before = str(module)
    assert ppu.prove_uniform_mla_loop(module) == int(case in ('arg', 'program', 'load', 'addptr', 'reduce'))
    assert str(module) == before
