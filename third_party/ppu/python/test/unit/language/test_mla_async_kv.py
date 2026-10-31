import pytest

from triton._C.libtriton import ir, ppu

MODULE = """
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [16, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 16 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @copy(%base: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<256x512xi8, #blocked> {
    %rows = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %columns = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %column = tt.expand_dims %columns {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xi32, #blocked>
    %stride = arith.constant dense<512> : tensor<256x1xi32, #blocked>
    %offset = arith.muli %row, %stride : tensor<256x1xi32, #blocked>
    %row_full = tt.broadcast %offset : tensor<256x1xi32, #blocked> -> tensor<256x512xi32, #blocked>
    %column_full = tt.broadcast %column : tensor<1x512xi32, #blocked> -> tensor<256x512xi32, #blocked>
    %offsets = arith.addi %row_full, %column_full : tensor<256x512xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<i8> -> tensor<256x512x!tt.ptr<i8>, #blocked>
    %pointers = tt.addptr %bases, %offsets : tensor<256x512x!tt.ptr<i8>, #blocked>, tensor<256x512xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<256x512xi1, #blocked>
    %zero = arith.constant dense<0> : tensor<256x512xi8, #blocked>
    %kv = tt.load %pointers, %mask, %zero : tensor<256x512x!tt.ptr<i8>, #blocked>
    %root = ttg.local_alloc %kv : (tensor<256x512xi8, #blocked>) -> !ttg.memdesc<256x512xi8, #shared, #smem>
    %value = ttg.local_load %root : !ttg.memdesc<256x512xi8, #shared, #smem> -> tensor<256x512xi8, #blocked>
    tt.return %value : tensor<256x512xi8, #blocked>
  }
}
"""


@pytest.mark.parametrize('case', ['aligned', 'unaligned', 'nonzero_padding', 'escaped_initializer', 'different_phase'])
def test_mla_async_kv(case, tmp_path):
    text = MODULE
    if case == 'unaligned':
        text = text.replace('tt.divisibility = 16', 'tt.divisibility = 1')
    elif case == 'nonzero_padding':
        text = text.replace('dense<0> : tensor<256x512xi8', 'dense<1> : tensor<256x512xi8')
    elif case == 'escaped_initializer':
        allocation = next(line for line in text.splitlines() if '%root = ttg.local_alloc' in line)
        tensor = 'tensor<256x512xi8, #blocked>'
        text = text.replace(allocation,
                            allocation + '\n    %extra = ttg.convert_layout %kv : ' + tensor + ' -> ' + tensor)
    elif case == 'different_phase':
        text = text.replace('maxPhase = 16', 'maxPhase = 8')
    else:
        assert case == 'aligned'
    context = ir.context()
    ir.load_dialects(context)
    ppu.load_dialects(context)
    path = tmp_path / 'module.ttgir'
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    before = str(module)
    changed = ppu.promote_mla_kv_copy(module)
    assert changed == int(case == 'aligned')
    if changed:
        rewritten = str(module)
        assert rewritten.count('ttg.async_copy_global_to_local') == 1
        assert rewritten.count('ttg.async_commit_group') == 1
        assert rewritten.count('ttg.async_wait') == 1
        assert ppu.promote_mla_kv_copy(module) == 0 and str(module) == rewritten
    else:
        assert str(module) == before
