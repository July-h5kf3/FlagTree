import pytest

from triton._C.libtriton import ir, ppu

MODULE = """
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [16, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 16], [8, 0], [0, 32], [0, 64], [0, 128], [0, 256], [128, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], warp = [[16, 0], [32, 0], [64, 0], [0, 0]], block = []}>
#mma = #ttg.ppu_mma<{versionMajor = 1, versionMinor = 0, warpsPerCTA = [2, 8], instrShape = [16, 16], vecSize = 1}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 16 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reuse(%kv: tensor<256x512xi8, #blocked>) -> (tensor<512x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<256x512xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) {
    %root = ttg.local_alloc %kv : (tensor<256x512xi8, #blocked>) -> !ttg.memdesc<256x512xi8, #shared, #smem>
    %converted = ttg.convert_layout %kv : tensor<256x512xi8, #blocked> -> tensor<256x512xi8, #linear>
    %key = tt.trans %converted {order = array<i32: 1, 0>} : tensor<256x512xi8, #linear> -> tensor<512x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %value = ttg.local_load %root : !ttg.memdesc<256x512xi8, #shared, #smem> -> tensor<256x512xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    tt.return %key, %value : tensor<512x256xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<256x512xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  }
}
"""


@pytest.mark.parametrize('case', [
    'reuse', 'duplicate_root', 'extra_view', 'extra_convert_user', 'different_layout', 'explicit_deallocation',
    'insufficient_alignment'
])
def test_mla_kv_views(case, tmp_path):
    text = MODULE
    allocation = next(line for line in text.splitlines() if '%root = ttg.local_alloc' in line)
    if case == 'duplicate_root':
        text = text.replace(allocation, allocation + '\n' + allocation.replace('%root =', '%extra_root ='))
    elif case == 'extra_view':
        descriptor = '!ttg.memdesc<256x512xi8, #shared, #smem>'
        text = text.replace(
            allocation, allocation + '\n    %view = ttg.memdesc_trans %root ' + '{order = array<i32: 0, 1>} : ' +
            descriptor + ' -> ' + descriptor)
    elif case == 'extra_convert_user':
        convert = next(line for line in text.splitlines() if '%converted = ttg.convert_layout' in line)
        tensor = 'tensor<256x512xi8, #linear>'
        text = text.replace(convert,
                            convert + '\n    %extra = ttg.convert_layout %converted : ' + tensor + ' -> ' + tensor)
    elif case == 'different_layout':
        text = text.replace('vec = 8, perPhase = 1', 'vec = 16, perPhase = 1')
    elif case == 'explicit_deallocation':
        value = next(line for line in text.splitlines() if '%value = ttg.local_load' in line)
        text = text.replace(value + '\n', '')
        text = text.replace(
            allocation,
            allocation + '\n' + value + '\n    ttg.local_dealloc %root : !ttg.memdesc<256x512xi8, #shared, #smem>')
    elif case == 'insufficient_alignment':
        text = text.replace('ttg.local_alloc %kv :', 'ttg.local_alloc %kv {alignment = 8 : i32} :')
    else:
        assert case == 'reuse'
    context = ir.context()
    ir.load_dialects(context)
    ppu.load_dialects(context)
    path = tmp_path / 'module.ttgir'
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    before = str(module)
    assert ppu.reuse_mla_kv_views(module) == int(case == 'reuse')
    if case == 'reuse':
        assert str(module).count('ttg.local_alloc') == 1
        assert str(module).count('ttg.memdesc_trans') == 1
        assert 'vec = 16, perPhase = 1, maxPhase = 16' in str(module)
        rewritten = str(module)
        assert ppu.reuse_mla_kv_views(module) == 0 and str(module) == rewritten
    else:
        assert str(module) == before


@pytest.mark.parametrize('geometry', ['h64', 'h32'])
@pytest.mark.parametrize('case', ['reuse', 'marker_off', 'duplicate_root', 'insufficient_alignment', 'mutable'])
def test_mla_k128_kv_views(case, geometry, tmp_path):
    text = MODULE.replace('256x512', '128x512').replace('512x256', '512x128')
    text = text.replace('[128, 0]', '[64, 0]').replace('warp = [[16, 0], [32, 0], [64, 0]',
                                                       'warp = [[16, 0], [32, 0], [0, 0]')
    text = text.replace('warpsPerCTA = [2, 8]', 'warpsPerCTA = [4, 4]')
    if geometry == 'h32':
        text = text.replace('warpsPerCTA = [16, 1]',
                            'warpsPerCTA = [8, 1]').replace('"ttg.num-warps" = 16', '"ttg.num-warps" = 8')
        text = text.replace('warpsPerCTA = [4, 4]', 'warpsPerCTA = [2, 4]')
        text = text.replace('warp = [[16, 0], [32, 0], [0, 0], [0, 0]]', 'warp = [[16, 0], [32, 0], [0, 0]]')
    if case != 'marker_off':
        text = text.replace('module attributes {', f'module attributes {{ppu.mla_{geometry}_k128_pv_layout = 1 : i32, ')
    allocation = next(line for line in text.splitlines() if '%root = ttg.local_alloc' in line)
    if case == 'duplicate_root':
        text = text.replace(allocation, allocation + '\n' + allocation.replace('%root =', '%extra_root ='))
    elif case == 'insufficient_alignment':
        text = text.replace('ttg.local_alloc %kv :', 'ttg.local_alloc %kv {alignment = 8 : i32} :')
    elif case == 'mutable':
        text = text.replace('#shared, #smem>', '#shared, #smem, mutable>')
    context = ir.context()
    ir.load_dialects(context)
    ppu.load_dialects(context)
    path = tmp_path / 'h64.ttgir'
    path.write_text(text)
    module = ir.parse_mlir_module(str(path), context)
    module.context = context
    before = str(module)
    assert ppu.reuse_mla_kv_views(module) == int(case == 'reuse')
    if case == 'reuse':
        assert 'vec = 16, perPhase = 1, maxPhase = 16' in str(module)
        after = str(module)
        assert ppu.reuse_mla_kv_views(module) == 0 and str(module) == after
    else:
        assert str(module) == before
