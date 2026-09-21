import importlib.util
import os
import re
from pathlib import Path

import pytest

triton = pytest.importorskip("triton")
pytest.importorskip("triton.backends.metax")
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from mm_source import mm_masked

BACKEND = Path(__file__).resolve().parents[2] / "backend"
SPEC = importlib.util.spec_from_file_location("metax_mm_adapter", BACKEND / "address_pass.py")
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)
pytestmark = pytest.mark.skipif(not (BACKEND / "shared-load-address").is_file(), reason="Build metax-mm-opt first")


@pytest.fixture(scope="module")
def source_cache(tmp_path_factory):
    os.environ["TRITON_CACHE_DIR"] = str(tmp_path_factory.mktemp("metax-mm-raw"))
    os.environ["TRITON_EXT_LIBDEVICE_PATH"] = str(
        Path(triton.__file__).parent / "backends/metax/lib/ext_maca_mathlib.bc")
    cache = {}

    def compile_source(shape):
        if shape not in cache:
            m, n, k = shape
            source = ASTSource(mm_masked, {"A": "*i8", "B": "*i8", "SA": "*fp32", "SB": "*fp32", "C": "*bf16"},
                               {"M": m, "N": n, "K": k, "BM": 128, "BN": 128, "BK": 256},
                               {(i, ): [["tt.divisibility", 16]]
                                for i in range(5)})
            kernel = triton.compile(
                source, target=GPUTarget("maca", 80, 64), options={
                    "num_warps": 4, "num_stages": 1, "pipeline": "cpasync", "pipeline_load_num": 2, "enable_fp_fusion":
                    False
                })
            cache[shape] = kernel.asm["llir"]
        return cache[shape]

    return compile_source


@pytest.mark.parametrize("rematerialize", [False, True])
@pytest.mark.parametrize("shape,group,splits", [
    ((256, 1024, 4096), 1, 1),
    ((97, 1024, 4096), 1, 1),
    ((98, 1024, 4096), 1, 4),
    ((255, 1024, 4096), 1, 1),
    ((384, 1024, 2048), 2, 1),
    ((8192, 152064, 3584), 32, 1),
    ((1848, 1536, 128256), 1, 1),
    ((1848, 1536, 151936), 1, 2),
    ((1848, 1536, 152064), 1, 2),
    ((98, 512, 3584), 1, 7),
])
def test_compile_structure(source_cache, shape, group, splits, rematerialize):
    output, _ = ADAPTER.transform(source_cache(shape), base_zero=True, pipeline=True, shape=shape, group_rows=group,
                                  split_partial_i32=splits, rematerialize_indices=rematerialize)
    assert "mm.steady" in output and "mm.tail" in output
    if splits > 1:
        assert output.count("call void @llvm.mxc.stg.predicator.v8i16") == 16
        assert "call i16 @llvm.mxc.cvt.f32tobf16" not in output
    else:
        assert output.count("call void @llvm.mxc.stg.predicator.i64") == 16


def test_tail_mask_mismatch(source_cache):
    shape = (1848, 1536, 151936)
    source = source_cache(shape)
    source, replacements = re.subn(r"(icmp slt i32 %\w+, )151936", r"\g<1>151920", source)
    assert replacements > 0
    with pytest.raises(ValueError, match="input packet row/reduction mask"):
        ADAPTER.transform(source, pipeline=True, shape=shape, split_partial_i32=2)


def test_wrong_shape(source_cache):
    with pytest.raises(ValueError, match="skipped_layout"):
        ADAPTER.transform(source_cache((98, 1024, 4096)), pipeline=True, shape=(102, 1024, 4096))


def test_unsafe_accumulation(source_cache):
    shape = (1848, 1536, 151936)
    with pytest.raises(ValueError, match="constant trip count"):
        ADAPTER.transform(source_cache(shape), pipeline=True, shape=shape)


def test_rematerialize_requires_layout():
    with pytest.raises(ValueError, match="requires pipeline and shape"):
        ADAPTER.transform("", pipeline=True, rematerialize_indices=True)
