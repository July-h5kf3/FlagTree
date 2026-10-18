# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List

from .. import _core as gl
from .._core import _unwrap_if_constexpr, builtin
from .._layouts import DistributedLayout, DotOperandLayout

__all__ = ["MacaMmaLayout", "mma"]


@dataclass(frozen=True)
class MacaMmaLayout(DistributedLayout):
    """Register layout of a row-major MetaX MMA accumulator on 64-lane warps.

    ``elements_per_thread`` is the MMA encoding's [M, N, K] fragment shape;
    [1, 1, 4] supports C550 INT8 and FP16/BF16 instructions. ``version`` is
    [major, minor], with [2, 0] selecting the C550 instruction encoding.
    Multi-CTA, transposed-load and column-major encodings are not supported.
    """

    warps_per_cta: List[int]
    elements_per_thread: List[int] = field(default_factory=lambda: [1, 1, 4])
    version: List[int] = field(default_factory=lambda: [2, 0])

    def __post_init__(self):
        warps = [_unwrap_if_constexpr(value) for value in _unwrap_if_constexpr(self.warps_per_cta)]
        elements = [_unwrap_if_constexpr(value) for value in _unwrap_if_constexpr(self.elements_per_thread)]
        if len(warps) != 2 or len(elements) != 3:
            raise ValueError("MetaX MMA requires two warp dimensions and three fragment dimensions")
        if any(value <= 0 or value & (value - 1) for value in warps + elements):
            raise ValueError("MetaX MMA dimensions must be positive powers of two")
        object.__setattr__(self, "warps_per_cta", warps)
        object.__setattr__(self, "elements_per_thread", elements)
        version = list(_unwrap_if_constexpr(self.version))
        if len(version) != 2 or version[0] != 2 or version[1] < 0:
            raise ValueError("MetaX MMA version must be [2, nonnegative minor]")
        object.__setattr__(self, "version", version)

    def _to_ir(self, builder):
        return builder.get_maca_mma_layout(self.warps_per_cta, self.elements_per_thread, self.version)

    def mangle(self):
        return f"MACAMMA_{self.version}_{self.warps_per_cta}_{self.elements_per_thread}_MACAMMA"

    def __hash__(self):
        return hash((tuple(self.version), tuple(self.warps_per_cta), tuple(self.elements_per_thread)))

    @property
    def rank(self):
        return 2


@builtin
def mma(a, b, acc, _semantic=None):
    """Compute a @ b + acc using explicitly distributed MetaX MMA fragments.

    Operands must use DotOperandLayout(0/1, acc.layout, 0). Signed INT8 inputs
    require an INT32 accumulator; FP16/BF16 inputs require FP32 accumulation.
    """
    if not isinstance(acc.type.layout, MacaMmaLayout):
        raise ValueError("acc must use MacaMmaLayout")
    if a.dtype != b.dtype or a.dtype not in (gl.int8, gl.float16, gl.bfloat16):
        raise TypeError("MetaX MMA inputs must have matching INT8, FP16 or BF16 dtypes")
    expected_dtype = gl.int32 if a.dtype == gl.int8 else gl.float32
    if acc.dtype != expected_dtype:
        raise TypeError(f"MetaX MMA requires {expected_dtype} accumulation for {a.dtype}")
    for index, operand in enumerate((a, b)):
        if operand.type.layout != DotOperandLayout(index, acc.type.layout, 0):
            raise ValueError("operands must use the accumulator's MetaX dot operand layouts")
    reduction_fragment = 4 * acc.type.layout.elements_per_thread[2]
    if a.shape[1] % reduction_fragment:
        raise ValueError("K must be a multiple of the MetaX MMA reduction fragment")
    result = _semantic.dot(a, b, acc, input_precision="ieee", max_num_imprecise_acc=None, out_dtype=acc.dtype)
    return gl.tensor(result.handle, acc.type)
