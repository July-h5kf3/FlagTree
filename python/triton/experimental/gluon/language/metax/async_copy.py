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
"""MetaX global-to-shared copies using standard Triton GPU operations.

The blocked source must map consecutive lane vectors to consecutive shared
addresses within each 64-lane warp. Copies require at least four aligned bytes
per lane. Global pointers must be contiguous across each shared swizzle span,
and predicates must be invariant across it. Unsupported layouts and masks are
rejected during lowering.

Copies issue immediately. ``commit_group`` does not emit a hardware operation;
``wait_group`` currently waits for all GVM copies and synchronizes the CTA,
including when a nonzero number of outstanding groups was requested.
"""

# These frontends build generic Triton GPU IR; the MetaX backend supplies the
# native copy and synchronization instructions.
from ..nvidia.ampere.async_copy import async_copy_global_to_shared, commit_group, wait_group
from .._core import builtin

__all__ = [
    "async_copy_global_to_shared",
    "commit_group",
    "wait_group",
    "wait_all",
    "block_barrier",
    "is_supported",
]


@builtin
def wait_all(_semantic=None):
    """Complete prior GVM requests using native counter wait instructions.

    Matches MACA arrive_gvmcnt(0) followed by barrier_inst. Requires a native
    plugin with supports_maca_native_barriers. Follow with block_barrier before
    consuming shared memory, since predicated copies can also issue BSM zero-fill.
    All CTA threads must participate.
    """
    _semantic.builder.create_maca_wait_all()


@builtin
def block_barrier(_semantic=None):
    """Complete BSM operations, then synchronize CTA threads with native barriers.

    Uses barrier_shared followed by barrier_inst, without a global memory fence.
    This does not guarantee completion of global-to-shared copies. Use wait_all
    before consuming those copies. All CTA threads must participate.
    """
    _semantic.builder.create_maca_block_barrier()


def is_supported() -> bool:
    """Check the loaded native plugin, which may predate these Python APIs."""
    try:
        from triton._C.libtriton.gluon_ir import supports_maca_async_copy
    except ImportError:
        return False
    return bool(supports_maca_async_copy)
