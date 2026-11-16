# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from triton.backends.ascend import compiler


@pytest.mark.parametrize("task_type,frontend_mode,loader_mode", [(10, "aiv", "aiv"), (20, "aiv", "aic"),
                                                                 (31, "aiv", "mix"), (42, "aiv", "mix"),
                                                                 (20, "aic", "aic"), (31, "mix", "mix")])
def test_void_cube_loader_uses_cann_task_kind(monkeypatch, task_type, frontend_mode, loader_mode):

    def compile_binary(linalg, metadata, options):
        metadata["bs_task_type"] = task_type
        return b"kernel"

    monkeypatch.setattr(compiler, "_USE_CUSTOM_COMPILE_OPT", None)
    monkeypatch.setattr(compiler, "linalg_to_bin_enable_npu_compile_A2_A3", compile_binary)
    metadata = {"mix_mode": frontend_mode}
    compiler._compile_linalg_to_npu_bin("", metadata, SimpleNamespace(compile_on_910_95=False))
    assert metadata["mix_mode"] == loader_mode
