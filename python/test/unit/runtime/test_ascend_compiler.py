# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch  # noqa: F401 -- initialize torch before Ascend's backend imports

compiler = pytest.importorskip("triton.backends.ascend.compiler")


@pytest.mark.parametrize("on_95", [False, True])
@pytest.mark.parametrize(
    "label,task_type,expected",
    [("aiv", 10, "aiv"), ("aiv", 20, "aic"), ("aiv", 30, "mix"), ("aiv", 32, "mix"), ("aiv", 40, "mix"),
     ("aiv", 0, "aiv"), ("aic", 20, "aic"), ("mix", 30, "mix")],
)
def test_generated_task_type_controls_loader(label, task_type, expected, on_95):
    function = ("linalg_to_bin_enable_npu_compile_910_95" if on_95 else "linalg_to_bin_enable_npu_compile_A2_A3")
    metadata = dict(mix_mode=label, bs_task_type=task_type)
    with patch.object(compiler, function, return_value=b"unchanged-binary"):
        binary = compiler._compile_linalg_to_npu_bin("", metadata, SimpleNamespace(compile_on_910_95=on_95))
    assert (binary, metadata["mix_mode"]) == (b"unchanged-binary", expected)
