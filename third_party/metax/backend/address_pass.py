import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def transform(source, base_zero=False, pipeline=False, shape=(), group_rows=1, split_partial_i32=1):
    # The available analysis SDK is LLVM 22; MACA's code generator is LLVM 19.
    # This adapter transports its vendor calling convention losslessly. Only
    # the C++ def-use transformation operates on instructions and addresses.
    if re.search(r"\bcc 1023\b", source):
        raise ValueError("Calling convention 1023 is reserved by this adapter")
    executable = Path(__file__).with_name("shared-load-address")
    if not executable.is_file():
        raise RuntimeError("Build the optional MetaX experimental MM target before enabling this optimization")
    normalized, kernels = re.subn(r"\bmetaxgpu_kernel\b", "cc 1023", source)
    if not kernels:
        raise ValueError("Expected at least one MACA kernel")
    with tempfile.TemporaryDirectory(prefix="metax-address-pass-") as temporary:
        input_path = Path(temporary) / "input.ll"
        output_path = Path(temporary) / "output.ll"
        input_path.write_text(normalized)
        command = [str(executable), str(input_path), "-o", str(output_path)]
        if pipeline:
            command.append("--matmul-pipeline")
        if group_rows != 1:
            if not shape or type(group_rows) is not int or not 1 <= group_rows <= 64:
                raise ValueError("CTA grouping requires shape and a group size in [1,64]")
            command.append("--group-rows=" + str(group_rows))
        if shape:
            if not pipeline or len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError("Layout requires a positive M,N,K specialization and pipeline")
            command.append("--layout-shape=" + ",".join(map(str, shape)))
        if split_partial_i32 != 1:
            if not shape or not pipeline or type(split_partial_i32) is not int or not 2 <= split_partial_i32 <= 64:
                raise ValueError("Split partial output requires shape, pipeline, and 2..64 splits")
            command.append("--split-partial-i32=" + str(split_partial_i32))
        if base_zero:
            command.append("--dynamic-shared-base-zero")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if pipeline and len(re.findall(r"^matched_mm=", result.stderr, re.MULTILINE)) != kernels:
            raise ValueError(result.stderr)
        output, restored = re.subn(r"\bcc1023\b|\bcc 1023\b", "metaxgpu_kernel", output_path.read_text())
        if restored != kernels:
            raise ValueError("Calling convention roundtrip count changed")
        output = output.replace("captures(none)", "nocapture")
        if "captures(" in output:
            raise ValueError("Unsupported capture attribute for MACA LLVM 19")
        count = int(re.search(r"shared_load_uses_rewritten=(\d+)", result.stderr)[1])
        return output, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dynamic-shared-base-zero", action="store_true")
    args = parser.parse_args()
    output, count = transform(args.input.read_text(), args.dynamic_shared_base_zero)
    args.output.write_text(output)
    print(f"shared_load_uses_rewritten={count}")
