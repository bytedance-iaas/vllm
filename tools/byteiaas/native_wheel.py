# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and identify the locked ByteIAAS native wheel against release PyTorch."""

import argparse
import dataclasses
import email
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def torch_contract() -> dict:
    import torch

    if torch.__version__.split("+")[0] != "2.13.0" or torch.version.cuda != "13.0":
        raise ValueError("ByteIAAS release native wheels require Torch2.13.0/CUDA13.0")
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
    }


def build(args: argparse.Namespace) -> None:
    lock = json.loads(args.lock.read_text())
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")):
        raise ValueError("Use an empty native wheel output directory")
    tree = subprocess.check_output(
        ["git", "-C", str(source), "write-tree"], text=True
    ).strip()
    if tree != lock["source_tree"]:
        raise ValueError(f"Native source differs from lock: {tree}")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code"], check=True)
    extras = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"],
        text=True,
    ).strip()
    if extras:
        raise ValueError(f"Unexpected native source build inputs: {extras}")
    contract = torch_contract()
    env = dict(os.environ)
    if args.kind == "deepgemm":
        env.update(DG_FORCE_BUILD="1", DG_USE_LOCAL_VERSION="1")
        command = [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output)]
    else:
        env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HUMMING_KERNELS"] = lock[
            "package_version"
        ]
        env["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
        subprocess.run(
            [sys.executable, "tools/build_native.py"],
            cwd=source,
            env=env,
            check=True,
        )
        command = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
        ]
    subprocess.run(
        command,
        cwd=source,
        env=env,
        check=True,
    )
    if args.kind == "humming":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "tags",
                "--remove",
                "--python-tag",
                "cp310",
                "--abi-tag",
                "abi3",
                "--platform-tag",
                f"manylinux_2_28_{platform.machine()}",
                *map(str, output.glob("*.whl")),
            ],
            check=True,
        )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one native wheel, found {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        if args.kind == "humming":
            native_root = f"humming/_native/{platform.machine()}/"
            expected = {
                native_root + name
                for name in (
                    "libhumming_launcher.so",
                    "libcubinpatch.so",
                    "_device_info.abi3.so",
                    "manifest.json",
                    "nvrtc_compile",
                )
            }
            if not expected.issubset(archive.namelist()):
                raise ValueError("Humming wheel is missing native build artifacts")
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Expected one native wheel METADATA")
        metadata = email.message_from_bytes(archive.read(metadata_files[0]))
        if metadata["Version"] != lock["package_version"]:
            raise ValueError(
                f"Unexpected native package version: {metadata['Version']}"
            )
    report = {
        "format_version": 1,
        "kind": args.kind,
        "repository": lock["repository"],
        "base_commit": lock["base_commit"],
        "source_commit": lock["source_commit"],
        "source_tree": tree,
        "package": metadata["Name"],
        "version": metadata["Version"],
        "wheel": wheel.name,
        "sha256": digest(wheel.read_bytes()),
        "lock_sha256": digest(args.lock.read_bytes()),
        "torch_contract": contract,
    }
    (output / "native-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "wheels.sha256").write_text(f"{report['sha256']}  {wheel.name}\n")
    print(json.dumps(report), flush=True)


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    lock = json.loads(args.lock.read_text())
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported native wheel manifest")
    if manifest["lock_sha256"] != digest(args.lock.read_bytes()):
        raise ValueError("Native wheel source lock checksum mismatch")
    if manifest["version"] != lock["package_version"]:
        raise ValueError("Native wheel version differs from source lock")
    for key in ("repository", "base_commit", "source_commit", "source_tree"):
        if manifest[key] != lock[key]:
            raise ValueError(f"Native wheel provenance mismatch: {key}")
    if manifest["torch_contract"] != torch_contract():
        raise ValueError("Native wheel PyTorch/ABI differs from runtime")
    wheel = args.manifest.parent / manifest["wheel"]
    if digest(wheel.read_bytes()) != manifest["sha256"]:
        raise ValueError("Native wheel checksum mismatch")
    distribution = importlib.metadata.distribution(manifest["package"])
    if distribution.version != lock["package_version"]:
        raise ValueError("Installed native version differs from source lock")
    verified = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            if ".data/" in name:
                raise ValueError(
                    f"Native verification needs a wheel relocation rule: {name}"
                )
            installed = Path(distribution.locate_file(name))
            if digest(installed.read_bytes()) != digest(archive.read(name)):
                raise ValueError(f"Installed native file differs from wheel: {name}")
            verified += 1
    if manifest["kind"] == "deepgemm":
        importlib.import_module("deep_gemm")
        import torch

        capabilities = {
            name: str(getattr(torch.ops.deep_gemm, name).default._schema)
            for name in (
                "fp8_fp4_mega_moe_sm90",
                "fp8_fp4_mega_moe_sm90_rs",
                "fp8_mega_moe",
                "_slice_symm_buffer_for_sm90_mega_moe",
                "get_symm_buffer_size_for_sm90_mega_moe",
            )
        }
    elif manifest["kind"] == "humming":
        os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"
        from humming.config import TuningConfig
        from humming.schema import HummingInputSchema, Mxfp4WeightSchema

        fields = {field.name: field for field in dataclasses.fields(TuningConfig)}
        if fields["stream_k_max_slices"].default != 10:
            raise ValueError("Humming ordered scheduler contract differs from lock")
        HummingInputSchema.from_config(
            {
                "quant_method": "humming",
                "dtype": "int8",
                "input_quant_mode": "dynamic_group",
                "input_scale_group_size": 128,
                "input_scale_dtype": "float32",
            }
        )
        Mxfp4WeightSchema()
        capabilities = {
            "ordered_scheduler_max_slices": 10,
            "int8_group128_fp32_descriptor": "PASS",
            "mx_fp4_descriptor": "PASS",
            "scope": "Package bytes and descriptors; GPU numerics checked separately.",
        }
    else:
        raise ValueError("Unsupported native dependency kind")
    report = {
        **manifest,
        "installed_files_verified": verified,
        "capabilities": capabilities,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--lock", type=Path, required=True)
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--kind", choices=["deepgemm", "humming"], default="deepgemm")
    builder.set_defaults(func=build)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--lock", type=Path, required=True)
    verifier.add_argument("--manifest", type=Path, required=True)
    verifier.add_argument("--report", type=Path)
    verifier.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
