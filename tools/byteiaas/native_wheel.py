# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and identify the locked ByteIAAS native wheel against release PyTorch."""

import argparse
import email
import hashlib
import importlib
import importlib.metadata
import json
import os
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
    env = dict(os.environ, DG_FORCE_BUILD="1", DG_USE_LOCAL_VERSION="1")
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(output),
        ],
        cwd=source,
        env=env,
        check=True,
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one native wheel, found {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
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
    importlib.import_module("deep_gemm")
    import torch

    schemas = {
        name: str(getattr(torch.ops.deep_gemm, name).default._schema)
        for name in (
            "fp8_fp4_mega_moe_sm90",
            "fp8_fp4_mega_moe_sm90_rs",
            "fp8_mega_moe",
            "_slice_symm_buffer_for_sm90_mega_moe",
            "get_symm_buffer_size_for_sm90_mega_moe",
        )
    }
    report = {**manifest, "installed_files_verified": verified, "schemas": schemas}
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
