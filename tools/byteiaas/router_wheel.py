# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and verify the fixed encoder-only Prefill router."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPOSITORY = "https://github.com/wangyicong52/router.git"
COMMIT = "ef339bc73a04872dc7999aebf761c5f86263af64"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if source.exists() or output.exists():
        raise ValueError("Use new router source and wheel directories")
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    subprocess.run(["git", "init", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "fetch", "--depth", "1", REPOSITORY, COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True
    )
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        raise ValueError("Unexpected router source commit")
    lock_hash = digest((source / "Cargo.lock").read_bytes())
    unrepaired = output / "unrepaired"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(unrepaired),
        ],
        cwd=source,
        check=True,
    )
    if digest((source / "Cargo.lock").read_bytes()) != lock_hash:
        raise ValueError("Router build changed Cargo.lock")
    wheels = list(unrepaired.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one unrepaired router wheel")
    subprocess.run(
        [
            "auditwheel",
            "repair",
            "--plat",
            f"manylinux_2_28_{platform.machine()}",
            "--wheel-dir",
            str(output),
            str(wheels[0]),
        ],
        check=True,
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1 or "manylinux_2_28" not in wheels[0].name:
        raise ValueError("Expected one repaired manylinux_2_28 router wheel")
    wheel = wheels[0]
    manifest = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "cargo_lock_sha256": lock_hash,
        "wheel": wheel.name,
        "sha256": digest(wheel.read_bytes()),
    }
    (output / "router-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "wheels.sha256").write_text(f"{manifest['sha256']}  {wheel.name}\n")
    shutil.rmtree(unrepaired)
    print(json.dumps(manifest))


def verify(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    if manifest["repository"] != REPOSITORY or manifest["commit"] != COMMIT:
        raise ValueError("Unexpected router source provenance")
    wheel = args.manifest.parent / manifest["wheel"]
    if digest(wheel.read_bytes()) != manifest["sha256"]:
        raise ValueError("Router wheel checksum mismatch")
    distribution = importlib.metadata.distribution("vllm-router")
    verified = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            if ".data/" in name:
                raise ValueError(f"Unsupported router wheel relocation: {name}")
            installed = Path(distribution.locate_file(name))
            if digest(installed.read_bytes()) != digest(archive.read(name)):
                raise ValueError(f"Installed router differs from wheel: {name}")
            verified += 1
    from vllm_router_rs import Router

    command = shutil.which("vllm-router")
    if command is None:
        raise ValueError("vllm-router command is missing")
    help_text = subprocess.check_output([command, "--help"], text=True)
    if "--encoder-only-prefill" not in help_text:
        raise ValueError("Router lacks encoder-only Prefill support")
    report = {
        **manifest,
        "version": distribution.version,
        "installed_files_verified": verified,
        "native_router": str(Router),
        "encoder_only_prefill_flag": "PASS",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.set_defaults(func=build)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--manifest", type=Path, required=True)
    verifier.add_argument("--report", type=Path, required=True)
    verifier.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
