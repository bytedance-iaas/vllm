# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seal and validate same-run ByteIAAS wheels before image builds."""

import argparse
import email
import hashlib
import json
import zipfile
from pathlib import Path

import router_wheel
import tokenizers_wheel
from prepare_native_source import package_version

COMPONENTS = {
    "vllm": ("workspace/dist", "vllm-*.whl", "wheel.sha256"),
    "deepgemm": (
        "workspace/byteiaas-native-dist/deepgemm",
        "deep_gemm-*.whl",
        "wheels.sha256",
    ),
    "humming": (
        "workspace/byteiaas-native-dist/humming",
        "humming_kernels-*.whl",
        "wheels.sha256",
    ),
    "tokenizers": (
        "workspace/byteiaas-tokenizers-dist",
        "tokenizers-*.whl",
        "wheels.sha256",
    ),
    "router": ("workspace/byteiaas-router-dist", "vllm_router-*.whl", "wheels.sha256"),
    "ep": ("tmp/ep_kernels_workspace/dist", "*.whl", "wheels.sha256"),
    "mooncake": (
        "tmp/mooncake-dist",
        "mooncake_transfer_engine_cuda13-*.whl",
        "wheel.sha256",
    ),
    "triton": ("tmp/triton_from_source_workspace/dist", "triton-*.whl", "wheel.sha256"),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
        return hasher.hexdigest()


def is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def source_env(path: Path) -> dict:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def inventory(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact symlinks are unsupported: {path}")
        if path.is_file() and path != root / "build-manifest.json":
            files[path.relative_to(root).as_posix()] = digest(path)
    return files


def check_provenance(repo: Path, kind: str, directory: Path, wheel: Path) -> None:
    if kind in ("deepgemm", "humming"):
        config_path = repo / "docker/byteiaas/native" / kind / "source.json"
        config = json.loads(config_path.read_text())
        lock_path = directory / "source-lock.json"
        lock = json.loads(lock_path.read_text())
        if (
            lock["format_version"] != 1
            or lock["repository"] != config["repository"]
            or lock["ref"] != config["ref"]
            or not is_hex(lock["source_commit"], 40)
            or not is_hex(lock["source_tree"], 40)
            or lock["package_version"] != package_version(config, lock["source_commit"])
        ):
            raise ValueError(f"Invalid resolved native source: {kind}")
        manifest = json.loads((directory / "native-manifest.json").read_text())
        expected = {
            "format_version": 1,
            "kind": kind,
            "lock_sha256": digest(lock_path),
            "version": lock["package_version"],
            **{
                key: lock[key]
                for key in ("repository", "ref", "source_commit", "source_tree")
            },
        }
        contract = manifest["torch_contract"]
        if (
            contract["torch"].split("+")[0] != "2.13.0"
            or contract["cuda"] != "13.0"
            or contract["cxx11_abi"] is not True
        ):
            raise ValueError(f"Native build ABI mismatch: {kind}")
        with zipfile.ZipFile(wheel) as archive:
            metadata_paths = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_paths) != 1:
                raise ValueError(f"Missing native wheel metadata: {kind}")
            metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
            if metadata["Version"] != lock["package_version"]:
                raise ValueError(f"Native wheel package version mismatch: {kind}")
            if kind == "humming":
                expected_files = {
                    "humming/_native/x86_64/" + name
                    for name in (
                        "libhumming_launcher.so",
                        "libcubinpatch.so",
                        "_device_info.abi3.so",
                        "manifest.json",
                        "nvrtc_compile",
                    )
                }
                if not expected_files.issubset(archive.namelist()):
                    raise ValueError("Humming artifact lacks compiled native files")
    elif kind in ("router", "tokenizers"):
        manifest = json.loads((directory / f"{kind}-manifest.json").read_text())
        if kind == "router":
            if not is_hex(manifest["commit"], 40):
                raise ValueError("Invalid router source commit")
            expected = {"repository": router_wheel.REPOSITORY, "ref": router_wheel.REF}
        else:
            expected = {
                "repository": tokenizers_wheel.REPOSITORY,
                "commit": tokenizers_wheel.COMMIT,
            }
    elif kind in ("mooncake", "ep"):
        versions = json.loads((repo / "docker/versions.json").read_text())["variable"]
        manifest = source_env(directory / "source.env")
        if kind == "mooncake":
            expected = {
                key.lower(): versions[f"MOONCAKE_GIT_{key}"]["default"]
                for key in ("REPOSITORY", "REF", "COMMIT")
            }
            expected["store_use_jemalloc"] = "ON"
        else:
            expected = {
                "repository": "https://github.com/deepseek-ai/DeepEP",
                **{
                    key: versions[arg]["default"]
                    for key, arg in (
                        ("commit", "DEEPEP_COMMIT_HASH"),
                        ("nvshmem_version", "NVSHMEM_VER"),
                        ("nccl_version", "NCCL_VERSION"),
                        ("cuda_version", "CUDA_VERSION"),
                    )
                },
            }
    else:
        return
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Artifact source provenance mismatch: {kind}")
    if kind in ("deepgemm", "humming", "router", "tokenizers") and (
        manifest["wheel"] != wheel.name or manifest["sha256"] != digest(wheel)
    ):
        raise ValueError(f"Artifact manifest wheel mismatch: {kind}")


def check_components(root: Path, repo: Path, normalize: bool = False) -> None:
    for kind, (relative, pattern, checksum_name) in COMPONENTS.items():
        directory = root / relative
        wheels = list(directory.glob("*.whl"))
        if kind == "triton":
            # Internal CUDA13 release workflows leave these source builders off.
            if wheels or (directory / "build-state").read_text().strip() != "disabled":
                raise ValueError(f"Expected disabled optional source build: {kind}")
            payload = directory / "build-state"
        else:
            if (
                len(wheels) != 1
                or list(directory.glob(pattern)) != wheels
                or (directory / "build-state").exists()
            ):
                raise ValueError(f"Expected one enabled component wheel: {kind}")
            payload = wheels[0]
            check_provenance(repo, kind, directory, payload)
        checksum = directory / checksum_name
        lines = checksum.read_text().splitlines()
        if len(lines) != 1:
            raise ValueError(f"Expected one checksum entry: {kind}")
        value, separator, filename = lines[0][:64], lines[0][64:66], lines[0][66:]
        if (
            not is_hex(value, 64)
            or separator not in ("  ", " *")
            or value != digest(payload)
        ):
            raise ValueError(f"Artifact checksum mismatch: {kind}")
        # Existing vLLM/EP producers emit build-stage paths. Normalize once
        # during export, checking their hashes before changing the path.
        if Path(filename).name != payload.name:
            raise ValueError(f"Artifact checksum payload mismatch: {kind}")
        if normalize:
            checksum.write_text(f"{value}  {payload.name}\n")
        elif filename != payload.name:
            raise ValueError(f"Artifact checksum path is not portable: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["seal", "validate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkout-ref", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    if not is_hex(args.checkout_ref, 40):
        raise ValueError("Expected an immutable source commit")
    if (root / "source-commit").read_text().strip() != args.checkout_ref:
        raise ValueError("Artifact source commit differs from image checkout")
    if (repo / ".github/byteiaas-build-contract").read_text().strip() != "5":
        raise ValueError("Expected ByteIAAS artifact build contract 5")
    identity = {
        "format_version": 5,
        "source_commit": args.checkout_ref,
        "run_id": args.run_id,
        "cuda_version": "13.0.3",
        "python_version": "3.12",
        "platform": "linux/amd64",
    }
    if args.command == "seal":
        # Reject links before touching checksum files.
        inventory(root)
        check_components(root, repo, normalize=True)
        manifest = {**identity, "files": inventory(root)}
        (root / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest = json.loads((root / "build-manifest.json").read_text())
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise ValueError("Artifact build identity differs from this workflow")
        if manifest["files"] != inventory(root):
            raise ValueError("Artifact file inventory or checksum differs from export")
        check_components(root, repo)
    print(json.dumps({"status": "PASS", **identity, "components": list(COMPONENTS)}))


if __name__ == "__main__":
    main()
