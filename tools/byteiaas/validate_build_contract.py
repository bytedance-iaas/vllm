# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject stale release build inputs before running expensive CI builds."""

import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    if (root / ".github/byteiaas-build-contract").read_text().strip() != "2":
        raise ValueError("Expected ByteIAAS build contract 2")
    dockerfile = (root / "docker/Dockerfile").read_text()
    for name, value in {
        "CUDA_VERSION": "13.0.3",
        "PYTHON_VERSION": "3.12",
        "UBUNTU_VERSION": "24.04",
    }.items():
        if f"ARG {name}={value}\n" not in dockerfile:
            raise ValueError(f"Unexpected release build default: {name}")
    if "torch==2.13.0\n" not in (root / "requirements/build/cuda.txt").read_text():
        raise ValueError("Unexpected native PyTorch build contract")
    sources = {}
    for kind in ("deepgemm", "humming"):
        path = root / "docker/byteiaas/native" / kind / "source-lock.json"
        lock = json.loads(path.read_text())
        if lock["format_version"] != 1:
            raise ValueError(f"Unsupported source lock: {kind}")
        for key in ("base_commit", "source_commit", "source_tree"):
            if re.fullmatch(r"[0-9a-f]{40}", lock[key]) is None:
                raise ValueError(f"Invalid source identity: {kind}/{key}")
        for patch in lock["patches"]:
            data = (path.parent / patch["file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != patch["sha256"]:
                raise ValueError(f"Native patch checksum mismatch: {patch['file']}")
        sources[kind] = {
            key: lock[key]
            for key in ("repository", "base_commit", "source_tree", "package_version")
        }
    print(json.dumps({"contract": 2, "native_sources": sources}, indent=2))


if __name__ == "__main__":
    main()
