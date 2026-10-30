# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collect effective dependency provenance in the built image."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/byteiaas"))
    parser.add_argument("--derive-base", help="Base image reference for a devel image")
    args = parser.parse_args()
    dependencies = {}
    base_identity = None
    if args.derive_base:
        base = json.loads((args.root / "image-metadata.json").read_text())
        if base["format_version"] != 2:
            raise ValueError("Devel image requires ByteIAAS metadata contract 2")
        dependencies = base["dependencies"]
        base_identity = {
            "reference": args.derive_base,
            **{
                key: base[key]
                for key in ("build_commit", "build_pipeline", "build_url", "image_tag")
            },
        }
    flags = (
        {}
        if args.derive_base
        else {
            "deepgemm": "BYTEIAAS_BUILD_NATIVE",
            "humming": "BYTEIAAS_BUILD_NATIVE",
            "tokenizers": "BYTEIAAS_BUILD_TOKENIZERS",
            "router": "BYTEIAAS_BUILD_ROUTER",
        }
    )
    for kind, flag in flags.items():
        required = os.environ.get(flag) == "true"
        path = args.root / "provenance" / f"{kind}-installed.json"
        if required:
            if not path.is_file():
                raise ValueError(f"Enabled dependency lacks installed proof: {kind}")
            dependencies[kind] = json.loads(path.read_text())
    metadata = {
        "format_version": 2,
        "release": "v0.32.0",
        "release_commit": "5df63a7203cdb9328d824a27b682e77c7b9928be",
        "build_commit": os.environ.get("VLLM_BUILD_COMMIT", "unknown"),
        "build_pipeline": os.environ.get("VLLM_BUILD_PIPELINE", "local"),
        "build_url": os.environ.get("VLLM_BUILD_URL", ""),
        "image_tag": os.environ.get("VLLM_IMAGE_TAG", ""),
        "vllm_version": importlib.metadata.version("vllm"),
        "dependencies": dependencies,
    }
    if base_identity:
        metadata["base_image_identity"] = base_identity
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "image-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
