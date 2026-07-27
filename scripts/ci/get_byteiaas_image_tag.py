#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TAG_SAFE_CHARS = frozenset(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_.-"
)


def is_tag_safe_suffix(value: str) -> bool:
    return (
        bool(value)
        and value[0].isalnum()
        and all(char in TAG_SAFE_CHARS for char in value)
    )


def is_vllm_version(value: str) -> bool:
    parts = value.split(".", 2)
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
        return False

    patch = parts[2]
    patch_digits = len(patch) - len(patch.lstrip("0123456789"))
    if patch_digits == 0:
        return False

    suffix = patch[patch_digits:]
    return all(char in TAG_SAFE_CHARS for char in suffix)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_version(version: str) -> str:
    normalized = version.strip().lstrip("v")
    if not is_vllm_version(normalized):
        raise SystemExit(
            f"invalid vLLM version: {version!r}; expected X.Y.Z with an "
            "optional Docker tag-safe suffix"
        )
    return normalized


def get_git_described_version(root: Path) -> str:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return get_latest_tag_version(root)
    return normalize_version(result.stdout.strip())


def get_latest_tag_version(root: Path) -> str:
    result = subprocess.run(
        ["git", "tag", "--list", "v[0-9]*", "--sort=-v:refname"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        for tag in result.stdout.splitlines():
            normalized = tag.strip().lstrip("v")
            if is_vllm_version(normalized):
                return normalized
    raise SystemExit(
        "failed to resolve vLLM version; pass --vllm-version or set "
        "BYTEIAAS_VLLM_VERSION"
    )


def get_vllm_version(version_arg: str) -> str:
    if version_arg:
        return normalize_version(version_arg)

    env_version = os.environ.get("BYTEIAAS_VLLM_VERSION", "")
    if env_version:
        return normalize_version(env_version)

    return get_git_described_version(repo_root())


def current_timestamp(timestamp_arg: str) -> str:
    if timestamp_arg:
        if len(timestamp_arg) != 12 or not timestamp_arg.isdigit():
            raise SystemExit("--timestamp must be in YYYYMMDDHHMM format")
        return timestamp_arg
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M")


def build_tag(
    *,
    mode: str,
    image_flavor: str,
    vllm_version: str,
    timestamp: str,
    tag_value: str,
    cuda_suffix: str,
) -> str:
    if mode == "dev":
        tag = f"v{vllm_version}.iaas.dev.{timestamp}"
    elif mode == "release":
        if not tag_value:
            raise SystemExit("--tag-value is required when --mode=release")
        if not is_tag_safe_suffix(tag_value):
            raise SystemExit("--tag-value must be a Docker tag-safe suffix")
        tag = f"v{vllm_version}.byted.{tag_value}.{timestamp}"
    else:
        raise SystemExit(f"unsupported mode: {mode}")

    if image_flavor == "openai":
        pass
    elif image_flavor == "openai-devel":
        tag = f"{tag}-openai-devel"
    else:
        raise SystemExit(f"unsupported image flavor: {image_flavor}")

    if cuda_suffix:
        tag = f"{tag}-{cuda_suffix}"
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ByteIAAS Volcengine CR image tags for vLLM."
    )
    parser.add_argument("--mode", choices=["dev", "release"], required=True)
    parser.add_argument(
        "--image-flavor",
        choices=["openai", "openai-devel"],
        default="openai",
        help="Image flavor. The default preserves the existing openai tag format.",
    )
    parser.add_argument(
        "--tag-value",
        default="",
        help="Internal tag value required for release mode.",
    )
    parser.add_argument(
        "--cuda-suffix",
        choices=["", "cu130"],
        default="cu130",
        help="CUDA suffix appended to the image tag.",
    )
    parser.add_argument(
        "--vllm-version",
        default="",
        help="Explicit vLLM version. Defaults to BYTEIAAS_VLLM_VERSION or git.",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Optional deterministic timestamp in YYYYMMDDHHMM format.",
    )
    args = parser.parse_args()

    tag = build_tag(
        mode=args.mode,
        image_flavor=args.image_flavor,
        vllm_version=get_vllm_version(args.vllm_version),
        timestamp=current_timestamp(args.timestamp),
        tag_value=args.tag_value,
        cuda_suffix=args.cuda_suffix,
    )
    print(tag)


if __name__ == "__main__":
    main()
