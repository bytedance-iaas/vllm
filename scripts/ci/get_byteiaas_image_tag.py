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
    return patch_digits > 0 and all(
        char in TAG_SAFE_CHARS for char in patch[patch_digits:]
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_version(version: str) -> str:
    normalized = version.strip().lstrip("v")
    if not is_vllm_version(normalized):
        raise SystemExit(f"invalid vLLM version: {version!r}")
    return normalized


def get_vllm_version(version_arg: str) -> str:
    if version_arg:
        return normalize_version(version_arg)
    if value := os.environ.get("BYTEIAAS_VLLM_VERSION", ""):
        return normalize_version(value)
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return normalize_version(result.stdout.strip())
    tags = subprocess.check_output(
        ["git", "tag", "--list", "v[0-9]*", "--sort=-v:refname"],
        cwd=repo_root(),
        text=True,
    )
    for tag in tags.splitlines():
        if is_vllm_version(tag.lstrip("v")):
            return normalize_version(tag)
    raise SystemExit("failed to resolve vLLM version")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "release"], required=True)
    parser.add_argument(
        "--image-flavor",
        choices=["openai", "openai-devel"],
        default="openai",
    )
    parser.add_argument("--tag-value", default="")
    parser.add_argument("--cuda-suffix", default="cu130")
    parser.add_argument("--vllm-version", default="")
    parser.add_argument("--format-suffix", default="")
    parser.add_argument("--timestamp", default="")
    args = parser.parse_args()
    version = get_vllm_version(args.vllm_version)
    timestamp = args.timestamp or datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
        "%Y%m%d%H%M"
    )
    if args.mode == "dev":
        tag = f"v{version}.iaas.dev.{timestamp}"
    else:
        if not is_tag_safe_suffix(args.tag_value):
            raise SystemExit("--tag-value is required and must be Docker tag-safe")
        tag = f"v{args.tag_value}.byted.{timestamp}"
    if args.image_flavor == "openai-devel":
        tag += "-openai-devel"
    if args.cuda_suffix:
        tag += f"-{args.cuda_suffix}"
    if args.format_suffix:
        if not is_tag_safe_suffix(args.format_suffix):
            raise SystemExit("--format-suffix must be Docker tag-safe")
        tag += f"-{args.format_suffix}"
    print(tag)


if __name__ == "__main__":
    main()
