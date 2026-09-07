#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate that a cached DeepGEMM checkout is safe to reuse."""

import argparse
import subprocess
import sys
from pathlib import Path


def _git(
    source_dir: Path,
    git_executable: str,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [git_executable, "-C", str(source_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def check_checkout(source_dir: Path, git_executable: str = "git") -> list[str]:
    errors = []

    status = _git(
        source_dir,
        git_executable,
        "status",
        "--porcelain",
        "--ignore-submodules=none",
    )
    if status.returncode != 0:
        errors.append(f"cannot inspect worktree status: {status.stderr.strip()}")
    elif status.stdout:
        errors.append(f"worktree is dirty:\n{status.stdout.rstrip()}")

    submodules = _git(
        source_dir,
        git_executable,
        "submodule",
        "status",
        "--recursive",
    )
    if submodules.returncode != 0:
        errors.append(
            f"cannot inspect recursive submodules: {submodules.stderr.strip()}"
        )
    else:
        invalid = [line for line in submodules.stdout.splitlines() if line[:1] in "-+U"]
        if invalid:
            errors.append(
                "submodules are not initialized at recorded commits:\n"
                + "\n".join(invalid)
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--git-executable", default="git")
    args = parser.parse_args()

    errors = check_checkout(args.source_dir, args.git_executable)
    if errors:
        print("DeepGEMM checkout is not reusable:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"DeepGEMM checkout is clean and reusable: {args.source_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
