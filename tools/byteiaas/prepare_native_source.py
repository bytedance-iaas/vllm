# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recreate a native dependency from an immutable base and checked-in patches."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def prepare(
    lock_path: Path, source: Path, local: Path | None, submodules: bool
) -> None:
    lock = json.loads(lock_path.read_text())
    if lock["format_version"] != 1:
        raise ValueError("Unsupported native source lock")
    for patch in lock["patches"]:
        path = lock_path.parent / patch["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]:
            raise ValueError(f"Native patch checksum mismatch: {path}")
    if source.exists():
        raise ValueError(f"Use a new source directory: {source}")
    source.mkdir(parents=True)
    subprocess.run(["git", "init", str(source)], check=True)
    repository = str(local.resolve()) if local else lock["repository"]
    git(source, "fetch", "--depth", "1", repository, lock["base_commit"])
    git(source, "checkout", "--detach", "FETCH_HEAD")
    if git(source, "rev-parse", "HEAD") != lock["base_commit"]:
        raise ValueError("Unexpected native base commit")
    for patch in lock["patches"]:
        git(
            source,
            "apply",
            "--index",
            str((lock_path.parent / patch["file"]).resolve()),
        )
    tree = git(source, "write-tree")
    if tree != lock["source_tree"]:
        raise ValueError(f"Unexpected native source tree: {tree}")
    if submodules and (source / ".gitmodules").exists():
        git(source, "submodule", "sync", "--recursive")
        git(source, "submodule", "update", "--init", "--recursive")
        status = git(source, "submodule", "status", "--recursive")
        if any(line.startswith(("-", "+", "U")) for line in status.splitlines()):
            raise ValueError("Native submodules differ from the locked gitlinks")
    print(
        json.dumps(
            {
                "repository": lock["repository"],
                "base_commit": lock["base_commit"],
                "ported_commit": lock["source_commit"],
                "source_tree": tree,
                "source": str(source),
                "submodules_initialized": submodules,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--local-repository", type=Path)
    parser.add_argument("--skip-submodules", action="store_true")
    args = parser.parse_args()
    prepare(
        args.lock.resolve(),
        args.source.resolve(),
        args.local_repository,
        not args.skip_submodules,
    )


if __name__ == "__main__":
    main()
