# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve a native dependency branch and record the source used by a build."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def resolve(repository: str, ref: str) -> str:
    branch = f"refs/heads/{ref}"
    git(Path.cwd(), "check-ref-format", branch)
    commit, actual_ref = git(
        Path.cwd(), "ls-remote", "--exit-code", repository, branch
    ).split()
    if actual_ref != branch or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Invalid branch resolution: {repository} {branch}")
    return commit


def checkout(repository: str, commit: str, source: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full source commit")
    source.mkdir(parents=True)
    git(source, "init")
    git(source, "remote", "add", "origin", repository)
    git(source, "fetch", "--no-tags", "--depth", "1", "origin", commit)
    git(source, "checkout", "--detach", "FETCH_HEAD")
    if git(source, "rev-parse", "HEAD") != commit:
        raise ValueError("Unexpected native source commit")


def package_version(config: dict, commit: str) -> str:
    version = config["package_version"]
    return f"{version}.g{commit[:7]}" if version.endswith("+unified") else version


def prepare(config: dict, commit: str, source: Path, lock: Path) -> None:
    checkout(config["repository"], commit, source)
    if (source / ".gitmodules").exists():
        git(source, "submodule", "update", "--init", "--recursive")
        status = git(source, "submodule", "status", "--recursive")
        if any(line.startswith(("-", "+", "U")) for line in status.splitlines()):
            raise ValueError("Native submodules differ from the checked-out gitlinks")
    lock.write_text(
        json.dumps(
            {
                "format_version": 1,
                "repository": config["repository"],
                "ref": config["ref"],
                "source_commit": commit,
                "source_tree": git(source, "rev-parse", "HEAD^{tree}"),
                "package_version": package_version(config, commit),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("resolve")
    builder = commands.add_parser("checkout")
    builder.add_argument("--commit", required=True)
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command == "resolve":
        print(resolve(config["repository"], config["ref"]))
    else:
        prepare(config, args.commit, args.source.resolve(), args.lock)


if __name__ == "__main__":
    main()
