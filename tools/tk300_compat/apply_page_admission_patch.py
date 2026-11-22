# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the version-pinned H20 API backport while retaining role compute files."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "h20-page-admission-manifest.json").read_text())
    current = {}
    for file in manifest["files"]:
        current[file] = hashlib.sha256((args.root / file).read_bytes()).hexdigest()
    if all(
        current[file] == record["candidate_sha256"]
        for file, record in manifest["files"].items()
    ):
        print("H20 page admission API already installed")
        return
    for file, record in manifest["files"].items():
        if current[file] != record["base_sha256"]:
            raise RuntimeError(f"H20 API base differs; preserve and inspect {file}")
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1"],
        input=(directory / "h20-page-admission.patch").read_text(),
        text=True,
        cwd=args.root,
        check=True,
    )
    for file, record in manifest["files"].items():
        actual = hashlib.sha256((args.root / file).read_bytes()).hexdigest()
        if actual != record["candidate_sha256"]:
            raise RuntimeError(f"H20 API backport readback differs: {file}")
    print("Verified all H20 page admission API files")


if __name__ == "__main__":
    main()
