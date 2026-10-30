# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and verify the pinned tokenizers wheel used for frontend offload."""

import argparse
import hashlib
import importlib
import json
import subprocess
import zipfile
from pathlib import Path

REPOSITORY = "https://github.com/wangyicong52/tokenizers.git"
COMMIT = "441393c720fbd27a4faea02b7af18ebe08a65a01"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(args: argparse.Namespace) -> None:
    source = args.source_dir.resolve()
    wheels = args.wheel_dir.resolve()
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(
            ["git", "-C", str(source), "remote", "add", "origin", REPOSITORY],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "fetch", "--depth", "1", "origin", COMMIT],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        raise ValueError(f"tokenizers source must be {COMMIT}, found {actual}")
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored",
        ],
        text=True,
    )
    if status:
        raise ValueError(
            f"Use a clean source tree with no extra build inputs:\n{status}"
        )
    wheels.mkdir(parents=True, exist_ok=True)
    if list(wheels.glob("*.whl")):
        raise ValueError("Use an empty wheel directory to avoid stale build artifacts")
    python = args.python.absolute()
    subprocess.run(
        [
            str(python),
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--compatibility",
            args.compatibility,
            "--interpreter",
            str(python),
            "--out",
            str(wheels),
        ],
        cwd=source / "bindings/python",
        check=True,
    )
    built = list(wheels.glob("*.whl"))
    if len(built) != 1:
        raise ValueError(f"Expected one tokenizers wheel, found {built}")
    wheel = built[0]
    digest = sha256(wheel.read_bytes())
    manifest = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "wheel": wheel.name,
        "sha256": digest,
        "compatibility": args.compatibility,
    }
    (wheels / "tokenizers-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (wheels / "wheels.sha256").write_text(f"{digest}  {wheel.name}\n")
    print(json.dumps(manifest))


def verify(args: argparse.Namespace) -> None:
    import tokenizers
    from tokenizers.decoders import ByteLevel, DecodeStream
    from tokenizers.models import WordLevel

    manifest = json.loads(args.manifest.read_text())
    if manifest["commit"] != COMMIT or manifest["repository"] != REPOSITORY:
        raise ValueError("Unexpected tokenizers source provenance")
    wheel = args.manifest.parent / manifest["wheel"]
    if sha256(wheel.read_bytes()) != manifest["sha256"]:
        raise ValueError("tokenizers wheel SHA256 mismatch")
    native = Path(importlib.import_module("tokenizers.tokenizers").__file__)
    with zipfile.ZipFile(wheel) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.startswith("tokenizers/tokenizers.") and name.endswith(".so")
        ]
        if len(matches) != 1 or sha256(native.read_bytes()) != sha256(
            archive.read(matches[0])
        ):
            raise ValueError("Installed native module differs from the pinned wheel")

    tokenizer = tokenizers.Tokenizer(WordLevel({"a": 0, "b": 1}, unk_token="a"))
    tokenizer.decoder = ByteLevel()
    stream = DecodeStream(ids=[0])
    if stream.prefill_pending is not True:
        raise ValueError("DecodeStream.prefill_pending is missing or incorrect")
    if stream.step(tokenizer, 1) != "b" or stream.prefill_pending is not False:
        raise ValueError("DecodeStream.step failed the retained-prefix contract")
    if DecodeStream().prefill_pending is not False:
        raise ValueError("Empty DecodeStream must not report pending prefill")
    report = {
        **manifest,
        "version": tokenizers.__version__,
        "native_module": str(native),
        "prefill_pending": "PASS",
        "installed_module_hash": "PASS",
    }
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--source-dir", type=Path, required=True)
    builder.add_argument("--wheel-dir", type=Path, required=True)
    builder.add_argument("--python", type=Path, required=True)
    builder.add_argument("--compatibility", default="manylinux_2_28")
    builder.set_defaults(func=build)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--manifest", type=Path, required=True)
    verifier.add_argument("--report", type=Path)
    verifier.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
