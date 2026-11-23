# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Package fixed producer modules while preserving the bound role hash policy."""

import argparse
import ast
import json
import subprocess
from hashlib import sha256
from pathlib import Path

PREFIX = "vllm/distributed/kv_transfer/kv_connector/v1/nixl/"
FILES = (
    "base_worker.py",
    "metadata.py",
    "connector.py",
    "push_scheduler.py",
    "push_worker.py",
)
RUNTIME_METADATA_SHA = (
    "7776e1be537fad6242dbecc4a0fa4a238aceda6d7b0ea4fa16d3dbbbfd8a8534"
)
BASE_COMMIT = "6f6b17e9dcfbc869e0f4f5b860fc5992763b82e9"
HASH_FUNCTION = "compute_nixl_compatibility_hash"


def function(text):
    return next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == HASH_FUNCTION
    )


def replace_function(text, source):
    node = function(text)
    lines = text.splitlines(keepends=True)
    return (
        "".join(lines[: node.lineno - 1])
        + source
        + "\n"
        + "".join(lines[node.end_lineno :])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--runtime-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    assert len(args.commit) == 40 and all(c in "0123456789abcdef" for c in args.commit)
    runtime_bytes = args.runtime_metadata.read_bytes()
    assert sha256(runtime_bytes).hexdigest() == RUNTIME_METADATA_SHA, (
        "Unrecognized bound runtime metadata"
    )
    runtime = runtime_bytes.decode()
    baseline = subprocess.check_output(
        ["git", "show", BASE_COMMIT + ":" + PREFIX + "metadata.py"], cwd=repository
    ).decode()
    # The role's only extra logic is this already-deployed compatibility policy.
    stripped_runtime = replace_function(
        runtime, "def compute_nixl_compatibility_hash():\n    pass"
    )
    stripped_baseline = replace_function(
        baseline, "def compute_nixl_compatibility_hash():\n    pass"
    )
    assert ast.dump(ast.parse(stripped_runtime)) == ast.dump(
        ast.parse(stripped_baseline)
    )
    preserved = ast.get_source_segment(runtime, function(runtime))
    assert preserved is not None
    manifest = {
        "vllm_commit": args.commit,
        "baseline_commit": BASE_COMMIT,
        "runtime_metadata_sha256": RUNTIME_METADATA_SHA,
        "preserved_function": HASH_FUNCTION,
        "roles": {"P": {}},
        "decoder": "Keep exact bound H20 modules and base_worker; no producer copy",
        "validation": "Source guards only; GPU serving is not yet validated",
    }
    for name in FILES:
        path = PREFIX + name
        raw = subprocess.check_output(
            ["git", "show", args.commit + ":" + path], cwd=repository
        )
        data = (
            replace_function(raw.decode(), preserved).encode()
            if name == "metadata.py"
            else raw
        )
        compile(data, path, "exec")
        target = args.output / "P" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data, "Immutable package differs"
        else:
            target.write_bytes(data)
        manifest["roles"]["P"][path] = {
            "canonical_sha256": sha256(raw).hexdigest(),
            "delivery_sha256": sha256(data).hexdigest(),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "commit": args.commit,
                "producer_files": len(FILES),
                "role_policy_preserved": True,
            }
        )
    )


if __name__ == "__main__":
    main()
