# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reject incompatible artifacts before the native cache loader can mutate state."""

import hashlib
import json

import pytest

from vllm.plugins.tk300_moe_cache import EXTRAS, SCHEMA, SHAPES, read_manifest


def artifacts(tmp_path):
    keys = [
        f"('trtllm::fused_moe::{stage}', 'MoERunner', {SHAPES}, {EXTRAS})"
        for stage in ("gemm1", "gemm2")
    ]
    cache = {"_metadata": {"gpu": "TK300 72G"}}
    for key, tactic in zip(keys, (10, 41), strict=True):
        cache[key] = [
            "MoERunner",
            tactic,
            ["cuda_graph_profile_replays", None, "l2_cache_policy", "hot"],
        ]
    raw = json.dumps(cache).encode()
    (tmp_path / "native.json").write_bytes(raw)
    manifest = {
        "schema": SCHEMA,
        "recipe": {"q": 8192},
        "trained_rows": [8192],
        "validated_rows": [6528, 8192],
        "native_bucket_interval": [4097, 8192],
        "native_profile_counts": [16, 28],
        "profile_ids": [10, 41],
        "cache_file": "native.json",
        "cache_sha256": hashlib.sha256(raw).hexdigest(),
        "native_file_keys": keys,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, cache


def test_verified_native_pair_can_be_loaded_without_rewriting_artifacts(tmp_path):
    path, manifest, cache = artifacts(tmp_path)
    raw, readback = read_manifest(path, {"q": 8192})
    assert readback["profile_ids"] == [10, 41]
    assert json.loads(raw) == cache
    assert json.loads(path.read_text()) == manifest


@pytest.mark.parametrize(
    "change",
    ["schema", "recipe", "checksum", "stage", "shape", "policy", "extra_op", "type"],
)
def test_rejects_unverified_cache_before_native_loading(tmp_path, change):
    path, manifest, cache = artifacts(tmp_path)
    if change == "schema":
        manifest["schema"] = "future-v2"
    elif change == "recipe":
        manifest["recipe"] = {"q": 4096}
    elif change == "checksum":
        manifest["cache_sha256"] = "0" * 64
    elif change == "stage":
        manifest["profile_ids"] = [41, 10]
    elif change == "shape":
        key = manifest["native_file_keys"][0]
        new_key = key.replace("(8192, 3072)", "(4096, 3072)")
        cache[new_key] = cache.pop(key)
        manifest["native_file_keys"][0] = new_key
    elif change == "policy":
        cache[manifest["native_file_keys"][0]][2][1] = 1
    elif change == "extra_op":
        cache["bf16_gemm"] = ["OtherRunner", 0]
    elif change == "type":
        cache[manifest["native_file_keys"][0]][1] = 10.0
    raw = json.dumps(cache).encode()
    (tmp_path / "native.json").write_bytes(raw)
    if change != "checksum":
        manifest["cache_sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_manifest(path, {"q": 8192})


def test_missing_manifest_leaves_native_cache_unloaded(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_manifest(tmp_path / "absent.json", {"q": 8192})
