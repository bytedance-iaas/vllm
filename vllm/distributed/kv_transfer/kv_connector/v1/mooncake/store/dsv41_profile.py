# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation boundary for the opt-in DSv4.1 PD/DRAM-store composition."""

import math
from typing import Any

from .dsv41_layout import FORMAT, PROFILE


def validate_store(config: Any) -> None:
    extra = config.kv_transfer_config.kv_connector_extra_config
    if extra.get("store_format") != FORMAT:
        raise ValueError("Canonical Store requires dsv41_canonical_v1")
    if extra.get("pd_role") not in ("prefill", "decode"):
        raise ValueError("Canonical Store requires pd_role=prefill or decode")
    if not all(
        isinstance(extra.get(k), str) and extra[k].strip()
        for k in ("cache_prefix", "model_revision")
    ):
        raise ValueError("Canonical Store requires cache_prefix and model_revision")
    parallel = config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("Canonical Store initially supports TP1/DCP1 only")
    if extra["pd_role"] == "decode" and extra.get("enable_lookup", False):
        raise ValueError("Decode Store lookup must remain disabled")
    if extra.get("enable_lookup", False) and not extra.get("load_async", True):
        raise ValueError("Canonical Store requires asynchronous loading")
    for key, default in {
        "max_pending_jobs": 32,
        "max_pending_lookups": 32,
        "max_pinned_blocks": 4096,
        "max_pinned_bytes": 512 * 1024**2,
        "max_inflight_bytes": 256 * 1024**2,
        "lookup_timeout_ms": 1000,
        "store_stall_timeout_s": 30,
        "failure_cooldown_s": 5,
    }.items():
        value = extra.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Canonical Store {key} must be finite and positive")
        if key not in (
            "store_stall_timeout_s",
            "failure_cooldown_s",
        ) and not isinstance(value, int):
            raise ValueError(f"Canonical Store {key} must be an integer")
    if config.cache_config.cache_dtype != "fp8_ds_mla":
        raise ValueError("Canonical Store requires explicit fp8_ds_mla")
    if not config.uses_dsv41_encoder_only_handoff or not config.use_v2_model_runner:
        raise ValueError("Canonical Store requires DSv4.1 encoder-only MRV2")


def validate_composition(config: Any) -> None:
    if not config.uses_dsv41_encoder_only_handoff:
        raise ValueError(
            "Canonical PD profile requires encoder-only mode on the parent"
        )
    root = config.kv_transfer_config
    extra = root.kv_connector_extra_config
    if (
        root.kv_connector != "MultiConnector"
        or extra.get("pd_store_profile") != PROFILE
    ):
        raise ValueError("Expected the explicit dsv41_flash_v1 MultiConnector profile")
    children = extra.get("connectors", [])
    if [c.get("kv_connector") for c in children] != [
        "MooncakeConnector",
        "MooncakeStoreConnector",
    ]:
        raise ValueError(
            "Profile requires MooncakeConnector then MooncakeStoreConnector"
        )
    if root.kv_role not in ("kv_producer", "kv_consumer"):
        raise ValueError("Profile requires dedicated producer/consumer roles")
    direct, store = children
    if direct.get("kv_role") != root.kv_role or not direct.get(
        "dsv41_encoder_only_prefill"
    ):
        raise ValueError("Direct Mooncake child must preserve the encoder-only PD role")
    expected = "prefill" if root.kv_role == "kv_producer" else "decode"
    store_extra = store.get("kv_connector_extra_config", {})
    if (
        store_extra.get("pd_role") != expected
        or store_extra.get("store_format") != FORMAT
    ):
        raise ValueError("Store child must declare the matching canonical PD role")
    if store.get("kv_role") != "kv_both" or not store.get("dsv41_encoder_only_prefill"):
        raise ValueError("Canonical Store child requires kv_both and encoder-only mode")
    if expected == "decode" and store_extra.get("enable_lookup", False):
        raise ValueError("Decode must receive exclusively from direct Mooncake")
    if expected == "decode" and root.kv_load_failure_policy != "fail":
        raise ValueError(
            "Decode direct PD transfer requires kv_load_failure_policy=fail"
        )
    if any(c.get("kv_connector_module_path") for c in children):
        raise ValueError(
            "Canonical profile requires the built-in connector implementations"
        )
    if config.cache_config.kv_offloading_size:
        raise ValueError("Do not combine canonical Store with --kv-offloading-size")
    if any(c.get("kv_buffer_device", "cuda") != "cuda" for c in children):
        raise ValueError("Both profile children require CUDA KV buffers")
    if (
        expected == "prefill"
        and store_extra.get("enable_lookup", False)
        and root.kv_load_failure_policy != "recompute"
    ):
        raise ValueError("Prefill Store loads require kv_load_failure_policy=recompute")
    p = config.parallel_config
    if p.distributed_executor_backend == "external_launcher":
        raise ValueError("Canonical profile requires one scheduler per DP engine")
    topology = (p.pipeline_parallel_size, p.prefill_context_parallel_size)
    if topology != ((2, 4) if expected == "prefill" else (1, 1)):
        raise ValueError("Profile requires P PP2/PCP4 and D PP1/PCP1")
