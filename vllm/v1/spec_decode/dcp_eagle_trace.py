# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch

from vllm.distributed.parallel_state import get_dcp_group, get_tp_group

_TRACE_FIELDS = {
    "block_table",
    "block_table_tensor",
    "causal",
    "dcp_context_kv_lens",
    "dcp_local_seq_lens",
    "decode",
    "decode_query_len",
    "global_kv_lens",
    "local_kv_lens",
    "max_dcp_context_kv_len",
    "max_query_len",
    "num_actual_tokens",
    "query_start_loc",
    "seq_lens",
    "slot_mapping",
}


def _trace_root() -> Path | None:
    value = os.getenv("VLLM_DCP_EAGLE_TRACE_DIR")
    return Path(value) if value else None


def _rank_enabled() -> bool:
    configured = os.getenv("VLLM_DCP_EAGLE_TRACE_TP_RANKS", "0,1")
    ranks = {int(value) for value in configured.split(",") if value}
    return get_tp_group().rank_in_group in ranks


def trace_enabled(round_idx: int, batch_size: int) -> bool:
    root = _trace_root()
    if root is None or batch_size != 1 or not _rank_enabled():
        return False
    max_rounds = int(os.getenv("VLLM_DCP_EAGLE_TRACE_MAX_ROUNDS", "32"))
    return round_idx < max_rounds


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.is_floating_point():
            tensor = tensor.float()
        return tensor.cpu()
    if dataclasses.is_dataclass(value):
        return {
            field.name: _snapshot(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name in _TRACE_FIELDS
        }
    if isinstance(value, dict):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if isinstance(value, (bool, float, int, str)) or value is None:
        return value
    return repr(value)


def snapshot_attention_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {"metadata": _snapshot(metadata)}

    result: dict[str, Any] = {}
    seen: set[int] = set()
    for layer_name, layer_metadata in metadata.items():
        identity = id(layer_metadata)
        if identity in seen:
            continue
        seen.add(identity)
        key = f"{type(layer_metadata).__name__}:{layer_name}"
        result[key] = _snapshot(layer_metadata)
    return result


def save_eagle_trace(
    round_idx: int,
    stage: str,
    batch_size: int,
    **payload: Any,
) -> None:
    root = _trace_root()
    if root is None or not trace_enabled(round_idx, batch_size):
        return

    tp_rank = get_tp_group().rank_in_group
    try:
        dcp_rank = get_dcp_group().rank_in_group
    except AssertionError:
        dcp_rank = 0
    rank_dir = root / f"tp{tp_rank}-dcp{dcp_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "round": round_idx,
        "stage": stage,
        "tp_rank": tp_rank,
        "dcp_rank": dcp_rank,
        **{key: _snapshot(value) for key, value in payload.items()},
    }
    torch.save(output, rank_dir / f"round-{round_idx:04d}-{stage}.pt")
