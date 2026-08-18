# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_tp_group,
    get_world_group,
)
from vllm.platforms import current_platform

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
    "max_decode_query_len",
    "max_query_len",
    "max_seq_len",
    "num_actual_tokens",
    "query_start_loc",
    "seq_lens",
    "slot_mapping",
}


def _trace_root() -> Path | None:
    value = os.getenv("VLLM_DCP_EAGLE_TRACE_DIR")
    return Path(value) if value else None


def eagle_trace_configured() -> bool:
    return _trace_root() is not None and _rank_enabled()


def _rank_enabled() -> bool:
    configured = os.getenv("VLLM_DCP_EAGLE_TRACE_TP_RANKS", "0,1")
    ranks = {int(value) for value in configured.split(",") if value}
    return get_tp_group().rank_in_group in ranks


def _rank_dir(root: Path) -> tuple[Path, int, int, int, int]:
    tp_rank = get_tp_group().rank_in_group
    world_rank = get_world_group().rank
    try:
        dcp_rank = get_dcp_group().rank_in_group
    except AssertionError:
        dcp_rank = 0
    pid = os.getpid()
    rank_dir = root / (f"world{world_rank}-pid{pid}-tp{tp_rank}-dcp{dcp_rank}")
    rank_dir.mkdir(parents=True, exist_ok=True)
    return rank_dir, world_rank, pid, tp_rank, dcp_rank


def trace_enabled(round_idx: int, batch_size: int) -> bool:
    root = _trace_root()
    if root is None or batch_size != 1 or not _rank_enabled():
        return False
    max_rounds = int(os.getenv("VLLM_DCP_EAGLE_TRACE_MAX_ROUNDS", "32"))
    return 0 <= round_idx < max_rounds


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


def _exact_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(key): _exact_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_exact_snapshot(item) for item in value]
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


def snapshot_dcp_topk(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        global_topk = getattr(module, "_dcp_eagle_trace_global_topk", None)
        local_topk = getattr(module, "_dcp_eagle_trace_local_topk", None)
        if global_topk is None and local_topk is None:
            continue
        result[module_name] = {
            "global_topk": _snapshot(global_topk),
            "local_topk": _snapshot(local_topk),
        }
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

    rank_dir, world_rank, pid, tp_rank, dcp_rank = _rank_dir(root)
    output = {
        "round": round_idx,
        "stage": stage,
        "world_rank": world_rank,
        "pid": pid,
        "tp_rank": tp_rank,
        "dcp_rank": dcp_rank,
        **{key: _snapshot(value) for key, value in payload.items()},
    }
    torch.save(output, rank_dir / f"round-{round_idx:04d}-{stage}.pt")


_TARGET_LAYER_TRIPWIRE_CAPTURED = False
_TARGET_LAYER_TRIPWIRE_ACTIVE = False
_DENSE_ATTN_DECOMP_CAPTURED = False
_TRIPWIRE_STAGES = (
    "A_attention_input",
    "B_attention_return",
    "C_ffn_input",
    "D_ffn_return",
)


@dataclasses.dataclass(frozen=True)
class TargetLayerTripwire:
    row: int
    position: int
    token_id: int
    max_layer: int


def dense_attn_decomp_enabled(
    layer_name: str,
    max_seq_len: int,
    num_actual_tokens: int,
) -> bool:
    configured_layer = os.getenv("VLLM_DCP_EAGLE_ATTN_DECOMP_LAYER")
    configured_position = os.getenv("VLLM_DCP_EAGLE_TRIPWIRE_POSITION")
    if (
        _trace_root() is None
        or configured_layer is None
        or configured_position is None
        or _DENSE_ATTN_DECOMP_CAPTURED
        or not _TARGET_LAYER_TRIPWIRE_ACTIVE
        or not _rank_enabled()
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(90)
        or num_actual_tokens != 1
        or max_seq_len != int(configured_position) + 1
    ):
        return False
    return f".layers.{int(configured_layer)}.self_attn.attn" in layer_name


def save_dense_attn_decomp(layer_name: str, **payload: Any) -> None:
    global _DENSE_ATTN_DECOMP_CAPTURED

    root = _trace_root()
    if root is None or _DENSE_ATTN_DECOMP_CAPTURED:
        return
    rank_dir, world_rank, pid, tp_rank, dcp_rank = _rank_dir(root)
    output = {
        "stage": "dense_attention_decomposition",
        "world_rank": world_rank,
        "pid": pid,
        "tp_rank": tp_rank,
        "dcp_rank": dcp_rank,
        "layer_name": layer_name,
        **{key: _exact_snapshot(value) for key, value in payload.items()},
    }
    layer_id = int(os.getenv("VLLM_DCP_EAGLE_ATTN_DECOMP_LAYER", "0"))
    position = int(os.getenv("VLLM_DCP_EAGLE_TRIPWIRE_POSITION", "-1"))
    torch.save(
        output,
        rank_dir / f"dense-attn-decomp-layer{layer_id}-pos{position}.pt",
    )
    _DENSE_ATTN_DECOMP_CAPTURED = True


def target_layer_tripwire_requested() -> bool:
    return (
        _trace_root() is not None
        and os.getenv("VLLM_DCP_EAGLE_TRIPWIRE_POSITION") is not None
        and not _TARGET_LAYER_TRIPWIRE_CAPTURED
    )


def prepare_target_layer_tripwire(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> TargetLayerTripwire | None:
    configured_position = os.getenv("VLLM_DCP_EAGLE_TRIPWIRE_POSITION")
    if not target_layer_tripwire_requested():
        return None
    assert configured_position is not None

    assert input_ids.device.type == "cpu"
    assert positions.device.type == "cpu"
    target_position = int(configured_position)
    rows = (positions == target_position).nonzero(as_tuple=False).flatten()
    if rows.numel() != 1:
        return None
    row = int(rows.item())
    token_id = int(input_ids[row].item())

    configured_token = os.getenv("VLLM_DCP_EAGLE_TRIPWIRE_TOKEN")
    if configured_token is not None and token_id != int(configured_token):
        return None
    return TargetLayerTripwire(
        row=row,
        position=target_position,
        token_id=token_id,
        max_layer=int(os.getenv("VLLM_DCP_EAGLE_TRIPWIRE_MAX_LAYER", "30")),
    )


def _tripwire_hidden(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    hidden_states = kwargs.get("hidden_states")
    if hidden_states is None:
        hidden_states = args[0]
    assert isinstance(hidden_states, torch.Tensor)
    return hidden_states


def _tripwire_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    assert isinstance(output, torch.Tensor)
    return output


def _target_decoder_layers(model: Any) -> Any:
    if hasattr(model, "get_language_model"):
        model = model.get_language_model()
    elif hasattr(model, "language_model"):
        model = model.language_model
    backbone = getattr(model, "model", None)
    return getattr(backbone, "layers", None)


@contextmanager
def capture_target_layer_tripwire(
    model: Any,
    tripwire: TargetLayerTripwire | None,
) -> Iterator[None]:
    global _TARGET_LAYER_TRIPWIRE_ACTIVE, _TARGET_LAYER_TRIPWIRE_CAPTURED

    if tripwire is None or _TARGET_LAYER_TRIPWIRE_CAPTURED:
        yield
        return

    if not _rank_enabled():
        _TARGET_LAYER_TRIPWIRE_ACTIVE = True
        try:
            yield
        finally:
            _TARGET_LAYER_TRIPWIRE_ACTIVE = False
            _TARGET_LAYER_TRIPWIRE_CAPTURED = True
        return

    root = _trace_root()
    assert root is not None
    layers = _target_decoder_layers(model)
    max_layer = tripwire.max_layer
    if layers is None or len(layers) <= max_layer:
        raise RuntimeError(
            "DCP EAGLE target tripwire could not find the requested decoder layers"
        )

    captures: dict[tuple[int, str], torch.Tensor] = {}
    ffn_output_reduced: list[bool] = []
    handles = []
    layer0_post_attention_residual: torch.Tensor | None = None

    def record(layer_idx: int, stage: str, tensor: torch.Tensor) -> None:
        captures[(layer_idx, stage)] = tensor[tripwire.row].detach().clone()

    for layer_idx, layer in enumerate(layers[: max_layer + 1]):
        ffn = layer.block_sparse_moe if layer.is_moe_layer else layer.mlp
        ffn_output_reduced.append(not layer.ffn_all_reduce_deferred)

        if layer_idx == 0:

            def layer_pre_hook(
                _module: Any,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> None:
                nonlocal layer0_post_attention_residual
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None:
                    hidden_states = args[1]
                residual = kwargs.get("residual")
                if residual is None and len(args) > 2:
                    residual = args[2]
                if residual is None:
                    residual = hidden_states
                assert isinstance(residual, torch.Tensor)
                layer0_post_attention_residual = residual[tripwire.row].detach().clone()

            handles.append(
                layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True)
            )

        def attention_pre_hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            layer_idx: int = layer_idx,
        ) -> None:
            record(layer_idx, "A_attention_input", _tripwire_hidden(args, kwargs))

        def attention_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            layer_idx: int = layer_idx,
        ) -> None:
            record(layer_idx, "B_attention_return", _tripwire_output(output))

        def ffn_pre_hook(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            layer_idx: int = layer_idx,
        ) -> None:
            record(layer_idx, "C_ffn_input", _tripwire_hidden(args, kwargs))

        def ffn_hook(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            layer_idx: int = layer_idx,
        ) -> None:
            record(layer_idx, "D_ffn_return", _tripwire_output(output))

        handles.extend(
            (
                layer.self_attn.register_forward_pre_hook(
                    attention_pre_hook, with_kwargs=True
                ),
                layer.self_attn.register_forward_hook(attention_hook),
                ffn.register_forward_pre_hook(ffn_pre_hook, with_kwargs=True),
                ffn.register_forward_hook(ffn_hook),
            )
        )

    _TARGET_LAYER_TRIPWIRE_ACTIVE = True
    try:
        yield
    finally:
        _TARGET_LAYER_TRIPWIRE_ACTIVE = False
        for handle in handles:
            handle.remove()

    missing = [
        (layer_idx, stage)
        for layer_idx in range(max_layer + 1)
        for stage in _TRIPWIRE_STAGES
        if (layer_idx, stage) not in captures
    ]
    if missing:
        raise RuntimeError(f"DCP EAGLE target tripwire missed checkpoints: {missing}")
    if layer0_post_attention_residual is None:
        raise RuntimeError("DCP EAGLE target tripwire missed the layer-0 residual")

    stacked = torch.stack(
        [
            captures[(layer_idx, stage)]
            for layer_idx in range(max_layer + 1)
            for stage in _TRIPWIRE_STAGES
        ]
    ).view(max_layer + 1, len(_TRIPWIRE_STAGES), -1)
    stacked = stacked.float().cpu()

    rank_dir, world_rank, pid, tp_rank, dcp_rank = _rank_dir(root)
    output = {
        "stage": "target_layer_tripwire",
        "world_rank": world_rank,
        "pid": pid,
        "tp_rank": tp_rank,
        "dcp_rank": dcp_rank,
        "position": tripwire.position,
        "token_id": tripwire.token_id,
        "layer_ids": list(range(max_layer + 1)),
        "checkpoint_names": list(_TRIPWIRE_STAGES),
        "ffn_output_reduced": ffn_output_reduced,
        "layer0_post_attention_residual": (
            layer0_post_attention_residual.detach().cpu()
        ),
        "checkpoints": stacked,
    }
    torch.save(
        output,
        rank_dir / f"target-layer-tripwire-pos{tripwire.position}.pt",
    )
    _TARGET_LAYER_TRIPWIRE_CAPTURED = True
