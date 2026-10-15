# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    ForwardContext,
    create_forward_context,
)
from vllm.logger import init_logger
from vllm.models.deepseek_v41.decoder_replay_layers import (
    DecoderReplayLayers,
    ReplayBatch,
)
from vllm.models.deepseek_v41.nvidia.decoder_replay_cudagraph import (
    DecoderReplayCudaGraphManager,
    DecoderReplayFrontCaptureDisabled,
    DecoderReplayFrontCudaGraphManager,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


@triton.jit
def _pad_replayed_slots_kernel(
    slot_mappings_ptr,  # [num_groups, num_tokens_padded]
    group_stride,
    cacheable_groups_ptr,  # [NUM_GROUPS] indices of the prefix-cacheable groups
    query_start_loc_ptr,  # [num_reqs + 1]
    positions_ptr,  # [num_tokens]
    replay_start_ptr,  # [num_reqs]
    window,
    pad_slot_id,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(replay_start_ptr + req)
    if start == 0:
        return
    begin = tl.load(query_start_loc_ptr + req)
    end = tl.load(query_start_loc_ptr + req + 1)
    for tok in range(begin, end, BLOCK):
        offs = tok + tl.arange(0, BLOCK)
        pos = tl.load(positions_ptr + offs, mask=offs < end, other=0)
        replayed = (offs < end) & (pos >= start) & (pos < start + window)
        for i in tl.static_range(NUM_GROUPS):
            group = tl.load(cacheable_groups_ptr + i)
            tl.store(
                slot_mappings_ptr + group * group_stride + offs,
                pad_slot_id,
                mask=replayed,
            )


@triton.jit
def _gather_replay_batch_kernel(
    query_start_loc_ptr,  # [num_reqs + 1]
    dropped_before_ptr,  # [num_reqs + 1] rows trimmed before each boundary
    seq_lens_ptr,  # [num_reqs]
    replay_start_ptr,  # [num_reqs] the encoder-side replay start
    positions_ptr,
    slot_mappings_ptr,  # [num_groups, num_tokens]
    slot_mappings_stride,
    rows_ptr,  # out: [num_kept] batch row of every kept row
    kept_query_start_loc_ptr,  # out: [num_reqs + 1]
    kept_positions_ptr,  # out: [num_kept]
    kept_slot_mappings_ptr,  # out: [num_groups, num_kept]
    kept_slot_mappings_stride,
    kept_kv_start_ptr,  # out: [num_reqs]
    window,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per request: its kept rows are the last ones before its
    boundary, and land at that boundary minus the rows trimmed before it."""
    req = tl.program_id(0)
    begin = tl.load(query_start_loc_ptr + req)
    end = tl.load(query_start_loc_ptr + req + 1)
    kept_begin = begin - tl.load(dropped_before_ptr + req)
    kept_end = end - tl.load(dropped_before_ptr + req + 1)
    if req == 0:
        tl.store(kept_query_start_loc_ptr, kept_begin)
    tl.store(kept_query_start_loc_ptr + req + 1, kept_end)

    # A trimmed request holds no replay-layer window KV below its kept rows.
    kv_start = tl.load(replay_start_ptr + req)
    if kept_end - kept_begin < end - begin:
        kv_start = tl.maximum(kv_start, tl.load(seq_lens_ptr + req) - window)
    tl.store(kept_kv_start_ptr + req, kv_start)

    first_row = end - (kept_end - kept_begin)
    for tok in range(kept_begin, kept_end, BLOCK):
        offs = tok + tl.arange(0, BLOCK)
        mask = offs < kept_end
        row = first_row + (offs - kept_begin)
        tl.store(rows_ptr + offs, row.to(tl.int64), mask=mask)
        tl.store(
            kept_positions_ptr + offs,
            tl.load(positions_ptr + row, mask=mask),
            mask=mask,
        )
        for g in tl.static_range(NUM_GROUPS):
            tl.store(
                kept_slot_mappings_ptr + g * kept_slot_mappings_stride + offs,
                tl.load(slot_mappings_ptr + g * slot_mappings_stride + row, mask=mask),
                mask=mask,
            )


class ReplayAttnMetadata(ModelSpecificAttnMetadata):
    """Hands the batch's replay starts to the sliding-window builders."""

    def __init__(self, replay_start: torch.Tensor) -> None:
        self.replay_start = replay_start

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: Any, num_reqs: int
    ) -> dict[str, Any]:
        if isinstance(attn_metadata_builder, DeepseekSparseSWAMetadataBuilder):
            return {"replay_start": self.replay_start[:num_reqs]}
        return {}


@dataclass
class OrdinaryDecodeBatch:
    """Inputs and context for running the ordinary Decode prefix by itself."""

    rows: torch.Tensor
    input_batch: InputBatch
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    forward_context: ForwardContext
    num_tokens_across_dp: torch.Tensor | None


@dataclass
class RetainedReplayBatch:
    """Owned later by the harness; built from one real scheduled replay step."""

    rows: torch.Tensor
    replay_start: torch.Tensor
    input_batch: InputBatch
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    forward_context: ForwardContext
    num_tokens_across_dp: torch.Tensor | None


@dataclass
class OrdinaryDecodePlan:
    num_ordinary_reqs: int
    has_ordinary: bool
    num_ordinary_tokens: int
    num_execution_tokens: int


@dataclass
class RetainedReplayPlan:
    kept_lens: np.ndarray
    local_candidate: bool
    num_tokens: int


@dataclass
class _ReplayCensus:
    interval: int
    waves: int = 0
    graph_runs: int = 0
    rows_513_576: int = 0
    actual_rows_sum: int = 0
    descriptor_capacity_sum: int = 0
    padding_rows_sum: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    dp_max_actual_rows: Counter[str] = field(default_factory=Counter)
    selected_descriptor_rows: Counter[str] = field(default_factory=Counter)

    @staticmethod
    def _row_bin(num_rows: int) -> str:
        if num_rows <= 0:
            return "0"
        start = ((num_rows - 1) // 32) * 32 + 1
        return f"{start}-{start + 31}"

    def record(
        self,
        *,
        outcome: str,
        actual_rows: int,
        dp_max_rows: int,
        dp_size: int,
        descriptor_rows: int | None,
    ) -> None:
        graph_run = outcome == "graph"
        capacity = (
            descriptor_rows * dp_size if descriptor_rows is not None else actual_rows
        )
        self.waves += 1
        self.graph_runs += int(graph_run)
        self.rows_513_576 += int(513 <= dp_max_rows <= 576)
        self.actual_rows_sum += actual_rows
        self.descriptor_capacity_sum += capacity
        self.padding_rows_sum += capacity - actual_rows
        self.dp_max_actual_rows[self._row_bin(dp_max_rows)] += 1
        self.selected_descriptor_rows[
            str(descriptor_rows) if graph_run else "eager"
        ] += 1
        if not graph_run:
            self.fallback_reasons[outcome] += 1
        if self.waves % self.interval != 0:
            return
        payload = {
            "schema": 1,
            "interval": self.interval,
            "mixed_waves": self.waves,
            "graph_runs": self.graph_runs,
            "eager_fallbacks": self.waves - self.graph_runs,
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "dp_max_actual_rows": dict(sorted(self.dp_max_actual_rows.items())),
            "selected_descriptor_rows": dict(
                sorted(self.selected_descriptor_rows.items())
            ),
            "dp_max_rows_513_576": self.rows_513_576,
            "actual_rows_sum": self.actual_rows_sum,
            "descriptor_capacity_sum": self.descriptor_capacity_sum,
            "padding_rows_sum": self.padding_rows_sum,
        }
        logger.info(
            "DSV41_REPLAY_CENSUS %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window and SWA bounded replay.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.

    SWA bounded replay (``CacheConfig.swa_bounded_replay``) keeps the
    sliding-window KV out of prefix caching and rebuilds it after a prefix hit
    by recomputing the hit's last window; the scheduler tells each request the
    position from which it holds window KV (``NewRequestData.replay_start``).
    ``prepare_attn`` gathers those per batch, pads the replayed tokens' slots
    in the prefix-cacheable groups so the cached KV stays as is, and hands the
    starts to the sliding-window metadata builders, whose kernels read no
    window KV below them.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            self.lookback_token_ids = torch.full(
                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

        # Per request state index; batches gather from it (see prepare_attn).
        self._replay_start_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self._replay_start = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._replay_start_staging = UvaBufferPool(self.max_num_reqs, torch.int32)
        self._global_replay_start = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._global_replay_start_staging = UvaBufferPool(
            self.max_num_reqs, torch.int32
        )
        self.pcp_world_size = getattr(
            vllm_config.parallel_config, "prefill_context_parallel_size", 1
        )
        # The replay window and the indices of the prefix-cacheable groups, from
        # the KV cache config on first use; None until then.
        self._replay: tuple[int, torch.Tensor] | None = None
        self._replay_kernels_warmed = False
        self._mixed_replay_graph_enabled = (
            os.environ.get("VLLM_DSV41_MIXED_REPLAY_GRAPH", "0") == "1"
        )
        self._mixed_replay_front_graph_enabled = (
            os.environ.get("VLLM_DSV41_MIXED_FRONT_GRAPH", "0") == "1"
        )
        if (
            self._mixed_replay_front_graph_enabled
            and not self._mixed_replay_graph_enabled
        ):
            raise ValueError(
                "VLLM_DSV41_MIXED_FRONT_GRAPH requires VLLM_DSV41_MIXED_REPLAY_GRAPH=1"
            )
        census_value = os.environ.get("VLLM_DSV41_REPLAY_CENSUS_INTERVAL", "0")
        try:
            census_interval = int(census_value)
        except ValueError as exc:
            raise ValueError(
                "VLLM_DSV41_REPLAY_CENSUS_INTERVAL must be an integer"
            ) from exc
        if census_interval < 0:
            raise ValueError("VLLM_DSV41_REPLAY_CENSUS_INTERVAL must not be negative")
        if census_interval and not self._mixed_replay_graph_enabled:
            raise ValueError(
                "VLLM_DSV41_REPLAY_CENSUS_INTERVAL requires "
                "VLLM_DSV41_MIXED_REPLAY_GRAPH=1"
            )
        parallel_config = vllm_config.parallel_config
        self._replay_census = (
            _ReplayCensus(census_interval)
            if census_interval and parallel_config.data_parallel_rank == 0
            else None
        )
        if self._replay_census is not None:
            logger.info(
                "DeepSeek V4.1 replay census enabled every %d mixed waves",
                census_interval,
            )

        self.decoder_replay_layers: DecoderReplayLayers | None = getattr(
            model, "decoder_replay_layers", None
        )
        if self.decoder_replay_layers is not None:
            # Requests whose every prompt row is read (prompt logprobs) never trim.
            self._keeps_rows_np = np.zeros(self.max_num_reqs, dtype=np.bool_)
            self._dropped_before = UvaBufferPool(self.max_num_reqs + 1, torch.int32)
            # The replay batch's inputs: its rows of the batch, query offsets,
            # positions, slot mappings (sized with the KV cache groups on first
            # use) and the position from which each request holds replay-layer
            # window KV.
            self._kept_rows = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self._kept_query_start_loc = torch.zeros(
                self.max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self._kept_positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self._kept_is_padding = torch.zeros(
                self.max_num_tokens, dtype=torch.bool, device=device
            )
            self._kept_slot_mappings: torch.Tensor | None = None
            self._kept_kv_start = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            # The replay layers' and the first one's SWA attention groups.
            self._replay_attn_groups: (
                tuple[list[list[AttentionGroup]], list[list[AttentionGroup]]] | None
            ) = None
            self._retained_replay_attn_groups: (
                tuple[list[list[AttentionGroup]], list[list[AttentionGroup]]] | None
            ) = None
            # Harness-only buffers are allocated lazily on explicit use.
            self._ordinary_input_buffers: InputBuffers | None = None
            self._ordinary_rows: torch.Tensor | None = None
            self._ordinary_slot_mappings: torch.Tensor | None = None
            self._ordinary_replay_start: torch.Tensor | None = None
            self._ordinary_attn_groups: list[list[AttentionGroup]] | None = None
            if self._mixed_replay_front_graph_enabled:
                self._front_attn_groups: list[list[AttentionGroup]] | None = None
                self._front_block_tables: tuple[torch.Tensor, ...] | None = None
                self._front_slot_mappings: torch.Tensor | None = None
                self._front_replay_start = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32, device=device
                )
                self._front_idx_mapping = torch.arange(
                    self.max_num_reqs, dtype=torch.int64, device=device
                )
                self._front_idx_mapping_np = np.arange(self.max_num_reqs, dtype=np.intp)
                self._front_query_start_loc_np = np.zeros(
                    self.max_num_reqs + 1, dtype=np.int32
                )
                self._front_num_scheduled_tokens = np.zeros(
                    self.max_num_reqs, dtype=np.int32
                )
                self._front_is_prefilling_np = np.ones(
                    self.max_num_reqs, dtype=np.bool_
                )
                self._front_seq_lens_cpu = torch.zeros(
                    self.max_num_reqs, dtype=torch.int32
                )
        layers = self.decoder_replay_layers
        compilation_config = vllm_config.compilation_config
        self.replay_cudagraphs: DecoderReplayCudaGraphManager | None = None
        self.replay_front_cudagraph: DecoderReplayFrontCudaGraphManager | None = None
        self._decoder_model: nn.Module | None = None
        if (
            layers is not None
            and compilation_config.cudagraph_decoder_replay
            and is_breakable_cudagraph_enabled()
            and get_pp_group().is_first_rank
            # The graphs' fixed inputs hold no LoRA mapping and no None input ids.
            and vllm_config.lora_config is None
            and not vllm_config.model_config.enable_prompt_embeds
        ):
            threshold = compilation_config.decoder_replay_trim_threshold
            if threshold <= layers.window:
                raise ValueError(
                    "decoder_replay_trim_threshold must exceed the replay window "
                    f"of {layers.window} tokens: smaller graphs have nothing to trim"
                )
            self.replay_cudagraphs = DecoderReplayCudaGraphManager(
                vllm_config, device, layers
            )
            if compilation_config.cudagraph_mode.has_piecewise_cudagraphs():
                layers.trim_threshold = threshold
        if self._mixed_replay_graph_enabled:
            blockers = []
            if compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                blockers.append("cudagraph_mode is not FULL_DECODE_ONLY")
            if parallel_config.tensor_parallel_size != 1:
                blockers.append("tensor_parallel_size is not 1")
            if parallel_config.pipeline_parallel_size != 1:
                blockers.append("pipeline_parallel_size is not 1")
            if self.replay_cudagraphs is None:
                blockers.append("dedicated replay CUDA graphs are unavailable")
            if blockers:
                raise ValueError(
                    "VLLM_DSV41_MIXED_REPLAY_GRAPH cannot be enabled: "
                    + "; ".join(blockers)
                )
        if self._mixed_replay_front_graph_enabled:
            language_model = getattr(model, "language_model", model)
            decoder_model = getattr(language_model, "model", None)
            blockers = []
            if decoder_model is None:
                blockers.append("decoder language model is unavailable")
            elif not hasattr(decoder_model, "_run_decoder_replay_front_graph"):
                blockers.append("decoder front boundary is unavailable")
            if self.decoder_replay_layers is None:
                blockers.append("decoder replay layers are unavailable")
            if self.replay_cudagraphs is None:
                blockers.append("dedicated replay CUDA graphs are unavailable")
            if self.max_num_reqs < 2:
                blockers.append("max_num_seqs must leave one padding request")
            if blockers:
                raise ValueError(
                    "VLLM_DSV41_MIXED_FRONT_GRAPH cannot be enabled: "
                    + "; ".join(blockers)
                )
            assert decoder_model is not None
            self._decoder_model = decoder_model
            self.replay_front_cudagraph = DecoderReplayFrontCudaGraphManager(
                vllm_config,
                device,
                decoder_model._run_decoder_replay_front_graph,
                use_inputs_embeds=(
                    self.supports_mm_inputs
                    or vllm_config.model_config.enable_prompt_embeds
                ),
                inputs_embeds_size=self.inputs_embeds_size,
                lookback_depth=depth,
                num_aux_outputs=len(
                    decoder_model._decoder_replay_front_aux_layer_ids()
                ),
                aux_hidden_size=vllm_config.model_config.hf_config.hidden_size,
            )
            logger.info("DeepSeek V4.1 mixed replay front graph enabled for 224 rows")

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self._replay_start_np[req_index] = new_req_data.replay_start
        if self.decoder_replay_layers is not None:
            params = new_req_data.sampling_params
            self._keeps_rows_np[req_index] = (
                params is not None and params.prompt_logprobs is not None
            )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            req_states.num_computed_tokens.gpu,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs

    def get_cudagraph_capture_attn_metadata(
        self, num_reqs: int
    ) -> ModelSpecificAttnMetadata:
        replay_start = self._replay_start[:num_reqs].zero_()
        return ReplayAttnMetadata(replay_start)

    def _get_replay_config(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[int, torch.Tensor]:
        if self._replay is None:
            specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
            self._replay = (
                max(spec.prefix_replay_tokens for spec in specs),
                torch.tensor(
                    [i for i, spec in enumerate(specs) if spec.prefix_cacheable],
                    dtype=torch.int32,
                    device=self.device,
                ),
            )
        return self._replay

    def preprocess_pcp_slot_mappings(
        self,
        global_input_batch: InputBatch,
        global_slot_mappings: torch.Tensor,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        window, cacheable_groups = self._get_replay_config(kv_cache_config)
        if not window:
            return

        num_reqs = global_input_batch.num_reqs
        replay_start_np = np.where(
            global_input_batch.is_prefilling_np[:num_reqs],
            self._replay_start_np[global_input_batch.idx_mapping_np[:num_reqs]],
            0,
        ).astype(np.int32)
        if not replay_start_np.any():
            return

        replay_start = self._global_replay_start_staging.copy_to_gpu(
            replay_start_np,
            out=self._global_replay_start[:num_reqs],
        )
        _pad_replayed_slots_kernel[(num_reqs,)](
            global_slot_mappings,
            global_slot_mappings.stride(0),
            cacheable_groups,
            global_input_batch.query_start_loc,
            global_input_batch.positions,
            replay_start,
            window,
            PAD_SLOT_ID,
            NUM_GROUPS=cacheable_groups.numel(),
            BLOCK=1024,
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        window, cacheable_groups = self._get_replay_config(kv_cache_config)
        if not self._replay_kernels_warmed:
            if self.decoder_replay_layers is not None:
                self._kept_slot_mappings = torch.zeros(
                    len(kv_cache_config.kv_cache_groups),
                    self.max_num_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )
            self._warm_up_replay_kernels(input_batch, slot_mappings)
            self._replay_kernels_warmed = True
        replay_start: torch.Tensor | None = None
        if window:
            num_reqs = input_batch.num_reqs
            # Decode rows sit above the hit, so only prefills carry a replay
            # start; dummy batches (captures, profiling) carry none.
            replay_start_np = np.where(
                input_batch.is_prefilling_np[:num_reqs],
                self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
                0,
            ).astype(np.int32)
            replay_start = self._replay_start_staging.copy_to_gpu(
                replay_start_np, out=self._replay_start[:num_reqs]
            )
            if replay_start_np.any() and self.pcp_world_size == 1:
                # The replayed tokens rebuild window KV only: their slots in the
                # prefix-cacheable groups are padded so the cached KV stays as is.
                _pad_replayed_slots_kernel[(num_reqs,)](
                    slot_mappings,
                    slot_mappings.stride(0),
                    cacheable_groups,
                    input_batch.query_start_loc,
                    input_batch.positions,
                    replay_start,
                    window,
                    PAD_SLOT_ID,
                    NUM_GROUPS=cacheable_groups.numel(),
                    BLOCK=1024,
                )
            assert model_specific_attn_metadata is None
            model_specific_attn_metadata = ReplayAttnMetadata(replay_start)
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        if self.decoder_replay_layers is not None:
            assert replay_start is not None
            self._prepare_replay_batch(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                replay_start,
            )
        return attn_metadata

    def _warm_up_replay_kernels(
        self, input_batch: InputBatch, slot_mappings: torch.Tensor
    ) -> None:
        """Compile the replay kernels on the buffers they run on: the startup
        dummy batches never replay or trim, so they would not launch them."""
        assert self._replay is not None
        window, cacheable_groups = self._replay
        if not window:
            return
        _pad_replayed_slots_kernel.warmup(
            slot_mappings,
            slot_mappings.stride(0),
            cacheable_groups,
            input_batch.query_start_loc,
            input_batch.positions,
            self._replay_start,
            window,
            PAD_SLOT_ID,
            NUM_GROUPS=cacheable_groups.numel(),
            BLOCK=1024,
            grid=(1,),
        )
        if self.decoder_replay_layers is None:
            return
        assert self._kept_slot_mappings is not None
        _gather_replay_batch_kernel.warmup(
            input_batch.query_start_loc,
            self._dropped_before.copy_to_uva(np.zeros(1, dtype=np.int32)),
            input_batch.seq_lens,
            self._replay_start,
            input_batch.positions,
            slot_mappings,
            slot_mappings.stride(0),
            self._kept_rows,
            self._kept_query_start_loc,
            self._kept_positions,
            self._kept_slot_mappings,
            self._kept_slot_mappings.stride(0),
            self._kept_kv_start,
            self.decoder_replay_layers.window,
            NUM_GROUPS=slot_mappings.shape[0],
            BLOCK=1024,
            grid=(1,),
        )

    def _prepare_replay_batch(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        replay_start: torch.Tensor,
        capture_desc: BatchExecutionDescriptor | None = None,
    ) -> None:
        """Set the replay layers' batch for this forward, or none. Eager steps trim;
        captures and PIECEWISE graphs at the trim threshold or above break out to a
        replay batch, trimmed or not; FULL and smaller PIECEWISE graphs keep the
        whole batch. Under DP every rank replays if any does."""
        layers = self.decoder_replay_layers
        assert layers is not None
        layers.replay_batch = None
        self._clear_replay_front_graph()
        force_replay_batch = capture_desc is not None or (
            cudagraph_mode == CUDAGraphMode.PIECEWISE
            and layers.graph_uses_replay_batch(input_batch.num_tokens_after_padding)
        )
        mixed_graph_coordination = (
            self._mixed_replay_graph_enabled
            and capture_desc is None
            and cudagraph_mode == CUDAGraphMode.NONE
        )
        if (
            not mixed_graph_coordination
            and cudagraph_mode != CUDAGraphMode.NONE
            and not force_replay_batch
        ):
            return

        num_reqs = input_batch.num_reqs
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        kept_lens = np.where(
            self._keeps_rows_np[input_batch.idx_mapping_np[:num_reqs]],
            query_lens,
            np.minimum(query_lens, layers.window),
        )
        # A batch of context-free non-prefill rows is a dummy (an idle DP rank's):
        # nothing reads it and its idx_mapping hits stale slots, so keep one row.
        seq_lens = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        is_dummy = not input_batch.is_prefilling_np[:num_reqs].any() and bool(
            (seq_lens == query_lens).all()
        )
        if is_dummy and capture_desc is None:
            kept_lens = np.zeros_like(kept_lens)
            kept_lens[0] = 1
        # Dummy batches (captures, an idle DP rank's) are not prefills: they
        # keep their rows unless a rank trims.
        needs_replay_batch = force_replay_batch or bool(
            (kept_lens < query_lens)[input_batch.is_prefilling_np[:num_reqs]].any()
        )
        num_tokens = int(kept_lens.sum())
        dp_tokens = None
        front_dp_tokens = None
        force_front_graph = False
        force_mixed_graph = (
            mixed_graph_coordination and self._is_mixed_replay_candidate(input_batch)
        )
        mixed_graph_eligible = (
            mixed_graph_coordination and self.replay_cudagraphs is not None
        )
        front_eligible, front_contexts = self._front_graph_shape(input_batch)
        front_num_tokens = input_batch.num_tokens_after_padding
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            if mixed_graph_coordination:
                (
                    needs_replay_batch,
                    force_mixed_graph,
                    dp_tokens,
                    force_front_graph,
                    front_dp_tokens,
                ) = self._agree_mixed_replay_across_dp(
                    needs_replay_batch,
                    force_mixed_graph,
                    mixed_graph_eligible,
                    num_tokens,
                    front_eligible,
                    front_num_tokens,
                    front_contexts,
                )
            else:
                needs_replay_batch, dp_tokens = self._agree_across_dp(
                    needs_replay_batch, num_tokens
                )
        elif force_mixed_graph:
            if not mixed_graph_eligible and self._replay_census is not None:
                self._record_replay_census(
                    outcome="peer_ineligible",
                    actual_rows=num_tokens,
                    dp_max_rows=num_tokens,
                    descriptor_rows=None,
                )
            force_mixed_graph = mixed_graph_eligible
            needs_replay_batch = needs_replay_batch or force_mixed_graph
            force_front_graph = (
                force_mixed_graph
                and front_eligible
                and front_num_tokens == DecoderReplayFrontCudaGraphManager.NUM_TOKENS
                and front_contexts == 1
            )
            if force_front_graph:
                front_dp_tokens = torch.tensor([front_num_tokens], dtype=torch.int32)
        if not needs_replay_batch:
            return

        cudagraphs, run_graph = self.replay_cudagraphs, None
        num_padded, cg_mode = num_tokens, CUDAGraphMode.NONE
        desc: BatchExecutionDescriptor | None = None
        dp_max_tokens = num_tokens if dp_tokens is None else int(dp_tokens.max())
        census_actual_rows = 0
        if force_mixed_graph and self._replay_census is not None:
            census_actual_rows = (
                num_tokens if dp_tokens is None else int(dp_tokens.sum())
            )
        if cudagraphs is not None:
            # Every rank runs the graph that fits the largest rank's batch.
            desc = capture_desc or cudagraphs.dispatch(num_reqs, dp_max_tokens, None, 0)
            if desc.cg_mode == CUDAGraphMode.PIECEWISE and (
                not force_mixed_graph
                or self._replay_graph_has_segments(cudagraphs, desc)
            ):
                num_padded, cg_mode = desc.num_tokens, desc.cg_mode
                run_graph = cudagraphs.run
                if dp_tokens is not None:
                    dp_tokens.fill_(num_padded)
        if force_mixed_graph:
            local_graph_ready = run_graph is not None
            front_context = None
            local_front_ready = not force_front_graph
            if force_front_graph:
                assert front_dp_tokens is not None
                manager = self.replay_front_cudagraph
                try:
                    if manager is not None and manager.has_graph():
                        front_context = self._prepare_replay_front_batch(
                            input_batch,
                            block_tables,
                            slot_mappings,
                            attn_groups,
                            kv_cache_config,
                            replay_start,
                            front_dp_tokens,
                        )
                except Exception:
                    logger.exception(
                        "Failed to prepare DeepSeek V4.1 replay front graph"
                    )
                local_front_ready = front_context is not None
            graph_ready, front_ready = local_graph_ready, local_front_ready
            if self.vllm_config.parallel_config.data_parallel_size > 1:
                graph_ready, front_ready = self._all_dp_flags(graph_ready, front_ready)
            if not graph_ready:
                if self._replay_census is not None:
                    assert cudagraphs is not None
                    self._record_replay_census(
                        outcome=self._replay_graph_fallback_reason(
                            cudagraphs,
                            desc,
                            dp_max_tokens,
                            local_graph_ready,
                        ),
                        actual_rows=census_actual_rows,
                        dp_max_rows=dp_max_tokens,
                        descriptor_rows=None,
                    )
                return
            if force_front_graph and front_ready:
                assert front_context is not None
                manager = self.replay_front_cudagraph
                decoder_model = self._decoder_model
                assert manager is not None and decoder_model is not None
                manager.prepare_runtime(front_context, front_num_tokens)
                decoder_model.decoder_replay_front_graph = manager.run
            assert desc is not None
            if self._replay_census is not None:
                self._record_replay_census(
                    outcome="graph",
                    actual_rows=census_actual_rows,
                    dp_max_rows=dp_max_tokens,
                    descriptor_rows=desc.num_tokens,
                )
        dp_metadata = None
        if dp_tokens is not None:
            dp_metadata = DPMetadata.make(
                self.vllm_config.parallel_config, num_padded, dp_tokens
            )
        replay_is_padding = self._kept_is_padding[:num_padded]
        replay_is_padding[:num_tokens].fill_(False)
        replay_is_padding[num_tokens:].fill_(True)
        kept_batch, kept_slot_mappings = self._kept_input_batch(
            input_batch, slot_mappings, replay_start, kept_lens, num_padded
        )
        if self._replay_attn_groups is None:
            self._replay_attn_groups = (
                self._copy_attn_groups(attn_groups, layers.metadata_prefixes),
                self._copy_attn_groups(attn_groups, {layers.first_swa_prefix}),
            )
        replay_groups, first_swa_groups = self._replay_attn_groups
        attn_metadata = super().prepare_attn(
            kept_batch,
            CUDAGraphMode.NONE,
            block_tables,
            kept_slot_mappings,
            replay_groups,
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(
                self._kept_kv_start[:num_reqs]
            ),
        )
        # The first replay layer wrote every row's window KV: its window starts
        # are the batch's.
        first_swa = layers.first_swa_prefix
        attn_metadata[first_swa] = super().prepare_attn(
            kept_batch,
            CUDAGraphMode.NONE,
            block_tables,
            kept_slot_mappings,
            first_swa_groups,
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(replay_start),
        )[first_swa]
        layers.replay_batch = ReplayBatch(
            self._kept_rows[:num_tokens],
            create_forward_context(
                attn_metadata,
                self.vllm_config,
                dp_metadata=dp_metadata,
                cudagraph_runtime_mode=cg_mode,
                batch_descriptor=BatchDescriptor(num_tokens=num_padded),
                slot_mapping=build_slot_mappings_by_layer(
                    kept_slot_mappings, kv_cache_config
                ),
                is_padding=replay_is_padding,
            ),
            run_graph,
        )

    def capture_inner_cudagraphs(
        self,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        def prepare(desc: BatchExecutionDescriptor) -> None:
            # Capture reuses the runtime path on a dummy batch, so the graph reads
            # the buffers a real step fills.
            num_reqs = min(desc.num_tokens, self.max_num_reqs)
            self._prepare_replay_batch(
                InputBatch.make_dummy(num_reqs, desc.num_tokens, input_buffers),
                CUDAGraphMode.NONE,
                block_tables.get_dummy_block_tables(num_reqs),
                block_tables.get_dummy_slot_mappings(desc.num_tokens),
                attn_groups,
                kv_cache_config,
                self._replay_start[:num_reqs].zero_(),
                capture_desc=desc,
            )

        if self.replay_cudagraphs is not None:
            # profile_only captures use pre-KV-cache groups: drop the cache.
            self._replay_attn_groups = None
            self.replay_cudagraphs.capture_replay_graphs(prepare)
        if self.replay_front_cudagraph is not None:
            self._front_attn_groups = None
            self._front_block_tables = None
            self._front_slot_mappings = None

            def prepare_front(
                desc: BatchExecutionDescriptor,
            ) -> ForwardContext:
                if (
                    desc.cg_mode != CUDAGraphMode.PIECEWISE
                    or desc.num_tokens != DecoderReplayFrontCudaGraphManager.NUM_TOKENS
                ):
                    raise RuntimeError(
                        "decoder replay front graph received an invalid descriptor"
                    )
                self.replay_front_cudagraph.prepare_capture_buffers()
                capture_buffers = InputBuffers(
                    self.max_num_reqs,
                    DecoderReplayFrontCudaGraphManager.NUM_TOKENS,
                    self.device,
                )
                num_decode_reqs = min(16, self.max_num_reqs - 2)
                generation_rows = np.zeros(num_decode_reqs, dtype=np.int32)
                if num_decode_reqs:
                    generation_rows.fill(96 // num_decode_reqs)
                    generation_rows[: 96 % num_decode_reqs] += 1
                query_lens = np.concatenate(
                    (generation_rows, np.array([128], dtype=np.int32))
                )
                capture_reqs = query_lens.size
                capture_batch = InputBatch.make_dummy(
                    capture_reqs,
                    DecoderReplayFrontCudaGraphManager.NUM_TOKENS,
                    capture_buffers,
                )
                query_start_loc_np = np.zeros(capture_reqs + 1, dtype=np.int32)
                np.cumsum(query_lens, out=query_start_loc_np[1:])
                seq_lens = np.concatenate(
                    (generation_rows + 512, np.array([128], dtype=np.int32))
                )
                capture_batch = replace(
                    capture_batch,
                    num_scheduled_tokens=query_lens,
                    query_start_loc=torch.tensor(
                        query_start_loc_np,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    query_start_loc_np=query_start_loc_np,
                    seq_lens=torch.tensor(
                        seq_lens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    seq_lens_cpu_upper_bound=torch.tensor(seq_lens),
                    is_prefilling_np=np.array(
                        [False] * num_decode_reqs + [True], dtype=np.bool_
                    ),
                    has_prefill=True,
                    max_query_len=128,
                )
                front_dp_tokens = torch.full(
                    (self.vllm_config.parallel_config.data_parallel_size,),
                    DecoderReplayFrontCudaGraphManager.NUM_TOKENS,
                    dtype=torch.int32,
                )
                return self._prepare_replay_front_batch(
                    capture_batch,
                    block_tables.get_dummy_block_tables(capture_reqs),
                    block_tables.get_dummy_slot_mappings(
                        DecoderReplayFrontCudaGraphManager.NUM_TOKENS
                    ),
                    attn_groups,
                    kv_cache_config,
                    self._front_replay_start[:capture_reqs].zero_(),
                    front_dp_tokens,
                )

            def agree_front_capture(local_ready: bool) -> bool:
                (ready,) = self._all_dp_flags(local_ready)
                return ready

            try:
                self.replay_front_cudagraph.capture_front_graphs(
                    prepare_front,
                    agree_front_capture,
                )
            except DecoderReplayFrontCaptureDisabled:
                self._mixed_replay_front_graph_enabled = False
                self.replay_front_cudagraph.clear_runtime()
                self.replay_front_cudagraph = None
                if self._decoder_model is not None:
                    self._decoder_model.decoder_replay_front_graph = None
                logger.warning(
                    "DeepSeek V4.1 replay front graph disabled; "
                    "retaining eager prefix and dedicated replay suffix"
                )

    def plan_ordinary_decode_batch(
        self,
        input_batch: InputBatch,
    ) -> OrdinaryDecodePlan:
        """Validate and size an ordinary-only batch without any collective."""
        num_reqs = input_batch.num_reqs
        is_prefilling = input_batch.is_prefilling_np[:num_reqs]
        prefill_indices = np.flatnonzero(is_prefilling)
        num_ordinary_reqs = (
            int(prefill_indices[0]) if prefill_indices.size else num_reqs
        )
        if not is_prefilling[num_ordinary_reqs:].all():
            raise ValueError(
                "ordinary Decode requests must precede every Prefill request"
            )
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        seq_lens = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        is_idle = not is_prefilling.any() and bool((seq_lens == query_lens).all())
        has_ordinary = num_ordinary_reqs > 0 and not is_idle
        num_ordinary_tokens = (
            int(input_batch.query_start_loc_np[num_ordinary_reqs])
            if has_ordinary
            else 0
        )
        return OrdinaryDecodePlan(
            num_ordinary_reqs=num_ordinary_reqs,
            has_ordinary=has_ordinary,
            num_ordinary_tokens=num_ordinary_tokens,
            num_execution_tokens=num_ordinary_tokens if has_ordinary else 1,
        )

    def build_ordinary_decode_batch(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        *,
        pad_to_dp_max: bool = False,
        plan: OrdinaryDecodePlan | None = None,
        global_has_ordinary: bool | None = None,
        agreed_num_tokens_across_dp: torch.Tensor | None = None,
    ) -> OrdinaryDecodeBatch | None:
        """Build an independent L0-to-boundary batch for ordinary Decode rows."""
        assert self.decoder_replay_layers is not None
        plan = plan or self.plan_ordinary_decode_batch(input_batch)
        num_reqs = input_batch.num_reqs
        is_prefilling = input_batch.is_prefilling_np[:num_reqs]
        num_ordinary_reqs = plan.num_ordinary_reqs
        has_ordinary = plan.has_ordinary
        num_ordinary_tokens = plan.num_ordinary_tokens
        num_execution_tokens = plan.num_execution_tokens

        num_tokens_across_dp = agreed_num_tokens_across_dp
        if global_has_ordinary is None and (
            self.vllm_config.parallel_config.data_parallel_size > 1
        ):
            any_ordinary, num_tokens_across_dp = self._agree_across_dp(
                has_ordinary, num_execution_tokens
            )
            if not any_ordinary:
                return None
        elif global_has_ordinary is None:
            if not has_ordinary:
                return None
        elif not global_has_ordinary:
            return None

        actual_num_tokens_across_dp = (
            None if num_tokens_across_dp is None else num_tokens_across_dp.clone()
        )
        num_padded = num_execution_tokens
        context_dp_tokens = num_tokens_across_dp
        if pad_to_dp_max and num_tokens_across_dp is not None:
            num_padded = int(num_tokens_across_dp.max())
            context_dp_tokens = num_tokens_across_dp.clone().fill_(num_padded)

        self._ensure_ordinary_buffers(slot_mappings.shape[0])
        buffers = self._ordinary_input_buffers
        rows_buffer = self._ordinary_rows
        ordinary_slot_mappings_buffer = self._ordinary_slot_mappings
        replay_start = self._ordinary_replay_start
        assert buffers is not None
        assert rows_buffer is not None
        assert ordinary_slot_mappings_buffer is not None
        assert replay_start is not None
        assert num_padded <= self.max_num_tokens

        ordinary_num_reqs = num_ordinary_reqs if has_ordinary else 1
        query_start_loc = buffers.query_start_loc[: ordinary_num_reqs + 1]
        input_ids = buffers.input_ids[:num_padded].zero_()
        positions = buffers.positions[:num_padded].zero_()
        is_padding = buffers.is_padding[:num_padded].fill_(True)
        ordinary_slot_mappings = ordinary_slot_mappings_buffer[:, :num_padded]
        ordinary_slot_mappings.fill_(PAD_SLOT_ID)

        if has_ordinary:
            query_start_loc.copy_(input_batch.query_start_loc[: num_ordinary_reqs + 1])
            input_ids[:num_ordinary_tokens].copy_(
                input_batch.input_ids[:num_ordinary_tokens]
            )
            positions[:num_ordinary_tokens].copy_(
                input_batch.positions[:num_ordinary_tokens]
            )
            is_padding[:num_ordinary_tokens].fill_(False)
            ordinary_slot_mappings[:, :num_ordinary_tokens].copy_(
                slot_mappings[:, :num_ordinary_tokens]
            )
            query_start_loc_np = input_batch.query_start_loc_np[
                : num_ordinary_reqs + 1
            ].copy()
            num_scheduled_tokens = input_batch.num_scheduled_tokens[:num_ordinary_reqs]
            ordinary_seq_lens = input_batch.seq_lens[:num_ordinary_reqs]
            ordinary_seq_lens_cpu = input_batch.seq_lens_cpu_upper_bound[
                :num_ordinary_reqs
            ]
            ordinary_is_prefilling = is_prefilling[:num_ordinary_reqs]
            max_query_len = int(num_scheduled_tokens.max())
        else:
            query_start_loc.zero_()
            query_start_loc[1] = 1
            query_start_loc_np = np.array([0, 1], dtype=np.int32)
            num_scheduled_tokens = np.ones(1, dtype=np.int32)
            ordinary_seq_lens = buffers.seq_lens[:1].fill_(1)
            ordinary_seq_lens_cpu = torch.ones(1, dtype=torch.int32)
            ordinary_is_prefilling = np.zeros(1, dtype=np.bool_)
            max_query_len = 1

        request_slice = slice(0, ordinary_num_reqs)
        ordinary_block_tables = (
            tuple(table[:ordinary_num_reqs] for table in block_tables)
            if has_ordinary
            else tuple(table[:1].new_zeros(table[:1].shape) for table in block_tables)
        )
        ordinary_input_batch = replace(
            input_batch,
            req_ids=input_batch.req_ids[request_slice],
            num_reqs=ordinary_num_reqs,
            num_reqs_after_padding=ordinary_num_reqs,
            idx_mapping=input_batch.idx_mapping[request_slice],
            idx_mapping_np=input_batch.idx_mapping_np[request_slice],
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_execution_tokens,
            num_tokens_after_padding=num_padded,
            num_draft_tokens_per_req=(
                None
                if input_batch.num_draft_tokens_per_req is None
                else input_batch.num_draft_tokens_per_req[request_slice]
            ),
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=ordinary_seq_lens,
            seq_lens_cpu_upper_bound=ordinary_seq_lens_cpu,
            dcp_local_seq_lens=(
                None
                if input_batch.dcp_local_seq_lens is None
                else input_batch.dcp_local_seq_lens[request_slice]
            ),
            num_computed_tokens_np=input_batch.num_computed_tokens_np[request_slice],
            prefill_len_np=input_batch.prefill_len_np[request_slice],
            num_computed_prefill_tokens_np=input_batch.num_computed_prefill_tokens_np[
                request_slice
            ],
            is_prefilling_np=ordinary_is_prefilling,
            has_prefill=False,
            input_ids=input_ids,
            positions=positions,
            is_padding=is_padding,
            prompt_lens=(
                None
                if input_batch.prompt_lens is None
                else input_batch.prompt_lens[request_slice]
            ),
            max_query_len=max_query_len,
            fast_prefill=None,
        )

        if self._ordinary_attn_groups is None:
            self._ordinary_attn_groups = self._copy_attn_groups(attn_groups, None)
        attn_metadata = super().prepare_attn(
            ordinary_input_batch,
            CUDAGraphMode.NONE,
            ordinary_block_tables,
            ordinary_slot_mappings,
            self._ordinary_attn_groups,
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(
                replay_start[:ordinary_num_reqs]
            ),
        )
        dp_metadata = None
        if context_dp_tokens is not None:
            dp_metadata = DPMetadata.make(
                self.vllm_config.parallel_config, num_padded, context_dp_tokens
            )
        return OrdinaryDecodeBatch(
            rows=rows_buffer[:num_ordinary_tokens],
            input_batch=ordinary_input_batch,
            block_tables=ordinary_block_tables,
            slot_mappings=ordinary_slot_mappings,
            forward_context=create_forward_context(
                attn_metadata,
                self.vllm_config,
                dp_metadata=dp_metadata,
                batch_descriptor=BatchDescriptor(num_tokens=num_padded),
                slot_mapping=build_slot_mappings_by_layer(
                    ordinary_slot_mappings, kv_cache_config
                ),
                is_padding=is_padding,
            ),
            num_tokens_across_dp=actual_num_tokens_across_dp,
        )

    def plan_retained_replay_batch(
        self,
        input_batch: InputBatch,
    ) -> RetainedReplayPlan:
        """Validate and size a retained suffix batch without any collective."""
        if self.decoder_replay_layers is None:
            raise RuntimeError("decoder replay layers are unavailable")
        layers = self.decoder_replay_layers
        num_reqs = input_batch.num_reqs
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        kept_lens = np.where(
            self._keeps_rows_np[input_batch.idx_mapping_np[:num_reqs]],
            query_lens,
            np.minimum(query_lens, layers.window),
        )
        is_prefilling = input_batch.is_prefilling_np[:num_reqs]
        seq_lens = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        is_idle = not is_prefilling.any() and bool((seq_lens == query_lens).all())
        if is_idle:
            kept_lens = np.zeros_like(kept_lens)
            kept_lens[0] = 1
        replay_starts = self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]]
        local_candidate = bool(
            is_prefilling.any()
            and not is_prefilling.all()
            and (is_prefilling & (replay_starts > 0)).any()
        )
        return RetainedReplayPlan(
            kept_lens=kept_lens,
            local_candidate=local_candidate,
            num_tokens=int(kept_lens.sum()),
        )

    def is_retained_replay_candidate(self, input_batch: InputBatch) -> bool:
        """Whether this real step has ordinary Decode plus resumed Prefill work."""
        if self.decoder_replay_layers is None:
            return False
        return self.plan_retained_replay_batch(input_batch).local_candidate

    def build_retained_replay_batch(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        *,
        plan: RetainedReplayPlan | None = None,
        global_has_candidate: bool | None = None,
        agreed_num_tokens_across_dp: torch.Tensor | None = None,
    ) -> RetainedReplayBatch | None:
        """Build an independent eager suffix context without installing it live."""
        layers = self.decoder_replay_layers
        assert layers is not None
        plan = plan or self.plan_retained_replay_batch(input_batch)
        num_reqs = input_batch.num_reqs
        is_prefilling = input_batch.is_prefilling_np[:num_reqs]
        kept_lens = plan.kept_lens
        num_tokens = plan.num_tokens
        num_tokens_across_dp = agreed_num_tokens_across_dp
        if global_has_candidate is None and (
            self.vllm_config.parallel_config.data_parallel_size > 1
        ):
            any_candidate, num_tokens_across_dp = self._agree_across_dp(
                plan.local_candidate, num_tokens
            )
            if not any_candidate:
                return None
        elif global_has_candidate is None:
            if not plan.local_candidate:
                return None
        elif not global_has_candidate:
            return None

        replay_start_np = np.where(
            is_prefilling,
            self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
            0,
        ).astype(np.int32)
        replay_start = self._replay_start_staging.copy_to_gpu(
            replay_start_np, out=self._replay_start[:num_reqs]
        )
        kept_batch, kept_slot_mappings = self._kept_input_batch(
            input_batch,
            slot_mappings,
            replay_start,
            kept_lens,
            num_tokens,
        )
        if self._retained_replay_attn_groups is None:
            self._retained_replay_attn_groups = (
                self._copy_attn_groups(attn_groups, layers.metadata_prefixes),
                self._copy_attn_groups(attn_groups, {layers.first_swa_prefix}),
            )
        replay_groups, first_swa_groups = self._retained_replay_attn_groups
        attn_metadata = super().prepare_attn(
            kept_batch,
            CUDAGraphMode.NONE,
            block_tables,
            kept_slot_mappings,
            replay_groups,
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(
                self._kept_kv_start[:num_reqs]
            ),
        )
        first_swa = layers.first_swa_prefix
        attn_metadata[first_swa] = super().prepare_attn(
            kept_batch,
            CUDAGraphMode.NONE,
            block_tables,
            kept_slot_mappings,
            first_swa_groups,
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(replay_start),
        )[first_swa]
        dp_metadata = None
        if num_tokens_across_dp is not None:
            dp_metadata = DPMetadata.make(
                self.vllm_config.parallel_config,
                num_tokens,
                num_tokens_across_dp,
            )
        is_padding = self._kept_is_padding[:num_tokens].zero_()
        return RetainedReplayBatch(
            rows=self._kept_rows[:num_tokens],
            replay_start=replay_start,
            input_batch=replace(
                kept_batch,
                input_ids=input_batch.input_ids.index_select(
                    0, self._kept_rows[:num_tokens]
                ),
                is_padding=is_padding,
            ),
            block_tables=block_tables,
            slot_mappings=kept_slot_mappings,
            forward_context=create_forward_context(
                attn_metadata,
                self.vllm_config,
                dp_metadata=dp_metadata,
                batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
                slot_mapping=build_slot_mappings_by_layer(
                    kept_slot_mappings, kv_cache_config
                ),
                is_padding=is_padding,
            ),
            num_tokens_across_dp=num_tokens_across_dp,
        )

    def _ensure_ordinary_buffers(self, num_kv_groups: int) -> None:
        if self._ordinary_input_buffers is None:
            self._ordinary_input_buffers = InputBuffers(
                self.max_num_reqs, self.max_num_tokens, self.device
            )
            self._ordinary_rows = torch.arange(
                self.max_num_tokens, dtype=torch.int64, device=self.device
            )
            self._ordinary_replay_start = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
        if self._ordinary_slot_mappings is None:
            self._ordinary_slot_mappings = torch.full(
                (num_kv_groups, self.max_num_tokens),
                PAD_SLOT_ID,
                dtype=torch.int64,
                device=self.device,
            )
        else:
            assert self._ordinary_slot_mappings.shape[0] == num_kv_groups

    def _prepare_replay_front_batch(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        replay_start: torch.Tensor,
        front_dp_tokens: torch.Tensor,
    ) -> ForwardContext:
        manager = self.replay_front_cudagraph
        decoder_model = self._decoder_model
        if manager is None or decoder_model is None:
            raise RuntimeError("decoder replay front graph is unavailable")

        capacity = manager.NUM_TOKENS
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens_after_padding
        if num_reqs >= self.max_num_reqs or not 0 < num_tokens <= capacity:
            raise ValueError(
                f"invalid replay front shape: reqs={num_reqs}, rows={num_tokens}"
            )
        cpu_terminal = int(input_batch.query_start_loc_np[num_reqs])
        if not 0 < cpu_terminal <= capacity:
            raise ValueError(
                f"invalid replay front CPU rows: {cpu_terminal}/{capacity}"
            )

        buffers = manager.input_buffers
        buffers.input_ids[:num_tokens].copy_(input_batch.input_ids[:num_tokens])
        buffers.input_ids[num_tokens:].zero_()
        buffers.positions[:num_tokens].copy_(input_batch.positions[:num_tokens])
        buffers.positions[num_tokens:].zero_()
        buffers.is_padding[:num_tokens].copy_(input_batch.is_padding[:num_tokens])
        buffers.is_padding[num_tokens:].fill_(True)

        buffers.query_start_loc.fill_(capacity)
        buffers.query_start_loc[: num_reqs + 1].copy_(
            input_batch.query_start_loc[: num_reqs + 1]
        )
        buffers.seq_lens.zero_()
        buffers.seq_lens[:num_reqs].copy_(input_batch.seq_lens[:num_reqs])
        buffers.seq_lens[num_reqs] = capacity - cpu_terminal

        self._front_query_start_loc_np.fill(capacity)
        self._front_query_start_loc_np[: num_reqs + 1] = input_batch.query_start_loc_np[
            : num_reqs + 1
        ]
        self._front_num_scheduled_tokens.fill(0)
        self._front_num_scheduled_tokens[:num_reqs] = input_batch.num_scheduled_tokens[
            :num_reqs
        ]
        self._front_num_scheduled_tokens[num_reqs] = capacity - cpu_terminal
        self._front_is_prefilling_np.fill(True)
        self._front_is_prefilling_np[:num_reqs] = input_batch.is_prefilling_np[
            :num_reqs
        ]
        self._front_seq_lens_cpu.zero_()
        self._front_seq_lens_cpu[:num_reqs].copy_(
            input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        )
        self._front_seq_lens_cpu[num_reqs] = capacity - cpu_terminal
        self._front_replay_start.zero_()
        self._front_replay_start[:num_reqs].copy_(replay_start[:num_reqs])

        if self._front_block_tables is None:
            self._front_block_tables = tuple(
                torch.zeros(
                    (self.max_num_reqs, *table.shape[1:]),
                    dtype=table.dtype,
                    device=table.device,
                )
                for table in block_tables
            )
        if len(self._front_block_tables) != len(block_tables):
            raise RuntimeError("decoder replay front KV group count changed")
        for front_table, table in zip(self._front_block_tables, block_tables):
            front_table[:num_reqs].copy_(table[:num_reqs])
            front_table[num_reqs].zero_()

        if self._front_slot_mappings is None:
            self._front_slot_mappings = torch.full(
                (slot_mappings.shape[0], capacity),
                PAD_SLOT_ID,
                dtype=slot_mappings.dtype,
                device=slot_mappings.device,
            )
        front_slot_mappings = self._front_slot_mappings
        if front_slot_mappings.shape[0] != slot_mappings.shape[0]:
            raise RuntimeError("decoder replay front slot group count changed")
        front_slot_mappings.fill_(PAD_SLOT_ID)
        front_slot_mappings[:, :num_tokens].copy_(slot_mappings[:, :num_tokens])

        padded_reqs = self.max_num_reqs
        front_batch = replace(
            input_batch,
            req_ids=input_batch.req_ids[:num_reqs]
            + [f"front_padding_{i}" for i in range(padded_reqs - num_reqs)],
            num_reqs=num_reqs + 1,
            num_reqs_after_padding=padded_reqs,
            idx_mapping=self._front_idx_mapping,
            idx_mapping_np=self._front_idx_mapping_np,
            num_scheduled_tokens=self._front_num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=capacity,
            query_start_loc=buffers.query_start_loc,
            query_start_loc_np=self._front_query_start_loc_np,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu_upper_bound=self._front_seq_lens_cpu,
            dcp_local_seq_lens=None,
            dcp_local_seq_lens_cpu_upper_bound=None,
            is_prefilling_np=self._front_is_prefilling_np,
            has_prefill=True,
            input_ids=buffers.input_ids,
            positions=buffers.positions,
            is_padding=buffers.is_padding,
            prompt_lens=None,
            max_query_len=max(
                input_batch.max_query_len or 0,
                int(self._front_num_scheduled_tokens.max()),
            ),
            fast_prefill=None,
        )

        if self._front_attn_groups is None:
            prefixes = decoder_model.decoder_replay_front_metadata_prefixes
            if not prefixes:
                raise RuntimeError("decoder replay front metadata set is empty")
            self._front_attn_groups = self._copy_attn_groups(attn_groups, prefixes)
        attn_metadata = super().prepare_attn(
            front_batch,
            CUDAGraphMode.FULL,
            self._front_block_tables,
            front_slot_mappings,
            self._front_attn_groups,
            kv_cache_config,
            model_specific_attn_metadata=ReplayAttnMetadata(self._front_replay_start),
        )
        context_dp_tokens = front_dp_tokens.clone().fill_(capacity)
        dp_metadata = DPMetadata.make(
            self.vllm_config.parallel_config,
            capacity,
            context_dp_tokens,
        )
        return create_forward_context(
            attn_metadata,
            self.vllm_config,
            dp_metadata=dp_metadata,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            batch_descriptor=BatchDescriptor(num_tokens=capacity),
            slot_mapping=build_slot_mappings_by_layer(
                front_slot_mappings, kv_cache_config
            ),
            is_padding=buffers.is_padding,
        )

    def _kept_input_batch(
        self,
        input_batch: InputBatch,
        slot_mappings: torch.Tensor,
        replay_start: torch.Tensor,
        kept_lens: np.ndarray,
        num_padded: int,
    ) -> tuple[InputBatch, torch.Tensor]:
        """Build the sub-``InputBatch`` of each request's last ``kept_lens``
        rows, and its slot mappings.

        The kept rows follow the device boundaries minus the rows trimmed before
        them: adaptive verification resizes the decodes on the GPU alone, and
        decodes never trim. Like a microbatch's, the sub-batch describes the
        forward only; its sampling fields are carried over and must not be read.
        """
        num_reqs = input_batch.num_reqs
        num_tokens = int(kept_lens.sum())
        kept_query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(kept_lens, out=kept_query_start_loc_np[1:])
        dropped_before = self._dropped_before.copy_to_uva(
            input_batch.query_start_loc_np[: num_reqs + 1] - kept_query_start_loc_np
        )
        assert self._kept_slot_mappings is not None
        kept_slot_mappings = self._kept_slot_mappings[:, :num_padded]
        kept_slot_mappings[:, num_tokens:].fill_(PAD_SLOT_ID)
        _gather_replay_batch_kernel[(num_reqs,)](
            input_batch.query_start_loc,
            dropped_before,
            input_batch.seq_lens,
            replay_start,
            input_batch.positions,
            slot_mappings,
            slot_mappings.stride(0),
            self._kept_rows,
            self._kept_query_start_loc,
            self._kept_positions,
            kept_slot_mappings,
            kept_slot_mappings.stride(0),
            self._kept_kv_start,
            self.decoder_replay_layers.window,  # type: ignore[union-attr]
            NUM_GROUPS=slot_mappings.shape[0],
            BLOCK=1024,
        )
        # Adaptive verification may lengthen the decodes on the device, up to
        # the batch's bound.
        max_query_len = max(int(kept_lens.max()), input_batch.max_query_len or 0)
        kept_batch = replace(
            input_batch,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_padded,
            query_start_loc=self._kept_query_start_loc[: num_reqs + 1],
            query_start_loc_np=kept_query_start_loc_np,
            max_query_len=max_query_len,
            positions=self._kept_positions[:num_padded],
            fast_prefill=None,
        )
        return kept_batch, kept_slot_mappings

    def _agree_across_dp(
        self, needs_replay_batch: bool, num_tokens: int
    ) -> tuple[bool, torch.Tensor | None]:
        """Whether any rank needs a replay batch and, if so, each rank's token count."""
        parallel_config = self.vllm_config.parallel_config
        agreed = torch.zeros(2, parallel_config.data_parallel_size, dtype=torch.int32)
        agreed[:, parallel_config.data_parallel_rank] = torch.tensor(
            [needs_replay_batch, num_tokens]
        )
        if not should_skip_dp_coordination():
            dist.all_reduce(agreed, group=get_dp_group().cpu_group)
        if not agreed[0].any():
            return False, None
        return True, agreed[1]

    def _is_mixed_replay_candidate(self, input_batch: InputBatch) -> bool:
        """Whether this step mixes Decode rows with resumed Prefill replay."""
        num_reqs = input_batch.num_reqs
        is_prefilling = input_batch.is_prefilling_np[:num_reqs]
        if not is_prefilling.any() or is_prefilling.all():
            return False
        replay_starts = self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]]
        return bool((is_prefilling & (replay_starts > 0)).any())

    def _front_graph_shape(self, input_batch: InputBatch) -> tuple[bool, int]:
        if (
            not self._mixed_replay_front_graph_enabled
            or self.replay_front_cudagraph is None
            or self._decoder_model is None
        ):
            return False, 0
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens_after_padding
        if (
            num_reqs >= self.max_num_reqs
            or not 0 < num_tokens <= DecoderReplayFrontCudaGraphManager.NUM_TOKENS
        ):
            return False, 0
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        if int(query_lens.sum()) > DecoderReplayFrontCudaGraphManager.NUM_TOKENS:
            return False, 0
        is_prefilling = input_batch.is_prefilling_np[:num_reqs]
        replay_starts = self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]]
        contexts = is_prefilling & (replay_starts > 0)
        context_count = int(contexts.sum())
        eligible = (
            context_count <= 1
            and int(is_prefilling.sum()) == context_count
            and bool((query_lens[contexts] == 128).all())
        )
        return eligible, context_count

    def _clear_replay_front_graph(self) -> None:
        if self.replay_front_cudagraph is not None:
            self.replay_front_cudagraph.clear_runtime()
        if self._decoder_model is not None:
            self._decoder_model.decoder_replay_front_graph = None

    def _agree_mixed_replay_across_dp(
        self,
        needs_replay_batch: bool,
        force_mixed_graph: bool,
        mixed_graph_eligible: bool,
        num_tokens: int,
        front_eligible: bool,
        front_num_tokens: int,
        front_contexts: int,
    ) -> tuple[
        bool,
        bool,
        torch.Tensor | None,
        bool,
        torch.Tensor | None,
    ]:
        """Agree whether an eager mixed step may force the dedicated replay graph."""
        parallel_config = self.vllm_config.parallel_config
        rank = parallel_config.data_parallel_rank
        agreed = torch.zeros(7, parallel_config.data_parallel_size, dtype=torch.int32)
        agreed[:, rank] = torch.tensor(
            [
                needs_replay_batch,
                force_mixed_graph,
                mixed_graph_eligible,
                num_tokens,
                front_eligible,
                front_num_tokens,
                front_contexts,
            ]
        )
        if should_skip_dp_coordination():
            if force_mixed_graph and not mixed_graph_eligible:
                if self._replay_census is not None:
                    self._record_replay_census(
                        outcome="peer_ineligible",
                        actual_rows=num_tokens,
                        dp_max_rows=num_tokens,
                        descriptor_rows=None,
                    )
                return False, False, None, False, None
            needs_replay_batch = needs_replay_batch or force_mixed_graph
            force_front_graph = (
                self._mixed_replay_front_graph_enabled
                and force_mixed_graph
                and front_eligible
                and front_num_tokens == DecoderReplayFrontCudaGraphManager.NUM_TOKENS
                and front_contexts == 1
            )
            return (
                needs_replay_batch,
                force_mixed_graph,
                agreed[3] if needs_replay_batch else None,
                force_front_graph,
                agreed[5] if force_front_graph else None,
            )
        dist.all_reduce(agreed, group=get_dp_group().cpu_group)
        force_requested = bool(agreed[1].any())
        if force_requested and not agreed[2].all():
            if self._replay_census is not None:
                self._record_replay_census(
                    outcome="peer_ineligible",
                    actual_rows=int(agreed[3].sum()),
                    dp_max_rows=int(agreed[3].max()),
                    descriptor_rows=None,
                )
            return False, False, None, False, None
        force_mixed_graph = force_requested
        needs_replay_batch = bool(agreed[0].any()) or force_mixed_graph
        force_front_graph = (
            self._mixed_replay_front_graph_enabled
            and force_mixed_graph
            and bool(agreed[4].all())
            and int(agreed[5].max()) == DecoderReplayFrontCudaGraphManager.NUM_TOKENS
            and int(agreed[6].sum()) == 1
        )
        return (
            needs_replay_batch,
            force_mixed_graph,
            agreed[3] if needs_replay_batch else None,
            force_front_graph,
            agreed[5] if force_front_graph else None,
        )

    def _all_dp_flags(self, *local_flags: bool) -> tuple[bool, ...]:
        parallel_config = self.vllm_config.parallel_config
        if should_skip_dp_coordination():
            return tuple(local_flags)
        ready = torch.zeros(
            len(local_flags),
            parallel_config.data_parallel_size,
            dtype=torch.int32,
        )
        ready[:, parallel_config.data_parallel_rank] = torch.tensor(
            local_flags, dtype=torch.int32
        )
        dist.all_reduce(ready, group=get_dp_group().cpu_group)
        return tuple(bool(flag.all()) for flag in ready)

    def _record_replay_census(
        self,
        *,
        outcome: str,
        actual_rows: int,
        dp_max_rows: int,
        descriptor_rows: int | None,
    ) -> None:
        census = self._replay_census
        if census is None:
            return
        census.record(
            outcome=outcome,
            actual_rows=actual_rows,
            dp_max_rows=dp_max_rows,
            dp_size=self.vllm_config.parallel_config.data_parallel_size,
            descriptor_rows=descriptor_rows,
        )

    @staticmethod
    def _replay_graph_fallback_reason(
        manager: DecoderReplayCudaGraphManager,
        descriptor: BatchExecutionDescriptor | None,
        dp_max_tokens: int,
        local_graph_ready: bool,
    ) -> str:
        if local_graph_ready:
            return "peer_graph_not_ready"
        if descriptor is not None and descriptor.cg_mode == CUDAGraphMode.PIECEWISE:
            return "no_graph_segment"
        if dp_max_tokens > manager.compilation_config.max_cudagraph_capture_size:
            return "above_capture_ceiling"
        return "dispatch_incompatible"

    @staticmethod
    def _replay_graph_has_segments(
        manager: DecoderReplayCudaGraphManager,
        descriptor: BatchExecutionDescriptor,
    ) -> bool:
        entry = manager.breakable_cg_runner.entries.get(
            BatchDescriptor(num_tokens=descriptor.num_tokens)
        )
        capture = entry.capture if entry is not None else None
        return capture is not None and capture.num_graphs >= 1

    def _copy_attn_groups(
        self,
        attn_groups: list[list[AttentionGroup]],
        prefixes: set[str] | None,
    ) -> list[list[AttentionGroup]]:
        """Copy matching groups, or every group when ``prefixes`` is None."""
        own_groups = []
        for groups in attn_groups:
            copies = []
            for group in groups:
                if prefixes is not None and prefixes.isdisjoint(group.layer_names):
                    continue
                copy = replace(group, metadata_builders=[])
                copy.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    group.metadata_builders[0].kernel_block_size,
                )
                copies.append(copy)
            own_groups.append(copies)
        return own_groups
