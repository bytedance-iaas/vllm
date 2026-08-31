# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import CrossAttentionSpec, MambaSpec
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def _uses_parallel_draft_full_k_warmup(model_runner: GPUModelRunner) -> bool:
    speculative_config = getattr(model_runner, "speculative_config", None)
    return (
        not model_runner.is_pooling_model
        and speculative_config is not None
        and speculative_config.method in ("dflash", "dspark")
        and model_runner.num_speculative_steps > 0
    )


def _parallel_draft_full_k_warmup_req_id(
    req_id_prefix: str = "_warmup_parallel_draft_full_k",
) -> str:
    return f"{req_id_prefix}_0_"


def _cleanup_warmup_requests(
    worker_execute_model: Callable[[SchedulerOutput], Any],
    req_ids: set[str],
) -> None:
    if not req_ids:
        return
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)


def _warmup_block_count(
    num_tokens: int,
    spec: Any,
    max_encoder_len: int,
) -> int:
    if isinstance(spec, CrossAttentionSpec):
        num_tokens = max_encoder_len
    num_blocks = cdiv(num_tokens, spec.block_size)
    if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
        num_blocks += spec.num_speculative_blocks
    return num_blocks


def _run_parallel_draft_full_k_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    *,
    req_id_prefix: str = "_warmup_parallel_draft_full_k",
) -> str:
    """Warm up the real one-target-token/full-K proposer path."""
    prompt_len = 2
    prompt_token_ids = list(range(prompt_len))
    req_id = _parallel_draft_full_k_warmup_req_id(req_id_prefix)
    sampling_params = SamplingParams.for_sampler_warmup()

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)
    lookahead_tokens = (
        model_runner.speculative_config.num_drafter_query_tokens
    )
    prefill_block_counts = [
        _warmup_block_count(
            prompt_len + lookahead_tokens,
            spec,
            max_encoder_len,
        )
        for spec in kv_cache_specs
    ]
    decode_block_counts = [
        _warmup_block_count(
            prompt_len + 1 + lookahead_tokens,
            spec,
            max_encoder_len,
        )
        for spec in kv_cache_specs
    ]
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, prefill_block_counts)
    ]

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prompt_token_ids,
        )
    ]
    prefill_output.num_scheduled_tokens = {req_id: prompt_len}
    prefill_output.total_num_scheduled_tokens = prompt_len
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cached_req = CachedRequestData.make_empty()
    cached_req.req_ids = [req_id]
    cached_req.num_computed_tokens = [prompt_len]
    cached_req.num_output_tokens = [1]
    cached_req.new_block_ids = [
        tuple(_alloc_blocks(n) for n in decode_block_deltas)
        if any(decode_block_deltas)
        else None
    ]

    decode_output = SchedulerOutput.make_empty()
    decode_output.scheduled_cached_reqs = cached_req
    decode_output.num_scheduled_tokens = {req_id: 1}
    decode_output.total_num_scheduled_tokens = 1
    decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups
    decode_output.num_spec_tokens_to_schedule = model_runner.num_speculative_steps

    worker_execute_model(prefill_output)
    worker_sample_tokens(None)
    worker_execute_model(decode_output)
    worker_sample_tokens(None)
    return req_id


def run_mixed_prefill_decode_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    num_tokens: int,
    *,
    mixed_step_context: AbstractContextManager[object] | None = None,
    req_id_prefix: str = "_v2_mixed_warmup",
) -> bool:
    """Run a V2 mixed prefill+decode step through normal scheduler inputs."""
    if model_runner.is_pooling_model or model_runner.max_num_reqs < 2 or num_tokens < 3:
        return False

    decode_req_id = f"{req_id_prefix}_decode_"
    prefill_req_id = f"{req_id_prefix}_prefill_"
    decode_prompt_len = 2
    decode_scheduled_tokens = 1
    prefill_len = num_tokens - decode_scheduled_tokens
    decode_token_ids = list(range(decode_prompt_len))
    prefill_token_ids = list(range(prefill_len))

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    decode_prefill_block_counts = [
        cdiv(decode_prompt_len, block_size) for block_size in group_block_sizes
    ]
    decode_block_counts = [
        cdiv(decode_prompt_len + decode_scheduled_tokens, block_size)
        for block_size in group_block_sizes
    ]
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, decode_prefill_block_counts)
    ]
    prefill_block_counts = [
        cdiv(prefill_len, block_size) for block_size in group_block_sizes
    ]
    required_blocks = sum(decode_block_counts) + sum(prefill_block_counts)
    if model_runner.kv_cache_config.num_blocks <= required_blocks:
        logger.warning(
            "Skipping V2 mixed prefill+decode warmup because only %d KV blocks "
            "are available for %d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            required_blocks,
        )
        return False

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        block_ids = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return block_ids

    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    decode_prefill_output = SchedulerOutput.make_empty()
    decode_prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=decode_req_id,
            prompt_token_ids=decode_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in decode_prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=decode_token_ids,
        ),
    ]
    decode_prefill_output.num_scheduled_tokens = {
        decode_req_id: decode_prompt_len,
    }
    decode_prefill_output.total_num_scheduled_tokens = decode_prompt_len
    decode_prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    decode_new_blocks = tuple(_alloc_blocks(n) for n in decode_block_deltas)
    cached_decode_req = CachedRequestData.make_empty()
    cached_decode_req.req_ids = [decode_req_id]
    cached_decode_req.num_computed_tokens = [decode_prompt_len]
    cached_decode_req.num_output_tokens = [1]
    cached_decode_req.new_block_ids = [
        decode_new_blocks if any(decode_block_deltas) else None
    ]

    mixed_output = SchedulerOutput.make_empty()
    mixed_output.scheduled_cached_reqs = cached_decode_req
    mixed_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=prefill_req_id,
            prompt_token_ids=prefill_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prefill_token_ids,
        ),
    ]
    mixed_output.num_scheduled_tokens = {
        decode_req_id: decode_scheduled_tokens,
        prefill_req_id: prefill_len,
    }
    mixed_output.total_num_scheduled_tokens = num_tokens
    mixed_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {decode_req_id, prefill_req_id}

    context = mixed_step_context or nullcontext()
    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(decode_prefill_output)
        worker_sample_tokens(None)
        with context:
            worker_execute_model(mixed_output)
            worker_sample_tokens(None)
        worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    return True


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run scheduler-realistic prefill and decode steps to JIT compile kernels.

    We must call the provided worker's execute_model for pipeline parallel
    coordination.
    """
    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    # Upper bound on the decode steps built in `decode_steps` below.
    num_decode_steps = 1
    if not model_runner.is_pooling_model:
        num_decode_steps = 5 if num_spec_steps > 0 else 3
    # Size the block allocation for the worst case: every request advancing
    # decode_query_len tokens on every decode step.
    decode_len = prompt_len + num_decode_steps * decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Encoder-decoder models: give each warmup request a dummy encoder input so
    # cross-attention warms up over a realistic, non-empty key sequence.
    # The dummy mm_feature is registered in the encoder cache and only its encoder
    # length is read (not the inputs themselves); the encoder itself is not scheduled.
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)
    warmup_mm_features: list[MultiModalFeatureSpec] = []
    if model_runner.is_encoder_decoder and max_encoder_len:
        warmup_mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="",
                identifier="_warmup_encoder",
                mm_position=PlaceholderRange(offset=0, length=max_encoder_len),
            )
        ]

    kv_cache_specs = [g.kv_cache_spec for g in kv_cache_groups]
    prefill_block_counts = [
        _warmup_block_count(prompt_len, spec, max_encoder_len)
        for spec in kv_cache_specs
    ]
    decode_block_counts = [
        _warmup_block_count(decode_len, spec, max_encoder_len)
        for spec in kv_cache_specs
    ]
    max_blocks_per_req = sum(decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
    )
    if max_blocks_per_req > 0:
        # Reserve block 0 (null block) and ensure we have enough blocks.
        # Encoder-only models allocate no KV blocks, so this cap doesn't apply.
        num_reqs = min(
            num_reqs,
            max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
        )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_task = model_runner.model_config.get_pooling_task(
            model_runner.get_supported_tasks()
        )
        pooling_params = PoolingParams(task=pooling_task)
        pooling_params.verify(model_runner.model_config)
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign distinct block IDs per request per group. 0 null block, start from 1.
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    # The KV-block zeroing kernel is driven by the scheduler's
    # new_block_ids_to_zero, so none of the steps below reach it.
    if model_runner.kv_block_zeroer is not None:
        model_runner.kv_block_zeroer.warmup(model_runner.kv_cache_config.num_blocks)

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(
                req_ids[i],
                prompt_token_ids,
                sampling_params,
                pooling_params,
                mm_features=warmup_mm_features,
            ),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    generic_pending = set(req_ids)
    parallel_pending: set[str] = set()
    parallel_warmup_attempted = False
    startup_num_spec_tokens = getattr(
        model_runner, "last_completed_num_spec_tokens_to_schedule", 0
    )
    main_succeeded = False

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(prefill_output)

        if not model_runner.is_pooling_model:
            # Warm up sampler and perform decode steps for non-pooling models.
            grammar_output = None
            if model_runner.is_last_pp_rank:
                # Exercise the structured output bitmask during prefill.
                vocab_size = model_runner.model_config.get_vocab_size()
                bitmask_width = (vocab_size + 31) // 32
                grammar_bitmask = np.full(
                    (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
                )
                grammar_output = GrammarOutput(
                    structured_output_request_ids=req_ids,
                    grammar_bitmask=grammar_bitmask,
                )

            worker_sample_tokens(grammar_output)

            # Per-request state carried across the decode steps.
            req_computed = [prompt_len] * num_reqs
            req_blocks = [list(prefill_block_counts) for _ in range(num_reqs)]

            def _run_decode_step(
                indices: list[int], spec_flags: list[bool]
            ) -> None:
                """Decode `indices`, spec-decoding requests marked by the flags."""
                cached_req_data = CachedRequestData.make_empty()
                cached_req_data.req_ids = [req_ids[i] for i in indices]
                cached_req_data.num_computed_tokens = [
                    req_computed[i] for i in indices
                ]
                cached_req_data.num_output_tokens = [1] * len(indices)
                cached_req_data.new_block_ids = []

                step_num_scheduled_tokens: dict[str, int] = {}
                step_spec_tokens: dict[str, list[int]] = {}
                for i, use_spec in zip(indices, spec_flags):
                    num_tokens = decode_query_len if use_spec else 1
                    after = req_computed[i] + num_tokens
                    deltas = [
                        _warmup_block_count(after, spec, max_encoder_len) - held
                        for spec, held in zip(kv_cache_specs, req_blocks[i])
                    ]
                    cached_req_data.new_block_ids.append(
                        tuple(_alloc_blocks(n) for n in deltas)
                        if any(deltas)
                        else None
                    )
                    req_blocks[i] = [
                        held + delta for held, delta in zip(req_blocks[i], deltas)
                    ]
                    step_num_scheduled_tokens[req_ids[i]] = num_tokens
                    if use_spec:
                        step_spec_tokens[req_ids[i]] = [0] * num_spec_steps

                decode_output = SchedulerOutput.make_empty()
                decode_output.scheduled_cached_reqs = cached_req_data
                decode_output.num_scheduled_tokens = step_num_scheduled_tokens
                decode_output.scheduled_spec_decode_tokens = step_spec_tokens
                decode_output.total_num_scheduled_tokens = sum(
                    step_num_scheduled_tokens.values()
                )
                decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

                worker_execute_model(decode_output)
                worker_sample_tokens(None)

                for i, use_spec in zip(indices, spec_flags):
                    req_computed[i] += decode_query_len if use_spec else 1

            all_indices = list(range(num_reqs))
            use_spec_decode = num_spec_steps > 0

            # Warm uniform, mixed-spec, and single-request decode shapes.
            decode_steps: list[tuple[list[int], list[bool]]] = [
                (all_indices, [use_spec_decode] * num_reqs),
            ]
            if num_reqs >= 2:
                decode_steps.append(([0, 1], [use_spec_decode, False]))
                if use_spec_decode:
                    decode_steps.append(([0, 1], [False, False]))
            if num_reqs > 1:
                decode_steps.append(([0], [use_spec_decode]))
                if use_spec_decode:
                    decode_steps.append(([0], [False]))
            elif use_spec_decode:
                decode_steps.append(([0], [False]))

            for step_indices, step_spec_flags in decode_steps:
                _run_decode_step(step_indices, step_spec_flags)

        _cleanup_warmup_requests(worker_execute_model, generic_pending)
        generic_pending.clear()

        if _uses_parallel_draft_full_k_warmup(model_runner):
            parallel_warmup_attempted = True
            parallel_req_id = _parallel_draft_full_k_warmup_req_id()
            parallel_pending.add(parallel_req_id)
            _run_parallel_draft_full_k_warmup(
                model_runner,
                worker_execute_model,
                worker_sample_tokens,
            )
            _cleanup_warmup_requests(worker_execute_model, parallel_pending)
            parallel_pending.clear()

        main_succeeded = True
    finally:
        cleanup_error: Exception | None = None
        for label, pending in (
            ("generic", generic_pending),
            ("parallel-draft", parallel_pending),
        ):
            if not pending:
                continue
            try:
                _cleanup_warmup_requests(worker_execute_model, pending)
            except Exception as exc:
                logger.exception("Failed to clean up %s warmup requests.", label)
                if cleanup_error is None:
                    cleanup_error = exc
        if parallel_warmup_attempted:
            model_runner.last_completed_num_spec_tokens_to_schedule = (
                startup_num_spec_tokens
            )
        try:
            model_runner.kv_connector.set_disabled(False)
        except Exception as exc:
            logger.exception("Failed to re-enable the KV connector after warmup.")
            if cleanup_error is None:
                cleanup_error = exc
        if main_succeeded and cleanup_error is not None:
            raise cleanup_error

    torch.accelerator.synchronize()
