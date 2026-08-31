#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu import dp_utils as gpu_dp_utils
from vllm.v1.worker.gpu import model_runner as gpu_model_runner
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as dflash_cudagraph

pytestmark = pytest.mark.cpu_test


def _create_vllm_config_for_dsd(
    max_num_seqs: int,
    max_spec_tokens: int,
    *,
    cudagraph_mode: str = "FULL_AND_PIECEWISE",
    use_dynamic_sd: bool = True,
    num_spec_per_batch_size: list[tuple[int, int, int]] | None = None,
    cudagraph_capture_sizes: list[int] | None = None,
    max_cudagraph_capture_size: int | None = None,
) -> MagicMock:
    """Create a minimal config that exercises DSD cudagraph dispatch.

    The test uses an exact capture-size grid so that every valid uniform decode
    shape has a directly matching FULL graph candidate.

    ``num_spec_per_batch_size`` lets a test supply an explicit DSD schedule of
    ``(range_start, range_end, num_speculative_tokens)`` tuples. When omitted,
    a schedule covering every query length in ``[1, max_decode_query_len]`` is
    generated.
    """

    max_decode_query_len = max_spec_tokens + 1
    max_capture_tokens = max_num_seqs * max_decode_query_len

    if cudagraph_capture_sizes is None:
        cudagraph_capture_sizes = list(range(1, max_capture_tokens + 1))
    if max_cudagraph_capture_size is None:
        max_cudagraph_capture_size = max_capture_tokens

    compilation_config = CompilationConfig(
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    compilation_config.max_cudagraph_capture_size = max_cudagraph_capture_size
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs,
    )
    vllm_config.parallel_config = ParallelConfig()
    # num_speculative_tokens is the max K (num_speculative_steps). The manager
    # recovers num_new_sampled_tokens_per_step as
    # decode_query_len - num_speculative_tokens; with decode_query_len =
    # max_spec_tokens + 1 this yields the normal per-step bonus of 1.
    vllm_config.num_speculative_tokens = max_spec_tokens

    speculative_config = MagicMock()
    speculative_config.uses_dynamic_speculative_decoding.return_value = use_dynamic_sd
    if use_dynamic_sd:
        # DSD reads the per-batch-size schedule; a schedule entry with K
        # speculative tokens maps to decode query length K + 1. By default
        # provide every query length in [1, max_decode_query_len] (i.e. K in
        # [0, max_spec_tokens]) so the manager captures a FULL decode graph for
        # each uniform shape.
        if num_spec_per_batch_size is None:
            num_spec_per_batch_size = [
                (qlen, qlen, qlen - 1) for qlen in range(1, max_decode_query_len + 1)
            ]
        speculative_config.num_speculative_tokens_per_batch_size = (
            num_spec_per_batch_size
        )
    else:
        speculative_config.num_speculative_tokens_per_batch_size = None
    vllm_config.speculative_config = speculative_config

    return vllm_config


def _patch_cudagraph_test_runtime(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: None,
    )


def _make_sparse_capture_sizes() -> list[int]:
    capture_sizes = [1, 2, 4]
    capture_sizes.extend(range(8, 256, 8))
    capture_sizes.extend(range(256, 513, 16))
    return capture_sizes


def _make_dynamic_sd_dummy_runner() -> tuple[object, dict[str, int]]:
    runner = object.__new__(gpu_model_runner.GPUModelRunner)
    runner.max_num_reqs = 8
    runner.num_speculative_steps = 7
    runner.decode_query_len = 8
    runner.last_completed_num_spec_tokens_to_schedule = 0
    runner.speculative_config = MagicMock()
    runner.speculative_config.uses_dynamic_speculative_decoding.return_value = True
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.kv_connector = SimpleNamespace(set_disabled=lambda *_: None)
    runner.is_first_pp_rank = True
    runner.is_last_pp_rank = True
    runner.intermediate_tensors = None
    runner.lora_config = None
    runner.execute_model_state = None
    runner.device = torch.device("cpu")
    runner.eplb = SimpleNamespace(step=lambda **kwargs: None)
    runner.req_states = SimpleNamespace(
        last_sampled_tokens=torch.zeros(runner.max_num_reqs, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(runner.max_num_reqs, dtype=torch.int64),
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(
                gpu=torch.zeros(runner.max_num_reqs, dtype=torch.float32)
            ),
            seeds=SimpleNamespace(
                gpu=torch.zeros(runner.max_num_reqs, dtype=torch.int64)
            ),
        )
    )
    runner.maybe_dummy_run_with_lora = lambda *args, **kwargs: nullcontext()

    recorded: dict[str, int] = {}

    def fake_execute_model(self, scheduler_output, **kwargs):
        scheduled_tokens = list(scheduler_output.num_scheduled_tokens.values())
        recorded["target_query_len"] = scheduled_tokens[0]
        recorded["target_total_tokens"] = scheduler_output.total_num_scheduled_tokens
        recorded["target_runtime_k"] = scheduler_output.num_spec_tokens_to_schedule
        self.execute_model_state = SimpleNamespace(
            input_batch=SimpleNamespace(
                num_reqs=len(scheduled_tokens),
                num_tokens=scheduler_output.total_num_scheduled_tokens,
                logits_indices=torch.tensor([0], dtype=torch.int64),
            ),
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=torch.zeros(
                scheduler_output.total_num_scheduled_tokens, 4, dtype=torch.float32
            ),
            aux_hidden_states=None,
            num_spec_tokens_to_schedule=scheduler_output.num_spec_tokens_to_schedule,
            finished_req_ids=set(),
        )

    class FakeSpeculator:
        supports_mm_inputs = False

        def propose(self, *args, runtime_num_speculative_tokens, **kwargs):
            recorded["proposer_runtime_k"] = int(runtime_num_speculative_tokens)
            input_batch = kwargs["input_batch"] if "input_batch" in kwargs else args[0]
            return torch.zeros(
                input_batch.num_reqs,
                runtime_num_speculative_tokens,
                dtype=torch.int64,
            )

    runner.execute_model = fake_execute_model.__get__(runner, type(runner))
    runner.speculator = FakeSpeculator()
    runner.model = SimpleNamespace()
    return runner, recorded


def test_dynamic_sd_target_capture_includes_scheduled_and_max_k_families(monkeypatch):
    max_num_seqs = 96
    max_spec_tokens = 7
    schedule = [(1, 16, 7), (17, 60, 5), (61, 96, 3)]

    _patch_cudagraph_test_runtime(monkeypatch)

    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config_for_dsd(
            max_num_seqs=max_num_seqs,
            max_spec_tokens=max_spec_tokens,
            cudagraph_mode="FULL_DECODE_ONLY",
            num_spec_per_batch_size=schedule,
            cudagraph_capture_sizes=_make_sparse_capture_sizes(),
            max_cudagraph_capture_size=512,
        ),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=max_spec_tokens + 1,
    )
    manager._graphs_captured = True

    for qlen, num_reqs, expected_tokens, expected_reqs in (
        (6, 13, 84, 14),
        (8, 10, 80, 10),
    ):
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs * qlen,
            uniform_token_count=qlen,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.uniform_token_count == qlen
        assert desc.num_tokens == expected_tokens
        assert desc.num_reqs == expected_reqs

    implicit_k0 = manager.dispatch(
        num_reqs=96,
        num_tokens=96,
        uniform_token_count=1,
        num_active_loras=0,
    )
    assert implicit_k0.cg_mode == CUDAGraphMode.NONE


def test_dynamic_sd_dspark_proposer_captures_runtime_k_families(monkeypatch):
    max_num_seqs = 96
    max_spec_tokens = 7

    _patch_cudagraph_test_runtime(monkeypatch)

    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config_for_dsd(
            max_num_seqs=max_num_seqs,
            max_spec_tokens=max_spec_tokens,
            cudagraph_mode="FULL_DECODE_ONLY",
            num_spec_per_batch_size=[
                (1, 16, 7),
                (17, 60, 5),
                (61, 96, 3),
            ],
        ),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=max_spec_tokens,
    )
    manager._graphs_captured = True

    for qlen, num_reqs in ((5, 9), (7, 8)):
        desc = manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs * qlen,
            uniform_token_count=qlen,
            num_active_loras=0,
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        assert desc.uniform_token_count == qlen
        assert desc.num_tokens == num_reqs * qlen
        assert desc.num_reqs == num_reqs

    implicit_k0 = manager.dispatch(
        num_reqs=9,
        num_tokens=0,
        uniform_token_count=0,
        num_active_loras=0,
    )
    assert implicit_k0.cg_mode == CUDAGraphMode.NONE


def test_dflash_capture_binds_descriptor_query_length(monkeypatch):
    manager = object.__new__(dflash_cudagraph.DFlashCudaGraphManager)
    manager.dp_size = 1
    manager.max_num_reqs = 16
    desc = gpu_cudagraph_utils.BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=12,
        num_reqs=3,
        uniform_token_count=4,
    )
    recorded: dict[str, int] = {}

    monkeypatch.setattr(
        dflash_cudagraph,
        "_prepare_dflash_inputs_to_capture",
        lambda *args, **kwargs: dflash_cudagraph.AttentionState(None, {}),
    )

    def fake_parent_capture(self, create_forward_fn, progress_bar_desc):
        create_forward_fn(desc, warmup=True)(CUDAGraphMode.NONE)

    monkeypatch.setattr(
        gpu_cudagraph_utils.CudaGraphManager,
        "capture",
        fake_parent_capture,
    )

    def forward_fn(*args, num_query_per_req, **kwargs):
        recorded["num_query_per_req"] = num_query_per_req

    dflash_cudagraph.DFlashCudaGraphManager.capture(
        manager,
        forward_fn,
        input_buffers=MagicMock(),
        block_tables=MagicMock(),
        attn_groups=[],
        kv_cache_config=MagicMock(),
        max_model_len=1024,
        causal=False,
    )

    assert recorded["num_query_per_req"] == 4


@pytest.mark.parametrize(
    ("previous_k", "current_k", "expected_target_qlen"),
    [(5, 7, 6), (7, 5, 8), (0, 5, 1)],
)
def test_dynamic_sd_dummy_run_uses_previous_target_k_and_current_proposer_k(
    previous_k: int,
    current_k: int,
    expected_target_qlen: int,
):
    runner, recorded = _make_dynamic_sd_dummy_runner()
    runner.last_completed_num_spec_tokens_to_schedule = previous_k

    gpu_model_runner.GPUModelRunner._dummy_run(
        runner,
        1,
        uniform_decode=True,
        num_spec_tokens_to_schedule=current_k,
        skip_eplb=True,
    )

    assert recorded == {
        "target_query_len": expected_target_qlen,
        "target_total_tokens": expected_target_qlen,
        "target_runtime_k": current_k,
        "proposer_runtime_k": current_k,
    }
    assert runner.last_completed_num_spec_tokens_to_schedule == current_k


def test_dynamic_sd_full_cudagraph_covers_all_uniform_decode_shapes(monkeypatch):
    """Dynamic SD should create FULL decode candidates for every k in [1, K+1].

    This validates the MRv2 CudaGraphManager path directly: once candidate
    shapes have been built, dispatch() should pick a FULL graph for every
    uniform decode batch shape produced by DSD up to max_num_seqs.
    """

    max_num_seqs = 512
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1

    _patch_cudagraph_test_runtime(monkeypatch)

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_decode_query_len,
    )

    # dispatch() only uses the precomputed candidate table after graphs are
    # considered captured. The actual graph objects are irrelevant here.
    manager._graphs_captured = True

    for num_reqs in range(1, max_num_seqs + 1):
        for max_query_len in range(1, max_decode_query_len + 1):
            # Uniform decode means every request contributes the same number of
            # tokens, so the total token count is exactly num_reqs * query_len.
            num_tokens = num_reqs * max_query_len
            uniform_tok_count = gpu_cudagraph_utils.get_uniform_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
            )

            # The scheduler should mark every one of these shapes as a uniform
            # decode batch, which is what enables FULL decode graph selection.
            assert uniform_tok_count == max_query_len

            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_tok_count,
                num_active_loras=0,
            )

            # With DSD enabled, MRv2 should have captured a FULL candidate for
            # every k in [1, K+1], so dispatch should stay on the FULL path.
            assert desc.cg_mode == CUDAGraphMode.FULL
            assert desc.uniform_token_count == max_query_len
            assert desc.num_tokens == num_tokens
            assert desc.num_reqs == num_reqs
            assert desc.num_active_loras == 0


def test_dynamic_sd_non_uniform_batch_falls_back_to_piecewise(monkeypatch):
    """DSD should use PIECEWISE when the batch is not a uniform decode batch.

    FULL DSD graphs are captured separately for each decode query length k.
    When runtime tokens are not uniform, uniform_token_count is None and those
    FULL candidates should be skipped in favor of the mixed-batch PIECEWISE
    graph under FULL_AND_PIECEWISE mode.
    """

    max_spec_tokens = 4

    _patch_cudagraph_test_runtime(monkeypatch)

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=512,
        max_spec_tokens=max_spec_tokens,
        cudagraph_mode="FULL_AND_PIECEWISE",
        use_dynamic_sd=True,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_spec_tokens + 1,
    )
    manager._graphs_captured = True

    # This shape is intentionally non-uniform: 3 tokens across 2 requests
    # cannot correspond to a single per-request query length.
    desc = manager.dispatch(
        num_reqs=2,
        num_tokens=3,
        uniform_token_count=None,
        num_active_loras=0,
    )

    assert desc.cg_mode == CUDAGraphMode.PIECEWISE
    assert desc.uniform_token_count is None
    assert desc.num_reqs is None
    assert desc.num_tokens == 3
    assert desc.num_active_loras == 0


def test_basic_sd_does_not_capture_shorter_full_decode_shapes(monkeypatch):
    """Without DSD, only the max decode query length should get FULL graphs.

    Basic SD captures FULL decode graphs only for decode_query_len = K + 1.
    Uniform batches with smaller query lengths should therefore miss the FULL
    path entirely when using FULL_AND_PIECEWISE.
    """

    max_num_seqs = 512
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1

    _patch_cudagraph_test_runtime(monkeypatch)

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
        cudagraph_mode="FULL_AND_PIECEWISE",
        use_dynamic_sd=False,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_decode_query_len,
    )
    manager._graphs_captured = True

    for num_reqs in range(1, max_num_seqs + 1):
        for max_query_len in range(1, max_decode_query_len):
            # These are still uniform decode batches, but basic SD should only
            # have FULL graphs for query_len == max_decode_query_len.
            num_tokens = num_reqs * max_query_len
            uniform_tok_count = gpu_cudagraph_utils.get_uniform_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
            )
            assert uniform_tok_count == max_query_len

            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_tok_count,
                num_active_loras=0,
            )

            assert desc.cg_mode == CUDAGraphMode.PIECEWISE
            assert desc.uniform_token_count is None
            assert desc.num_tokens == num_tokens
            assert desc.num_reqs is None
            assert desc.num_active_loras == 0


def test_dynamic_sd_only_captures_scheduled_query_lengths(monkeypatch):
    """DSD should capture scheduled query lengths plus the Kmax family.

    With a partial schedule of ``(1, 32, 4)`` and ``(33, 128, 3)``, only the
    scheduled speculative-token counts (K = 4 and K = 3) become decode query
    lengths (K + 1 = 5 and 4). The Kmax family adds query length 8. Runtime
    K=0 is not captured unless explicitly scheduled.
    """

    max_num_seqs = 128
    max_spec_tokens = 7
    max_decode_query_len = max_spec_tokens + 1

    # (range_start, range_end, num_speculative_tokens): K = 4 and K = 3 are
    # scheduled. Kmax is retained for initial target/dummy execution.
    num_spec_per_batch_size = [(1, 32, 4), (33, 128, 3)]
    scheduled_query_lens = {max_decode_query_len}
    scheduled_query_lens.update(entry[2] + 1 for entry in num_spec_per_batch_size)

    _patch_cudagraph_test_runtime(monkeypatch)

    vllm_config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
        cudagraph_mode="FULL_AND_PIECEWISE",
        use_dynamic_sd=True,
        num_spec_per_batch_size=num_spec_per_batch_size,
    )
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        decode_query_len=max_decode_query_len,
    )
    manager._graphs_captured = True

    for num_reqs in range(1, max_num_seqs + 1):
        for max_query_len in range(1, max_decode_query_len + 1):
            num_tokens = num_reqs * max_query_len
            uniform_tok_count = gpu_cudagraph_utils.get_uniform_token_count(
                num_reqs,
                num_tokens,
                max_query_len,
            )
            assert uniform_tok_count == max_query_len

            desc = manager.dispatch(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                uniform_token_count=uniform_tok_count,
                num_active_loras=0,
            )

            if max_query_len in scheduled_query_lens:
                # Scheduled query lengths get a dedicated FULL decode graph.
                assert desc.cg_mode == CUDAGraphMode.FULL
                assert desc.uniform_token_count == max_query_len
                assert desc.num_tokens == num_tokens
                assert desc.num_reqs == num_reqs
            else:
                # Unscheduled query lengths (including the lower values 1 and 2)
                # have no FULL candidate and must fall back to PIECEWISE.
                assert desc.cg_mode == CUDAGraphMode.PIECEWISE
                assert desc.uniform_token_count is None
                assert desc.num_tokens == num_tokens
                assert desc.num_reqs is None
            assert desc.num_active_loras == 0


def test_dynamic_sd_sparse_capture_grid_dispatches_by_query_length(monkeypatch):
    max_num_seqs = 96
    max_spec_tokens = 7
    schedule = [(1, 16, 7), (17, 60, 5), (61, 96, 3)]

    _patch_cudagraph_test_runtime(monkeypatch)

    config = _create_vllm_config_for_dsd(
        max_num_seqs=max_num_seqs,
        max_spec_tokens=max_spec_tokens,
        cudagraph_mode="FULL_DECODE_ONLY",
        num_spec_per_batch_size=schedule,
        cudagraph_capture_sizes=_make_sparse_capture_sizes(),
        max_cudagraph_capture_size=512,
    )
    target_manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=max_spec_tokens + 1,
    )
    target_manager._graphs_captured = True

    target_desc = target_manager.dispatch(
        num_reqs=13,
        num_tokens=13 * 6,
        uniform_token_count=6,
        num_active_loras=0,
    )
    assert target_desc.cg_mode == CUDAGraphMode.FULL
    assert target_desc.uniform_token_count == 6
    assert target_desc.num_tokens == 84
    assert target_desc.num_reqs == 14

    for num_reqs in (84, 85):
        tail_desc = target_manager.dispatch(
            num_reqs=num_reqs,
            num_tokens=num_reqs * 6,
            uniform_token_count=6,
            num_active_loras=0,
        )
        assert tail_desc.cg_mode == CUDAGraphMode.FULL
        assert tail_desc.uniform_token_count == 6
        assert tail_desc.num_tokens == 510
        assert tail_desc.num_reqs == 85

    proposer_manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=config,
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=max_spec_tokens,
    )
    proposer_manager._graphs_captured = True

    proposer_desc = proposer_manager.dispatch(
        num_reqs=27,
        num_tokens=27 * 5,
        uniform_token_count=5,
        num_active_loras=0,
    )
    assert proposer_desc.cg_mode == CUDAGraphMode.FULL
    assert proposer_desc.uniform_token_count == 5
    assert proposer_desc.num_tokens == 140
    assert proposer_desc.num_reqs == 28


def test_dynamic_sd_sparse_grid_dispatch_survives_dp_sync(monkeypatch):
    max_num_seqs = 96
    max_spec_tokens = 7
    schedule = [(1, 16, 7), (17, 60, 5), (61, 96, 3)]

    _patch_cudagraph_test_runtime(monkeypatch)
    monkeypatch.setattr(
        gpu_dp_utils,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=None),
    )

    def make_manager(decode_query_len: int):
        config = _create_vllm_config_for_dsd(
            max_num_seqs=max_num_seqs,
            max_spec_tokens=max_spec_tokens,
            cudagraph_mode="FULL_DECODE_ONLY",
            num_spec_per_batch_size=schedule,
            cudagraph_capture_sizes=_make_sparse_capture_sizes(),
            max_cudagraph_capture_size=512,
        )
        manager = gpu_cudagraph_utils.CudaGraphManager(
            vllm_config=config,
            device=torch.device("cpu"),
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_len=decode_query_len,
        )
        manager._graphs_captured = True
        return manager

    def sync_dispatch(
        manager,
        *,
        num_reqs: int,
        num_tokens: int,
        query_len: int,
        peer_num_tokens: int,
        peer_query_len: int | None = None,
    ):
        def fake_all_reduce(tensor, group):
            tensor[0][1] = peer_num_tokens
            tensor[1][1] = CUDAGraphMode.FULL.value
            tensor[2][1] = peer_query_len or query_len

        monkeypatch.setattr(gpu_dp_utils.dist, "all_reduce", fake_all_reduce)
        desired_desc = manager.dispatch(
            num_reqs,
            num_tokens,
            query_len,
            num_active_loras=0,
        )
        return gpu_dp_utils.sync_cudagraph_and_dp_padding(
            manager,
            desired_desc,
            num_tokens,
            num_reqs,
            query_len,
            dp_size=2,
            dp_rank=0,
        )[0]

    target_desc = sync_dispatch(
        make_manager(max_spec_tokens + 1),
        num_reqs=12,
        num_tokens=12 * 6,
        query_len=6,
        peer_num_tokens=13 * 6,
    )
    assert target_desc.cg_mode == CUDAGraphMode.FULL
    assert target_desc.uniform_token_count == 6
    assert target_desc.num_tokens == 84

    proposer_desc = sync_dispatch(
        make_manager(max_spec_tokens),
        num_reqs=26,
        num_tokens=26 * 5,
        query_len=5,
        peer_num_tokens=27 * 5,
    )
    assert proposer_desc.cg_mode == CUDAGraphMode.FULL
    assert proposer_desc.uniform_token_count == 5
    assert proposer_desc.num_tokens == 140

    mixed_width_desc = sync_dispatch(
        make_manager(max_spec_tokens + 1),
        num_reqs=12,
        num_tokens=12 * 6,
        query_len=6,
        peer_num_tokens=10 * 8,
        peer_query_len=8,
    )
    assert mixed_width_desc.cg_mode == CUDAGraphMode.NONE
    assert mixed_width_desc.uniform_token_count is None


@pytest.mark.parametrize(
    ("actual_loras", "captured_loras"),
    [(0, 0), (1, 2), (2, 2), (3, 4), (5, 4)],
)
def test_dynamic_sd_uniform_families_preserve_lora_buckets(
    monkeypatch,
    actual_loras: int,
    captured_loras: int,
):
    _patch_cudagraph_test_runtime(monkeypatch)
    manager = gpu_cudagraph_utils.CudaGraphManager(
        vllm_config=_create_vllm_config_for_dsd(
            max_num_seqs=16,
            max_spec_tokens=3,
            cudagraph_mode="FULL_DECODE_ONLY",
            num_spec_per_batch_size=[(1, 8, 3), (9, 16, 1)],
            cudagraph_capture_sizes=[4, 8, 16, 32, 64],
            max_cudagraph_capture_size=64,
        ),
        device=torch.device("cpu"),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=4,
        lora_capture_cases=[0, 2, 4],
    )
    manager._graphs_captured = True

    desc = manager.dispatch(
        num_reqs=3,
        num_tokens=6,
        uniform_token_count=2,
        num_active_loras=actual_loras,
    )

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == 2
    assert desc.num_tokens == 8
    assert desc.num_active_loras == captured_loras
