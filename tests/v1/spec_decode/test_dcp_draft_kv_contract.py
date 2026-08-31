# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheChildPageMapping,
    KVCacheRegionIdentity,
    KVCacheTemporalLayout,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import (
    get_eagle3_draft_attention_layer_name,
    get_prompt_draft_kv_coverage,
)
from vllm.v1.worker import utils as worker_utils


def _make_replicated_draft_proposer_for_validation(
    *,
    connector: str = "MooncakeConnector",
    role: str = "kv_consumer",
    module_path: str | None = None,
    load_failure_policy: str = "fail",
    dcp_size: int = 2,
    enforce_eager: bool = True,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    enable_prefix_caching: bool = False,
) -> SpecDecodeBaseProposer:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.replicated_draft_kv = True
    proposer.method = "eagle3"
    proposer.parallel_drafting = False
    proposer.needs_extra_input_slots = False
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=64,
            num_key_value_heads=64,
            head_dim=128,
        )
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=8,
        decode_context_parallel_size=dcp_size,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_size=1,
        use_ubatching=False,
        cp_kv_cache_interleave_size=128,
    )
    model_config = SimpleNamespace(
        architectures=["MiniMaxM3ForCausalLM"],
        enforce_eager=enforce_eager,
        get_num_layers=lambda _: 60,
    )
    proposer.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=model_config,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
            kv_offloading_size=None,
            block_size=128,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector=connector,
            kv_role=role,
            kv_connector_module_path=module_path,
            kv_load_failure_policy=load_failure_policy,
        ),
    )
    return proposer


def test_minimax_sparse_and_draft_temporal_layouts_are_distinct() -> None:
    assert KVCacheTemporalLayout.SHARDED_DCP != (KVCacheTemporalLayout.FULL_TEMPORAL)
    assert KVCacheTemporalLayout.SHARDED_DCP.value == "sharded_dcp"
    assert KVCacheTemporalLayout.FULL_TEMPORAL.value == "full_temporal"


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_replicated_draft_accepts_builtin_mooncake_consumer(
    dcp_size: int,
) -> None:
    proposer = _make_replicated_draft_proposer_for_validation(dcp_size=dcp_size)

    proposer._validate_replicated_draft_kv_config()


def test_replicated_draft_rejects_dcp3() -> None:
    proposer = _make_replicated_draft_proposer_for_validation(dcp_size=3)

    with pytest.raises(
        ValueError,
        match="decode_context_parallel_size must be 1 or 2",
    ):
        proposer._validate_replicated_draft_kv_config()


@pytest.mark.parametrize(
    ("enforce_eager", "cudagraph_mode"),
    [
        (True, CUDAGraphMode.NONE),
        (False, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_replicated_draft_execution_mode_allowlist(
    enforce_eager: bool,
    cudagraph_mode: CUDAGraphMode,
) -> None:
    proposer = _make_replicated_draft_proposer_for_validation(
        enforce_eager=enforce_eager,
        cudagraph_mode=cudagraph_mode,
    )

    proposer._validate_replicated_draft_kv_config()


def test_replicated_draft_allows_atomic_prefix_caching() -> None:
    proposer = _make_replicated_draft_proposer_for_validation(
        enable_prefix_caching=True
    )

    proposer._validate_replicated_draft_kv_config()


@pytest.mark.parametrize(
    ("enforce_eager", "cudagraph_mode", "error"),
    [
        (True, CUDAGraphMode.FULL_DECODE_ONLY, "eager execution requires"),
        (False, CUDAGraphMode.NONE, "non-eager execution requires"),
        (False, CUDAGraphMode.PIECEWISE, "non-eager execution requires"),
        (False, CUDAGraphMode.FULL, "non-eager execution requires"),
        (False, CUDAGraphMode.FULL_AND_PIECEWISE, "non-eager execution requires"),
    ],
)
def test_replicated_draft_rejects_unsupported_graph_modes(
    enforce_eager: bool,
    cudagraph_mode: CUDAGraphMode,
    error: str,
) -> None:
    proposer = _make_replicated_draft_proposer_for_validation(
        enforce_eager=enforce_eager,
        cudagraph_mode=cudagraph_mode,
    )

    with pytest.raises(ValueError, match=error):
        proposer._validate_replicated_draft_kv_config()


def test_replicated_draft_metadata_reuses_graph_buffers() -> None:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.dcp_world_size = 2
    proposer.max_model_len = 512
    proposer.block_size = 128
    proposer.cp_kv_cache_interleave_size = 128
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.positions = torch.tensor([0, 127, 128, 255], dtype=torch.int64)
    proposer._replicated_draft_block_table = torch.zeros((2, 4), dtype=torch.int32)
    proposer._replicated_draft_reject_mask = torch.ones(8, dtype=torch.bool)
    proposer._slot_mapping_buffer = torch.zeros(8, dtype=torch.int64)

    def metadata(
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> CommonAttentionMetadata:
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens,
            _seq_lens_cpu=seq_lens.cpu().clone(),
            seq_lens_cpu_upper_bound=seq_lens.cpu().clone(),
            num_reqs=seq_lens.numel(),
            num_actual_tokens=int(query_start_loc[-1]),
            max_query_len=int((query_start_loc[1:] - query_start_loc[:-1]).max()),
            max_seq_len=int(seq_lens.max()),
            block_table_tensor=block_table,
            slot_mapping=torch.empty(0, dtype=torch.int64),
        )

    first = proposer._build_replicated_draft_metadata(
        metadata(
            torch.tensor([[3, 5], [7, 9]], dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int32),
            torch.tensor([128, 256], dtype=torch.int32),
        ),
        num_tokens=4,
    )
    block_ptr = first.block_table_tensor.data_ptr()
    slot_ptr = first.slot_mapping.data_ptr()
    assert first.block_table_tensor.tolist() == [[6, 7, 10, 11], [14, 15, 18, 19]]
    assert not proposer._replicated_draft_reject_mask[:4].any()

    proposer.positions[0] = 256
    second = proposer._build_replicated_draft_metadata(
        metadata(
            torch.tensor([[11, 13]], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([257], dtype=torch.int32),
        ),
        num_tokens=1,
    )
    assert second.block_table_tensor.data_ptr() == block_ptr
    assert second.slot_mapping.data_ptr() == slot_ptr
    assert second.block_table_tensor.tolist() == [[22, 23, 26, 27]]
    assert second.slot_mapping.tolist() == [3328]


@pytest.mark.parametrize(
    ("connector", "role", "module_path", "load_failure_policy"),
    [
        ("MooncakeConnector", "kv_producer", None, "fail"),
        ("MooncakeConnector", "kv_consumer", "custom.connector", "fail"),
        ("MooncakeConnector", "kv_consumer", None, "recompute"),
        ("NixlConnector", "kv_consumer", None, "fail"),
    ],
)
def test_replicated_draft_rejects_unsupported_kv_transfer(
    connector: str,
    role: str,
    module_path: str | None,
    load_failure_policy: str,
) -> None:
    proposer = _make_replicated_draft_proposer_for_validation(
        connector=connector,
        role=role,
        module_path=module_path,
        load_failure_policy=load_failure_policy,
    )

    with pytest.raises(ValueError, match="built-in Mooncake consumer"):
        proposer._validate_replicated_draft_kv_config()


def test_attention_spec_preserves_temporal_layout_when_block_size_changes() -> None:
    spec = FullAttentionSpec(
        block_size=128,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
    )

    resized = spec.copy_with_new_block_size(64)

    assert resized.block_size == 64
    assert resized.temporal_layout == KVCacheTemporalLayout.FULL_TEMPORAL


def test_attention_spec_merge_rejects_mixed_temporal_layouts() -> None:
    common = {
        "block_size": 128,
        "num_kv_heads": 8,
        "head_size": 128,
        "dtype": torch.float16,
    }
    sharded = FullAttentionSpec(
        **common,
        temporal_layout=KVCacheTemporalLayout.SHARDED_DCP,
    )
    full = FullAttentionSpec(
        **common,
        temporal_layout=KVCacheTemporalLayout.FULL_TEMPORAL,
    )

    with pytest.raises(AssertionError):
        FullAttentionSpec.merge([sharded, full])


def test_parent_block_copy_moves_both_full_temporal_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.arange(3 * 2, dtype=torch.uint8).reshape(3, 2)
    draft = torch.arange(6 * 2, dtype=torch.uint8).reshape(6, 2)
    target_before = target.clone()
    draft_before = draft.clone()
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda value, device: torch.from_numpy(value).to(device),
    )

    worker_utils.copy_kv_cache_blocks_inplace(
        [target, draft],
        num_blocks=3,
        kv_cache_block_copies=[(0, 2)],
    )

    assert torch.equal(target[2], target_before[0])
    assert torch.equal(draft[4:6], draft_before[0:2])


def test_region_identity_includes_temporal_layout() -> None:
    layer_name = "model.layers.60.self_attn.attn"
    sharded = KVCacheRegionIdentity(
        layer_name,
        KVCacheTemporalLayout.SHARDED_DCP,
        protocol_version=3,
        child_page_mapping=KVCacheChildPageMapping.IDENTITY,
        child_page_factor=1,
    )
    full = KVCacheRegionIdentity(
        layer_name,
        KVCacheTemporalLayout.FULL_TEMPORAL,
        protocol_version=3,
        child_page_mapping=KVCacheChildPageMapping.GLOBAL_PAGE_MODULO,
        child_page_factor=2,
    )

    assert sharded != full
    assert sharded.protocol_key == (
        layer_name,
        "sharded_dcp",
        3,
        "identity",
        1,
    )
    assert full.protocol_key == (
        layer_name,
        "full_temporal",
        3,
        "global_page_modulo",
        2,
    )


def test_region_identity_rejects_protocol_or_child_mapping_mismatch() -> None:
    base = {
        "layer_name": "model.layers.60.self_attn.attn",
        "temporal_layout": KVCacheTemporalLayout.FULL_TEMPORAL,
        "protocol_version": 3,
        "child_page_mapping": KVCacheChildPageMapping.GLOBAL_PAGE_MODULO,
        "child_page_factor": 2,
    }
    expected = KVCacheRegionIdentity(**base)

    assert expected != KVCacheRegionIdentity(**{**base, "protocol_version": 2})
    assert expected != KVCacheRegionIdentity(
        **{**base, "child_page_mapping": KVCacheChildPageMapping.IDENTITY}
    )
    assert expected != KVCacheRegionIdentity(**{**base, "child_page_factor": 1})


def test_eagle3_draft_layer_identity_is_global_under_pp1_and_pp2() -> None:
    assert get_eagle3_draft_attention_layer_name(60) == (
        "model.layers.60.self_attn.attn"
    )
    assert get_eagle3_draft_attention_layer_name(60, 1) == (
        "model.layers.61.self_attn.attn"
    )


@pytest.mark.parametrize(
    ("total_target_layers", "draft_layer_index"),
    [(0, 0), (-1, 0), (30, 0), (61, 0), (60, -1)],
)
def test_invalid_draft_layer_identity_fails_closed(
    total_target_layers: int,
    draft_layer_index: int,
) -> None:
    with pytest.raises(ValueError):
        get_eagle3_draft_attention_layer_name(
            total_target_layers,
            draft_layer_index,
        )


def test_prompt_draft_kv_final_token_is_decode_owned() -> None:
    coverage = get_prompt_draft_kv_coverage(
        prompt_tokens=257,
        target_prefix_tokens=0,
        compatible_draft_prefix_tokens=0,
    )

    assert coverage.transfer_start_token == 0
    assert coverage.transfer_end_token_exclusive == 256
    assert coverage.transfer_token_count == 256
    assert coverage.decode_recompute_position == 256


def test_prompt_draft_kv_compatible_prefix_transfers_exact_suffix() -> None:
    coverage = get_prompt_draft_kv_coverage(
        prompt_tokens=3500,
        target_prefix_tokens=1024,
        compatible_draft_prefix_tokens=768,
    )

    assert coverage.transfer_start_token == 768
    assert coverage.transfer_end_token_exclusive == 3499
    assert coverage.transfer_token_count == 2731


def test_target_only_prefix_does_not_shrink_draft_suffix() -> None:
    coverage = get_prompt_draft_kv_coverage(
        prompt_tokens=256,
        target_prefix_tokens=128,
        compatible_draft_prefix_tokens=0,
    )

    assert coverage.transfer_start_token == 0
    assert coverage.transfer_end_token_exclusive == 255


def test_full_draft_prefix_still_recomputes_final_token() -> None:
    coverage = get_prompt_draft_kv_coverage(
        prompt_tokens=128,
        target_prefix_tokens=128,
        compatible_draft_prefix_tokens=128,
    )

    assert coverage.transfer_start_token == 127
    assert coverage.transfer_end_token_exclusive == 127
    assert coverage.transfer_token_count == 0
    assert coverage.decode_recompute_position == 127


@pytest.mark.parametrize(
    ("prompt_tokens", "target_prefix_tokens", "draft_prefix_tokens"),
    [
        (0, 0, 0),
        (128, -1, 0),
        (128, 129, 0),
        (128, 64, -1),
        (128, 64, 65),
    ],
)
def test_invalid_prompt_or_prefix_coverage_fails_closed(
    prompt_tokens: int,
    target_prefix_tokens: int,
    draft_prefix_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        get_prompt_draft_kv_coverage(
            prompt_tokens,
            target_prefix_tokens,
            draft_prefix_tokens,
        )
