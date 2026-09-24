# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def test_materialize_context_kv_profile_does_not_generate_drafts():
    observed = {}

    class Model:
        def combine_hidden_states(self, states):
            observed["combined"] = states.clone()
            return states[:, :2]

        def precompute_and_store_context_kv(self, states, positions):
            observed["states"] = states.clone()
            observed["positions"] = positions.clone()

    speculator = object.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 3
    speculator.max_model_len = 128
    speculator.hidden_states = torch.zeros(4, 2)
    speculator.context_positions = torch.tensor([7, 8, 0, 0])
    speculator.model = Model()
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=2,
        seq_lens_cpu_upper_bound=np.array([8]),
    )
    aux_hidden_states = [torch.ones(2, 1), torch.full((2, 1), 2.0)]

    DFlashSpeculator.materialize_context_kv(
        speculator,
        input_batch=input_batch,
        last_hidden_states=torch.zeros(2, 2),
        aux_hidden_states=aux_hidden_states,
        num_sampled=torch.ones(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=torch.zeros(1, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
        temperature=torch.zeros(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        dummy_run=True,
        skip_attn_for_dummy_run=True,
    )

    assert speculator.draft_max_seq_len == 11
    assert observed["combined"].tolist() == [[1.0, 2.0], [1.0, 2.0]]
    assert observed["states"].tolist() == [[1.0, 2.0], [1.0, 2.0]]
    assert observed["positions"].tolist() == [7, 8]


def test_profile_draft_does_not_reuse_target_dp_token_counts():
    observed = {}
    speculator = object.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 3
    speculator.num_speculative_steps = 2
    speculator.draft_tokens = torch.zeros(1, 2, dtype=torch.int64)
    speculator.materialize_context_kv = lambda *args, **kwargs: None
    speculator._prepare_eplb_forward = lambda num_tokens: None

    def generate_draft(*args, **kwargs):
        observed.update(kwargs)

    speculator._generate_draft = generate_draft
    input_batch = SimpleNamespace(num_reqs=1)
    target_dp_sync = SimpleNamespace(num_tokens_across_dp=torch.tensor([8192]))

    result = DFlashSpeculator.propose(
        speculator,
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros(1, 2),
        aux_hidden_states=None,
        num_sampled=torch.ones(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=torch.zeros(1, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
        temperature=torch.zeros(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        dp_sync=target_dp_sync,
        dummy_run=True,
        skip_attn_for_dummy_run=True,
    )

    assert observed["num_tokens_across_dp"] is None
    assert observed["cudagraph_runtime_mode"] == CUDAGraphMode.NONE
    assert result.shape == (1, 2)
