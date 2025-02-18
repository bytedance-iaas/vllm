# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The InfinistoreConnector transfers KV caches between prefill vLLM worker (KV cache 
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
import array
import hashlib
import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.distributed.kv_transfer.kv_pipe.infinistore_pipe import InfinistoreTransporter

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

def compute_prompt_id_hashes(prompt_ids: List[int], prev_hash : str = "") -> str:
    tokens_bytes = array.array('l', prompt_ids).tobytes()
    hash_input = prev_hash.encode('utf-8') + tokens_bytes
    
    return hashlib.sha256(hash_input).hexdigest()

def compute_block_hashes(prompt_ids: List[int], prompt_seq_lengths: List[int],
                         block_size: int) -> List[str]:

    hashes = []
    seq_index = 0

    for seq_len in prompt_seq_lengths:
        seq_tokens = prompt_ids[seq_index:seq_index + seq_len]
        num_pages = math.ceil(seq_len / block_size)
        prev_hash = ""

        # Loop over each page within the current sequence
        for page_num in range(num_pages):
            start_token = page_num * block_size
            end_token = min((page_num + 1) * block_size, seq_len)
            tokens_in_page = seq_tokens[start_token:end_token]

            # Compute hash for the current page
            current_hash = compute_prompt_id_hashes(tokens_in_page, prev_hash)

            prev_hash = current_hash

            hashes.append(current_hash)

        seq_index += seq_len

    return hashes


class InfinistoreConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        
        try:
            import infinistore
        except ImportError as e:
            raise ImportError(
                "Please install infinistore (pip install infinistore)  " 
                "to run vLLM with InfinistoreConnector.") from e
        
        if config.kv_connector != "InfinistoreConnector":
            raise ValueError(
                "InfinistoreConnector does not support kv_connector %s",
                config.kv_connector)
        
        self.infinistore_transporter =  InfinistoreTransporter(VllmConfig)

    
    def _compute_kv_cache_offsets(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ):
        offsets = []
        attn_metadata = model_input.attn_metadata
        k_or_v_total_size = kv_caches[0].numel()
        page_size = kv_caches[0][0].numel()
        seq_start_index = 0  # Added missing initialization

        for seq_length in attn_metadata.seq_lens:
            num_pages = math.ceil(seq_length / self.block_size)

            for page_num in range(num_pages):
                start_token_idx = page_num * self.block_size
                slot_mapping_value = attn_metadata.slot_mapping[seq_start_index + start_token_idx].item()
                block_id = slot_mapping_value // self.block_size
                k_offset = block_id * page_size
                offsets.append((k_offset, k_offset + k_or_v_total_size))

            seq_start_index += seq_length  # Ensure correct index tracking

        return offsets

    def _prepare_kv_cache_transfer(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ):
        """Prepares offsets and block hashes for KV cache transfer."""
        offsets = self._compute_kv_cache_offsets(model_input, kv_caches)
        block_hashes = compute_block_hashes(model_input.input_tokens, model_input.attn_metadata.seq_lens, self.block_size)
        assert len(block_hashes) == len(offsets)
        return offsets, block_hashes

    def _transfer_kv_caches(
        self,
        model_executable: torch.nn.Module,
        kv_caches: List[torch.Tensor],
        model_input: "ModelInputForGPUWithSamplingMetadata",
        send: bool
    ):
        """Handles sending or receiving KV caches based on `send` flag."""
        offsets, block_hashes = self._prepare_kv_cache_transfer(model_input, kv_caches)
        start_layer, end_layer = model_executable.model.start_layer, model_executable.model.end_layer
        page_size = kv_caches[0][0].numel()

        for layer_idx in range(start_layer, end_layer):
            block_offsets: List[Tuple[str, int]] = [
                (self._get_kv_cache_key(hash_val, layer_idx), offset_val)
                for hash_val, offset in zip(block_hashes, offsets)
                for offset_val in offset  # Expands to (k_cache_offset, v_cache_offset)
            ]
            
            transporter = self.infinistore_transporter
            tensor = kv_caches[layer_idx - start_layer]
            if send:
                transporter.send_tensor(tensor, block_offsets, page_size)
            else:
                transporter.read_tensor(tensor, block_offsets, page_size)

    def send_kv_caches(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ):
        """Sends KV cache tensors to InfiniStore."""
        self._transfer_kv_caches(model_executable, kv_caches, model_input, send=True)

    def recv_kv_caches(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ):
        """Receives KV cache tensors from InfiniStore."""
        self._transfer_kv_caches(model_executable, kv_caches, model_input, send=False)

    def send_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        attn_metadata = model_input.attn_metadata
        input_tokens = model_input.input_tokens.tolist()
        hidden_size = hidden_or_intermediate_states.size(-1)


        start_pos = 0
        for seq_length in attn_metadata.seq_lens:
            end_pos = start_pos + seq_length
            seq_hash = compute_prompt_id_hashes(input_tokens[start_pos:end_pos])
            self.infinistore_transporter.send_tensor(hidden_or_intermediate_states, [(seq_hash, start_pos * hidden_size)], hidden_size*seq_length)
            start_pos = end_pos

    def recv_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
    ) -> None:
        attn_metadata = model_input.attn_metadata
        input_tokens = model_input.input_tokens.tolist()
        model_config = model_executable.model.config
        hidden_size = model_config.hf_config.hidden_size

        hidden_states = torch.zeros((attn_metadata.total_seq_len, hidden_size), dtype=torch.float32)

        start_pos = 0
        for seq_length in attn_metadata.seq_lens:
            end_pos = start_pos + seq_length
            seq_hash = compute_prompt_id_hashes(input_tokens[start_pos:end_pos])
            self.infinistore_transporter.recv_tensor(hidden_states, [(seq_hash, start_pos * hidden_size)], hidden_size*seq_length)
            start_pos = end_pos

        return hidden_states

    
    def _get_kv_cache_key(self, page_hash: str,
                         layer_idx: int) -> Tuple[str, str]:
        k_cache_key = self.kv_key_initial + f"{layer_idx}_{page_hash}_k"
        v_cache_key = self.kv_key_initial + f"{layer_idx}_{page_hash}_v"

        return k_cache_key, v_cache_key
    
    def synchronize_transporter(self):
        self.infinistore_transporter.synchronize()

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        
        self.send_kv_cache(model_executable, model_input, kv_caches)

        self.send_hidden_states(model_executable, model_input, hidden_or_intermediate_states)

        self.synchronize_transporter()


    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        try:
            self.recv_kv_caches(model_executable, model_input, kv_caches)

            hidden_states = self.recv_hidden_states(model_executable, model_input)
            bypass_model_exec = False
        except Exception as e:
            logger.error("Failed to receive KV caches and hidden ", e)
            return None, True, model_input

        logger.debug(
            "[rank%d]: Successfully received all KVs and hidden "
            "states, skip model forwarding.", torch.distributed.get_rank())

        return hidden_states, bypass_model_exec, model_input

    def close(self):
        pass
