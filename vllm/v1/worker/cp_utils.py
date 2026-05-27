# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.v1.utils import CpuGpuBuffer

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


class PCPManager:
    """Build per-rank token metadata for Prefill Context Parallelism.

    PCP splits long prefill requests across ranks using a head/tail chunk
    assignment. Decode requests are not split; they are replicated on every PCP
    rank so mixed decode+prefill batches keep decode semantics unchanged.

    The manager owns the small CPU/GPU buffers needed by the model runner to:

    * replace scheduled token counts with the local PCP-rank token counts;
    * build the local token positions for this PCP rank;
    * mask padding after PCP all-gather;
    * restore all-gathered hidden/KV tensors back to original request order.
    """

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        max_buffer_num_tokens: int,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        assert pcp_world_size > 1
        assert 0 <= pcp_rank < pcp_world_size
        self.pcp_world_size = pcp_world_size
        self.pcp_rank = pcp_rank

        self.pcp_allgather_restore_idx = CpuGpuBuffer(
            max_buffer_num_tokens,
            dtype=torch.int64,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_padded_slot_mapping = torch.empty(
            (max_buffer_num_tokens,),
            dtype=torch.int64,
            device=device,
        )
        self.num_pcp_pads_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int64
        )
        self.num_pcp_pads_cpu = self.num_pcp_pads_cpu_tensor.numpy()
        self.pcp_unpad_mask = CpuGpuBuffer(
            (max_buffer_num_tokens,),
            dtype=torch.bool,
            device=device,
            pin_memory=pin_memory,
        )
        self.pcp_unpad_mask_cpu_tensor = self.pcp_unpad_mask.cpu
        self.pcp_unpad_mask_gpu_tensor = self.pcp_unpad_mask.gpu
        self.pcp_unpad_mask_cpu = self.pcp_unpad_mask_cpu_tensor.numpy()

    @staticmethod
    def _get_cumsum_and_arange(
        num_scheduled_tokens: np.ndarray,
        arange_np: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cumulative token counts and per-request aranges.

        Example: [2, 5, 3] -> ([2, 7, 10],
        [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]).
        """
        cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        cumsums_offsets = np.repeat(
            cu_num_tokens - num_scheduled_tokens, num_scheduled_tokens
        )
        arange = arange_np[:total_num_tokens] - cumsums_offsets
        return cu_num_tokens, arange

    def update_tokens_for_pcp(
        self,
        tokens: np.ndarray,
        arange_np: np.ndarray,
        num_reqs: int,
        reorder_batch_threshold: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update token counts and positions for this PCP rank.

        Args:
            tokens: Scheduled token counts per request before PCP splitting.
            arange_np: Reusable arange buffer large enough for the padded batch.
            num_reqs: Number of active requests in the prefix of ``tokens``.
            reorder_batch_threshold: Decode/prefill split threshold used by MLA
                metadata builders. Requests with scheduled tokens less than or
                equal to this threshold are treated as decode requests.

        Returns:
            ``(pcp_tokens, pcp_positions)`` for this rank.
        """
        assert reorder_batch_threshold is not None, (
            "PCP depends on reorder batch to split decode and prefill requests."
        )
        tokens = tokens[:num_reqs]
        num_decode_reqs = int(np.sum(tokens <= reorder_batch_threshold))
        num_decode_tokens = int(np.sum(tokens[:num_decode_reqs]))

        num_padded_scheduled_tokens = (
            np.ceil(tokens / (2 * self.pcp_world_size)).astype(np.int32)
            * (2 * self.pcp_world_size)
        )
        num_padded_scheduled_tokens[:num_decode_reqs] = (
            tokens[:num_decode_reqs] * self.pcp_world_size
        )

        self.num_pcp_pads_cpu[:num_reqs] = num_padded_scheduled_tokens - tokens

        cu_padded_tokens, pcp_padded_arange = self._get_cumsum_and_arange(
            num_padded_scheduled_tokens, arange_np
        )
        self.pcp_unpad_mask_cpu[: pcp_padded_arange.shape[0]] = (
            pcp_padded_arange < np.repeat(tokens, num_padded_scheduled_tokens)
        )
        self.pcp_unpad_mask.copy_to_gpu(pcp_padded_arange.shape[0])

        pcp_tokens = num_padded_scheduled_tokens // self.pcp_world_size
        pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
        pcp_chunk_sizes[:num_decode_reqs] = pcp_tokens[:num_decode_reqs]

        _, pcp_arange = self._get_cumsum_and_arange(pcp_tokens, arange_np)
        _, pcp_chunk_arange = self._get_cumsum_and_arange(
            pcp_chunk_sizes, arange_np
        )
        pcp_head_chunk_mask = pcp_arange < np.repeat(pcp_chunk_sizes, pcp_tokens)

        def get_current_rank_positions(
            positions_start_loc: int | np.ndarray, rank: int
        ) -> np.ndarray:
            positions = np.zeros(len(pcp_head_chunk_mask), dtype=np.int32)
            head_start_loc = positions_start_loc + rank * pcp_chunk_sizes
            tail_start_loc = (
                positions_start_loc
                + (2 * self.pcp_world_size - rank - 1) * pcp_chunk_sizes
            )
            positions[pcp_head_chunk_mask] = pcp_chunk_arange + np.repeat(
                head_start_loc, pcp_chunk_sizes
            )
            positions[~pcp_head_chunk_mask] = (
                pcp_chunk_arange[num_decode_tokens:]
                + np.repeat(tail_start_loc, pcp_chunk_sizes)[num_decode_tokens:]
            )
            return positions

        positions = get_current_rank_positions(0, self.pcp_rank)
        if num_decode_reqs > 0:
            positions[:num_decode_tokens] = self._get_cumsum_and_arange(
                tokens[:num_decode_reqs], arange_np
            )[1]

        padded_pos_start_loc = np.roll(cu_padded_tokens, 1)
        padded_pos_start_loc[0] = 0
        all_positions = np.concatenate(
            [
                get_current_rank_positions(padded_pos_start_loc, rank)
                for rank in range(self.pcp_world_size)
            ]
        )
        restore_idx = all_positions.argsort()
        self.pcp_allgather_restore_idx.np[: restore_idx.shape[0]] = restore_idx
        self.pcp_allgather_restore_idx.copy_to_gpu(restore_idx.shape[0])

        return pcp_tokens[:num_reqs], positions


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


def get_total_cp_world_size():
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP might not be initialized in testing
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size
