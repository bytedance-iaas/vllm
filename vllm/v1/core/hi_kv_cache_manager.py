# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlock,
                                         hash_request_tokens)
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheManager

logger = init_logger(__name__)


class HiKVCacheManager(KVCacheManager):

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
        num_preallocate_tokens: int = 64,
        meta_manager = None,
        log_stats: bool = False
    ) -> None:
        super().__init__(block_size, num_gpu_blocks, max_model_len,  sliding_window, enable_caching, num_preallocate_tokens, log_stats)

        self.meta_manager = meta_manager

        # For scheduler output
        self.swap_in_req_map: dict[str, dict[int, int]] = {}
        self.swap_out_map: dict[int, int] = {}

        # Used to track blocks that are currently being swapped out;
        # these blocks cannot be evicted during this process.
        self.kv_cache_saving_blocks: dict[int, int] = {}

    def get_computed_blocks(
            self, request: Request) -> tuple[list[KVCacheBlock], list[BlockHashType], int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        if not self.enable_caching:
            # Prefix caching is disabled.
            return [], [], 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        self.prefix_cache_stats.requests += 1
        if request.sampling_params.prompt_logprobs is None:
            # Check for cache hits
            computed_blocks = []
            computed_extended_block_hash = []
            for block_hash in block_hashes:
                # block_hashes is a chain of block hashes. If a block hash
                # is not in the cached_block_hash_to_id, the following
                # block hashes are not computed yet for sure.
                if cached_block := self.block_pool.get_cached_block(
                        block_hash):
                    computed_blocks.append(cached_block)

                # check whether the cache hits in the extended storage.
                elif self.meta_manager.exist(str(block_hash.hash_value)):
                    computed_extended_block_hash.append(block_hash)
                else:
                    break

            self.prefix_cache_stats.queries += len(block_hashes)
            # TODO(luchangqi) Whether to distinguish hit rates across different storage media
            self.prefix_cache_stats.hits += len(computed_blocks) + len(computed_extended_block_hash)

            # NOTE(woosuk): Since incomplete blocks are not eligible for
            # sharing, `num_computed_tokens` is always a multiple of
            # `block_size`.
            num_computed_tokens = (len(computed_blocks) + len(computed_extended_block_hash)) * self.block_size
            return computed_blocks, computed_extended_block_hash,  num_computed_tokens
        else:
            # Skip cache hits for prompt logprobs
            return [],[], 0

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
        new_computed_extended_blocks: Optional[list[BlockHashType]] = None,
        saved_blocks: Optional[set[int]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        # remove the saved blocks.
        if saved_blocks is not None:
            for block_id in saved_blocks:
                assert block_id in self.kv_cache_saving_blocks
                self.kv_cache_saving_blocks[block_id] -= 1
                assert self.kv_cache_saving_blocks[block_id] >= 0
                if self.kv_cache_saving_blocks[block_id] == 0:
                    del self.kv_cache_saving_blocks[block_id]
            saved_blocks.clear()

        if num_tokens == 0:
            raise ValueError("num_tokens must be greater than 0")

        new_computed_blocks = new_computed_blocks or []
        new_computed_extended_blocks = new_computed_extended_blocks or []

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        has_computed_blocks = len(new_computed_blocks) + len(new_computed_extended_blocks)
        has_computed_tokens = request.num_computed_tokens + has_computed_blocks * self.block_size
        total_tokens = has_computed_tokens + num_tokens
        num_required_blocks = cdiv(total_tokens, self.block_size)
        req_blocks = self.req_to_blocks[request.request_id]
        num_new_blocks = (num_required_blocks - len(req_blocks) -
                          len(new_computed_blocks))
        
        # If a computed block of a request is an eviction candidate (in the
        # free queue and ref_cnt == 0), it cannot be counted as a free block
        # when allocating this request.
        num_evictable_computed_blocks = sum(1 for blk in new_computed_blocks
                                            if blk.ref_cnt == 0)
        if (num_new_blocks > self.block_pool.get_num_free_blocks(len(self.kv_cache_saving_blocks)) -
                num_evictable_computed_blocks):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not new_computed_blocks, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        req_blocks.extend(new_computed_blocks)

        # Start to handle new blocks

        if num_new_blocks <= 0:
            # No new block is needed.
            new_blocks = []
        else:
            # Get new blocks from the free block pool considering
            # preallocated blocks.
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(len(self.kv_cache_saving_blocks)),
                # Should not exceed the maximum number of blocks per request.
                # This is especially because the block table has the shape
                # [..., max_num_blocks_per_req].
                self.max_num_blocks_per_req - len(req_blocks),
            )
            assert num_new_blocks > 0

            # Concatenate the computed block IDs and the new block IDs.
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks, self.kv_cache_saving_blocks)
            req_blocks.extend(new_blocks)

        if not self.enable_caching:
            return new_blocks
        
        # Swap in the cpu blocks. Metadata update happens in
        # cache_full_blocks below.
        assert (len(new_computed_extended_blocks) < len(new_blocks)
                if new_computed_blocks else True)

        # Use `new_computed_blocks` for a new request, and `num_cached_block`
        # for a running request.
        num_cached_blocks = self.num_cached_block.get(request.request_id,
                                                      len(new_computed_blocks))
        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        num_full_blocks_after_append = (total_tokens - len(
            request.spec_token_ids)) // self.block_size

        new_cached_blk_list = self.block_pool.cache_full_blocks(
            request=request,
            blocks=req_blocks,
            block_hashes=self.req_to_block_hashes[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks_after_append,
            block_size=self.block_size,
        )

        # get scheduler output swap in
        if len(new_computed_extended_blocks) > 0:
            block_mapping = ({
                extended_block.hash_value: new_block.block_id
                for extended_block, new_block in zip(new_computed_extended_blocks,
                                                new_blocks)
            })

            self.swap_in_req_map[request.request_id] = block_mapping

        # TODO(luchangqi): add metadata test, avoid repeated swap out
        if new_cached_blk_list:
            for block in new_cached_blk_list:
                self.swap_out_map[block.block_id] = block.block_hash.hash_value
                if block.block_id not in self.kv_cache_saving_blocks:
                    self.kv_cache_saving_blocks[block.block_id] = 1
                else:
                    self.kv_cache_saving_blocks[block.block_id] += 1

        self.num_cached_block[
            request.request_id] = num_full_blocks_after_append
        return new_blocks
    
    def end_schedule_step(self) -> None:
        """A callback hook that is called when a scheduling step ends."""
        self.swap_in_req_map.clear()
        self.swap_out_map.clear()