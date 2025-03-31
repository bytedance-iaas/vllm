# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Iterable
from typing import Optional

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
                                         KVCacheBlock,
                                         generate_block_hash_extra_keys,
                                         hash_block_tokens)
from vllm.v1.request import Request
from vllm.v1.core.block_pool import BlockPool

logger = init_logger(__name__)


class GPUCPUBlockPool(BlockPool):
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The 
    free_block_queue stores the free blocks in eviction order to enable 
    allocation, free, and cache eviction. The cached_block_hash_to_block 
    maps between block hash and cached block to support finding cached blocks 
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, enable_caching: bool):
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.enable_caching = enable_caching
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        self.cpu_block_pool: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_cpu_blocks)
        ]

        self.next_available_cpu_block_id = 0
    
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # {block_hash: {block ID: block}}. A cached block is
        # a full block with a block hash that can be used for prefix caching.
        # The cached block may be used by running requests or in the
        # free_block_queue that could potentially be evicted.
        # NOTE: We currently don't de-duplicate the blocks in the cache,
        # meaning that if a block becomes full and is cached, we don't check
        # if there is already an identical block in the cache. This is because
        # we want to make sure the allocated block IDs won't change so that
        # block tables are append-only.
        self.cached_block_hash_to_block: dict[BlockHashType, dict[
            int, KVCacheBlock]] = defaultdict(dict)
        
        self.cached_block_hash_to_cpu_block: dict[BlockHashType,
                                                  KVCacheBlock] = {}
        
        self.step_d2h_swap_map: dict[int, int] = {}
        self.step_h2d_swap_map: dict[int, int] = {}
        # CPU blocks in use in this scheduling step i.e. source block for
        # swap-in and destination block for swap-out.
        self.step_cpu_block_in_use: set[int] = set()
    
    def get_cached_cpu_block(
            self, block_hash: BlockHashType) -> Optional[KVCacheBlock]:
        """Get a cached cpu block by the block hash, or None if cache miss.
        Args:
            block_hash: The hash value of the block.
        Returns:
            The cached cpu block if it exists, or None.
        """
        return self.cached_block_hash_to_cpu_block.get(block_hash, None)

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash and block_hash in self.cached_block_hash_to_block:
            if (block_hash not in self.cached_block_hash_to_cpu_block
                    and (target_cpu_block := self._get_new_cpu_block())):
                target_cpu_block_id = target_cpu_block.block_id
                self.step_d2h_swap_map[block.block_id] = target_cpu_block_id
                self.step_cpu_block_in_use.add(target_cpu_block_id)
                self.cpu_block_pool[target_cpu_block_id].block_hash = block_hash
                self.cached_block_hash_to_cpu_block[block_hash] = (
                    self.cpu_block_pool[target_cpu_block_id])
            block.reset_hash()
            del self.cached_block_hash_to_block[block_hash][block.block_id]

            if len(self.cached_block_hash_to_block[block_hash]) == 0:
                del self.cached_block_hash_to_block[block_hash]

            return True
        return False

    def _get_new_cpu_block(self) -> Optional[KVCacheBlock]:
        """Get a new cpu block from the cpu block pool, evict
        it from the cpu cache if it is cached.
        TODO(meng): currently we use a simple round-robin strategy
        to get a new cpu block. Implement LRU.
        Returns:
            A new cpu block, or None if all cpu blocks are used.
        """
        if len(self.step_cpu_block_in_use) == len(self.cpu_block_pool):
            return None

        while self.next_available_cpu_block_id in self.step_cpu_block_in_use:
            self.next_available_cpu_block_id = (
                self.next_available_cpu_block_id + 1) % len(
                    self.cpu_block_pool)

        target_cpu_block_id = self.next_available_cpu_block_id
        self.next_available_cpu_block_id = (self.next_available_cpu_block_id +
                                            1) % len(self.cpu_block_pool)

        # Evict the cpu block if it is cached.
        block_hash = self.cpu_block_pool[target_cpu_block_id].block_hash
        if block_hash is not None:
            del self.cached_block_hash_to_cpu_block[block_hash]
            self.cpu_block_pool[target_cpu_block_id].reset_hash()

        return self.cpu_block_pool[target_cpu_block_id]