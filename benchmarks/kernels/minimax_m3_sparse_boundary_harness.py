# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 sparse-attention minimal-boundary equivalence harness.

This is an offline diagnostic for the Sparse Attention output boundary.  It
does not execute qkv projection, RoPE, KV insertion, the indexer, compiler
matchers, or serving instrumentation.  It constructs frozen q/index_q/top-k/KV
state, runs the current main sparse-attention implementation twice, and checks
that the comparison plumbing catches mismatches through:

* q / index_q / top-k snapshots;
* main sparse-attention output;
* rank-local o_proj-style output;
* global all-reduced o_proj-style output;
* post-attention RMSNorm output;
* a synthetic logit probe derived from the post-attention hidden state.

The candidate is intentionally `self-check` for Round 68.  Future candidate
custom ops should plug into `run_candidate_boundary` without widening the
boundary to qkv projection, RoPE, KV insertion, or indexer state mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

DTYPE = torch.bfloat16
HEAD_DIM = 128
SPARSE_BLOCK_SIZE = 128
TOTAL_Q_HEADS = 64
TOTAL_KV_HEADS = 4
HIDDEN_SIZE = 6144
TOPK_BLOCKS = 16
LOGIT_PROBE_SIZE = 128


def layer_name_from_id(layer_id: int) -> str:
    if layer_id < 3:
        raise ValueError("MiniMax-M3 sparse attention starts at layer 3")
    return f"model.layers.{layer_id}.self_attn.attn"


@dataclass(frozen=True)
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]

    def __post_init__(self) -> None:
        if len(self.seq_lens) != len(self.query_lens):
            raise ValueError("seq_lens and query_lens must have the same length")

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lens)


class HarnessLayer:
    def __init__(
        self,
        *,
        layer_name: str,
        topk_indices_buffer: torch.Tensor,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
    ) -> None:
        self.layer_name = layer_name
        self.topk_indices_buffer = topk_indices_buffer
        self._k_scale = k_scale
        self._v_scale = v_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline MiniMax-M3 sparse-attention boundary checker."
    )
    parser.add_argument("--model", default=None, help="Accepted for runbook parity.")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--candidate", default="self-check", choices=("self-check",))
    parser.add_argument("--scenario", choices=("decode", "prefill"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=("auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2"),
    )
    parser.add_argument(
        "--indexer-kv-dtype",
        default="auto",
        help="Accepted for future real-indexer parity; currently not used.",
    )
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument(
        "--inject-mismatch",
        choices=("none", "attn_output", "topk", "global_o_proj"),
        default="none",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device, Any]:
    from vllm.config import CompilationConfig, ParallelConfig
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    if not torch.cuda.is_available():
        raise RuntimeError("This harness requires CUDA GPUs.")
    if not current_platform.is_cuda() or current_platform.is_rocm():
        raise RuntimeError("This harness currently supports NVIDIA CUDA only.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    nnodes = max(1, (world_size + local_world_size - 1) // local_world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    parallel_config = ParallelConfig(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        disable_custom_all_reduce=True,
        distributed_executor_backend="external_launcher",
        nnodes=nnodes,
    )
    # Keep vLLM's distributed initializers on the simple non-MoE path.
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        model_config=SimpleNamespace(is_moe=False),
        compilation_config=CompilationConfig(),
        speculative_config=None,
        kv_transfer_config=None,
    )

    # Store the config globally for the duration of the process.  This script is
    # a short-lived diagnostic, so there is no need for a nested context manager.
    from vllm.config import vllm as vllm_config_module

    vllm_config_module._current_vllm_config = vllm_config
    init_method = "env://"
    if world_size == 1 and (
        "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ
    ):
        init_method = f"file:///tmp/minimax_m3_sparse_boundary_{os.getpid()}"
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=init_method,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, local_rank, world_size, device, vllm_config


def make_common_metadata(
    batch: BatchSpec,
    device: torch.device,
) -> Any:
    from vllm.v1.attention.backend import CommonAttentionMetadata

    query_start_loc = torch.zeros(
        batch.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc[1:] = torch.tensor(
        batch.query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    seq_lens = torch.tensor(batch.seq_lens, dtype=torch.int32, device=device)
    context_lens = [
        seq_len - query_len
        for seq_len, query_len in zip(batch.seq_lens, batch.query_lens)
    ]
    max_blocks = (max(batch.seq_lens) + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
    block_table = torch.arange(
        batch.batch_size * max_blocks, dtype=torch.int32, device=device
    ).view(batch.batch_size, max_blocks)
    slot_mapping = torch.arange(batch.num_tokens, dtype=torch.int64, device=device)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        _seq_lens_cpu=seq_lens.cpu(),
        _num_computed_tokens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        num_reqs=batch.batch_size,
        num_actual_tokens=batch.num_tokens,
        max_query_len=max(batch.query_lens),
        max_seq_len=max(batch.seq_lens),
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
    )


def make_batch(args: argparse.Namespace) -> BatchSpec:
    if args.context_len <= TOPK_BLOCKS * SPARSE_BLOCK_SIZE:
        raise ValueError(
            "context_len must exceed top-k coverage so sparse selection is non-trivial"
        )
    if args.scenario == "decode":
        if args.query_len != 1:
            raise ValueError(
                "decode scenario requires query_len=1; use prefill for "
                "multi-token query validation"
            )
        return BatchSpec(
            seq_lens=[args.context_len + args.query_len] * args.batch_size,
            query_lens=[args.query_len] * args.batch_size,
        )

    if args.batch_size != 1:
        raise ValueError("prefill scenario currently expects --batch-size 1")
    if args.query_len <= 1:
        raise ValueError("prefill scenario needs query_len > 1")
    return BatchSpec(
        seq_lens=[args.context_len + args.query_len],
        query_lens=[args.query_len],
    )


def make_topk(
    batch: BatchSpec,
    num_kv_heads: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    base = torch.arange(TOPK_BLOCKS, dtype=torch.int32, device=device)
    head_offsets = torch.arange(
        num_kv_heads,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(1)
    for seq_len, query_len in zip(batch.seq_lens, batch.query_lens):
        context_len = seq_len - query_len
        for local_q in range(query_len):
            token_index = len(rows)
            absolute_pos = context_len + local_q
            max_valid_block = max(0, absolute_pos // SPARSE_BLOCK_SIZE)
            num_valid_blocks = max_valid_block + 1
            row = (
                base.unsqueeze(0) + token_index * 3 + head_offsets * 7
            ) % num_valid_blocks
            rows.append(row.to(torch.int32).contiguous())
    return torch.stack(rows, dim=0).contiguous()


def make_kv_cache(
    *,
    num_pages: int,
    num_kv_heads: int,
    device: torch.device,
    dtype_name: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseBackend
    from vllm.platforms import current_platform

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = MiniMaxM3SparseBackend.get_kv_cache_shape(
        num_pages,
        SPARSE_BLOCK_SIZE,
        num_kv_heads,
        HEAD_DIM,
    )
    stride_order = MiniMaxM3SparseBackend.get_kv_cache_stride_order()
    physical_shape = tuple(shape[i] for i in stride_order)
    inv_order = [stride_order.index(i) for i in range(len(stride_order))]
    bf16_physical = (
        torch.randn(
            physical_shape,
            dtype=DTYPE,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    if dtype_name in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        if dtype_name == "fp8_e5m2":
            fp8_dtype = (
                torch.float8_e5m2fnuz
                if current_platform.is_fp8_fnuz()
                else torch.float8_e5m2
            )
        else:
            fp8_dtype = current_platform.fp8_dtype()
        scale = torch.ones((), dtype=torch.float32, device=device)
        return bf16_physical.to(fp8_dtype).permute(*inv_order), scale, scale
    return bf16_physical.permute(*inv_order), None, None


def tensor_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    if lhs.shape != rhs.shape:
        return {
            "shape_match": False,
            "lhs_shape": list(lhs.shape),
            "rhs_shape": list(rhs.shape),
        }
    if lhs.dtype == torch.int32 or rhs.dtype == torch.int32:
        mismatch = lhs != rhs
        return {
            "shape_match": True,
            "mismatch_count": int(mismatch.sum().item()),
            "exact": bool(not mismatch.any().item()),
        }
    diff = lhs.detach().float() - rhs.detach().float()
    abs_diff = diff.abs()
    max_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
    mean_abs = float(abs_diff.mean().item()) if abs_diff.numel() else 0.0
    return {
        "shape_match": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "exact": max_abs == 0.0,
    }


def gemma_rms_norm(hidden: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
    added = hidden + residual
    var = added.float().pow(2).mean(dim=-1, keepdim=True)
    out = added.float() * torch.rsqrt(var + 1e-6)
    out = out * (1.0 + weight.float())
    return out.to(hidden.dtype), added


def run_main_sparse_attn(
    *,
    impl: Any,
    layer: HarnessLayer,
    metadata: dict[str, Any],
    vllm_config: Any,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
) -> torch.Tensor:
    from vllm.forward_context import set_forward_context

    output = torch.empty_like(q)
    with set_forward_context(metadata, vllm_config, num_tokens=q.shape[0]):
        return impl.forward(layer, q, kv_cache, output).clone()


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    from vllm.config import set_current_vllm_config
    from vllm.models.minimax_m3.common.sparse_attention import (
        MiniMaxM3SparseMetadataBuilder,
        select_main_impl_cls,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    rank, _local_rank, world_size, device, vllm_config = setup_distributed()
    batch = make_batch(args)
    layer_name = layer_name_from_id(args.layer)
    tp_size = world_size
    if TOTAL_Q_HEADS % tp_size != 0:
        raise ValueError(
            f"TOTAL_Q_HEADS={TOTAL_Q_HEADS} is not divisible by TP={tp_size}"
        )
    rank_seed = args.seed + rank * 100_000
    local_q_heads = TOTAL_Q_HEADS // tp_size
    local_kv_heads = max(1, TOTAL_KV_HEADS // tp_size)
    q_size = local_q_heads * HEAD_DIM
    common = make_common_metadata(batch, device)
    max_blocks = common.block_table_tensor.shape[1]
    num_pages = batch.batch_size * max_blocks

    spec = FullAttentionSpec(
        block_size=SPARSE_BLOCK_SIZE,
        num_kv_heads=local_kv_heads,
        head_size=HEAD_DIM,
        dtype=DTYPE,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(rank_seed)
    q = torch.randn(
        batch.num_tokens,
        q_size,
        dtype=DTYPE,
        device=device,
        generator=generator,
    )
    index_q = torch.randn(
        batch.num_tokens,
        local_kv_heads * HEAD_DIM,
        dtype=DTYPE,
        device=device,
        generator=generator,
    )
    kv_cache, k_scale, v_scale = make_kv_cache(
        num_pages=num_pages,
        num_kv_heads=local_kv_heads,
        device=device,
        dtype_name=args.kv_cache_dtype,
        seed=rank_seed + 1,
    )
    topk = make_topk(batch, local_kv_heads, device)
    layer = HarnessLayer(
        layer_name=layer_name,
        topk_indices_buffer=topk,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    with set_current_vllm_config(vllm_config):
        builder = MiniMaxM3SparseMetadataBuilder(
            spec,
            [layer_name],
            vllm_config,
            device,
        )
        metadata = {layer_name: builder.build(0, common)}
        impl_cls = select_main_impl_cls(
            topk_blocks=TOPK_BLOCKS,
            kv_cache_dtype=args.kv_cache_dtype,
            num_kv_heads=local_kv_heads,
        )
        impl = impl_cls(
            num_heads=local_q_heads,
            head_size=HEAD_DIM,
            scale=HEAD_DIM**-0.5,
            num_kv_heads=local_kv_heads,
            kv_cache_dtype=args.kv_cache_dtype,
            topk_blocks=TOPK_BLOCKS,
            sparse_block_size=SPARSE_BLOCK_SIZE,
        )

    q_snapshot = q.clone()
    index_q_snapshot = index_q.clone()
    topk_snapshot = topk.clone()
    kv_snapshot = kv_cache.clone()

    baseline_attn = run_main_sparse_attn(
        impl=impl,
        layer=layer,
        metadata=metadata,
        vllm_config=vllm_config,
        q=q,
        kv_cache=kv_cache,
    )
    candidate_attn = run_main_sparse_attn(
        impl=impl,
        layer=layer,
        metadata=metadata,
        vllm_config=vllm_config,
        q=q,
        kv_cache=kv_cache,
    )

    generator.manual_seed(rank_seed + 2)
    o_proj_weight = (
        torch.randn(
            q_size,
            HIDDEN_SIZE,
            dtype=DTYPE,
            device=device,
            generator=generator,
        )
        * 0.01
    )
    baseline_local = baseline_attn @ o_proj_weight
    candidate_local = candidate_attn @ o_proj_weight

    if args.inject_mismatch == "attn_output":
        candidate_attn.flatten()[0].add_(1.0)
        candidate_local = candidate_attn @ o_proj_weight
    elif args.inject_mismatch == "topk":
        topk.flatten()[0].add_(1)
    elif args.inject_mismatch == "global_o_proj":
        candidate_local.flatten()[0].add_(1.0)

    baseline_global = baseline_local.clone()
    candidate_global = candidate_local.clone()
    dist.all_reduce(baseline_global, op=dist.ReduceOp.SUM)
    dist.all_reduce(candidate_global, op=dist.ReduceOp.SUM)

    generator.manual_seed(rank_seed + 3)
    residual = torch.randn(
        batch.num_tokens,
        HIDDEN_SIZE,
        dtype=DTYPE,
        device=device,
        generator=generator,
    )
    norm_weight = (
        torch.randn(
            HIDDEN_SIZE,
            dtype=DTYPE,
            device=device,
            generator=generator,
        )
        * 0.01
    )
    baseline_post, _ = gemma_rms_norm(baseline_global, residual.clone(), norm_weight)
    candidate_post, _ = gemma_rms_norm(candidate_global, residual.clone(), norm_weight)

    generator.manual_seed(rank_seed + 4)
    logit_probe_weight = (
        torch.randn(
            HIDDEN_SIZE,
            LOGIT_PROBE_SIZE,
            dtype=DTYPE,
            device=device,
            generator=generator,
        )
        * 0.01
    )
    baseline_probe = baseline_post @ logit_probe_weight
    candidate_probe = candidate_post @ logit_probe_weight

    num_tokens = batch.num_tokens
    metrics = {
        "q": tensor_stats(q_snapshot[:num_tokens], q[:num_tokens]),
        "index_q": tensor_stats(index_q_snapshot[:num_tokens], index_q[:num_tokens]),
        "topk": tensor_stats(topk_snapshot[:num_tokens], topk[:num_tokens]),
        "kv_cache": tensor_stats(kv_snapshot, kv_cache),
        "attn_output": tensor_stats(
            baseline_attn[:num_tokens],
            candidate_attn[:num_tokens],
        ),
        "o_proj_local": tensor_stats(
            baseline_local[:num_tokens],
            candidate_local[:num_tokens],
        ),
        "o_proj_global": tensor_stats(
            baseline_global[:num_tokens],
            candidate_global[:num_tokens],
        ),
        "post_attention_hidden": tensor_stats(
            baseline_post[:num_tokens],
            candidate_post[:num_tokens],
        ),
        "synthetic_logit_probe": tensor_stats(
            baseline_probe[:num_tokens],
            candidate_probe[:num_tokens],
        ),
    }
    exact = all(metric.get("exact", False) for metric in metrics.values())
    local_pass_tensor = torch.tensor(
        [1 if exact else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(local_pass_tensor, op=dist.ReduceOp.MIN)
    passed = bool(local_pass_tensor.item())

    rank_record = {
        "rank": rank,
        "layer_name": layer_name,
        "scenario": args.scenario,
        "num_tokens": batch.num_tokens,
        "num_padded_tokens": batch.num_tokens,
        "local_q_heads": local_q_heads,
        "local_kv_heads": local_kv_heads,
        "metrics": metrics,
        "passed": exact,
    }
    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rank_record)

    result = {
        "passed": passed,
        "require_exact": bool(args.require_exact),
        "candidate": args.candidate,
        "inject_mismatch": args.inject_mismatch,
        "world_size": world_size,
        "ranks": gathered,
        "notes": {
            "logits": (
                "Layer 3 has no real LM logits; synthetic_logit_probe is the "
                "nearest safe downstream sensitivity check."
            ),
            "boundary": (
                "main sparse attention output only; qkv/rope/kv_insert/indexer excluded"
            ),
        },
    }
    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True))
    dist.barrier()
    return result


def main() -> int:
    args = parse_args()
    try:
        result = run_case(args)
    finally:
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
    if args.require_exact and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
