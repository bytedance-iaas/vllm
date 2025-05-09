# SPDX-License-Identifier: Apache-2.0

import torch
import torch.utils.benchmark as benchmark
from typing import Optional
from benchmark_shapes import WEIGHT_SHAPES_MOE

from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.cutlass_w4a8_moe import cutlass_w4a8_moe
from vllm.utils import F, FlexibleArgumentParser
from vllm import _custom_ops as ops


def bench_run(results: list[benchmark.Measurement], model: str,
              num_experts: int, topk: int, per_act_token: bool,
              per_out_ch: bool, mkn: tuple[int, int, int]):
    label = "Quant Matmul"

    sub_label = (
        "{}, num_experts={}, topk={}, per_act_token={} per_out_ch={}, "
        "MKN=({})".format(model, num_experts, topk, per_act_token, per_out_ch,
                          mkn))

    print(f"Testing: {sub_label}")

    (m, k, n) = mkn

    dtype = torch.bfloat16

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

    num_weights_in_32_bits = 8
    assert n % num_weights_in_32_bits == 0, f"n must be a multiple of {num_weights_in_32_bits}"
    unprocessed_int_weight_1 = torch.randint(
        -2**31,
        2**31, (num_experts, k, n * 2 // num_weights_in_32_bits),
        dtype=torch.int32,
        device="cuda")
    unprocessed_int_weight_2 = torch.randint(
        -2**31,
        2**31, (num_experts, n, k // num_weights_in_32_bits),
        dtype=torch.int32,
        device="cuda")
    a1_scale = torch.randn(k, dtype=dtype, device="cuda")
    a2_scale = torch.randn(n, dtype=dtype, device="cuda")

    group_size = 128

    scale_1 = torch.randn(num_experts,
                          k // group_size,
                          n * 2,
                          dtype=torch.bfloat16,
                          device="cuda") * 0.1
    scale_2 = torch.randn(
        num_experts, n // group_size, k, dtype=torch.bfloat16,
        device="cuda") * 0.1

    unprocessed_weight_1 = unprocessed_int_weight_1.view(torch.int8)
    unprocessed_weight_2 = unprocessed_int_weight_2.view(torch.int8)

    unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    ref_q_weight_1 = unpacker(unprocessed_weight_1.cpu()).cuda()
    ref_q_weight_2 = unpacker(unprocessed_weight_2.cpu()).cuda()
    ref_weight_1 = ref_q_weight_1 * scale_1.repeat_interleave(group_size,
                                                              dim=1)
    ref_weight_2 = ref_q_weight_2 * scale_2.repeat_interleave(group_size,
                                                              dim=1)

    weight_1 = ref_q_weight_1.permute(0, 2, 1).contiguous()
    weight_2 = ref_q_weight_2.permute(0, 2, 1).contiguous()
    weight_1 = packer(weight_1.cpu()).cuda()
    weight_2 = packer(weight_2.cpu()).cuda()
    w1_q = weight_1.view((num_experts, n * 2, k // 2)).view(torch.int8)
    w2_q = weight_2.view((num_experts, k, n // 2)).view(torch.int8)

    ###############################################################
    # scale interleave
    # scale_1 [E, K, N]
    scale_1 = scale_1.permute(0, 2, 1)  # [E, N, K]
    scale_1_interleaved = scale_1.reshape(scale_1.shape[0], scale_1.shape[1],
                                          (scale_1.shape[2] // 4),
                                          4)  # [E, N, K/4, 4]
    scale_1_interleaved = scale_1_interleaved.permute(0, 2, 1,
                                                      3)  # [E, K/4, N, 4]
    scale_1_interleaved = scale_1_interleaved.reshape(
        scale_1.shape[0], scale_1.shape[2] // 4,
        scale_1.shape[1] * 4)  # [E, K/4, N*4]
    w1_scale = scale_1_interleaved.contiguous()

    scale_2 = scale_2.permute(0, 2, 1)  # [E, N, K]
    scale_2_interleaved = scale_2.reshape(scale_2.shape[0], scale_2.shape[1],
                                          (scale_2.shape[2] // 4),
                                          4)  # [E, N, K/4, 4]
    scale_2_interleaved = scale_2_interleaved.permute(0, 2, 1,
                                                      3)  # [E, K/4, N, 4]
    scale_2_interleaved = scale_2_interleaved.reshape(
        scale_2.shape[0], scale_2.shape[2] // 4,
        scale_2.shape[1] * 4)  # [E, K/4, N*4]
    w2_scale = scale_2_interleaved.contiguous()

    ###############################################################

    device = "cuda"
    a_strides1 = torch.empty((num_experts, 3),
                             dtype=torch.int64,
                             device=device)
    b_strides1 = torch.empty((num_experts, 3),
                             dtype=torch.int64,
                             device=device)
    c_strides1 = torch.empty((num_experts, 3),
                             dtype=torch.int64,
                             device=device)
    a_strides2 = torch.empty((num_experts, 3),
                             dtype=torch.int64,
                             device=device)
    b_strides2 = torch.empty((num_experts, 3),
                             dtype=torch.int64,
                             device=device)
    c_strides2 = torch.empty((num_experts, 3),
                             dtype=torch.int64,
                             device=device)
    s_strides13 = c_strides1
    s_strides2 = c_strides2
    a_strides1[:, 0].fill_(k)
    a_strides1[:, 1].fill_(1)
    a_strides1[:, 2].zero_()
    b_strides1[:, 0].fill_(k // 2)
    b_strides1[:, 1].fill_(1)
    b_strides1[:, 2].zero_()
    c_strides1[:, 0].fill_(1)
    c_strides1[:, 1].fill_(2 * n)
    c_strides1[:, 2].zero_()
    a_strides2[:, 0].fill_(n)
    a_strides2[:, 1].fill_(1)
    a_strides2[:, 2].zero_()
    b_strides2[:, 0].fill_(n // 2)
    b_strides2[:, 1].fill_(1)
    b_strides2[:, 2].zero_()
    c_strides2[:, 0].fill_(1)
    c_strides2[:, 1].fill_(k)
    c_strides2[:, 2].zero_()

    score = torch.randn((m, num_experts), device="cuda", dtype=dtype)

    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=a,
        router_logits=score,
        use_grouped_topk=True,
        top_k=topk,
        renormalize=False,
        topk_group=1,
        num_expert_group=num_experts,
        custom_routing_function=None,
        scoring_func="sigmoid",
        e_score_correction_bias=None,
    )

    expert_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")

    apply_router_weight_on_input = False

    # ref
    def ref_moe(a: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                ref_weight_1: torch.Tensor,
                ref_weight_2: torch.Tensor,
                has_pre_quant: bool = False,
                has_alpha: bool = False,
                pre_quant_scale_1: Optional[torch.Tensor] = None,
                pre_quant_scale_2: Optional[torch.Tensor] = None,
                alpha_1: Optional[torch.Tensor] = None,
                alpha_2: Optional[torch.Tensor] = None):
        inputs = a.cuda().float()
        inputs_merged = inputs.view(-1, inputs.shape[-1])
        results = torch.zeros_like(inputs_merged)
        for i, (scales, experts) in enumerate(zip(topk_weights, topk_ids)):
            scales /= sum(scales)
            input = inputs_merged[i, :]
            for scale, expert in zip(scales, experts):
                input = inputs_merged[i, :]
                fc1_qd = ref_weight_1[expert].cuda().float()
                if has_pre_quant:
                    input = input * pre_quant_scale_1.squeeze()
                if has_alpha:
                    input = input.to(torch.float8_e4m3fn).float()
                    fc1_qd = fc1_qd.to(torch.float8_e4m3fn).float()
                    fc1 = torch.matmul(input, fc1_qd) * alpha_1[expert]
                else:
                    fc1 = torch.matmul(input, fc1_qd)
                fc1, gate = fc1.chunk(2, dim=-1)
                fc1 = fc1 * torch.nn.functional.silu(gate)
                fc2_qd = ref_weight_2[expert].cuda().float()
                if has_pre_quant:
                    fc1 = fc1 * pre_quant_scale_2.squeeze()
                if has_alpha:
                    fc1 = fc1.to(torch.float8_e4m3fn).float()
                    fc2_qd = fc2_qd.to(torch.float8_e4m3fn).float()
                    final = torch.matmul(fc1, fc2_qd) * alpha_2[expert]
                else:
                    final = torch.matmul(fc1, fc2_qd)
                results[i] += scale * final
        ref = results.view(*inputs.shape).to(dtype)
        return ref

    def run_cutlass_w4a8_moe(
        num_repeats: int,
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a_strides1: torch.Tensor,
        b_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides13: torch.Tensor,
        s_strides2: torch.Tensor,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
    ):
        for _ in range(num_repeats - 1):
            cutlass_w4a8_moe(
                a,
                w1_q,
                w2_q,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                a_strides1,
                b_strides1,
                c_strides1,
                a_strides2,
                b_strides2,
                c_strides2,
                s_strides13,
                s_strides2,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input)
        # Return the output from the last run
        return cutlass_w4a8_moe(
            a,
            w1_q,
            w2_q,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            a_strides1,
            b_strides1,
            c_strides1,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides13,
            s_strides2,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input)

    def run_cutlass_from_graph(
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a_strides1: torch.Tensor,
        b_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides13: torch.Tensor,
        s_strides2: torch.Tensor,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
    ):
        with set_current_vllm_config(
                VllmConfig(parallel_config=ParallelConfig(
                    pipeline_parallel_size=1))):
            return cutlass_w4a8_moe(
                a,
                w1_q,
                w2_q,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                a_strides1,
                b_strides1,
                c_strides1,
                a_strides2,
                b_strides2,
                c_strides2,
                s_strides13,
                s_strides2,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input)

    def replay_graph(graph, num_repeats):
        for _ in range(num_repeats):
            graph.replay()
        torch.cuda.synchronize()

    cutlass_stream = torch.cuda.Stream()
    cutlass_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cutlass_graph, stream=cutlass_stream):
        run_cutlass_from_graph(
            a,
            w1_q,
            w2_q,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            a_strides1,
            b_strides1,
            c_strides1,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides13,
            s_strides2,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input)
    torch.cuda.synchronize()

    min_run_time = 5
    num_warmup = 5
    num_runs = 1
    # Warmup
    ref_moe(a, topk_weights, topk_ids, ref_weight_1, ref_weight_2, True,
            False, a1_scale, a2_scale)

    globals = {
        # Baseline params
        "ref_weight_1": ref_weight_1,
        "ref_weight_2": ref_weight_2,
        "score": score,
        "topk": topk,
        # Cutlass params
        "a1_scale": a1_scale,
        "a2_scale": a2_scale,
        "w1_q": w1_q,
        "w2_q": w2_q,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "a_strides1": a_strides1,
        "b_strides1": b_strides1,
        "c_strides1": c_strides1,
        "a_strides2": a_strides2,
        "b_strides2": b_strides2,
        "c_strides2": c_strides2,
        "s_strides13": s_strides13,
        "s_strides2": s_strides2,
        # cuda graph params
        "cutlass_graph": cutlass_graph,
        # Gen params
        "a": a,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "num_runs": num_runs,
        "expert_map": expert_map,
        "apply_router_weight_on_input": apply_router_weight_on_input,
        # Kernels
        "run_cutlass_w4a8_moe": run_cutlass_w4a8_moe,
        "replay_graph": replay_graph,
        "ref_moe": ref_moe,
    }

    print(f"a shape: {a.shape}, dtype: {a.dtype}")
    print(f"w1_q shape: {w1_q.shape}, dtype: {w1_q.dtype}")
    print(f"w2_q shape: {w2_q.shape}, dtype: {w2_q.dtype}")
    print(f"w1_scale shape: {w1_scale.shape}, dtype: {w1_scale.dtype}")
    print(f"w2_scale shape: {w2_scale.shape}, dtype: {w2_scale.dtype}")
    print(f"a_strides1 shape: {a_strides1.shape}, dtype: {a_strides1.dtype}")
    print(f"b_strides1 shape: {b_strides1.shape}, dtype: {b_strides1.dtype}")
    print(f"c_strides1 shape: {c_strides1.shape}, dtype: {c_strides1.dtype}")
    print(f"a_strides2 shape: {a_strides2.shape}, dtype: {a_strides2.dtype}")
    print(f"b_strides2 shape: {b_strides2.shape}, dtype: {b_strides2.dtype}")
    print(f"c_strides2 shape: {c_strides2.shape}, dtype: {c_strides2.dtype}")
    print(
        f"s_strides13 shape: {s_strides13.shape}, dtype: {s_strides13.dtype}")
    print(f"s_strides2 shape: {s_strides2.shape}, dtype: {s_strides2.dtype}")
    print(
        f"topk_weights shape: {topk_weights.shape}, dtype: {topk_weights.dtype}"
    )
    print(f"topk_ids shape: {topk_ids.shape}, dtype: {topk_ids.dtype}")
    print(f"a1_scale shape: {a1_scale.shape}, dtype: {a1_scale.dtype}")
    print(f"a2_scale shape: {a2_scale.shape}, dtype: {a2_scale.dtype}")
    print(f"expert_map shape: {expert_map.shape}, dtype: {expert_map.dtype}")

    #Warmup
    ref_moe(
        a,
        topk_weights,
        topk_ids,
        ref_weight_1,
        ref_weight_2,
        has_pre_quant=True,
        has_alpha=False,
        pre_quant_scale_1=a1_scale,
        pre_quant_scale_2=a2_scale,
    )

    results.append(
        benchmark.Timer(
            stmt=
            "ref_moe(a, topk_weights, topk_ids, ref_weight_1, ref_weight_2, has_pre_quant=True, has_alpha=False, pre_quant_scale_1=a1_scale, pre_quant_scale_2=a2_scale)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="ref_moe",
        ).blocked_autorange(min_run_time=min_run_time))

    # Warmup
    run_cutlass_w4a8_moe(
        num_warmup,
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input)

    results.append(
        benchmark.Timer(
            stmt=
            "run_cutlass_w4a8_moe(num_runs, a, w1_q, w2_q, w1_scale, w2_scale, topk_weights, topk_ids, a_strides1, b_strides1, c_strides1, a_strides2, b_strides2, c_strides2, s_strides13, s_strides2, a1_scale=a1_scale, a2_scale=a2_scale, expert_map=expert_map, apply_router_weight_on_input=apply_router_weight_on_input)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="grouped_gemm_moe",
        ).blocked_autorange(min_run_time=min_run_time))

    # Warmup
    replay_graph(cutlass_graph, num_warmup)

    results.append(
        benchmark.Timer(
            stmt="replay_graph(cutlass_graph, num_runs)",
            globals=globals,
            label=label,
            sub_label=sub_label,
            description="grouped_gemm_moe_cuda_graphs",
        ).blocked_autorange(min_run_time=min_run_time))

    # correctness verification
    ref_out = ref_moe(
        a,
        topk_weights,
        topk_ids,
        ref_weight_1,
        ref_weight_2,
        has_pre_quant=True,
        has_alpha=False,
        pre_quant_scale_1=a1_scale,
        pre_quant_scale_2=a2_scale,
    )

    cutlass_out = run_cutlass_w4a8_moe(
        1,  # Single run for verification
        a,
        w1_q,
        w2_q,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        a_strides1,
        b_strides1,
        c_strides1,
        a_strides2,
        b_strides2,
        c_strides2,
        s_strides13,
        s_strides2,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input)
    
    torch.cuda.synchronize()

    print("ref_out: ", ref_out)
    print("cutlass_out: ", cutlass_out)
    print("max diff: ", torch.max(torch.abs(ref_out - cutlass_out)))

def main():

    results: list[benchmark.Measurement] = []

    model = "dpsk-w4a8"
    num_experts = 8
    topk = 8
    per_act_token = True
    per_out_ch = False
    size_m = 1
    size_k = 7168
    size_n = 4096
    mkn = (size_m, size_k, size_n)
    bench_run(results, model, num_experts, topk, per_act_token, per_out_ch,
              mkn)
    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    main()
