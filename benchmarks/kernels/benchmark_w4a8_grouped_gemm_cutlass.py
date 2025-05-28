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


def print_tensor_info(name, tensor):
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  values: {tensor.flatten()}[:10]")  # Print first 10 values


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

    a = torch.ones((m, k), dtype=torch.bfloat16, device='cuda')
    unprocessed_int_weight_1 = torch.ones((num_experts, n * 2, k),
                                          dtype=torch.int8,
                                          device="cuda")
    unprocessed_int_weight_2 = torch.ones((num_experts, k, n),
                                          dtype=torch.int8,
                                          device="cuda")
    a1_scale = torch.ones(k, dtype=dtype, device="cuda")
    a2_scale = torch.ones(n, dtype=dtype, device="cuda")

    group_size = 128

    scale_1 = torch.ones(num_experts,
                         n * 2,
                         k // group_size,
                         dtype=torch.bfloat16,
                         device="cuda")
    scale_2 = torch.ones(num_experts,
                         k,
                         n // group_size,
                         dtype=torch.bfloat16,
                         device="cuda")

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    ref_weight_1 = unprocessed_int_weight_1 * scale_1.repeat_interleave(
        group_size, dim=2)
    ref_weight_2 = unprocessed_int_weight_2 * scale_2.repeat_interleave(
        group_size, dim=2)

    weight_1 = packer(unprocessed_int_weight_1.cpu()).cuda()
    weight_2 = packer(unprocessed_int_weight_2.cpu()).cuda()
    w1_q = weight_1.view((num_experts, n * 2, k // 2)).view(torch.int8)
    w2_q = weight_2.view((num_experts, k, n // 2)).view(torch.int8)

    ###############################################################
    # scale interleave
    scale_1_interleaved = scale_1.reshape(scale_1.shape[0], scale_1.shape[1],
                                          (scale_1.shape[2] // 4),
                                          4)  # [E, N, K/4, 4]
    scale_1_interleaved = scale_1_interleaved.permute(0, 2, 1,
                                                      3)  # [E, K/4, N, 4]
    scale_1_interleaved = scale_1_interleaved.reshape(
        scale_1.shape[0], scale_1.shape[2] // 4,
        scale_1.shape[1] * 4)  # [E, K/4, N*4]
    w1_scale = scale_1_interleaved.contiguous()

    scale_2_interleaved = scale_2.reshape(scale_2.shape[0], scale_2.shape[1],
                                          (scale_2.shape[2] // 4), 4)
    scale_2_interleaved = scale_2_interleaved.permute(0, 2, 1, 3)
    scale_2_interleaved = scale_2_interleaved.reshape(scale_2.shape[0],
                                                      scale_2.shape[2] // 4,
                                                      scale_2.shape[1] * 4)
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
    print_tensor_info("score", score)

    topk_weights, topk_ids = FusedMoE.select_experts(
        hidden_states=a,
        router_logits=score,
        use_grouped_topk=True,
        top_k=topk,
        renormalize=False,
        topk_group=1,
        num_expert_group=1,
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
                    inv_scale = 1.0 / pre_quant_scale_1.squeeze()
                    input = input * inv_scale
                if has_alpha:
                    input = input.to(torch.float8_e4m3fn).float()
                    fc1_qd = fc1_qd.to(torch.float8_e4m3fn).float()
                    fc1 = torch.matmul(input, fc1_qd.T) * alpha_1[expert]
                else:
                    fc1 = torch.matmul(input, fc1_qd.T)
                # print(f"fc1: {fc1[:10]}")
                gate, fc1 = fc1.chunk(2, dim=-1)
                # print(f"gate: {gate[:10]}")
                fc1 = fc1 * torch.nn.functional.silu(gate)
                # print(f"fc1_silu: {fc1[:10]}")
                fc2_qd = ref_weight_2[expert].cuda().float()
                # print(f"fc2_qd: {fc2_qd[:10]}")
                if has_pre_quant:
                    inv_scale = 1.0 / pre_quant_scale_2.squeeze()
                    fc1 = fc1 * inv_scale
                if has_alpha:
                    fc1 = fc1.to(torch.float8_e4m3fn).float()
                    fc2_qd = fc2_qd.to(torch.float8_e4m3fn).float()
                    final = torch.matmul(fc1, fc2_qd.T) * alpha_2[expert]
                else:
                    final = torch.matmul(fc1, fc2_qd.T)
                    # print(f"final: {final[:10]}")
                results[i] += scale * final
                # print(f"results[i]: {results[i]}")
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

    print_tensor_info("a", a)
    print_tensor_info("ref_weight_1", ref_weight_1)
    print_tensor_info("ref_weight_2", ref_weight_2)
    print_tensor_info("w1_q", w1_q)
    print_tensor_info("w2_q", w2_q)
    print_tensor_info("w1_scale", w1_scale)
    print_tensor_info("w2_scale", w2_scale)
    print_tensor_info("a_strides1", a_strides1)
    print_tensor_info("b_strides1", b_strides1)
    print_tensor_info("c_strides1", c_strides1)
    print_tensor_info("a_strides2", a_strides2)
    print_tensor_info("b_strides2", b_strides2)
    print_tensor_info("c_strides2", c_strides2)
    print_tensor_info("s_strides13", s_strides13)
    print_tensor_info("s_strides2", s_strides2)
    print_tensor_info("topk_weights", topk_weights)
    print_tensor_info("topk_ids", topk_ids)
    print_tensor_info("a1_scale", a1_scale)
    print_tensor_info("a2_scale", a2_scale)
    print_tensor_info("expert_map", expert_map)

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

    print("ref_out: ", ref_out[:10])
    print("cutlass_out: ", cutlass_out[:10])
    print_tensor_info("ref_out", ref_out)
    print_tensor_info("cutlass_out", cutlass_out)
    print("max diff: ", torch.max(torch.abs(ref_out - cutlass_out)))
    print("mean diff: ", torch.mean(torch.abs(ref_out - cutlass_out)))
    print("relative diff: ",
          torch.mean(torch.abs(ref_out - cutlass_out) / torch.abs(ref_out)))

def main():

    results: list[benchmark.Measurement] = []

    model = "dpsk-w4a8"
    num_experts = 1
    topk = 1
    per_act_token = True
    per_out_ch = False
    size_m = 1
    size_k = 1024
    size_n = 512
    mkn = (size_m, size_k, size_n)
    bench_run(results, model, num_experts, topk, per_act_token, per_out_ch,
              mkn)
    compare = benchmark.Compare(results)
    compare.print()


def postprocess(w):
    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
    unpacked_tensor = unpacker(w.cpu().T.contiguous())
    print(
        f"unpacked_tensor: {unpacked_tensor}, unpacked_tensor.shape: {unpacked_tensor.shape}, unpacked_tensor.dtype: {unpacked_tensor.dtype}"
    )
    unpacked_tensor_t = unpacked_tensor.T.contiguous()
    print(
        f"unpacked_tensor_t: {unpacked_tensor_t}, unpacked_tensor_t.shape: {unpacked_tensor_t.shape}, unpacked_tensor_t.dtype: {unpacked_tensor_t.dtype}"
    )
    packed_tensor = packer(unpacked_tensor_t)
    print(
        f"packed_tensor: {packed_tensor}, packed_tensor.shape: {packed_tensor.shape}, packed_tensor.dtype: {packed_tensor.dtype}"
    )
    return packed_tensor


def test_weight_pack():
    a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 6]], dtype=torch.int8)
    print(f"a: {a}, a.shape: {a.shape}, a.dtype: {a.dtype}")
    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    a_pack = packer(a.T.cpu().contiguous())
    print(
        f"a_pack: {a_pack}, a_pack.shape: {a_pack.shape}, a_pack.dtype: {a_pack.dtype}"
    )
    a_pack_t = a_pack.T
    print(
        f"a_pack_t: {a_pack_t}, a_pack_t.shape: {a_pack_t.shape}, a_pack_t.dtype: {a_pack_t.dtype}"
    )

    a_q = postprocess(a_pack_t)
    print(f"a_q: {a_q}, a_q.shape: {a_q.shape}, a_q.dtype: {a_q.dtype}")


if __name__ == "__main__":
    main()
    # test_weight_pack()
