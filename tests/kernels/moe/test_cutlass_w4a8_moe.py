# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from typing import Optional

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.cutlass_w4a8_moe import cutlass_w4a8_moe

# debug = True
debug = False

def print_tensor_info(name, tensor):
    if not debug:
        return
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  values: {tensor.flatten()[:10]}")  # Print first 10 values


def pack_interleave(num_experts, ref_weight, ref_scale):
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    weight = packer(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    w_q = w_q.contiguous()

    ###############################################################
    # scale interleave, [E, K, N]
    scale = ref_scale.permute(0, 2, 1)  # [E, N, K]
    scale_interleaved = scale.reshape(scale.shape[0], scale.shape[1],
                                      (scale.shape[2] // 4),
                                      4)  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        scale.shape[0], scale.shape[2] // 4,
        scale.shape[1] * 4)  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)

@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_moe_w4afp8(dtype):

    m = 4
    k = 7168
    n = 2048
    group_size = 128
    num_experts = 32
    topk = 8
    # a = torch.ones((m, k), dtype=dtype, device='cuda') * 0.01
    # a[1:] = 0.02
    a = torch.randn(m, k, dtype=dtype, device='cuda') * 0.1
    # print_tensor_info("a1", a[0])
    # print_tensor_info("a2", a[1])
    dtype = torch.bfloat16

    affine_coeff = 0.005
    # ref_weight_1 = torch.ones((num_experts, n * 2, k),
    #                           dtype=torch.int8,
    #                           device="cuda")
    # ref_weight_2 = torch.ones((num_experts, k, n),
    #                           dtype=torch.int8,
    #                           device="cuda")
    ref_weight_1 = torch.randint(-8,
                                 8, (num_experts, n * 2, k),
                                 dtype=torch.int8,
                                 device='cuda')
    ref_weight_2 = torch.randint(-8,
                                 8, (num_experts, k, n),
                                 dtype=torch.int8,
                                 device='cuda')
    a1_scale = torch.randn(1, dtype=torch.float32, device="cuda")
    a2_scale = torch.randn(1, dtype=torch.float32, device="cuda")
    # a1_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    # a2_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    scale_1 = torch.randn(
        num_experts, k // group_size, n * 2, dtype=dtype,
        device="cuda") * affine_coeff
    scale_2 = torch.randn(
        num_experts, n // group_size, k, dtype=dtype,
        device="cuda") * affine_coeff
    # scale_1 = torch.ones(
    #     (num_experts, k // group_size, n * 2), dtype=dtype,
    #     device="cuda")
    # scale_2 = torch.ones(
    #     (num_experts, n // group_size, k), dtype=dtype,
    #     device="cuda")

    # ref_weight_1 = unprocessed_int_weight_1 * scale_1.repeat_interleave(
    #     group_size, dim=2)
    # ref_weight_2 = unprocessed_int_weight_2 * scale_2.repeat_interleave(
    #     group_size, dim=2)

    w1_q, w1_scale = pack_interleave(num_experts, ref_weight_1, scale_1)
    w2_q, w2_scale = pack_interleave(num_experts, ref_weight_2, scale_2)

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
    s_strides13 = torch.empty((num_experts, 3),
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
    s_strides2 = torch.empty((num_experts, 3),
                             dtype=torch.int64,
                             device=device)
    # s_strides13 = c_strides1
    # s_strides2 = c_strides2

    a_strides1[:, 0].fill_(k)
    a_strides1[:, 1].fill_(1)
    a_strides1[:, 2].zero_()

    b_strides1[:, 0].fill_(k)
    b_strides1[:, 1].fill_(1)
    b_strides1[:, 2].zero_()

    c_strides1[:, 0].fill_(1)
    c_strides1[:, 1].fill_(2 * n)
    c_strides1[:, 2].zero_()

    s_strides13[:, 0].fill_(1)
    s_strides13[:, 1].fill_(2 * n)
    s_strides13[:, 2].zero_()

    a_strides2[:, 0].fill_(n)
    a_strides2[:, 1].fill_(1)
    a_strides2[:, 2].zero_()

    b_strides2[:, 0].fill_(n)
    b_strides2[:, 1].fill_(1)
    b_strides2[:, 2].zero_()

    c_strides2[:, 0].fill_(1)
    c_strides2[:, 1].fill_(k)
    c_strides2[:, 2].zero_()

    s_strides2[:, 0].fill_(1)
    s_strides2[:, 1].fill_(k)
    s_strides2[:, 2].zero_()

    score = torch.randn((m, num_experts), device="cuda", dtype=dtype)
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
    # topk_weights = torch.tensor([[0.8086, 0.4180], [0.7695, 0.6523]], dtype=torch.float, device='cuda')
    # topk_ids = torch.tensor([[2, 1], [0, 2]], dtype=torch.int32, device='cuda')
    # topk_ids = torch.tensor([[0], [0]], dtype=torch.int32, device='cuda')
    # topk_ids = torch.tensor([[1], [1]], dtype=torch.int32, device='cuda')
    print_tensor_info("topk_weights", topk_weights)
    print_tensor_info("topk_ids", topk_ids)
    expert_map = torch.arange(num_experts, dtype=torch.int32, device="cuda")

    def ref(x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            ref_weight_1: torch.Tensor,
            ref_weight_2: torch.Tensor,
            ref_weight_scale_1: torch.Tensor,
            ref_weight_scale_2: torch.Tensor,
            has_pre_quant: bool = False,
            has_alpha: bool = False,
            pre_quant_scale_1: Optional[torch.Tensor] = None,
            pre_quant_scale_2: Optional[torch.Tensor] = None,
            alpha_1: Optional[torch.Tensor] = None,
            alpha_2: Optional[torch.Tensor] = None):
        results = torch.zeros_like(x)
        # selected_experts, final_scales = routing_method.apply(router_logits)
        # unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        tensors_collector = []
        aggregated_tensors_lists = {
            "c1": [],
            "silu_intermediate": [],
            "intermediate_q": [],
            "c2": [],
            "delta_results":
            []  # Stores the contribution of each expert to the results
        }
        for e_idx in range(num_experts):
            print(f"==================expert {e_idx}======================")
            mask = topk_ids == e_idx
            activated_tokens = mask.sum(1).bool()
            act = x[activated_tokens, :]
            if act.shape[0] == 0:
                continue
            final_scale = (topk_weights *
                           mask).sum(1)[activated_tokens].unsqueeze(1)

            act = torch.clamp((act / pre_quant_scale_1.float()), -448.0,
                              448.0).to(torch.float8_e4m3fn).to(dtype)
            print_tensor_info("act", act)
            w3_w1 = ref_weight_1[e_idx]
            ref_w_scale_repeat = ref_weight_scale_1[e_idx].t(
            ).repeat_interleave(128, dim=1).to(float)
            print_tensor_info("ref_w_scale_repeat1", ref_w_scale_repeat)
            w3_w1 = (w3_w1.to(float) * ref_w_scale_repeat).to(dtype)
            fc1 = ((torch.matmul(act, w3_w1.T)) * alpha_1).to(
                torch.float16)
            print_tensor_info("fc1", fc1)
            aggregated_tensors_lists["c1"].append(fc1.clone().detach())
            # tensors_collector.append({
            #     "name": "c1",
            #     "tensor": fc1.clone().detach()
            # })

            gate, fc1 = fc1.chunk(2, dim=-1)
            print_tensor_info("gate", gate)
            print_tensor_info("fc1", fc1)
            fc1 = fc1 * torch.nn.functional.silu(gate)
            print_tensor_info("fc1 after silu", fc1)
            # tensors_collector.append({
            #     "name": "silu_intermediate",
            #     "tensor": fc1.clone().detach()
            # })
            aggregated_tensors_lists["silu_intermediate"].append(
                fc1.clone().detach())

            # act = torch.clamp((fc1 / pre_quant_scale_2[e_idx].float()), -448.0,
            #                   448.0).to(torch.float8_e4m3fn).to(dtype)
            print_tensor_info("pre_quant_scale_2", pre_quant_scale_2)
            act = (fc1 / pre_quant_scale_2.float()).to(
                torch.float8_e4m3fn)
            # torch.save(act, "ref_intermediate_q_fp8")
            act = act.to(dtype)
            print_tensor_info("act2", act)
            # tensors_collector.append({
            #     "name": "intermediate_q",
            #     "tensor": act.clone().detach()
            # })
            aggregated_tensors_lists["intermediate_q"].append(
                act.clone().detach())

            # act = torch.load("ref_intermediate_q_fp8").to(dtype)
            # tensors_collector.append({
            #     "name": "intermediate_q",
            #     "tensor": act.clone().detach()
            # })

            w2 = ref_weight_2[e_idx]
            ref_w_scale_repeat = ref_weight_scale_2[e_idx].t(
            ).repeat_interleave(128, dim=1).to(float)
            print_tensor_info("ref_w_scale_repeat2", ref_w_scale_repeat)
            w2 = (w2.to(float) * ref_w_scale_repeat).to(dtype)
            fc2 = (torch.matmul(act, w2.T) * alpha_2).to(torch.float16)
            print_tensor_info("fc2", fc2)
            # tensors_collector.append({
            #     "name": "c2",
            #     "tensor": fc2.clone().detach()
            # })
            aggregated_tensors_lists["c2"].append(fc2.clone().detach())

            results[activated_tokens, :] += (fc2 * final_scale).to(
                results.dtype)
            print_tensor_info("results", results)
            # tensors_collector.append({
            #     "name": "results",
            #     "tensor": results.clone().detach()
            # })

        for name, tensor_list in aggregated_tensors_lists.items():
            non_empty_tensors = [t for t in tensor_list if t.numel() > 0]
            if non_empty_tensors:
                aggregated_tensor = torch.cat(non_empty_tensors, dim=0)
                tensors_collector.append({"name": name, "tensor": aggregated_tensor})
            elif name in ["c1", "silu_intermediate", "intermediate_q", "c2", "delta_results"]:
                print(f"Warning: All tensors for step '{name}' were empty or skipped. Appending an empty tensor.")
                # Determine a representative dtype and device
                ref_dtype = x.dtype
                ref_device = x.device
                expected_k_dim = 0
                if name == "c1": expected_k_dim = n * 2
                elif name == "silu_intermediate" or name == "intermediate_q": expected_k_dim = n
                elif name == "c2" or name == "delta_results": expected_k_dim = k

                tensors_collector.append({
                    "name": name,
                    "tensor": torch.empty((0, expected_k_dim), dtype=ref_dtype, device=ref_device)
                })
        return results, tensors_collector

    output, cutlass_tensors = cutlass_w4a8_moe(
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
        apply_router_weight_on_input=False)

    ref_output, ref_tensors = ref(
        a,
        topk_weights,
        topk_ids,
        ref_weight_1,
        ref_weight_2,
        scale_1,
        scale_2,
        has_pre_quant=True,
        has_alpha=True,
        pre_quant_scale_1=a1_scale,
        pre_quant_scale_2=a2_scale,
        alpha_1=a1_scale,
        alpha_2=a2_scale,
    )

    # compare
    torch.cuda.synchronize()

    compare_intermediate_val(cutlass_tensors, ref_tensors)

    # compare final output
    print("\nComparing final output tensors...")
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)
    # woq_assert_near_eq(ref_output, output, 2)
    print("SUCCESS: Final output tensors are close.")


def compare_intermediate_val(cutlass_tensors, ref_tensors):
    cutlass_tensors_map = {}
    if cutlass_tensors: # Check if cutlass_tensors is not None and not empty
        cutlass_tensors_map = {item["name"]: item["tensor"] for item in cutlass_tensors}


    for ref_item in ref_tensors:
        ref_name = ref_item["name"]
        ref_tensor_val = ref_item["tensor"]

        print(f"\nComparing tensor: '{ref_name}'")

        if ref_name in cutlass_tensors_map:
            cutlass_tensor_val = cutlass_tensors_map[ref_name]
            try:
                if cutlass_tensor_val.device != ref_tensor_val.device:
                    print(f"  WARNING: Tensor '{ref_name}' devices differ. Ref: {ref_tensor_val.device}, Cutlass: {cutlass_tensor_val.device}. Moving Cutlass tensor to Ref tensor's device.")
                    cutlass_tensor_val = cutlass_tensor_val.to(ref_tensor_val.device)

                torch.testing.assert_close(cutlass_tensor_val, ref_tensor_val, rtol=1e-2, atol=0.1)
                print(f"  SUCCESS: '{ref_name}' tensors are close.")
            except AssertionError as e:
                # torch.set_printoptions(threshold=10_000)
                print(f"  FAILURE: '{ref_name}' tensors are NOT close.")
                print(f"    Ref tensor: {ref_tensor_val.flatten()}")
                print(f"    Cutlass tensor: {cutlass_tensor_val.flatten()}")
                print(f"    Max absolute difference: {torch.max(torch.abs(cutlass_tensor_val.to(ref_tensor_val.dtype) - ref_tensor_val))}")
                print(f"    Mean absolute difference: {torch.mean(torch.abs(cutlass_tensor_val.to(ref_tensor_val.dtype) - ref_tensor_val))}")
                print(f"    AssertionError: {e}")
                raise
        else:
            print(f"  WARNING: Tensor '{ref_name}' not found in cutlass_tensors output.")
