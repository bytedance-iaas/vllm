# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
from typing import Optional

import torch

from vllm import _custom_ops as ops


def cutlass_w4a8_moe(
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
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """
    This function computes a w4a8-quantized Mixture of Experts (MoE) layer
    using two sets of quantized weights, w1_q and w2_q, and top-k gating
    mechanism. The matrix multiplications are implemented with CUTLASS
    grouped gemm.

    Parameters:
    - a (torch.Tensor): The input tensor to the MoE layer.
        Shape: [M, K]
    - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
        Shape: [num_experts, K // 2, N * 2]
        (the weights are passed transposed and int4-packed)
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, N // 2, K]
        (the weights are passed transposed and int4-packed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - ab_strides1 (torch.Tensor): The input and weights strides of the first
        grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - ab_strides2 (torch.Tensor): The input and weights strides of the second
        grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    - out_dtype (torch.Tensor): The output tensor type.

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    """

    # print(
    #     f"a.shape, dtype: {a.shape, a.dtype}, w1_q.shape, dtype: {w1_q.shape, w1_q.dtype}, w2_q.shape, dtype: {w2_q.shape, w2_q.dtype}, w1_scale.shape, dtype: {w1_scale.shape, w1_scale.dtype}, w2_scale.shape, dtype: {w2_scale.shape, w2_scale.dtype}, topk_weights.shape: {topk_weights.shape}, topk_ids.shape: {topk_ids.shape}, ab_strides1.shape: {ab_strides1.shape}, c_strides1.shape: {c_strides1.shape}, ab_strides2.shape: {ab_strides2.shape}, c_strides2.shape: {c_strides2.shape}"
    # )

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert w1_q.dtype == torch.int8
    assert w2_q.dtype == torch.int8
    assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
    assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
    assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
    assert w1_q.shape[0] == w1_scale.shape[
        0], "w1 scales expert number mismatch"
    assert w1_q.shape[0] == w2_scale.shape[
        0], "w2 scales expert number mismatch"
    assert w1_scale.shape[1] == w1_q.shape[2] * 2 / 512 and \
        w1_scale.shape[2] == w1_q.shape[1] * 4, "W1 scale shape mismatch"
    assert w2_scale.shape[1] == w2_q.shape[2] * 2 / 512 and \
        w2_scale.shape[2] == w2_q.shape[1] * 4, "W2 scale shape mismatch"

    # assert a1_scale is None or a1_scale.dim(
    # ) == 0 or a1_scale.shape[0] == 1 or a1_scale.shape[0] == a.shape[
    #     0], "Input scale shape mismatch"
    # assert a2_scale is None or a1_scale is None or a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501
    # assert ab_strides1.shape[0] == w1_q.shape[
    #     0], "AB Strides 1 expert number mismatch"
    # assert c_strides1.shape[0] == w1_q.shape[
    #     0], "C Strides 1 expert number mismatch"
    # assert ab_strides2.shape[0] == w2_q.shape[
    #     0], "AB Strides 2 expert number  mismatch"
    # assert c_strides2.shape[0] == w2_q.shape[
    #     0], "C Strides 2 expert number mismatch"

    # w1_q = w1_q.transpose(1, 2)
    # w2_q = w2_q.transpose(1, 2)
    # w1_scale = w1_scale.transpose(1, 2)
    # w2_scale = w1_scale.transpose(1, 2)
    w1_scale = w1_scale.to(torch.float32)
    w2_scale = w2_scale.to(torch.float32)

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed

    topk = topk_ids.size(1)

    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)
    if apply_router_weight_on_input:
        assert topk == 1, \
            "apply_router_weight_on_input is only implemented for topk=1"
        # TODO: this only works for topK=1, will need to update for topK>1
        a = a * topk_weights.to(torch.half)

    a_q, a1_scale = ops.scaled_fp8_quant(
        a, a1_scale.float(), use_per_token_if_dynamic=per_act_token)
    device = a_q.device

    expert_offsets = torch.empty((num_experts + 1),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes1 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)
    problem_sizes2 = torch.empty((num_experts, 3),
                                 dtype=torch.int32,
                                 device=device)

    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    # if expert_offsets.device == "cuda:0":
    #     print(f"topk_ids {topk_ids}, expert_offsets {expert_offsets}, problem_sizes1 {problem_sizes1}")
    #     print(f"problem_sizes2 {problem_sizes2}, a_map {a_map}, c_map {c_map}")
    #     print(f"num_experts {num_experts}, n {n}, k {k}")
    # a1_scale and a2_scale are not supported
    # print(f"before get_cutlass_moe_mm_data")
    # print_tensor("topk_ids", topk_ids)

    ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n,
                                k)

    # problem_sizes1 = problem_sizes1[:, [1, 0, 2]].contiguous()
    # problem_sizes2 = problem_sizes2[:, [1, 0, 2]].contiguous()

    problem_sizes1_swapped = torch.empty_like(problem_sizes1)
    problem_sizes2_swapped = torch.empty_like(problem_sizes2)
    problem_sizes1_swapped[:, 0] = problem_sizes1[:, 1]
    problem_sizes1_swapped[:, 1] = problem_sizes1[:, 0]
    problem_sizes1_swapped[:, 2] = problem_sizes1[:, 2]
    problem_sizes2_swapped[:, 0] = problem_sizes2[:, 1]
    problem_sizes2_swapped[:, 1] = problem_sizes2[:, 0]
    problem_sizes2_swapped[:, 2] = problem_sizes2[:, 2]
    problem_sizes1 = problem_sizes1_swapped
    problem_sizes2 = problem_sizes2_swapped

    # torch.cuda.synchronize()
    # print(f"finish get_cutlass_moe_mm_data")
    # print_tensor("topk_ids", topk_ids)
    # print_tensor("expert_offsets", expert_offsets)
    # print_tensor("problem_sizes1", problem_sizes1)
    # print_tensor("problem_sizes2", problem_sizes2)
    # print_tensor("a_map", a_map)
    # print_tensor("c_map", c_map)

    # print(f"ssss num_experts {num_experts}, ssss n {n}, ssss k {k}")
    # print(f"ssss m {m}")

    # if expert_offsets.device == "cuda:0":
    #     print("expert offsets", expert_offsets)
    # if problem_sizes1.device == "cuda:0":
    #     print("problem_sizes1", problem_sizes1)
    # device_id = a.get_device()
    # torch.save(a1_scale, f'/nvme0n1/{device_id}_a1_scale.tensor')
    # torch.save(a_map, f'/nvme0n1/{device_id}_a_map.tensor')
    # torch.save(a_q, f'/nvme0n1/{device_id}_a_q.tensor')

    a_map = torch.clamp(a_map,
                        min=0,
                        max=min(a1_scale.size(0) - 1,
                                a_q.size(0) - 1))
    # print("finish a_map")
    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    # print(f"yy a_map {a_map}")
    rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

    # print(f"m {m}, n {n}, k {k}, topk {topk}")
    # m 32768, n 256, k 7168, topk 8
    # print("CHECKPOINT5")

    # print(f"rep_a_q.shape {rep_a_q.shape}, w1_q.shape {w1_q.shape}, w1_scale.shape {w1_scale.shape}")
    # print(f"expert_offsets.shape {expert_offsets.shape}, problem_sizes1.shape {problem_sizes1.shape}")
    # print(
    #     f"ab_strides1.shape {ab_strides1.shape}, c_strides1.shape {c_strides1.shape}"
    # )
    # EP:
    # rep_a_q.shape torch.Size([262144, 7168]), w1_q.shape torch.Size([32, 4096, 3584]), w1_scale.shape torch.Size([32, 14, 16384])
    # expert_offsets.shape torch.Size([33]), problem_sizes1.shape torch.Size([32, 3])

    # torch.save(rep_a_q, f'/nvme0n1/{device_id}_rep_a_q.tensor')
    # torch.save(w1_q, f'/nvme0n1/{device_id}_w1_q.tensor')
    # torch.save(rep_a1_scales, f'/nvme0n1/{device_id}_rep_a1_scales.tensor')
    # torch.save(w1_scale, f'/nvme0n1/{device_id}_w1_scale.tensor')
    # torch.save(expert_offsets, f'/nvme0n1/{device_id}_expert_offsets.tensor')
    # torch.save(problem_sizes1, f'/nvme0n1/{device_id}_problem_sizes1.tensor')
    # torch.save(a_strides1, f'/nvme0n1/{device_id}_a_strides1.tensor')
    # torch.save(b_strides1, f'/nvme0n1/{device_id}_b_strides1.tensor')
    # torch.save(c_strides1, f'/nvme0n1/{device_id}_c_strides1.tensor')
    # torch.save(s_strides13, f'/nvme0n1/{device_id}_s_strides13.tensor')

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.half)
    c2 = torch.empty((m * topk, k), device=device, dtype=torch.half)

    ops.int4_fp8_grouped_gemm(c1, rep_a_q, w1_q, rep_a1_scales, w1_scale,
                              expert_offsets[:-1], problem_sizes1, a_strides1,
                              b_strides1, c_strides1, s_strides13, 128)
    # print("c1", c1)
    # print("c1.shape", c1.shape)
    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.half)
    torch.ops._C.silu_and_mul(intermediate, c1)

    intemediate_q, a2_scale = ops.scaled_fp8_quant(
        intermediate, a2_scale.float(), use_per_token_if_dynamic=per_act_token)

    ops.int4_fp8_grouped_gemm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
                              expert_offsets[:-1], problem_sizes2, a_strides2,
                              b_strides2, c_strides2, s_strides2, 128)
    # print("c2", c2)
    # print("c2.shape", c2.shape)

    # c2 = torch.randn((m * topk, k), dtype=torch.half, device=device)

    # Gather tokens
    # print(f"xx c_map {c_map}")
    c_map = torch.clamp(c_map, min=0, max=c2.size(0) - 1)
    # print(f"ssss_after c_map {c_map}")
    c2 = c2[c_map].view(m, topk, k)
    # print(f"yy c_map {c_map}")

    if not apply_router_weight_on_input:
        c2 = c2 * topk_weights.view(m, topk, 1).to(torch.half)
    c2 = c2.to(torch.bfloat16)
    return c2.sum(dim=1)


def print_tensor(name: str, t: torch.Tensor):
    print(f"{name}: shape {t.shape}, dtype {t.dtype}, device {t.device}")
    elements_to_print = min(20, t.numel())
    values_cpu = t.flatten()[:elements_to_print].cpu()
    print(f"{name} values (first {elements_to_print}): {values_cpu}")
