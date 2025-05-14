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
    topk_ids_: torch.Tensor,
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
        Shape: [num_experts, N * 2,  K // 2]
        (the weights are passed transposed and int4-packed)
    - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
        Shape: [num_experts, K, N // 2]
        (the weights are passed transposed and int4-packed)
    - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
        Shape: [num_experts, K // 512, N * 8]
    - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
        Shape: [num_experts, N // 512, K * 4]
    - topk_weights (torch.Tensor): The weights of each token->expert mapping.
    - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
    - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
    - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
    - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
    - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
    - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
    - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
    - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
    - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
        Shape: scalar or [1, K]
    - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
        quantize the intermediate result between the gemms.
        Shape: scalar or [1, N]
    - expert_map (Optional[torch.Tensor]): In the case of Expert parallel,
        every Rank is responsible for a subset of experts. expert_map is a
        mapping from global expert-id to local expert-id. When expert_map[i]
        is -1, it means that this Rank is not responsible for global
        expert-id i.
    - apply_router_weight_on_input (bool): When true, the topk weights are
        applied directly on the inputs. This is only applicable when topk is 1.

    Returns:
    - torch.Tensor: The fp8 output tensor after applying the MoE layer.
    """

    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
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

    # assert a1_scale is None or a1_scale.dim() == 0 or \
    #     a1_scale.shape[0] == 1 or \
    #     a1_scale.shape[0] == a.shape[0], "Input scale shape mismatch"
    # assert a2_scale is None or a1_scale is None or \
    #     a2_scale.shape == a1_scale.shape, "Intermediate scale shape mismatch"  # noqa: E501
    assert a_strides1.shape[0] == w1_q.shape[
        0], "A Strides 1 expert number mismatch"
    assert b_strides1.shape[0] == w1_q.shape[
        0], "B Strides 1 expert number mismatch"
    # assert c_strides1.shape[0] == w1_q.shape[
    #     0], "C Strides 1 expert number mismatch"
    assert a_strides2.shape[0] == w2_q.shape[
        0], "A Strides 2 expert number  mismatch"
    assert b_strides2.shape[0] == w2_q.shape[
        0], "B Strides 2 expert number mismatch"
    # assert c_strides2.shape[0] == w2_q.shape[
    #     0], "C Strides 2 expert number mismatch"

    num_experts = w1_q.size(0)
    m = a.size(0)
    k = w1_q.size(2) * 2  # w1_q is transposed and packed
    n = w2_q.size(2) * 2  # w2_q is transposed and packed

    local_topk_ids = topk_ids_
    if expert_map is not None:
        "Translate info from expert_map to topk_ids"
        local_topk_ids = torch.where(expert_map[topk_ids_] != -1,
                                     expert_map[topk_ids_], -1)
    
    topk = local_topk_ids.size(1)

    per_act_token = a1_scale.numel() != 1 if a1_scale is not None else (
        a2_scale.numel() != 1 if a2_scale is not None else False)
    if apply_router_weight_on_input:
        assert topk == 1, \
            "apply_router_weight_on_input is only implemented for topk=1"
        # TODO: this only works for topK=1, will need to update for topK>1
        a = a * topk_weights.to(torch.half)

    # print_tensor("a", a)
    # print_tensor("a1_scale (before)", a1_scale)
    # print_tensor("a2_scale (before)", a2_scale)
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
    
    a_map_initializer = torch.empty
    c2_initializer = torch.empty
    if expert_map is not None:
        # With expert_map each Rank processes only a subset of experts. As
        # a result not all of a_map and c2 tensors are filled. We fill it
        # zeros for correctness.
        a_map_initializer = torch.zeros
        c2_initializer = torch.zeros

    a_map = a_map_initializer((local_topk_ids.numel()),
                              dtype=torch.int32,
                              device=device)
    c_map = torch.empty((local_topk_ids.numel()),
                        dtype=torch.int32,
                        device=device)

    # print(f"topk_ids {topk_ids}, expert_offsets {expert_offsets}, problem_sizes1 {problem_sizes1}")
    # print(f"problem_sizes2 {problem_sizes2}, a_map {a_map}, c_map {c_map}")
    # print(f"num_experts {num_experts}, n {n}, k {k}")
    # a1_scale and a2_scale are not supported

    # print(f"before get_cutlass_moe_mm_data")
    ops.get_cutlass_moe_mm_data(local_topk_ids, expert_offsets, problem_sizes1,
                                problem_sizes2, a_map, c_map, num_experts, n,
                                k)
    # print(f"finish get_cutlass_moe_mm_data")
    # print_tensor("topk_ids", topk_ids)
    # print_tensor("expert_offsets", expert_offsets)
    # print_tensor("problem_sizes1", problem_sizes1)
    # print_tensor("problem_sizes2", problem_sizes2)
    # print_tensor("a_map", a_map)
    # print_tensor("c_map", c_map)
    # print(f"n {n}, k {k}")

    # a_map = torch.clamp(a_map,
    #                     min=0,
    #                     max=min(a1_scale.size(0) - 1,
    #                             a_q.size(0) - 1))
    rep_a_q = a_q.view(dtype=torch.uint8)[a_map].view(dtype=a_q.dtype)
    #rep_a1_scales = a1_scale[a_map] if per_act_token else a1_scale

    # print_tensor("a_q", a_q)
    # print_tensor("a1_scale (after)", a1_scale)
    # print_tensor("a_map", a_map)
    # print_tensor("rep_a_q", rep_a_q)
    # print_tensor("rep_a1_scales", rep_a1_scales)

    # device_id = a.get_device()
    # save_dir = f"/nvme0n1/w4a8_debug_tensors/group_gemm_param/device_{device_id}"
    # import os
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    #     torch.save(a_map, f'{save_dir}/a_map.tensor')
    #     torch.save(rep_a_q, f'{save_dir}/rep_a_q.tensor')
    #     torch.save(w1_q, f'{save_dir}/w1_q.tensor')
    #     torch.save(a1_scale, f'{save_dir}/a1_scale.tensor')
    #     torch.save(w1_scale, f'{save_dir}/w1_scale.tensor')
    #     torch.save(expert_offsets, f'{save_dir}/expert_offsets.tensor')
    #     torch.save(problem_sizes1, f'{save_dir}/problem_sizes1.tensor')
    #     torch.save(a_strides1, f'{save_dir}/a_strides1.tensor')
    #     torch.save(b_strides1, f'{save_dir}/b_strides1.tensor')
    #     torch.save(c_strides1, f'{save_dir}/c_strides1.tensor')
    #     torch.save(s_strides13, f'{save_dir}/s_strides13.tensor')

    c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.half)
    c2 = c2_initializer((m * topk, k), device=device, dtype=torch.half)

    ops.cutlass_w4a8_moe_mm(c1, rep_a_q, w1_q, a1_scale, w1_scale,
                              expert_offsets[:-1], problem_sizes1, a_strides1,
                              b_strides1, c_strides1, s_strides13, 128, m)

    # print("c1", c1)
    # print("c1.shape", c1.shape)
    intermediate = torch.empty((m * topk, n), device=device, dtype=torch.half)
    torch.ops._C.silu_and_mul(intermediate, c1)

    intemediate_q, a2_scale = ops.scaled_fp8_quant(
        intermediate, a2_scale.float(), use_per_token_if_dynamic=per_act_token)

    ops.cutlass_w4a8_moe_mm(c2, intemediate_q, w2_q, a2_scale, w2_scale,
                              expert_offsets[:-1], problem_sizes2, a_strides2,
                              b_strides2, c_strides2, s_strides2, 128, m)
    # print("c2", c2)
    # print("c2.shape", c2.shape)

    # Gather tokens
    # print(f"xx c_map {c_map}")
    # c_map = torch.clamp(c_map, min=0, max=c2.size(0) - 1)
    # print(f"ssss_after c_map {c_map}")
    c2 = c2[c_map].view(m, topk, k)

    if not apply_router_weight_on_input:
        c2 = c2 * topk_weights.view(m, topk, 1)

    return c2.sum(dim=1).to(torch.bfloat16)


def print_tensor(name: str, t: torch.Tensor):
    print(f"{name}: shape {t.shape}, dtype {t.dtype}, device {t.device}")
    elements_to_print = min(20, t.numel())
    values_cpu = t.flatten()[:elements_to_print].cpu()
    print(f"{name} values (first {elements_to_print}): {values_cpu}")
