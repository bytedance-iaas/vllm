import torch
import pytest
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoE

def print_tensor_info(name, tensor):
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  values: {tensor.flatten()[:10]}")  # Print first 10 values


def pack_interleave(num_experts, ref_weight, ref_scale):
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    weight = packer(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    # w_q = w_q.contiguous().transpose(1, 2)
    w_q = w_q.contiguous()

    ###############################################################
    # scale interleave, [E, K, N]
    # scale = ref_scale.permute(0, 2, 1)  # [E, N, K]
    scale = ref_scale
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


@pytest.mark.parametrize("batch_size", [1])
def test_int4_fp8_grouped_gemm_single_expert(batch_size):
    # Test parameters
    num_experts = 1
    m = batch_size  # batch size
    k = 7168  # input dimension
    n = 2048  # output dimension
    #torch.manual_seed(0)
    dtype = torch.bfloat16

    print(f"\nTesting with batch_size={batch_size}")

    # Create input tensors with ones
    a = torch.randn(m, k, dtype=dtype, device='cuda')
    # a = torch.abs(torch.randn(m, k, dtype=torch.bfloat16, device='cuda'))
    # a = torch.ones(m, k, dtype=torch.bfloat16, device='cuda')
    #a = torch.full((m, k), -1, dtype=torch.bfloat16, device='cuda')
    ref_w = torch.randint(-8,
                          8, (num_experts, n, k),
                          dtype=torch.int8,
                          device='cuda')
    # ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device='cuda')

    # Create scales with ones
    affine_coeff = 0.005
    # a_scale = torch.ones(1, k, dtype=torch.float, device='cuda')
    a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
    # ref_w_scale = torch.ones(num_experts,
    #                          k // 128,
    #                          n,
    #                          dtype=dtype,
    #                          device='cuda') * affine_coeff
    # ref_w_scale = torch.randn(
    #     num_experts, k // 128, n, dtype=dtype, device='cuda') * affine_coeff
    ref_w_scale = torch.randn(
        num_experts, n, k // 128, dtype=dtype, device='cuda') * affine_coeff
    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)

    # Create expert offsets and problem sizes
    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device='cuda')
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device='cuda')

    device = "cuda"
    a_strides = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
    b_strides = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
    c_strides = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
    s_strides = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
    # s_strides = c_strides
    a_strides[:, 0].fill_(k)
    a_strides[:, 1].fill_(1)
    a_strides[:, 2].zero_()
    b_strides[:, 0].fill_(k)
    b_strides[:, 1].fill_(1)
    b_strides[:, 2].zero_()
    # b_strides[:, 0].fill_(1)
    # b_strides[:, 1].fill_(k)
    # b_strides[:, 2].zero_()
    c_strides[:, 0].fill_(n)
    c_strides[:, 1].fill_(1)
    c_strides[:, 2].zero_()
    s_strides[:, 0].fill_(n)
    s_strides[:, 1].fill_(1)
    s_strides[:, 2].zero_()

    # Print all input parameters
    print_tensor_info("Input a", a)
    print_tensor_info("Weights w", w)
    print_tensor_info("Ref Weights ref_w", ref_w)
    print_tensor_info("Input scale a_scale", a_scale)
    print_tensor_info("Weight scale w_scale", w_scale)
    print_tensor_info("Expert offsets", expert_offsets)
    print_tensor_info("Problem sizes", problem_sizes)
    print_tensor_info("A strides", a_strides)
    print_tensor_info("B strides", b_strides)
    print_tensor_info("C strides", c_strides)
    print_tensor_info("S strides", s_strides)

    # Quantize input
    a_q, a_scale = ops.scaled_fp8_quant(a, a_scale)
    print_tensor_info("Quantized input a_q", a_q)
    print_tensor_info("Updated input scale a_scale", a_scale)

    # Create output tensor
    c = torch.empty((m, n), dtype=torch.float16, device='cuda')

    # Run the operator
    # w = w.view(torch.quint4x2)
    ops.cutlass_w4a8_moe_mm(c, a_q, w, a_scale, w_scale, expert_offsets[:-1],
                            problem_sizes, a_strides, b_strides, c_strides,
                            s_strides, 128, m)
    c = c.to(dtype)

    print_tensor_info("Output c", c)

    # Reference implementation
    a = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn)
    ref_w_scale_repeat = ref_w_scale[0].repeat_interleave(128,
                                                              dim=1).to(float)
    print_tensor_info("ref_w_scale_repeat", ref_w_scale_repeat)
    ref_w_one_expert = (ref_w[0].to(float) * ref_w_scale_repeat).to(dtype)
    print_tensor_info("ref_w_one_expert", ref_w_one_expert)
    c_ref = torch.matmul(a.to(dtype), ref_w_one_expert.t().to(dtype)) * a_scale
    c_ref = c_ref.to(dtype)
    print_tensor_info("Reference output c_ref", c_ref)

    # Compare results
    max_diff = torch.max(torch.abs(c - c_ref))
    mean_diff = torch.mean(torch.abs(c - c_ref))
    print(f"\nMax difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print("relative diff: ",
          torch.mean(torch.abs(c - c_ref) / torch.abs(c_ref)))

    # woq_assert_near_eq(c_ref, c, 2)
    torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=0.1)

    # Basic shape checks
    assert c.shape == (m, n)
    assert not torch.isnan(c).any()
    assert not torch.isinf(c).any()

    # Assert close


if __name__ == "__main__":
    pytest.main([__file__])
