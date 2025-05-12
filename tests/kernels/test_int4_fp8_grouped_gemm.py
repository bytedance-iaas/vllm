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

@pytest.mark.parametrize("batch_size", [1])
def test_int4_fp8_grouped_gemm_single_expert(batch_size):
    # Test parameters
    num_experts = 1
    m = batch_size  # batch size
    k = 512  # input dimension
    n = 512  # output dimension
    
    print(f"\nTesting with batch_size={batch_size}")
    
    # Create input tensors with ones
    a = torch.ones(m, k, dtype=torch.bfloat16, device='cuda')
    ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device='cuda')
    
    # Create scales with ones
    a_scale = torch.ones(1, k, dtype=torch.float, device='cuda')
    ref_w_scale = torch.ones(num_experts, k // 128, n, dtype=torch.bfloat16, device='cuda')

    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)
    
    # Create expert offsets and problem sizes
    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device='cuda')
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device='cuda')

    device = "cuda"
    a_strides = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
    b_strides = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
    c_strides = torch.empty((num_experts, 3), dtype=torch.int64, device=device)
    s_strides = c_strides
    a_strides[:, 0].fill_(k)
    a_strides[:, 1].fill_(1)
    a_strides[:, 2].zero_()
    b_strides[:, 0].fill_(k // 2)
    b_strides[:, 1].fill_(1)
    b_strides[:, 2].zero_()
    c_strides[:, 0].fill_(1)
    c_strides[:, 1].fill_(n)
    c_strides[:, 2].zero_()
    
    # Print all input parameters
    print_tensor_info("Input a", a)
    print_tensor_info("Weights w", w)
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
    ops.int4_fp8_grouped_gemm(c, a_q, w, a_scale, w_scale,
                             expert_offsets[:-1], problem_sizes,
                             a_strides, b_strides, c_strides,
                             s_strides, 128)
    
    print_tensor_info("Output c", c)
    
    # Reference implementation
    c_ref = torch.matmul(a, ref_w[0].t().to(torch.bfloat16))  # Using .t() property instead of .T()
    print_tensor_info("Reference output c_ref", c_ref)
    
    # Compare results
    max_diff = torch.max(torch.abs(c - c_ref))
    mean_diff = torch.mean(torch.abs(c - c_ref))
    print(f"\nMax difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    # Basic shape checks
    assert c.shape == (m, n)
    assert not torch.isnan(c).any()
    assert not torch.isinf(c).any()

if __name__ == "__main__":
    pytest.main([__file__]) 
