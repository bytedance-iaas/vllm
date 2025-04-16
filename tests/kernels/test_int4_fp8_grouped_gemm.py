import torch
import numpy as np
from vllm._custom_ops import int4_fp8_grouped_gemm

def test_int4_fp8_grouped_gemm():
    # Set device

    for device_id in range(3, 4):
        #torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("TP %d" % device_id)
        device = 'cuda:%d' % device_id
        rep_a_q = torch.load(f'/nvme0n1/{device_id}_rep_a_q.tensor', weights_only=False).to(torch.device(device))
        w1_q = torch.load(f'/nvme0n1/{device_id}_w1_q.tensor', weights_only=False).to(torch.device(device))
        w1_scale = torch.load(f'/nvme0n1/{device_id}_w1_scale.tensor', weights_only=False).to(torch.device(device))
        expert_offsets = torch.load(f'/nvme0n1/{device_id}_expert_offsets.tensor', weights_only=False).to(torch.device(device))
        problem_sizes1 = torch.load(f'/nvme0n1/{device_id}_problem_sizes1.tensor', weights_only=False).to(torch.device(device))
    
        # Run the kernel
        d_tensor = int4_fp8_grouped_gemm(
            rep_a_q,
            w1_q,
            w1_scale,
            expert_offsets[:-1],
            problem_sizes1,
            128
        )
        
        # Print output tensor information
        print("\nOutput tensor D:")
        print(f"  Shape: {list(d_tensor.shape)}")
        print(f"  Strides: {list(d_tensor.stride())}")
        
        # Verify output shape
        assert d_tensor.shape == (262144, 1024), f"Expected output shape (262144, 1024), got {d_tensor.shape}"
        assert d_tensor.dtype == torch.float16, f"Expected output dtype float16, got {d_tensor.dtype}"

        print(d_tensor)
        
        print("\nTest passed successfully!")

if __name__ == "__main__":
    test_int4_fp8_grouped_gemm() 
