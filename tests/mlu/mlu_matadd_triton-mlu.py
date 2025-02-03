import triton
import triton.language as tl
import torch
import torch_mlu

# Simple kernel for matrix addition (2x2 matrix)
@triton.jit
def matrix_add_kernel(A, B, C):
    # Hardcoded for 2x2 matrices
    row = tl.arange(0, 2)
    col = tl.arange(0, 2)
    
    # Load values from A and B
    a = tl.load(A + row[:, None] * 2 + col[None, :])
    b = tl.load(B + row[:, None] * 2 + col[None, :])
    
    # Add the matrices
    c = a + b
    
    # Store the result in C
    tl.store(C + row[:, None] * 2 + col[None, :], c)

# Initialize 2x2 matrices A and B with random values
A = torch.rand((2, 2), dtype=torch.float32).mlu()
B = torch.rand((2, 2), dtype=torch.float32).mlu()
C = torch.empty((2, 2), dtype=torch.float32).mlu()

# Print matrices A and B
print("Matrix A:")
print(A)
print("Matrix B:")
print(B)

# Launch the kernel
matrix_add_kernel[(1, 1)](A, B, C)

# Print the result matrix C
print("Matrix C (A + B):")
print(C)
