import torch
import torch_mlu

t0 = torch.randn(2, 2, device='mlu')
t1 = torch.randn(2, 2, device='mlu')
print('t0 = ', t0)
print('t1 = ', t1)
print('t0 + t1 =', t0 + t1)
