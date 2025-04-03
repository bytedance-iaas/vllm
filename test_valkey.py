from vllm import _custom_ops as ops
import torch

client = ops.valkey_connect("127.0.0.1", 6380, False)
key = "4"

kv_cache = torch.randn((2, 10), dtype=torch.float16, device="cuda:0")
print(f"kv_cache is {kv_cache}")
ops.valkey_set(client, key, kv_cache)

get_cache = torch.zeros((2, 10), dtype=torch.float16, device="cuda:0")
ops.valkey_get(client, key, get_cache)

print(f"get_cache is {kv_cache}")