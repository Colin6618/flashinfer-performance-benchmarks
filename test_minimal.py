import torch
import flashinfer

q = torch.randn(32, 128, device="cuda", dtype=torch.float16)
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

print("start")
o = flashinfer.single_decode_with_kv_cache(q, k, v)
print("output shape:", o.shape)
print("output dtype:", o.dtype)
print("output device:", o.device)
