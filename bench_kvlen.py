import csv
import time
import torch
import flashinfer

DTYPE = torch.float16
NUM_HEADS = 32
HEAD_DIM = 128
KV_LENS = [512, 1024, 2048, 4096, 8192]
WARMUP = 10
ITERS = 50
OUT_CSV = "results/kvlen_results.csv"


def bench_once(kv_len: int) -> float:
    q = torch.randn(NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    k = torch.randn(kv_len, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    v = torch.randn(kv_len, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)

    for _ in range(WARMUP):
        _ = flashinfer.single_decode_with_kv_cache(q, k, v)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = flashinfer.single_decode_with_kv_cache(q, k, v)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / ITERS * 1000.0


print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("dtype:", DTYPE)
print("num_heads:", NUM_HEADS)
print("head_dim:", HEAD_DIM)
print("warmup:", WARMUP, "iters:", ITERS)

rows = []
for kv_len in KV_LENS:
    avg_ms = bench_once(kv_len)
    rows.append({"kv_len": kv_len, "avg_ms": avg_ms})
    print(f"kv_len={kv_len}, avg_ms={avg_ms:.4f}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["kv_len", "avg_ms"])
    writer.writeheader()
    writer.writerows(rows)

print(f"saved results to {OUT_CSV}")
