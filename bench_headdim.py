import csv
import time
import torch
import flashinfer

DTYPE = torch.float16
NUM_HEADS = 32
KV_LEN = 2048
HEAD_DIMS = [64, 128, 256]
WARMUP = 10
ITERS = 50
OUT_CSV = "results/headdim_results.csv"


def bench_once(head_dim: int) -> float:
    q = torch.randn(NUM_HEADS, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(KV_LEN, NUM_HEADS, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn(KV_LEN, NUM_HEADS, head_dim, device="cuda", dtype=DTYPE)

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
print("kv_len:", KV_LEN)
print("warmup:", WARMUP, "iters:", ITERS)

rows = []
for head_dim in HEAD_DIMS:
    avg_ms = bench_once(head_dim)
    rows.append({"head_dim": head_dim, "avg_ms": avg_ms})
    print(f"head_dim={head_dim}, avg_ms={avg_ms:.4f}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["head_dim", "avg_ms"])
    writer.writeheader()
    writer.writerows(rows)

print(f"saved results to {OUT_CSV}")
