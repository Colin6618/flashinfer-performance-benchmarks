import csv
import time
import torch
import flashinfer

DTYPE = torch.float16
NUM_HEADS = 32
HEAD_DIM = 128
KV_LEN = 2048
STEADY_ITERS = 50
OUT_CSV = "results/jit_overhead_results.csv"


def make_inputs():
    q = torch.randn(NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    k = torch.randn(KV_LEN, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    v = torch.randn(KV_LEN, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    return q, k, v


def run_once(q, k, v):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    o = flashinfer.single_decode_with_kv_cache(q, k, v)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return o, (t1 - t0) * 1000.0


print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("dtype:", DTYPE)
print("num_heads:", NUM_HEADS)
print("head_dim:", HEAD_DIM)
print("kv_len:", KV_LEN)
print("steady_iters:", STEADY_ITERS)

q, k, v = make_inputs()

out, first_ms = run_once(q, k, v)
print("first_call_ms:", f"{first_ms:.4f}")
print("output shape:", tuple(out.shape))

steady_times = []
for _ in range(STEADY_ITERS):
    _, ms = run_once(q, k, v)
    steady_times.append(ms)

steady_avg_ms = sum(steady_times) / len(steady_times)
steady_min_ms = min(steady_times)
steady_max_ms = max(steady_times)

print("steady_avg_ms:", f"{steady_avg_ms:.4f}")
print("steady_min_ms:", f"{steady_min_ms:.4f}")
print("steady_max_ms:", f"{steady_max_ms:.4f}")
print("first_vs_steady_ratio:", f"{first_ms / steady_avg_ms:.2f}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "first_call_ms",
            "steady_avg_ms",
            "steady_min_ms",
            "steady_max_ms",
            "first_vs_steady_ratio",
            "kv_len",
            "num_heads",
            "head_dim",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "first_call_ms": first_ms,
            "steady_avg_ms": steady_avg_ms,
            "steady_min_ms": steady_min_ms,
            "steady_max_ms": steady_max_ms,
            "first_vs_steady_ratio": first_ms / steady_avg_ms,
            "kv_len": KV_LEN,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
        }
    )

print(f"saved results to {OUT_CSV}")
