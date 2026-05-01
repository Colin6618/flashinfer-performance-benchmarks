import csv
import time
import torch
import flashinfer

DTYPE = torch.float16
NUM_HEADS = 32
KV_LEN = 2048
HEAD_DIM = 128
BATCH_SIZE = 8
PAGE_SIZE = 16
WARMUP = 5
ITERS = 20
OUT_CSV = "results/decode_results.csv"



def make_inputs(batch_size, num_heads, kv_len, head_dim):
    q = torch.randn(batch_size, num_heads, head_dim, device="cuda", dtype=DTYPE)

    num_pages_per_seq = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_num_pages = batch_size * num_pages_per_seq
    kv_data = torch.randn(
        total_num_pages,
        2,
        PAGE_SIZE,
        num_heads,
        head_dim,
        device="cuda",
        dtype=DTYPE,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % PAGE_SIZE + 1,
        device="cuda",
        dtype=torch.int32,
    )
    return q, kv_data, kv_indptr, kv_indices, kv_last_page_len



def bench_decode():
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len = make_inputs(
        BATCH_SIZE, NUM_HEADS, KV_LEN, HEAD_DIM
    )
    workspace_buffer = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    decode_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        NUM_HEADS,
        NUM_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        data_type=DTYPE,
        q_data_type=DTYPE,
    )

    for _ in range(WARMUP):
        _ = decode_wrapper.run(q, kv_data)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = decode_wrapper.run(q, kv_data)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / ITERS * 1000.0
    return avg_ms


print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("dtype:", DTYPE)
print("num_heads:", NUM_HEADS)
print("head_dim:", HEAD_DIM)
print("kv_len:", KV_LEN)
print("batch_size:", BATCH_SIZE)
print("page_size:", PAGE_SIZE)
print("warmup:", WARMUP, "iters:", ITERS)

avg_ms = bench_decode()
print(f"batch_decode avg_ms={avg_ms:.4f}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["batch_size", "kv_len", "page_size", "avg_ms"]
    )
    writer.writeheader()
    writer.writerow(
        {
            "batch_size": BATCH_SIZE,
            "kv_len": KV_LEN,
            "page_size": PAGE_SIZE,
            "avg_ms": avg_ms,
        }
    )

print(f"saved results to {OUT_CSV}")
