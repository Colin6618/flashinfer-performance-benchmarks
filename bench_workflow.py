import csv
import time
from pathlib import Path

import torch
import flashinfer

from flashinfer.telemetry import flush, record_event


DTYPE = torch.float16
NUM_HEADS = 32
HEAD_DIM = 128
KV_LEN = 2048
BATCH_SIZE = 8
PAGE_SIZE = 16
SINGLE_STEADY_ITERS = 50
BATCH_WARMUP = 5
BATCH_ITERS = 20
OUT_CSV = Path("results/workflow_results.csv")


def phase_marker(name, state, **fields):
    record_event("benchmark_phase", phase=name, state=state, **fields)


def synchronize():
    torch.cuda.synchronize()


def elapsed_ms(fn):
    synchronize()
    t0 = time.perf_counter()
    result = fn()
    synchronize()
    return result, (time.perf_counter() - t0) * 1000.0


def make_single_decode_inputs(kv_len=KV_LEN, head_dim=HEAD_DIM):
    q = torch.randn(NUM_HEADS, head_dim, device="cuda", dtype=DTYPE)
    k = torch.randn(kv_len, NUM_HEADS, head_dim, device="cuda", dtype=DTYPE)
    v = torch.randn(kv_len, NUM_HEADS, head_dim, device="cuda", dtype=DTYPE)
    return q, k, v


def make_batch_decode_inputs():
    q = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    num_pages_per_seq = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    total_num_pages = BATCH_SIZE * num_pages_per_seq
    kv_data = torch.randn(
        total_num_pages,
        2,
        PAGE_SIZE,
        NUM_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )
    kv_indptr = (
        torch.arange(0, BATCH_SIZE + 1, device="cuda", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (BATCH_SIZE,),
        (KV_LEN - 1) % PAGE_SIZE + 1,
        device="cuda",
        dtype=torch.int32,
    )
    return q, kv_data, kv_indptr, kv_indices, kv_last_page_len


def single_decode_workflow(rows):
    phase_marker(
        "single_decode_cold",
        "start",
        kv_len=KV_LEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    q, k, v = make_single_decode_inputs()
    out, first_ms = elapsed_ms(lambda: flashinfer.single_decode_with_kv_cache(q, k, v))
    phase_marker(
        "single_decode_cold",
        "end",
        kv_len=KV_LEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        output_shape=str(tuple(out.shape)),
        phase_duration_ms=first_ms,
    )
    rows.append(
        {
            "phase": "single_decode_cold",
            "metric": "first_call_ms",
            "value_ms": first_ms,
            "iters": 1,
            "batch_size": "",
            "kv_len": KV_LEN,
            "page_size": "",
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
        }
    )

    phase_marker(
        "single_decode_steady",
        "start",
        iters=SINGLE_STEADY_ITERS,
        kv_len=KV_LEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    steady_times = []
    for _ in range(SINGLE_STEADY_ITERS):
        _, ms = elapsed_ms(lambda: flashinfer.single_decode_with_kv_cache(q, k, v))
        steady_times.append(ms)
    steady_avg_ms = sum(steady_times) / len(steady_times)
    phase_marker(
        "single_decode_steady",
        "end",
        iters=SINGLE_STEADY_ITERS,
        kv_len=KV_LEN,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        phase_duration_ms=sum(steady_times),
        avg_ms=steady_avg_ms,
    )
    rows.append(
        {
            "phase": "single_decode_steady",
            "metric": "avg_call_ms",
            "value_ms": steady_avg_ms,
            "iters": SINGLE_STEADY_ITERS,
            "batch_size": "",
            "kv_len": KV_LEN,
            "page_size": "",
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
        }
    )


def batch_decode_workflow(rows):
    q, kv_data, kv_indptr, kv_indices, kv_last_page_len = make_batch_decode_inputs()
    workspace_buffer = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )

    phase_marker(
        "batch_decode_plan",
        "start",
        batch_size=BATCH_SIZE,
        kv_len=KV_LEN,
        page_size=PAGE_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )

    _, plan_ms = elapsed_ms(
        lambda: decode_wrapper.plan(
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
    )
    phase_marker(
        "batch_decode_plan",
        "end",
        batch_size=BATCH_SIZE,
        kv_len=KV_LEN,
        page_size=PAGE_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        phase_duration_ms=plan_ms,
    )
    rows.append(
        {
            "phase": "batch_decode_plan",
            "metric": "plan_call_ms",
            "value_ms": plan_ms,
            "iters": 1,
            "batch_size": BATCH_SIZE,
            "kv_len": KV_LEN,
            "page_size": PAGE_SIZE,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
        }
    )

    for _ in range(BATCH_WARMUP):
        _ = decode_wrapper.run(q, kv_data)
    synchronize()

    phase_marker(
        "batch_decode_run",
        "start",
        iters=BATCH_ITERS,
        batch_size=BATCH_SIZE,
        kv_len=KV_LEN,
        page_size=PAGE_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    def run_batch_iters():
        out = None
        for _ in range(BATCH_ITERS):
            out = decode_wrapper.run(q, kv_data)
        return out

    _, total_run_ms = elapsed_ms(run_batch_iters)
    avg_run_ms = total_run_ms / BATCH_ITERS
    phase_marker(
        "batch_decode_run",
        "end",
        iters=BATCH_ITERS,
        batch_size=BATCH_SIZE,
        kv_len=KV_LEN,
        page_size=PAGE_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        phase_duration_ms=total_run_ms,
        avg_ms=avg_run_ms,
    )
    rows.append(
        {
            "phase": "batch_decode_run",
            "metric": "avg_call_ms",
            "value_ms": avg_run_ms,
            "iters": BATCH_ITERS,
            "batch_size": BATCH_SIZE,
            "kv_len": KV_LEN,
            "page_size": PAGE_SIZE,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
        }
    )


def main():
    print("device:", torch.cuda.get_device_name(0), flush=True)
    print("capability:", torch.cuda.get_device_capability(0), flush=True)
    print("dtype:", DTYPE, flush=True)
    print("num_heads:", NUM_HEADS, flush=True)
    print("head_dim:", HEAD_DIM, flush=True)
    print("kv_len:", KV_LEN, flush=True)
    print("batch_size:", BATCH_SIZE, flush=True)
    print("page_size:", PAGE_SIZE, flush=True)
    print("single_steady_iters:", SINGLE_STEADY_ITERS, flush=True)
    print("batch_warmup:", BATCH_WARMUP, "batch_iters:", BATCH_ITERS, flush=True)

    rows = []
    print("[phase] single_decode_cold + single_decode_steady", flush=True)
    single_decode_workflow(rows)
    print("[phase] batch_decode_plan + batch_decode_run", flush=True)
    batch_decode_workflow(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "metric",
                "value_ms",
                "iters",
                "batch_size",
                "kv_len",
                "page_size",
                "num_heads",
                "head_dim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved workflow results to {OUT_CSV}", flush=True)
    for row in rows:
        print(f"{row['phase']} {row['metric']}={row['value_ms']:.4f} ms", flush=True)
    flush()


if __name__ == "__main__":
    main()
