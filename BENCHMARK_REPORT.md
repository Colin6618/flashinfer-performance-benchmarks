# FlashInfer Decode Benchmarking on NOTS Cluster

## Project Scope
This report summarizes a small, reproducible benchmark study of FlashInfer decode kernels on the NOTS cluster.

- Focus kernel path: `single_decode_with_kv_cache`
- Focus scenario: decode-side KV-cache growth in autoregressive serving
- Goal: validate systems-level behavior trends, not claim official FlashInfer performance
- Hardware caveat: Tesla V100 (SM70) is below FlashInfer's official SM75+ support

Important context:
- FlashInfer officially targets SM75+. It is highly likely that V100 cards will be allocated on the NOTS cluster; other GPU, such as the L40, are sometimes available.
- These runs used an experimental bypass in `flashinfer/jit/core.py` plus `FLASHINFER_EXPERIMENTAL_VOLTA=1`.
- All numbers in this report are exploratory and unofficial.

## Implementation
Benchmarks in this repository:

- `test_minimal.py`: minimal feasibility check for decode on V100
- `bench_kvlen.py`: main KV-cache-length latency sweep
- `bench_jit_overhead.py`: first-call versus steady-state latency
- `bench_headdim.py`: supplementary head-dimension sweep

Generated result files:

- `kvlen_results.csv`
- `jit_overhead_results.csv`
- `headdim_results.csv`

## Experimental Setup
Platform and software:

- Cluster: NOTS
- GPU: Tesla V100-PCIE-32GB
- GPU capability: SM70
- PyTorch: 2.6.0
- CUDA: 12.6
- Data type: float16

Common benchmark settings (from scripts):

- Warmup iterations: 10 (for sweeps)
- Timed iterations: 50
- CUDA synchronization around timed regions

Workloads:

1. KV-cache sweep (main): fixed `num_heads=32`, `head_dim=128`, vary `kv_len` in `{512, 1024, 2048, 4096, 8192}`
2. JIT overhead: fixed `num_heads=32`, `head_dim=128`, `kv_len=2048`
3. Head-dim sweep (supplementary): fixed `num_heads=32`, `kv_len=2048`, vary `head_dim` in `{64, 128, 256}`


## Feasibility Check
After enabling `FLASHINFER_EXPERIMENTAL_VOLTA=1`, minimal decode ran successfully.

Representative output:

- device: Tesla V100-PCIE-32GB
- capability: (7, 0)
- output shape: `torch.Size([32, 128])`
- output dtype: `torch.float16`
- output device: `cuda:0`

Interpretation:
- FlashInfer can run on this V100 setup through an experimental path.
- This does not change official support status (still SM75+).

## Results Summary
From recorded CSV files in this repo:

- Decode latency increases with KV-cache length (main result).
- First-call latency is much higher than steady-state latency due to JIT/setup.
- Supplementary head-dim sweep shows monotonic latency increase with larger head dimensions.

### Main Result: Decode Latency vs KV-Cache Length
Source: `kvlen_results.csv`

| KV-Cache Length | Average Decode Latency (ms) |
| ---: | ---: |
| 512  | 0.026206 |
| 1024 | 0.034270 |
| 2048 | 0.056226 |
| 4096 | 0.098318 |
| 8192 | 0.184424 |

Takeaway:
- Decode latency increases with KV-cache length, consistent with KV-cache-intensive autoregressive decoding.

### JIT Insight: First Call vs Steady-State
Source: `jit_overhead_results.csv`

| Metric | Value |
| --- | ---: |
| First call latency | 170.29 ms |
| Steady-state average latency | 0.0767 ms |
| Steady-state minimum latency | 0.0727 ms |
| Steady-state maximum latency | 0.1584 ms |
| Cold-start / steady-state ratio | 2219.95x |

Takeaway:
- FlashInfer pays a large one-time JIT cost, but repeated decode calls are much faster after specialization.

### Supplementary: Head-Dimension Sweep
Source: `headdim_results.csv`

| Head Dimension | Average Decode Latency (ms) |
| ---: | ---: |
| 64  | 0.036901 |
| 128 | 0.056171 |
| 256 | 0.098093 |

## Observations

- KV-cache growth is the dominant trend in this microbenchmark: larger cache length increases decode latency.
- The cold-start effect is significant and should be separated from steady-state behavior in any serving discussion.
- Even on unsupported hardware, the experiments reproduce expected systems-level patterns (cache growth cost + JIT warmup effect).

## Limitations

- Hardware limitation: Tesla V100 (SM70) is outside FlashInfer's official SM75+ support range.
- Compatibility path is experimental (`flashinfer/jit/core.py` bypass and `FLASHINFER_EXPERIMENTAL_VOLTA=1`).
- Workloads are synthetic microbenchmarks, not end-to-end serving traces.
- No controlled comparison here against official FlashInfer numbers on supported GPUs.

## Reproducibility
Run order:

```bash
source setup_flashinfer_env.sh
python test_minimal.py
python bench_kvlen.py
python bench_jit_overhead.py
python bench_headdim.py
```

Expected output artifacts:

- `kvlen_results.csv`
- `jit_overhead_results.csv`
- `headdim_results.csv`

## Conclusion
This project validates core serving-oriented behavior of FlashInfer decode kernels under an exploratory V100 setup on NOTS.


