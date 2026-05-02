# Lightweight Telemetry for FlashInfer JIT and Decode Profiling

This repository contains a lab project that adds lightweight observability to
FlashInfer and uses it to profile JIT compilation, module loading, batch decode
planning, and repeated decode execution.

The main benchmark measures:

- cold single-decode execution,
- warm single-decode execution,
- paged-KV batch decode planning,
- repeated paged-KV batch decode runs.

A report is also included in this repo [Lightweight Telemetry for FlashInfer.pdf](<Lightweight Telemetry for FlashInfer.pdf>).

## Changes and Key Files

- `bench_workflow.py` - primary workflow benchmark used in the report.
- `run_all_benchmarks.sh` - runs the workflow benchmark and writes one telemetry log.
- `setup_flashinfer_env.sh` - HPC environment setup and telemetry defaults.
- `analyze_flashinfer_logs.py` - converts JSONL telemetry into CSV summaries.
- `plot_flashinfer_logs.py` - generates report figures from benchmark outputs.
- `src/flashinfer/flashinfer/telemetry.py` - lightweight telemetry helpers.
- `src/flashinfer/flashinfer/jit/core.py` - JIT-path instrumentation.
- `src/flashinfer/flashinfer/decode.py` - decode wrapper instrumentation.

scripts such as `bench_jit_overhead.py`, `bench_headdim.py`,
`bench_kvlen.py`, and `bench_decode.py` are retained as supporting
microbenchmarks.

## Running on the Rice HPC Cluster

Request an interactive GPU node first:

```bash
srun --pty --time=02:59:59 --gpus=1 --mem=64G $SHELL
```

Then run:

```bash
source setup_flashinfer_env.sh
bash run_all_benchmarks.sh

latest=$(ls -t logs/obs_*_workflow.jsonl | head -1)
python analyze_flashinfer_logs.py "$latest"
python plot_flashinfer_logs.py "$latest"
```

The setup script enables FlashInfer observability by default:

- `FLASHINFER_OBS_ENABLE=1`
- `FLASHINFER_OBS_SAMPLE_RATE=1.0`
- `FLASHINFER_OBS_FORMAT=jsonl`

Also avoid the home quota issue.

## Outputs

Typical generated outputs include:

- `logs/obs_*_workflow.jsonl` - raw telemetry events.
- `logs/obs_*_workflow.summary.csv` - flattened event records.
- `logs/obs_*_workflow.event_summary.csv` - per-event timing summary.
- `results/workflow_results.csv` - phase-level benchmark measurements.
- `figures/*.pdf` and `figures/*.png` - report figures.


## Public files on GitHub

 Env caches, build artifacts, or HPC specific logs are not public on Github.

## Environment Used for the telemetry

- GPU: NVIDIA L40S
- Compute capability: 8.9 (`sm_89`)
- PyTorch: 2.6.0
- CUDA: 12.6
- FlashInfer: 0.6.7 with telemetry injection

## Author

Yuzhi Han, Rice University
