# FlashInfer Performance Benchmarks

A comprehensive benchmark suite for [FlashInfer](https://github.com/flashinfer-ai/flashinfer), a high-performance GPU kernel library for LLM inference. This project analyzes key performance characteristics of FlashInfer's single-decode attention kernels across varying model dimensions and input shapes.

## Overview

FlashInfer provides state-of-the-art GPU kernels for attention computation in LLM inference. This benchmark suite evaluates:

- **Head Dimension Impact**: Latency across different attention head dimensions (64, 128, 256 dimensions)
- **KV Cache Length Scaling**: Performance with varying K-V cache sequence lengths (512-8192 tokens)
- **JIT Compilation Overhead**: First-call latency vs. steady-state performance

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.6+ support
- Python 3.10+
- PyTorch with CUDA support

### Setup

```bash
# Load environment and activate virtual environment
source setup_flashinfer_env.sh
```

This script:
- Loads HPC modules (GCC 13.3.0, PyTorch 2.6.0-CUDA-12.6.0)
- Activates the Python 3.12 virtual environment
- Configures environment variables for GPU caches
- Verifies PyTorch/CUDA installation

### Running Benchmarks

```bash
# Test minimal functionality
python test_minimal.py

# Run head dimension benchmark
python bench_headdim.py

# Run KV cache length benchmark
python bench_kvlen.py

# Run JIT overhead analysis
python bench_jit_overhead.py
```

Each benchmark script generates a CSV file with results.

## Benchmark Details

### test_minimal.py
Simple correctness test verifying FlashInfer's single-decode attention kernel:
- Query: (32, 128) fp16 tensor
- Key/Value: (2048, 32, 128) fp16 tensors
- Validates output shape and dtype

### bench_headdim.py
Measures latency impact of varying attention head dimensions:
- **Fixed parameters**: 32 heads, 2048 KV length
- **Variable**: head_dim ∈ {64, 128, 256}
- **Output**: `headdim_results.csv`

### bench_kvlen.py
Analyzes KV cache length scaling:
- **Fixed parameters**: 32 heads, 128 head_dim
- **Variable**: kv_len ∈ {512, 1024, 2048, 4096, 8192}
- **Output**: `kvlen_results.csv`

### bench_jit_overhead.py
Quantifies JIT compilation overhead:
- First call (includes JIT compilation) vs. steady-state performance
- Uses same dimensions: 32 heads, 128 head_dim, 2048 KV length
- **Output**: `jit_overhead_results.csv`

## Benchmark Configuration

All benchmarks use:
- **Data type**: float16 (fp16)
- **Warmup iterations**: 10
- **Measurement iterations**: 50
- **GPU synchronization**: Enabled for accurate timing

## Results

See [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) for detailed analysis and performance characteristics.

### Environment

Tests were conducted on:
- GPU Architecture: NVIDIA H100/A100 class
- CUDA Version: 12.6.0
- PyTorch: 2.6.0

## Project Structure

```
.
├── test_minimal.py              # Functionality test
├── bench_headdim.py            # Head dimension benchmark
├── bench_kvlen.py              # KV cache length benchmark
├── bench_jit_overhead.py       # JIT compilation overhead
├── setup_flashinfer_env.sh     # Environment setup script
├── headdim_results.csv         # Head dimension results
├── kvlen_results.csv           # KV length results
├── jit_overhead_results.csv    # JIT overhead results
├── BENCHMARK_REPORT.md         # Detailed analysis
├── README.md                   # This file
└── src/flashinfer/            # FlashInfer library source

Cache Directories (auto-created, in .gitignore):
├── pip-cache/        # pip package cache
├── torch-cache/      # PyTorch model cache
├── triton-cache/     # Triton JIT cache
├── cuda-cache/       # CUDA compilation cache
└── tmp/              # Temporary files
```

## Performance Analysis

FlashInfer's `single_decode_with_kv_cache` kernel shows:

1. **Minimal JIT overhead**: ~2200x first-call latency vs steady-state (indicates one-time compilation cost)
2. **Efficient scaling**: Sub-linear latency increase with KV cache length
3. **Head dimension impact**: Linear relationship between head_dim and latency
4. **Steady-state consistency**: ~5-10% variance in repeated calls

For complete analysis, see [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).

## References

- [FlashInfer GitHub Repository](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer Documentation](https://docs.flashinfer.ai)
- [FlashAttention Papers](https://github.com/Dao-AILab/flash-attention)

## Author

Colin6618 - Performance Analysis

