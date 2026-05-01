#!/usr/bin/env bash
# Run the primary FlashInfer workflow benchmark.
# Usage: source setup_flashinfer_env.sh && bash run_all_benchmarks.sh

set -e
mkdir -p logs results
export BASE="${BASE:-/scratch/$USER/flashinfer_demo}"

export FLASHINFER_WORKSPACE_BASE="$BASE"
export FLASHINFER_OBS_ENABLE="${FLASHINFER_OBS_ENABLE:-1}"
export FLASHINFER_OBS_SAMPLE_RATE="${FLASHINFER_OBS_SAMPLE_RATE:-1.0}"
export FLASHINFER_OBS_FORMAT="${FLASHINFER_OBS_FORMAT:-jsonl}"
export FLASHINFER_OBS_OUTPUT="logs/obs_$(date +%Y%m%d_%H%M%S)_workflow.jsonl"
echo "FlashInfer workspace base: $FLASHINFER_WORKSPACE_BASE"
echo "FlashInfer observability log: $FLASHINFER_OBS_OUTPUT"

echo "[1/1] Running bench_workflow.py ..."
python -u bench_workflow.py

echo "Workflow benchmark finished. Logs: $FLASHINFER_OBS_OUTPUT"
