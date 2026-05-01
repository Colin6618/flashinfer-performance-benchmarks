#!/usr/bin/env bash
source /etc/profile

module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.6.0-CUDA-12.6.0

export BASE=/scratch/$USER/flashinfer_demo

source $BASE/venv312sys/bin/activate
export PYTHONPATH=$BASE/src/flashinfer:$PYTHONPATH

# 强制所有 cache 路径指向 /scratch/$USER/flashinfer_demo 下，避免 home quota 爆满
export TMPDIR=$BASE/tmp
export PIP_CACHE_DIR=$BASE/pip-cache
export TORCH_HOME=$BASE/torch-cache
export XDG_CACHE_HOME=$BASE/.cache
export CUDA_CACHE_PATH=$BASE/cuda-cache
export TRITON_CACHE_DIR=$BASE/triton-cache
export FLASHINFER_CACHE_DIR=$BASE/flashinfer-cache
export FLASHINFER_WORKSPACE_BASE=$BASE
export MPLCONFIGDIR=$BASE/matplotlib-cache

# 清理 home 目录下的 flashinfer 和 pip cache，防止 quota 爆满
rm -rf ~/.cache/flashinfer/*
rm -rf ~/.cache/pip/*


export TMPDIR=$BASE/tmp
export PIP_CACHE_DIR=$BASE/pip-cache
export TORCH_HOME=$BASE/torch-cache
export XDG_CACHE_HOME=$BASE/.cache
export CUDA_CACHE_PATH=$BASE/cuda-cache
export TRITON_CACHE_DIR=$BASE/triton-cache
export FLASHINFER_WORKSPACE_BASE=$BASE
export MPLCONFIGDIR=$BASE/matplotlib-cache
export FLASHINFER_EXPERIMENTAL_VOLTA=1

# FlashInfer observability: enabled by default for this profiling project.
export FLASHINFER_OBS_ENABLE=1
export FLASHINFER_OBS_SAMPLE_RATE=1.0
export FLASHINFER_OBS_OUTPUT=${FLASHINFER_OBS_OUTPUT:-$BASE/logs/obs.jsonl}
export FLASHINFER_OBS_FORMAT=jsonl
mkdir -p "$(dirname "$FLASHINFER_OBS_OUTPUT")"
mkdir -p "$MPLCONFIGDIR"

if [[ "$FLASHINFER_OBS_ENABLE" == "1" ]]; then
    echo "FlashInfer observability: enabled"
    echo "  sample_rate=$FLASHINFER_OBS_SAMPLE_RATE"
    echo "  output=$FLASHINFER_OBS_OUTPUT"
    echo "  format=$FLASHINFER_OBS_FORMAT"
    echo "  workspace_base=$FLASHINFER_WORKSPACE_BASE"
else
    echo "FlashInfer observability: disabled"
fi

echo "Environment ready"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
