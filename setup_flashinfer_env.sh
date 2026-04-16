#!/usr/bin/env bash
source /etc/profile

module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.6.0-CUDA-12.6.0

export BASE=/scratch/$USER/flashinfer_demo
source $BASE/venv312sys/bin/activate
export PYTHONPATH=/scratch/$USER/flashinfer_demo/src/flashinfer:$PYTHONPATH


export TMPDIR=$BASE/tmp
export PIP_CACHE_DIR=$BASE/pip-cache
export TORCH_HOME=$BASE/torch-cache
export XDG_CACHE_HOME=$BASE/.cache
export CUDA_CACHE_PATH=$BASE/cuda-cache
export TRITON_CACHE_DIR=$BASE/triton-cache
export FLASHINFER_EXPERIMENTAL_VOLTA=1

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
