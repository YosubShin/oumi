#!/bin/bash
#SBATCH --job-name=oumi
#SBATCH --partition=gpuA100x4-preempt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH -A bchy-delta-gpu

# Partitions:
# gpuA100x4-preempt (Charge Factor: 0.5, 16 cores, 250GB host memory)
# gpuA40x4-preempt (Charge Factor: 0.25, 16 cores, 62.5GB host memory)
# gpuA40x4 (Charge Factor: 0.5, 16 cores, 62.5GB host memory)
# gpuH200x8 (Charge Factor: 3.0, 12 cores, 250GB host memory)


set -euo pipefail

module purge >/dev/null 2>&1 || true

echo "Node: $(hostname)"

# # 1. Create a big scratch directory on node-local SSD
# export TMPDIR=/tmp/$USER-${SLURM_JOB_ID}
# mkdir -p $TMPDIR
# echo "Using TMPDIR = $TMPDIR"

# # Optional but recommended:
# export UV_CACHE_DIR=$TMPDIR/uv-cache
# export PIP_CACHE_DIR=$TMPDIR/pip-cache
# mkdir -p $UV_CACHE_DIR $PIP_CACHE_DIR

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"
export RESULTS_DIR
export REPO_DIR

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Writing outputs to ${RESULTS_DIR}"

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "KOA_PROJECT_ROOT=${KOA_PROJECT_ROOT:-unset}"
echo "KOA_ML_CODE_ROOT=${KOA_ML_CODE_ROOT:-unset}"
echo "KOA_RUN_DIR=${KOA_RUN_DIR:-unset}"
echo "KOA_RUN_METADATA_DIR=${KOA_RUN_METADATA_DIR:-unset}"
echo "KOA_SHARED_ENV=${KOA_SHARED_ENV:-unset}"
echo

echo "==== GPU Info ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo

echo "==== GPU Memory Detection ===="
# Query first GPU's name and total VRAM (in MB)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)

echo "Detected GPU: ${GPU_NAME} with ${GPU_MEM} MB VRAM"

# echo "==== Python Environment ====="
# if command -v python >/dev/null 2>&1; then
#   which python
#   python --version
# else
#   echo "python not found"
# fi
# echo

# Load necessary modules
# We need to load these modules to build flash-attn from source
# module load gcc/11.4.0
# module load ninja

# bash scripts/tmp_health.sh

# source "scripts/setup_env_delta.sh"

# srun --mpi=none uv oumi train -c configs/projects/dcvlr/starter_kit/qwenvl/qwenvl-oumi-walton-exclude-geometry-biology.yaml --training.output_dir "${RESULTS_DIR}/checkpoints"

apptainer exec --nv \
--bind /work/hdd/bchy/shin1:/work/hdd/bchy/shin1 \
/sw/external/NGC/pytorch_25.08-py3.sif \
bash -c "
    export PATH=\"\$HOME/.local/bin:\$PATH\"

    echo Inside container: \$HOSTNAME
    echo RESULTS_DIR=\$RESULTS_DIR
    echo REPO_DIR=\$REPO_DIR

    pip install --no-deps oumi
pip install \
  accelerate \
  aiofiles \
  anthropic \
  anyio \
  attrs \
  boto3 \
  cachetools \
  click \
  datasets \
  dill \
  docker \
  fsspec \
  huggingface_hub \
  jmespath \
  jsonschema \
  loguru \
  markupsafe \
  multiprocess \
  numexpr \
  numpy \
  omegaconf \
  packaging \
  pandas \
  pillow \
  pydantic \
  peft \
  python-dotenv \
  pyyaml \
  regex \
  requests \
  rich \
  rsa \
  s3transfer \
  safetensors \
  scipy \
  setproctitle \
  tiktoken \
  tokenizers \
  tomli \
  tqdm \
  transformers \
  trl \
  typer \
  urllib3 \
  wandb \
  jsonlines \
  mlflow

    # Run the actual training command
    oumi train \
        -c configs/projects/dcvlr/starter_kit/qwenvl/qwenvl-oumi-walton-exclude-geometry-biology.yaml \
        --training.output_dir \"\$RESULTS_DIR/checkpoints\"
"