#!/bin/bash
#SBATCH --job-name=2b-read-benefit
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=runs/2b-read-benefit-%j.slurm.log
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
test -n "${SLURM_JOB_ID:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache" TMPDIR="$ROOT/cache/tmp"
export TORCH_HOME="$ROOT/cache/torch" CUDA_CACHE_PATH="$ROOT/cache/cuda"
export RECENT_EVIDENCE_MEDIA_ROOT=/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
ulimit -f 16384
PY="$ROOT/envs/read-benefit-v1/bin/python"
CODE=experiments/2b-read-benefit-v1
LOCK=runs/read_benefit_execution_prepare_20261028_01/execution_lock.json
"$PY" "$CODE/freeze_features.py" --verify "$LOCK"
"$PY" "$CODE/feature_worker.py" --output "$ROOT/runs/2b-read-benefit-$SLURM_JOB_ID" --lock "$LOCK"
