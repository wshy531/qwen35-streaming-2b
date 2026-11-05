#!/usr/bin/env bash
#SBATCH --job-name=q35-hard-mask-profile
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/hard-mask-profile-%j.out
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
test -n "${SLURM_JOB_ID:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ulimit -f 65536
timeout --signal=TERM 13m "$ROOT/envs/hard-mask-prune-v1/bin/python" experiments/2b-hard-mask-prune-v1/gpu_profile.py --out "$ROOT/runs/hard-mask-profile-$SLURM_JOB_ID"
