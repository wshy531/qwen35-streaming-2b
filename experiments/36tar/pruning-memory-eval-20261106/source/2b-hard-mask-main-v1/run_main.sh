#!/usr/bin/env bash
#SBATCH --job-name=q35-hard-mask-main
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/hard-mask-main-%j.out
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
test -n "${SLURM_JOB_ID:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ulimit -f 131072
timeout --signal=TERM 179m "$ROOT/envs/hard-mask-main-v1/bin/python" experiments/2b-hard-mask-main-v1/main_train.py --out "$ROOT/runs/hard-mask-main-$SLURM_JOB_ID"
