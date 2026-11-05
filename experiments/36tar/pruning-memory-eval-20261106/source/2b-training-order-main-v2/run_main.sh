#!/usr/bin/env bash
#SBATCH --job-name=q35-order-main
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/training-order-main-%j.out
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
test -n "${SLURM_JOB_ID:-}"
case "${1:-}" in sequential|interleaved) ;; *) exit 2;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ulimit -f 131072
timeout --signal=TERM 89m "$ROOT/envs/delayed-readiness-v1/bin/python" \
 experiments/2b-training-order-main-v2/run_main.py --arm "$1" --out "$ROOT/runs/training-order-main-$SLURM_JOB_ID"
