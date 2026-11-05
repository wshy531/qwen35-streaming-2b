#!/usr/bin/env bash
#SBATCH --job-name=q35-utility-train
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/history-utility-main-%j.out
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ulimit -f 131072
PYTHON="$ROOT/envs/delayed-readiness-v1/bin/python"
TRAIN="$ROOT/runs/history-utility-train-$SLURM_JOB_ID"
DEV="$ROOT/runs/history-utility-dev-$SLURM_JOB_ID"
timeout --signal=TERM 138m "$PYTHON" experiments/2b-history-utility-main-v1/score_panel.py --split remaining_train --out "$TRAIN"
timeout --signal=TERM 68m "$PYTHON" experiments/2b-history-utility-main-v1/score_panel.py --split development --out "$DEV"
CUDA_VISIBLE_DEVICES=-1 timeout --signal=TERM 10m "$PYTHON" experiments/2b-history-utility-v1/fit.py --train-run runs/history-utility-220503 --train-run "$TRAIN" --dev-run "$DEV" --out "$ROOT/runs/history-utility-fit-$SLURM_JOB_ID"
