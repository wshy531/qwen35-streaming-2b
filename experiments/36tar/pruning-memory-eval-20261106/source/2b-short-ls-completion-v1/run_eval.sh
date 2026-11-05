#!/usr/bin/env bash
#SBATCH --job-name=q35-short-ls-fill
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/short-ls-fill-%j.out
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ulimit -f 65536
timeout --signal=TERM 59m envs/delayed-readiness-v1/bin/python experiments/2b-short-ls-completion-v1/run_eval.py --out "$ROOT/runs/short-ls-fill-$SLURM_JOB_ID"
