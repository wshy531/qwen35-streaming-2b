#!/usr/bin/env bash
#SBATCH --job-name=2b-sember-pruned-ls
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/sember-pruned-ls-%j.out
set -euo pipefail
PROJECT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$PROJECT"
unset SEMBER_MEDIA_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface" XDG_CACHE_HOME="$PROJECT/cache" TMPDIR="$PROJECT/cache/tmp"
ulimit -f 65536
timeout --signal=TERM 179m envs/delayed-readiness-v1/bin/python experiments/2b-sember-pruned-ls-v1/run_eval.py --out "$PROJECT/runs/sember-pruned-ls-$SLURM_JOB_ID"
