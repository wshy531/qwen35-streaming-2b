#!/usr/bin/env bash
#SBATCH --job-name=q35-evidence-scale-N
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/evidence-scale-nextqa-v2-%j.out
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
test -n "${SLURM_JOB_ID:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ulimit -f 65536
timeout --signal=TERM 59m "$ROOT/envs/evidence-scale-v1/bin/python" experiments/2b-evidence-scale-v1/worker_v2.py --dataset NExT-QA --minutes 60 --out "$ROOT/runs/evidence-scale-nextqa-v2-$SLURM_JOB_ID"
