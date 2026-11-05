#!/usr/bin/env bash
#SBATCH --job-name=q35-memory-prune-panel150
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/memory-prune-panel150-%j.out
set -euo pipefail
cd /home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
test -n "${SLURM_JOB_ID:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
ulimit -f 65536
exec timeout --signal=TERM 359m envs/memory-prune-panel150-v1/bin/python experiments/2b-memory-prune-panel150-v1/worker.py --out "runs/memory-prune-panel150-$SLURM_JOB_ID"
