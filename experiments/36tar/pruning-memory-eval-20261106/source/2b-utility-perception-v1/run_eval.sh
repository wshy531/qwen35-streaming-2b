#!/bin/bash
#SBATCH --job-name=2b-utility-perception
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/utility-perception-%j.out
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ulimit -f 65536
envs/delayed-readiness-v1/bin/python experiments/2b-utility-perception-v1/evaluate.py --out "runs/utility-perception-$SLURM_JOB_ID"
