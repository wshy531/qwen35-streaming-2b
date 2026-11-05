#!/bin/bash
#SBATCH --job-name=2b-readability
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
PROJECT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$PROJECT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface" TMPDIR="$PROJECT/cache/tmp"
"$PROJECT/envs/qwen35-2b-fp32/bin/python" "$PROJECT/experiments/2b-memory-readability-v1/evaluate.py" \
 --run "$PROJECT/runs/2b-readability-inputs-20261020-01" \
 --out "$PROJECT/runs/2b-readability-${SLURM_JOB_ID}"
