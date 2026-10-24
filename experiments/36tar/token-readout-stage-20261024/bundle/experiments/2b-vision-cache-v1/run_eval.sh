#!/bin/bash
#SBATCH --job-name=2b-vision-cache-v1
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
PROJECT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$PROJECT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface" TMPDIR="$PROJECT/cache/tmp"
timeout 2450 "$PROJECT/envs/qwen35-2b-fp32/bin/python" experiments/2b-vision-cache-v1/evaluate.py \
  --run "$PROJECT/runs/2b_vision_cache_prepare_20261024_01/inputs" \
  --out "$PROJECT/runs/2b-vision-cache-$SLURM_JOB_ID"
