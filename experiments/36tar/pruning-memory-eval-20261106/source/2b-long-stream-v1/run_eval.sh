#!/bin/bash
#SBATCH --job-name=2b-long-stream
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
PROJECT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$PROJECT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface" TMPDIR="$PROJECT/cache/tmp"
PY="$PROJECT/envs/qwen35-2b-fp32/bin/python"
CODE="$PROJECT/experiments/2b-long-stream-v1/evaluate.py"
INPUT="$PROJECT/runs/2b-long-stream-inputs-20261020-01"
OUTPUT="$PROJECT/runs/2b-long-stream-${SLURM_JOB_ID}"
mkdir "$OUTPUT"
"$PY" "$CODE" --run "$INPUT" --stage native --out "$OUTPUT/native"
"$PY" "$CODE" --run "$INPUT" --stage memory --out "$OUTPUT/memory"
