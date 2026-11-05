#!/bin/bash
#SBATCH --job-name=36tar-b02-cap4
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/2b-short-event-inputs-20261020-01/slurm-%j.log
set -euo pipefail
PROJECT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$PROJECT"
: "${SLURM_JOB_ID:?Slurm allocation required}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PROJECT/cache/huggingface" TMPDIR="$PROJECT/cache/tmp"
PY="$PROJECT/envs/qwen35-2b-fp32/bin/python"
RUN="$PROJECT/runs/2b-short-event-inputs-20261020-01"
OUT="$PROJECT/runs/2b-short-event-${SLURM_JOB_ID}"
test -f "$RUN/authorization.json"
mkdir "$OUT"
timeout --signal=TERM --kill-after=15 1530 "$PY" experiments/2b-short-event-v1/evaluate.py worker --run "$RUN" --stage diagnostic --out "$OUT/diagnostic"
timeout --signal=TERM --kill-after=15 1530 "$PY" experiments/2b-short-event-v1/evaluate.py worker --run "$RUN" --stage main --out "$OUT/main"
