#!/bin/bash
#SBATCH --job-name=2b-perception-interference
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/2b-perception-interference-%j.slurm.log
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -f 16384
PY="$ROOT/envs/qwen35-2b-fp32/bin/python"
OUT="$ROOT/runs/2b-perception-interference-$SLURM_JOB_ID"
PREP="$ROOT/runs/perception_interference_prepare_20261027_01"
"$PY" experiments/2b-perception-interference-v1/evaluate.py --out "$OUT" --lock "$PREP/execution_lock.json" --media-root /home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos
"$PY" experiments/2b-perception-interference-v1/analyze.py --run "$OUT"
