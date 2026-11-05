#!/bin/bash
#SBATCH --job-name=2b-temporal-router
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/2b-temporal-router-%j.slurm.log
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache" TMPDIR="$ROOT/cache/tmp"
export TORCH_HOME="$ROOT/cache/torch" CUDA_CACHE_PATH="$ROOT/cache/cuda"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
ulimit -f 16384
PY="$ROOT/envs/temporal-router-v1/bin/python"
OUT="$ROOT/runs/2b-temporal-router-$SLURM_JOB_ID"
"$PY" experiments/2b-temporal-router-v1/router_worker.py --out "$OUT" --lock "$ROOT/runs/temporal_router_prepare_20261027_01/execution_lock.json"
"$PY" experiments/2b-temporal-router-v1/route_analysis.py --run "$OUT" --output "$OUT/analysis.json"
