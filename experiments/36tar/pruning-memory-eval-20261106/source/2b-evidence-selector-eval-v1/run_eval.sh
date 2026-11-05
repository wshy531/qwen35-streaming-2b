#!/bin/bash
#SBATCH --job-name=selector-qa-fixed
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/selector-qa-%j.slurm.log
set -euo pipefail
ROOT=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache"
ulimit -f 65536
"$ROOT/envs/evidence-selector-eval-v1/bin/python" "$ROOT/experiments/2b-evidence-selector-eval-v1/selector_qa_eval.py" --out "$ROOT/runs/selector-qa-$SLURM_JOB_ID"
