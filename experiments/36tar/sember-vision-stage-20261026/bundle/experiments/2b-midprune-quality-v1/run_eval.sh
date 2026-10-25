#!/bin/bash
#SBATCH --job-name=2b-midprune-quality
#SBATCH --partition=g078t
#SBATCH --comment=gx_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
cd /home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache/xdg"
export TMPDIR="$PWD/cache/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec envs/qwen35-2b-fp32/bin/python experiments/2b-midprune-quality-v1/launch.py
