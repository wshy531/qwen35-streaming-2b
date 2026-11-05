#!/bin/bash
#SBATCH --job-name=2b-sember-profile
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/cache/huggingface" TMPDIR="$PROJECT/cache/tmp"
export SEMBER_MEDIA_ROOT=/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23
PYTHON="$PROJECT/envs/qwen35-2b-fp32/bin/python"
RESULT="$PROJECT/runs/2b-sember-profile-$SLURM_JOB_ID"
mkdir "$RESULT"
"$PYTHON" experiments/2b-sember-v1/evaluate_profile.py --mode bounded --out "$RESULT/bounded"
"$PYTHON" experiments/2b-sember-v1/evaluate_profile.py --mode reference --out "$RESULT/reference"
