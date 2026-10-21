#!/usr/bin/env bash
# Submit once with 1 GPU, 4 CPU, 24 GiB, 45 minutes, --no-requeue.
set -euo pipefail
: "${QWEN_REPO:?set QWEN_REPO}"
: "${QWEN_RUN:?set the CPU-preflight run}"
: "${QWEN_PYTHON:?set the same environment python}"
: "${SLURM_JOB_ID:?allocated Slurm job required}"
: "${CUDA_VISIBLE_DEVICES:?use Slurm-assigned GPU; do not unset or override it}"
cd "$QWEN_REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$QWEN_PYTHON" -B scripts/run_2b_memory_eval.py --stage gpu --out "$QWEN_RUN" \
  --preflight "$QWEN_RUN/cpu"
"$QWEN_PYTHON" -B scripts/run_2b_memory_eval.py --stage score --out "$QWEN_RUN"
