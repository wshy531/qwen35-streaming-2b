#!/bin/sh
set -eu
cluster_project=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
test "$(realpath "$cluster_project")" = /home/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
test "$(id -un)" = u2025010462
test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
cd "$cluster_project"
export HF_HOME="$cluster_project/cache/huggingface" XDG_CACHE_HOME="$cluster_project/cache/xdg" TORCH_HOME="$cluster_project/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$cluster_project/cache/torchinductor" TRITON_CACHE_DIR="$cluster_project/cache/triton" TMPDIR="$cluster_project/cache/tmp" CUDA_CACHE_PATH="$cluster_project/cache/cuda"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cluster_code="$cluster_project/experiments/2b-natural-eval-v1"
cluster_python="$cluster_project/envs/qwen35-2b-fp32/bin/python"
cluster_run="$cluster_project/runs/2b-natural-eval-inputs-20261020-01"
cluster_output="$cluster_project/runs/2b-natural-eval-${SLURM_JOB_ID}"
mkdir "$cluster_output"
printf '%s\n' "$SLURM_JOB_ID" > "$cluster_output/job_id.txt"
hostname > "$cluster_output/hostname.txt"
timeout 1800 "$cluster_python" "$cluster_code/evaluate.py" worker native --run "$cluster_run" --out "$cluster_output/native" > "$cluster_output/native.log" 2>&1
timeout 1800 "$cluster_python" "$cluster_code/evaluate.py" worker memory --run "$cluster_run" --out "$cluster_output/memory" > "$cluster_output/memory.log" 2>&1
