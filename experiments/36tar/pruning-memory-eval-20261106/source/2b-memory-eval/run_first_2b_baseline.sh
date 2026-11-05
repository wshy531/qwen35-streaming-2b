#!/bin/sh
# Run only inside a separately approved Slurm allocation. This file never allocates GPUs.
set -eu
cluster_project=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
test "$(realpath "$cluster_project")" = /home/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
test "$(id -un)" = u2025010462
test -n "${SLURM_JOB_ID:-}"
test -n "${CUDA_VISIBLE_DEVICES:-}"
cd "$cluster_project"
export HF_HOME="$cluster_project/cache/huggingface" XDG_CACHE_HOME="$cluster_project/cache/xdg" TORCH_HOME="$cluster_project/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$cluster_project/cache/torchinductor" TRITON_CACHE_DIR="$cluster_project/cache/triton" TMPDIR="$cluster_project/cache/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cluster_code="$cluster_project/repo/experiments/36tar/2b-memory-eval"
cluster_python="$cluster_project/envs/qwen35-2b-fp32/bin/python"
cluster_proof="$cluster_project/runs/first-2b-preflight-20261016-02"
cluster_output="$cluster_project/runs/first-2b-gpu-${SLURM_JOB_ID}"
mkdir "$cluster_output"
printf '%s\n' "$SLURM_JOB_ID" > "$cluster_output/job_id.txt"
hostname > "$cluster_output/hostname.txt"
timeout 420 "$cluster_python" "$cluster_code/first_2b_baseline.py" --stage gpu --implementation original \
  --repo "$cluster_project/repo" --model-dir "$cluster_project/models/Qwen3.5-2B" \
  --media "$cluster_project/data/ovo-pilot-dev4" --preflight "$cluster_proof/original" \
  --out "$cluster_output/original" > "$cluster_output/original.log" 2>&1
timeout 420 "$cluster_python" "$cluster_code/first_2b_baseline.py" --stage gpu --implementation stream \
  --repo "$cluster_project/repo" --model-dir "$cluster_project/models/Qwen3.5-2B" \
  --media "$cluster_project/data/ovo-pilot-dev4" --preflight "$cluster_proof/stream" \
  --out "$cluster_output/stream" > "$cluster_output/stream.log" 2>&1
timeout 30 "$cluster_python" "$cluster_code/analyze_first_baseline.py" --original "$cluster_output/original" \
  --stream "$cluster_output/stream" --out "$cluster_output/comparison" > "$cluster_output/comparison.log" 2>&1
