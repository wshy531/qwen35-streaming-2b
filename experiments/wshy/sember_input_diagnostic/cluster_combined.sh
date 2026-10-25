#!/usr/bin/env bash
# Approved 2026-10-24: one submission, no automatic retry.
#SBATCH --job-name=sember-input
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=g078t
#SBATCH --gres=gpu:1
#SBATCH --comment=gx_llm
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/share/czy/wsy/logs/sember_input_%j.out
#SBATCH --error=/home/share/czy/wsy/logs/sember_input_%j.err
set -euo pipefail
base=/home/share/czy/wsy
: "${SLURM_JOB_ID:?approved allocation required}"
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_RESTART_COUNT:-0}" == 0 ]] || exit 1
if [[ "${QWEN_INPUT_TIMED:-}" != 1 ]]; then
    export QWEN_INPUT_TIMED=1 QWEN_DIAGNOSTIC_STARTED="$(date +%s)"
    exec timeout --signal=TERM --kill-after=15s 14100s bash "$0"
fi
repo="$base/projects/qwen35-streaming-2b"
model="$base/models/Qwen3.5-2B"
environment="$base/envs/qwen35-2b-fp32-sember"
run="$base/runs/sember-input-diagnostic-20261024-01"
for path in "$repo" "$model" "$environment"; do
    [[ "$(realpath -e "$path")" == "$path" ]] || exit 1
done
[[ ! -e "$run" ]] || exit 1
export QWEN_WRITE_ROOT="$base" QWEN_CACHE_ROOT="$base/cache/sember-input-diagnostic"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
assigned_gpu=${CUDA_VISIBLE_DEVICES:?Slurm GPU required}
cd "$repo"
python="$environment/bin/python"
export CUDA_VISIBLE_DEVICES=''
"$python" -B scripts/run_2b_sember_input_diagnostic.py --stage plan --out "$run"
"$python" -B scripts/run_2b_sember_input_diagnostic.py --stage cpu --out "$run" --model-dir "$model"
export CUDA_VISIBLE_DEVICES="$assigned_gpu"
"$python" -B scripts/run_2b_sember_input_diagnostic.py --stage gpu --out "$run"
export CUDA_VISIBLE_DEVICES=''
"$python" -B scripts/run_2b_sember_input_diagnostic.py --stage score --out "$run"
