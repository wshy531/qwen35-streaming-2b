#!/usr/bin/env bash
# Submit once only after this specific run has been approved.
#SBATCH --job-name=sember102
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=g078t
#SBATCH --gres=gpu:1
#SBATCH --comment=gx_llm
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/home/share/czy/wsy/logs/sember102_%j.out
#SBATCH --error=/home/share/czy/wsy/logs/sember102_%j.err
set -euo pipefail
base=/home/share/czy/wsy
: "${SLURM_JOB_ID:?approved Slurm allocation required}"
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 && "${SLURM_RESTART_COUNT:-0}" == 0 ]] || exit 1
export QWEN_REPO="$base/projects/qwen35-streaming-2b"
export QWEN_MODEL="$base/models/Qwen3.5-2B"
export QWEN_RUN="$base/runs/sember-102-eval-20261024-01"
export QWEN_PYTHON="$base/envs/qwen35-2b-fp32-sember/bin/python"
[[ "$(realpath -e "$QWEN_REPO")" == "$base/projects/qwen35-streaming-2b" ]] || exit 1
[[ "$(realpath -e "$QWEN_MODEL")" == "$base/models/Qwen3.5-2B" ]] || exit 1
[[ "$(realpath -e "$base/envs/qwen35-2b-fp32-sember")" == "$base/envs/qwen35-2b-fp32-sember" ]] || exit 1
[[ ! -e "$QWEN_RUN" && ! -e "$base/runs/sember-102-baselines-20261024-01/sampling" ]] || exit 1
export QWEN_WRITE_ROOT="$base" QWEN_CACHE_ROOT="$base/cache/sember-102-2b"
export TMPDIR="$base/cache/tmp" XDG_CACHE_HOME="$base/cache/xdg" HF_HOME="$base/cache/hf"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
assigned_gpu=${CUDA_VISIBLE_DEVICES:?Slurm-assigned GPU required}
cd "$QWEN_REPO"
export CUDA_VISIBLE_DEVICES=''
"$QWEN_PYTHON" -B scripts/run_2b_sember_102.py --stage media --out "$QWEN_RUN"
"$QWEN_PYTHON" -B scripts/run_2b_sember_102.py --stage prepare --out "$QWEN_RUN" --model-dir "$QWEN_MODEL"
"$QWEN_PYTHON" -B scripts/run_2b_sember_102.py --stage cpu --out "$QWEN_RUN"
export CUDA_VISIBLE_DEVICES="$assigned_gpu"
"$QWEN_PYTHON" -B scripts/run_2b_sember_102.py --stage gpu --out "$QWEN_RUN"
"$QWEN_PYTHON" -B scripts/run_2b_sember_102.py --stage score --out "$QWEN_RUN"
