#!/usr/bin/env bash
# CPU execution, including an explicitly approved GPU-node reservation.
set -euo pipefail
: "${SLURM_JOB_ID:?run inside the approved CPU allocation}"
[[ "${SLURM_CPUS_PER_TASK:-0}" == 4 ]] || { echo 'exactly four CPUs required' >&2; exit 1; }
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || { echo 'restarts are forbidden' >&2; exit 1; }
reservation="${QWEN_CPU_ALLOW_GPU_RESERVATION:-0}"
[[ "$reservation" == 0 || "$reservation" == 1 ]] || { echo 'invalid reservation flag' >&2; exit 1; }
if [[ -n "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" && "$reservation" != 1 ]]; then
  echo 'GPU reservation requires explicit preflight authorization' >&2
  exit 1
fi
: "${QWEN_REPO:?set QWEN_REPO}"
: "${QWEN_MODEL:?set QWEN_MODEL}"
: "${QWEN_DATASET:?set QWEN_DATASET}"
: "${QWEN_RUN:?set a new QWEN_RUN}"
: "${QWEN_PYTHON:?set the existing environment python}"
cd "$QWEN_REPO"
export CUDA_VISIBLE_DEVICES=''
printf 'CPU preflight: job=%s reserved_GPUs=%s CUDA_VISIBLE_DEVICES=empty pretrained_loads=0\n' \
  "$SLURM_JOB_ID" "${SLURM_JOB_GPUS:-none}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
comparison_args=()
case "${QWEN_EVAL_COMPARISON:-0}" in
  0) ;;
  1) comparison_args=(--comparison) ;;
  *) echo 'invalid comparison mode' >&2; exit 1 ;;
esac
"$QWEN_PYTHON" -B scripts/run_2b_memory_eval.py --stage prepare \
  --model-dir "$QWEN_MODEL" --dataset-root "$QWEN_DATASET" --out "$QWEN_RUN" "${comparison_args[@]}"
# The runner validates QWEN_RUN before producing any files. Inherited regressions
# are part of preflight and their receipt is mandatory at the GPU gate.
"$QWEN_PYTHON" -B scripts/run_2b_memory_eval.py --stage preflight --out "$QWEN_RUN"
# The offline audit is a separate suite; it does not change the 28-contract
# proof consumed by the 171-question GPU replay gate.
"$QWEN_PYTHON" -B -m unittest discover -s experiments/wshy/long_stream_audit/tests -v \
  > "$QWEN_RUN/cpu/long_stream_audit_contracts.txt" 2>&1
"$QWEN_PYTHON" -B -c 'import os; from pathlib import Path; from experiments.wshy.online_memory_2b.receipts import successful_counts; successful_counts((Path(os.environ["QWEN_RUN"])/"cpu/long_stream_audit_contracts.txt").read_text(encoding="utf-8"), [8])'
