#!/usr/bin/env bash
# Preparation only. Requires separate approval for the downloads below; no GPU jobs.
set -euo pipefail
project=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
physical_project=/home/share/czy/swt/my_pape/qwen35-streaming-2b-36tar
test "$(whoami)" = u2025010462
test "$(realpath "$project")" = "$physical_project"
test "${DOWNLOAD_APPROVED:-}" = yes || { echo 'Download approval is required.' >&2; exit 20; }
case "${BOOTSTRAP_STAGE:-all}" in install|all) ;; *) exit 25;; esac
test -n "${BOOTSTRAP_RUN:-}" || { echo 'Set a fresh BOOTSTRAP_RUN name.' >&2; exit 21; }
case "$BOOTSTRAP_RUN" in *[!a-zA-Z0-9_-]*|'') exit 22;; esac
run_dir="$project/runs/$BOOTSTRAP_RUN"
test ! -e "$run_dir"
mkdir "$run_dir"
export TMPDIR="$project/cache"
export PIP_CACHE_DIR="$project/cache/pip"
export HF_HOME="$project/cache/huggingface"
export XDG_CACHE_HOME="$project/cache/xdg"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
env_dir="$project/envs/qwen35-2b-fp32"
test ! -e "$env_dir" || { echo 'Environment exists; inspect before resuming.' >&2; exit 23; }
test "$(git -C "$project/repo" branch --show-current)" = 36tar/2b-memory-eval
test "$(git -C "$project/repo" rev-parse HEAD)" = e13ed8babfb8af525991be96c0d5f5369cc1f173
/usr/bin/python3.11 -m venv "$env_dir"
py="$env_dir/bin/python"
exec > >(tee "$run_dir/bootstrap.log") 2>&1
echo 'Approved preparation: install binary packages and verify transferred pinned model; no inference.'
timeout 120 "$py" -m pip install --only-binary=:all: --retries 1 --timeout 20 pip==25.3
timeout 1200 "$py" -m pip install --only-binary=:all: --retries 1 --timeout 20 \
  torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
timeout 600 "$py" -m pip install --only-binary=:all: --retries 1 --timeout 20 \
  -r "$project/repo/requirements-cluster.txt"
"$py" -m pip check
"$py" -m pip freeze > "$run_dir/packages.txt"
# Hugging Face is unreachable from this login node. Transfer verified assets first.
if test "${BOOTSTRAP_STAGE:-all}" = install; then
  echo 'Package installation complete. Transferred model verification is still required.'
  exit 0
fi
"$py" "$project/repo/scripts/prepare_model.py" \
  --model-dir "$project/models/Qwen3.5-2B" --verify-only
echo 'Preparation complete. CPU regression and GPU execution require their own preflight.'
