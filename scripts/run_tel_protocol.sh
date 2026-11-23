#!/usr/bin/env bash
# Run the protocol controls on one GPU host. Keep runtime writes under $XPER_HOME or /tmp.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
KH=${XPER_HOME:-$(dirname "$ROOT")}
PY=${XPER_PYTHON:-$KH/miniconda3/envs/xper/bin/python}

if [[ -n ${XPER_HOST:-} && $(hostname -s) != "$XPER_HOST" ]]; then
  echo "Run this launcher on $XPER_HOST." >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
export XDG_CACHE_HOME=$KH/.cache XDG_CONFIG_HOME=$KH/.config
export HF_HOME=$KH/.cache/huggingface TORCH_HOME=$KH/.cache/torch
export VLLM_CACHE_ROOT=/tmp/xper-tel-protocol-cache/vllm VLLM_CONFIG_ROOT=$KH/.config/vllm
export TRITON_CACHE_DIR=/tmp/xper-tel-protocol-cache/triton CUDA_CACHE_PATH=/tmp/xper-tel-protocol-cache/cuda
export TORCHINDUCTOR_CACHE_DIR=/tmp/xper-tel-protocol-cache/inductor
export NUMBA_CACHE_DIR=$KH/.cache/numba MPLCONFIGDIR=$KH/.cache/matplotlib
export OUTLINES_CACHE_DIR=$KH/.cache/outlines
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Even a one-call episode spends >=128 calls/round; this cap cannot stop a
# 44,414-call run early. The shared script's default 40 rounds can do so.
export MAX_IT=400
mkdir -p logs/queue "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME"

# Only this new queue is locked; existing queues and other users are untouched.
exec 9>logs/queue/tel_protocol.74.lock
flock -n 9 || { echo 'Protocol queue is already locked.' >&2; exit 1; }
exec "$PY" -B scripts/gpu_queue.py \
  --jobs logs/queue/jobs_tel_protocol.jsonl \
  --state logs/queue/tel_protocol.74.state.json \
  --claim-dir logs/queue/claim_tel_protocol --host-label 74 "$@"
