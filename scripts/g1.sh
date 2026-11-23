#!/usr/bin/env bash
# Gate G1 (headroom) + stores for G2 (coverage) on telecom/test, 8 samples per task.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
M=${XPER_HOME}/models/hf
OUT=results/g1; mkdir -p "$OUT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

run() {  # name model extra-args...
  local name=$1 model=$2; shift 2
  local GPU
  GPU=$(scripts/wait_gpus.sh 1) || exit 1
  echo "$(date +%H:%M) $name on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/rollout.py --model "$model" --model-id "$name" \
    --out "$OUT/$name.db" --gpu-util 0.8 --env-workers 32 "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  grep '^{"' "$OUT/$name.log" | tail -2
  [ $rc -eq 0 ] || { echo "$name FAILED (rc=$rc)"; tail -6 "$OUT/$name.log"; }
  return $rc
}

run qwen3-4b-inst   "$M/Qwen3-4B-Instruct-2507" --n 8
run qwen3-8b-nothink "$M/Qwen3-8B"             --n 8 --no-think
echo "=== coverage (G2 preview) ==="
$PY scripts/coverage.py --src "$OUT/qwen3-4b-inst.db" --tgt "$OUT/qwen3-8b-nothink.db"
echo "=== self-coverage control (4b-inst vs 4b-inst) ==="
$PY scripts/coverage.py --src "$OUT/qwen3-4b-inst.db" --tgt "$OUT/qwen3-4b-inst.db"
