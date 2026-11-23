#!/usr/bin/env bash
# Math env: source (4B-Instruct) explores the train split once -> the experience store the target
# may reuse. Its env calls are the budget the target's self-exploration is matched against.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
H=${XPER_HOME}/models/hf
read -ra G <<< "${GPUS:-0 1}"
NG=${#G[@]}
N=${N:-2000}
R=results/math/src
mkdir -p $R
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
for g in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g busy (${mem}MiB)"
done
echo "=== $(date +%H:%M) math source: 4B-Instruct on $N train problems, n=1"
pids=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY scripts/rollout.py --env math --chatml --model $H/Qwen3-4B-Instruct-2507 \
    --model-id math-src-4b-inst --split train --stratify $N --shard $i/$NG --n 1 --seed 700 \
    --max-steps 12 --max-tokens 2048 --env-workers 48 \
    --out $R/4b-inst.shard$i.db > $R/4b-inst.shard$i.log 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || die "shard $i failed, see $R/4b-inst.shard$i.log"; done
$PY scripts/count_success.py "$R/4b-inst.shard*.db" | tee $R/stats.json
echo "=== $(date +%H:%M) done"
