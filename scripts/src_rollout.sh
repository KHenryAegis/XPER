#!/usr/bin/env bash
# Source experience on leak-free telecom train_full: 4B-Instruct, stratified tasks, 8 GPU shards.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
M=${XPER_HOME}/models/hf/Qwen3-4B-Instruct-2507
NT=${NT:-600}; N=${N:-4}
pids=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup $PY scripts/rollout.py --model $M --model-id qwen3-4b-inst \
    --chatml --split train_full --stratify $NT --shard $i/8 --n $N --env-workers 24 --gpu-util 0.8 \
    --out results/src/4b-inst.shard$i.db > results/src/4b-inst.shard$i.log 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || exit 1; done
grep -h '^{"model' results/src/4b-inst.shard*.log
