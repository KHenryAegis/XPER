#!/bin/bash
# Re-eval B2 seed models with --no-think (matching how they were trained)
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -u
cd "$(dirname "$0")/.."
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
O=${XPER_HOME}/models/xper
START=${XPER_HOME}/models/hf/Qwen3-8B
G=(2 3 4 5 6 7)
pids=()
n=0

# Start model
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/rollout.py --chatml --no-think --model $START \
  --model-id 8b-instruct-start-nothink --n 8 --out results/eval/8b-instruct-start-nothink.db \
  > results/eval/8b-instruct-start-nothink.log 2>&1 &
pids+=($!)
n=1

# 6 seed models
for s in 401 402 403; do
  for arm in xfer selfexp; do
    gpu=${G[$((n % 6))]}
    m=8b-instruct-$arm-s$s
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/rollout.py --chatml --no-think --model $O/$m/final \
      --model-id $m-nothink --n 8 --out results/eval/$m-nothink.db > results/eval/$m-nothink.log 2>&1 &
    pids+=($!)
    n=$((n + 1))
  done
done

for p in "${pids[@]}"; do wait "$p" || exit 1; done
grep '^{"model' results/eval/8b-instruct-start-nothink.log
for s in 401 402 403; do
  for arm in xfer selfexp; do
    grep '^{"model' results/eval/8b-instruct-$arm-s$s-nothink.log
  done
done
