#!/usr/bin/env bash
# ICL robustness on the strong start (8B-Instruct thinking-off): example draws (icl seed) and k.
# Usage: GPUS="6 7" bash scripts/run_icl_sweep.sh "6:1" "6:2" "3:0" "12:0"   (k:seed per run)
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
H=${XPER_HOME}/models/hf
read -ra G <<< "${GPUS:-6 7}"
runs=("$@")
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }

i=0
while [ $i -lt ${#runs[@]} ]; do
  pids=(); names=()
  for g in "${G[@]}"; do
    [ $i -lt ${#runs[@]} ] || break
    IFS=: read -r k s <<< "${runs[$i]}"
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$mem" -lt 1000 ] || die "GPU $g already has ${mem}MiB in use"
    m=8b-instruct-icl${k}-s$s
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rollout.py --chatml --no-think --model $H/Qwen3-8B \
      --model-id $m --n 8 --icl-src 'results/src/4b-inst.shard*.db' --icl-k $k --icl-seed $s \
      --out results/eval/$m.db > results/eval/$m.log 2>&1 &
    pids+=($!); names+=($m); i=$((i + 1))
  done
  for j in "${!pids[@]}"; do wait "${pids[$j]}" || die "${names[$j]} failed"; done
  sleep 10
done
for r in "${runs[@]}"; do
  IFS=: read -r k s <<< "$r"
  grep -h -E '^ICL|^\{"model' results/eval/8b-instruct-icl${k}-s$s.log
done
echo "=== $(date +%H:%M) done"
