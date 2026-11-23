#!/usr/bin/env bash
# Do SFT transfer and ICL stack? Evaluate the B2-seeds transfer checkpoints with the same k=6
# ICL block (seed 0) used for the 8B-Instruct ICL baseline.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
O=${XPER_HOME}/models/xper
read -ra G <<< "${GPUS:-6 7}"
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
seeds=(401 402 403); i=0
while [ $i -lt ${#seeds[@]} ]; do
  pids=(); names=()
  for g in "${G[@]}"; do
    [ $i -lt ${#seeds[@]} ] || break
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$mem" -lt 1000 ] || die "GPU $g already has ${mem}MiB in use"
    m=8b-instruct-xfer-s${seeds[$i]}-icl6
    CUDA_VISIBLE_DEVICES=$g $PY scripts/rollout.py --chatml --no-think \
      --model $O/8b-instruct-xfer-s${seeds[$i]}/final --model-id $m --n 8 \
      --icl-src 'results/src/4b-inst.shard*.db' --icl-k 6 --icl-seed 0 \
      --out results/eval/$m.db > results/eval/$m.log 2>&1 &
    pids+=($!); names+=($m); i=$((i + 1))
  done
  for j in "${!pids[@]}"; do wait "${pids[$j]}" || die "${names[$j]} failed"; done
  sleep 10
done
for s in "${seeds[@]}"; do grep -h '^{"model' results/eval/8b-instruct-xfer-s$s-icl6.log; done
echo "=== $(date +%H:%M) done"
