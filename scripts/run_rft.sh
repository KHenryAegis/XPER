#!/usr/bin/env bash
# Iterated rejection-sampling fine-tuning (RFT / STaR-style) self-exploration baseline.
# Each round: current policy samples 600 stratified train_full tasks (n=1, ~11.7k env calls),
# then SFT *from the start model* on all successes collected so far (no compounding drift),
# and that model samples the next round. Stops once the source's budget (44,414 calls) is spent.
#
# Usage: START=<hf dir> TAG=<name> THINK=--no-think|"" SEED=501 bash scripts/run_rft.sh
#   strong start: START=models/hf/Qwen3-8B TAG=rft-instruct THINK=--no-think
#   weak start:   START=models/xper/8b-commonsft/final TAG=rft-commonsft THINK=""
# Other envs: ENVARGS (sampling + eval, e.g. "--env math --max-steps 12 --max-tokens 2048
#   --env-workers 48"), SPLIT (train split name), EVALARGS (default "--n 8"), BUDGET.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
TR=$(dirname "$PY")/torchrun
O=${XPER_HOME}/models/xper
: "${START:?}" "${TAG:?}"; THINK=${THINK-}; SEED=${SEED:-501}
BUDGET=${BUDGET:-44414}; PORT0=${PORT0:-29700}; SPLIT=${SPLIT:-train_full}; STRAT=${STRAT:-600}
ENVARGS=${ENVARGS-}; EVALARGS=${EVALARGS:---n 8}
IFS=',' read -ra G <<< "${GPUS:-2,3,4,5,6,7}"
NG=${#G[@]}
T=$(IFS=,; echo "${G[*]:0:4}"); NT=$(( NG < 4 ? NG : 4 ))  # FSDP world size
R=results/$TAG.s$SEED
mkdir -p $R results/eval
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
for g in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g already has ${mem}MiB in use"
done

policy=$START; round=0; spent=0
echo "=== $(date +%H:%M) RFT $TAG seed $SEED from $START on GPUs ${G[*]}"
while [ "$spent" -lt "$BUDGET" ] && [ $round -lt ${MAX_ROUNDS:-8} ]; do
  pids=()
  for i in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES=${G[$i]} $PY scripts/rollout.py --chatml $THINK --model $policy \
      --model-id $TAG-r$round --split $SPLIT --stratify $STRAT --shard $i/$NG \
      --n 1 --seed $((SEED + round)) --env-workers 24 --gpu-util 0.8 $ENVARGS \
      --out $R/r$round.shard$i.db > $R/r$round.shard$i.log 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || die "round $round shard $i failed, see $R/r$round.shard$i.log"
  done
  stats=$($PY scripts/count_success.py "$R/r$round.shard*.db")
  spent=$($PY scripts/count_success.py "$R/r*.shard*.db" | $PY -c "import json,sys; print(json.load(sys.stdin)['env_calls'])")
  echo "  round $round: $(echo $stats | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['successes'], 'successes /', d['episodes'], 'episodes')"), cumulative env calls $spent"
  total=$($PY scripts/count_success.py "$R/r*.shard*.db" | $PY -c "import json,sys; print(json.load(sys.stdin)['successes'])")
  if [ "$total" -gt 0 ] && [ "$total" != "${last_total:-0}" ]; then
    sleep 10
    out=$O/$TAG-s$SEED-r$round
    CUDA_VISIBLE_DEVICES=$T $TR --nproc_per_node $NT --master_port $((PORT0 + round % 50)) scripts/bc_train.py --full \
      --src "$R/r*.shard*.db" --model $START --out $out --epochs 2 --lr 1e-5 --grad-accum 2 \
      > $R/sft.r$round.log 2>&1 || die "SFT round $round failed, see $R/sft.r$round.log"
    sleep 15
    policy=$out/final
  else
    echo "  round $round: no new successes, keep sampling with the current policy"
  fi
  last_total=$total; round=$((round + 1))
done
$PY scripts/count_success.py "$R/r*.shard*.db" > $R/stats.json; cat $R/stats.json

m=$TAG-s$SEED
echo "=== $(date +%H:%M) eval $policy"
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/rollout.py --chatml $THINK --model $policy \
  --model-id $m $ENVARGS $EVALARGS --out results/eval/$m.db > results/eval/$m.log 2>&1 || die "eval failed"
grep -h '^{"model' results/eval/$m.log
echo "=== $(date +%H:%M) done"
