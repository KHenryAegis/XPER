#!/usr/bin/env bash
# B2 seeds: repeat the strong-start comparison with uniform "bare" rendering and 3 seeds.
#
#   start = Qwen3-8B-Instruct, thinking off, re-measured bare (--no-think off) for comparability
#   arm A (self-exp): own rollouts, one round, seed 401+s, then SFT on its own successes
#   arm B (transfer): SFT on the source's 223 train_full successes (44,414 calls charged to source)
#
# Every arm trains with bare assistant turns and is evaluated the same way, so the
# empty-think-block shortcut of the earlier single-seed pipeline cannot appear. SFT recipe held fixed
# (full-param FSDP, 2 ep, lr 1e-5, grad-accum 2); the seeds vary the rollout data.
# GPUs 0-1 belong to another user. Any failure aborts.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
TR=$(dirname "$PY")/torchrun
H=${XPER_HOME}/models/hf
O=${XPER_HOME}/models/xper
START=$H/Qwen3-8B
R=results/pipeline_b2s
BUDGET=44414
IFS=',' read -ra G <<< "${GPUS:-2,3,4,5,6,7}"
NG=${#G[@]}
SEEDS=(401 402 403)
mkdir -p $R results/eval
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }

for i in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
  [ "$mem" -lt 1000 ] || die "GPU $i already has ${mem}MiB in use"
done
echo "=== $(date +%H:%M) B2-seeds on GPUs ${G[*]}"

echo "=== $(date +%H:%M) start model, bare eval"
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/rollout.py --chatml --model $START \
  --model-id 8b-instruct-start-bare --n 8 --out results/eval/8b-instruct-start-bare.db \
  > results/eval/8b-instruct-start-bare.log 2>&1 || die "start eval failed"

for s in "${!SEEDS[@]}"; do
  seed=${SEEDS[$s]}
  echo "=== $(date +%H:%M) seed $seed: self-exploration from 8B-Instruct until $BUDGET env calls"
  round=0; spent=0
  while [ "$spent" -lt "$BUDGET" ] && [ $round -lt 8 ]; do
    pids=()
    for i in "${!G[@]}"; do
      CUDA_VISIBLE_DEVICES=${G[$i]} $PY scripts/rollout.py --chatml --no-think --model $START \
        --model-id 8b-instruct-selfexp --split train_full --stratify 600 --shard $i/$NG \
        --n 1 --seed $((seed + round)) --env-workers 24 --gpu-util 0.8 \
        --out $R/selfexp.s$seed.r$round.shard$i.db > $R/selfexp.s$seed.r$round.shard$i.log 2>&1 &
      pids+=($!)
    done
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || die "s$seed round $round shard $i failed, see $R/selfexp.s$seed.r$round.shard$i.log"
    done
    r=$R/selfexp.s$seed.r*.shard*.db
    spent=$($PY scripts/count_success.py "$r" | $PY -c "import json,sys; print(json.load(sys.stdin)['env_calls'])")
    echo "  seed $seed round $round: cumulative env calls $spent"
    round=$((round + 1))
  done
  $PY scripts/count_success.py "$r" > $R/selfexp.s$seed.stats.json
  cat $R/selfexp.s$seed.stats.json
done

T=$(IFS=,; echo "${G[*]:0:4}")
echo "=== $(date +%H:%M) SFT on GPUs $T (${#SEEDS[@]} transfer + ${#SEEDS[@]} self-exp runs)"
port=29660
for s in "${!SEEDS[@]}"; do
  seed=${SEEDS[$s]}
  for arm in xfer selfexp; do
    src='results/src/4b-inst.shard*.db'
    [ $arm = selfexp ] && src="$R/selfexp.s$seed.r*.shard*.db"
    out=$O/8b-instruct-$arm-s$seed
    log=$R/${arm}_s$seed.train.log
    CUDA_VISIBLE_DEVICES=$T $TR --nproc_per_node 4 --master_port $port scripts/bc_train.py --full \
      --src "$src" --model $START --out "$out" --epochs 2 --lr 1e-5 --grad-accum 2 \
      > "$log" 2>&1 || die "SFT $out failed, see $log"
    port=$((port + 1))
  done
done

echo "=== $(date +%H:%M) evals"
pids=()
n=0
for s in "${!SEEDS[@]}"; do
  for arm in xfer selfexp; do
    gpu=${G[$((n % NG))]}
    m=8b-instruct-$arm-s${SEEDS[$s]}
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/rollout.py --chatml --model $O/$m/final \
      --model-id $m --n 8 --out results/eval/$m.db > results/eval/$m.log 2>&1 &
    pids+=($!)
    n=$((n + 1))
  done
done
for p in "${pids[@]}"; do wait "$p" || die "an eval failed"; done
for s in "${!SEEDS[@]}"; do
  for arm in xfer selfexp; do
    m=8b-instruct-$arm-s${SEEDS[$s]}
    grep -h '^{"model' results/eval/$m.log || echo "$m eval FAILED"
  done
done
grep -h '^{"model' results/eval/8b-instruct-start-bare.log
echo "=== $(date +%H:%M) done"