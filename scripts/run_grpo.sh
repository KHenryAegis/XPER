#!/usr/bin/env bash
# Online RL self-exploration baseline: group-relative policy gradient (GRPO without ratio clipping
# and without KL; one pass per batch, so the policy that sampled is the one being updated).
# Each iteration: TASKS train_full tasks x G samples with the current policy -> group-normalized
# advantages (zero-variance groups dropped) -> one weighted-CE pass (bc_train --weighted) from the
# current policy -> next policy. Stops when the target has spent BUDGET env calls.
#
# Usage: START=<hf dir> TAG=<name> THINK=--no-think|"" bash scripts/run_grpo.sh
# INJECT=<source store glob>: off-policy source successes into groups (INJECT_MODE fail|all; grpo_advantages.py).
#   INJECT_SELF=1: prefer the target's own earlier successes; SRC_TASK_FRAC: source kept for this task fraction.
# Other envs: ENVARGS (sampling + eval, e.g. "--env math --max-steps 12 --max-tokens 2048
#   --env-workers 48"), SPLIT (train split name), EVALARGS (default "--n 8"), BUDGET.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK_KH=${XPER_HOME:-$(dirname "$PWD")}
PY=${XPER_PYTHON:-$TASK_KH/miniconda3/envs/xper/bin/python}
[ -x "$PY" ] || PY=$(command -v python)
TR=${XPER_TORCHRUN:-$(dirname "$PY")/torchrun}
O=${XPER_MODEL_DIR:-$TASK_KH/models/xper}
: "${START:?}" "${TAG:?}"; THINK=${THINK-}; SEED=${SEED:-601}
INJECT=${INJECT:-}; INJECT_MODE=${INJECT_MODE:-fail}; INJECT_CONTROL=${INJECT_CONTROL:-none}
INJECT_LIMIT=${INJECT_LIMIT:--1}  # cumulative source trajectories across all rounds; -1 = unlimited
[[ "$INJECT_LIMIT" =~ ^(-1|[0-9]+)$ ]] || { echo 'Invalid INJECT_LIMIT' >&2; exit 1; }
src_used=0
# LEAN=1: delete each iteration's convs/train jsonl once consumed (stores + logs kept) and keep only the final policy dir
LEAN=${LEAN:-}
INJECT_SELF=${INJECT_SELF:-}; SRC_TASK_FRAC=${SRC_TASK_FRAC:-1.0}  # bootstrapping, see grpo_advantages.py
PORT0=${PORT0:-29800}; BUDGET=${BUDGET:-44414}; SPLIT=${SPLIT:-train_full}; ENVARGS=${ENVARGS-}; EVALARGS=${EVALARGS:---n 8}; TASKS=${TASKS:-64}; GROUP=${GROUP:-8}; LR=${LR:-1e-6}; MAX_IT=${MAX_IT:-40}
# gpu_queue.py passes space-separated GPUs, convert to array
read -ra G <<< "${GPUS:-2 3 4 5 6 7}"
NG=${#G[@]}
T=$(IFS=,; echo "${G[*]:0:4}"); NT=$(( NG < 4 ? NG : 4 ))  # FSDP world size
R=results/$TAG.s$SEED
mkdir -p $R results/eval
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
for g in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g already has ${mem}MiB in use"
done

echo "=== $(date +%H:%M) GRPO $TAG from $START: $TASKS tasks x $GROUP per iter, lr $LR, budget $BUDGET"
policy=$START; it=0; spent=0
while [ "$spent" -lt "$BUDGET" ] && [ $it -lt $MAX_IT ]; do
  shard() {  # one rollout shard; retried once below (vLLM's internal port can collide with another engine)
    rm -f $R/it$it.shard$1.db $R/it$it.shard$1.db-wal $R/it$it.shard$1.db-shm $R/it$it.shard$1.convs.jsonl
    CUDA_VISIBLE_DEVICES=${G[$1]} $PY scripts/rollout.py --chatml $THINK --model $policy \
      --model-id $TAG-it$it --split $SPLIT --stratify $TASKS --shard $1/$NG --n $GROUP \
      --seed $((SEED + it)) --env-workers 24 --gpu-util 0.8 $ENVARGS \
      --save-convs $R/it$it.shard$1.convs.jsonl --out $R/it$it.shard$1.db > $R/it$it.shard$1.log 2>&1
  }
  pids=()
  for i in "${!G[@]}"; do shard $i & pids+=($!); done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" && continue
    mv $R/it$it.shard$i.log $R/it$it.shard$i.try1.log
    echo "  it $it shard $i failed once, retrying (see $R/it$it.shard$i.try1.log)"
    sleep 20; shard $i || die "it $it shard $i rollout failed twice, see $R/it$it.shard$i.log"
  done
  spent=$($PY scripts/count_success.py "$R/it*.shard*.db" | $PY -c "import json,sys; print(json.load(sys.stdin)['env_calls'])")
  CAP=()
  [ "$INJECT_LIMIT" -ge 0 ] && CAP=(--max-src-injections $((INJECT_LIMIT - src_used)))
  adv=$($PY scripts/grpo_advantages.py --convs $R/it$it.shard*.convs.jsonl --out $R/it$it.train.jsonl "${CAP[@]}" \
    ${INJECT:+--inject-src "$INJECT" --inject-mode $INJECT_MODE --inject-control $INJECT_CONTROL --seed $it --src-task-frac $SRC_TASK_FRAC} \
    ${INJECT_SELF:+--inject-convs "$R/it*.shard*.convs.jsonl"}) || die "it $it advantage construction failed"
  used_now=$(echo "$adv" | $PY -c "import json,sys; r=json.load(sys.stdin); print(r['injected']-r['injected_own'])") || die "invalid injection statistics"
  src_used=$((src_used + used_now))
  [ "$INJECT_LIMIT" -ge 0 ] && [ "$src_used" -gt "$INJECT_LIMIT" ] && die "source injection cap exceeded"
  echo "  it $it: $adv | cumulative env calls $spent"
  n=$(echo "$adv" | $PY -c "import json,sys; print(json.load(sys.stdin)['train_seqs'])")
  if [ "$n" -gt 0 ]; then
    sleep 10
    # ping-pong between two dirs, always writing the one that does NOT hold the current policy
    # (iteration parity is wrong once an iteration makes no update)
    out=$O/$TAG-s$SEED-0
    [ "$policy" = "$out/final" ] && out=$O/$TAG-s$SEED-1
    rm -rf "$out"
    # a port the OS reports free right now: fixed PORT0+it ports collided with other users' listeners
    # on the shared box (EADDRINUSE on 30080 / 30246, 2026-11-01)
    port=$($PY -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
    CUDA_VISIBLE_DEVICES=$T $TR --nproc_per_node $NT --master_port $port scripts/bc_train.py --full \
      --weighted --src $R/it$it.train.jsonl --model $policy --out $out --epochs 1 --lr $LR --grad-accum 2 \
      > $R/it$it.train.log 2>&1 || die "it $it update failed, see $R/it$it.train.log"
    sleep 15
    policy=$out/final
  fi
  [ -n "$LEAN" ] && { rm -f $R/it$it.train.jsonl; [ -z "$INJECT_SELF" ] && rm -f $R/it$it.shard*.convs.jsonl; }
  it=$((it + 1))
done
[ "$spent" -lt "$BUDGET" ] && die "iteration limit reached below budget: $spent / $BUDGET"
$PY scripts/count_success.py "$R/it*.shard*.db" > $R/stats.json; cat $R/stats.json
if [ "$policy" = "$START" ]; then
  echo "  no update was made (no mixed groups); evaluating the start model"
  evalm=$START
else
  final=$O/$TAG-s$SEED-final
  rm -rf "$final"
  if [ -n "$LEAN" ]; then
    mv "$(dirname $policy)" "$final"; rm -rf $O/$TAG-s$SEED-0 $O/$TAG-s$SEED-1
  else
    cp -r "$(dirname $policy)" "$final"
  fi
  evalm=$final/final
fi
m=$TAG-s$SEED
echo "=== $(date +%H:%M) eval $evalm"
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/rollout.py --chatml $THINK --model $evalm \
  --model-id $m $ENVARGS $EVALARGS --out results/eval/$m.db > results/eval/$m.log 2>&1 || die "eval failed"
grep -h '^{"model' results/eval/$m.log
# CLEAN_FINAL=1: the eval DB is the deliverable; drop the trained weights (never the start model)
[ -n "${CLEAN_FINAL:-}" ] && [ "$evalm" != "$START" ] && rm -rf "$O/$TAG-s$SEED-final"
echo "=== $(date +%H:%M) done"
