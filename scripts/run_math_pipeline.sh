#!/usr/bin/env bash
# Math env, strong target (Qwen3-8B thinking-off), same protocol as telecom B2:
#   self-exp : target explores the same train problems as the source (same seed, n=1), then SFT on
#              its own successes (budget matched to the source's env calls)
#   transfer : SFT on the source's (4B-Instruct) successes
#   icl      : source successes as in-context examples, no training
# Bare SFT (2 ep, lr 1e-5), eval with --no-think on the 300 test problems x4.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
TR=$(dirname "$PY")/torchrun
H=${XPER_HOME}/models/hf
O=${XPER_HOME}/models/xper
START=$H/Qwen3-8B
read -ra G <<< "${GPUS:-0 1 6}"
NG=${#G[@]}
T=$(IFS=,; echo "${G[*]}")
N=${N:-2000}
R=results/math
EV="--env math --chatml --no-think --split test --n 4 --max-steps 12 --max-tokens 2048 --env-workers 48"
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
[ -s $R/src/stats.json ] || die "run scripts/run_math_source.sh first"
for g in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g busy (${mem}MiB)"
done
echo "=== $(date +%H:%M) source: $(cat $R/src/stats.json)"

echo "=== $(date +%H:%M) self-exploration: 8B on the same $N train problems (seed 700, n=1)"
mkdir -p $R/selfexp
pids=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY scripts/rollout.py --env math --chatml --no-think --model $START \
    --model-id math-8b-selfexp --split train --stratify $N --shard $i/$NG --n 1 --seed 700 \
    --max-steps 12 --max-tokens 2048 --env-workers 48 \
    --out $R/selfexp/8b.shard$i.db > $R/selfexp/8b.shard$i.log 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || die "self-exp shard $i failed"; done
$PY scripts/count_success.py "$R/selfexp/8b.shard*.db" | tee $R/selfexp/stats.json

sft() {  # src out port
  CUDA_VISIBLE_DEVICES=$T $TR --nproc_per_node $NG --master_port $3 scripts/bc_train.py --full \
    --src "$1" --model $START --out $2 --epochs 2 --lr 1e-5 --grad-accum 2 \
    > $2.train.log 2>&1 || die "SFT $2 failed, see $2.train.log"
  sleep 15
}
mkdir -p $O/math
echo "=== $(date +%H:%M) SFT transfer, then SFT self-exp ($NG-GPU FSDP)"
sft "$R/src/4b-inst.shard*.db" $O/math/8b-xfer 29750
sft "$R/selfexp/8b.shard*.db" $O/math/8b-selfexp 29751

echo "=== $(date +%H:%M) eval: transfer | self-exp | ICL k=4 (source examples)"
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/rollout.py $EV --model $O/math/8b-xfer/final --model-id math-8b-xfer \
  --out $R/math-8b-xfer.db > $R/math-8b-xfer.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=${G[1]} $PY scripts/rollout.py $EV --model $O/math/8b-selfexp/final --model-id math-8b-selfexp \
  --out $R/math-8b-selfexp.db > $R/math-8b-selfexp.log 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=${G[2]} $PY scripts/rollout.py $EV --model $START --model-id math-8b-icl4 \
  --icl-src "$R/src/4b-inst.shard*.db" --icl-k 4 --icl-obs-chars 600 \
  --out $R/math-8b-icl4.db > $R/math-8b-icl4.log 2>&1 & p3=$!
for p in $p1 $p2 $p3; do wait $p || die "an eval failed"; done
for m in math-8b-instruct math-4b-inst math-8b-xfer math-8b-selfexp math-8b-icl4; do
  grep -h -E '^ICL|^\{"model' $R/$m.log
done
echo "=== $(date +%H:%M) done"
