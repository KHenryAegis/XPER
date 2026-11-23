#!/usr/bin/env bash
# Math env, WEAK target (Qwen3-8B-Base eos-init, 0.75% zero-shot): the cold-start test on a second
# environment. Same protocol as the strong pipeline: self-exp on the same 2,000 train problems
# (seed 700, n=1) vs SFT on the 4B source's successes; bare SFT (2 ep, lr 1e-5), eval 300x4.
# Sampling uses --max-tokens 1024 (the base model mostly runs to the cap; 2048 made a turn ~500 s);
# all evals use 2048 like the strong pipeline.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
TR=$(dirname "$PY")/torchrun
O=${XPER_HOME}/models/xper
START=$O/qwen3-8b-base-init-eos
read -ra G <<< "${GPUS:-0 1 6}"
NG=${#G[@]}
T=$(IFS=,; echo "${G[*]}")
R=results/math
EV="--env math --chatml --split test --n 4 --max-steps 12 --max-tokens 2048 --env-workers 48"
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
for g in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g busy (${mem}MiB)"
done

echo "=== $(date +%H:%M) weak self-exploration: 8B-Base on the same 2000 train problems"
mkdir -p $R/selfexp-weak
pids=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} $PY scripts/rollout.py --env math --chatml --model $START \
    --model-id math-8b-base-selfexp --split train --stratify 2000 --shard $i/$NG --n 1 --seed 700 \
    --max-steps 12 --max-tokens 1024 --env-workers 48 \
    --out $R/selfexp-weak/8b-base.shard$i.db > $R/selfexp-weak/8b-base.shard$i.log 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || die "weak self-exp shard $i failed"; done
$PY scripts/count_success.py "$R/selfexp-weak/8b-base.shard*.db" | tee $R/selfexp-weak/stats.json
n_self=$($PY -c "import json; print(json.load(open('$R/selfexp-weak/stats.json'))['successes'])")

sft() {  # src out port
  CUDA_VISIBLE_DEVICES=$T $TR --nproc_per_node $NG --master_port $3 scripts/bc_train.py --full \
    --src "$1" --model $START --out $2 --epochs 2 --lr 1e-5 --grad-accum 2 \
    > $2.train.log 2>&1 || die "SFT $2 failed, see $2.train.log"
  sleep 15
}
mkdir -p $O/math
echo "=== $(date +%H:%M) SFT weak transfer"
sft "$R/src/4b-inst.shard*.db" $O/math/8b-base-xfer 29760
pids=()
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/rollout.py $EV --model $O/math/8b-base-xfer/final \
  --model-id math-8b-base-xfer --out $R/math-8b-base-xfer.db > $R/math-8b-base-xfer.log 2>&1 &
pids+=($!)
if [ "$n_self" -gt 0 ]; then
  echo "=== $(date +%H:%M) SFT weak self-exp on its $n_self successes (GPUs ${G[*]:1})"
  T2=$(IFS=,; echo "${G[*]:1}")
  CUDA_VISIBLE_DEVICES=$T2 $TR --nproc_per_node $((NG - 1)) --master_port 29761 scripts/bc_train.py --full \
    --src "$R/selfexp-weak/8b-base.shard*.db" --model $START --out $O/math/8b-base-selfexp \
    --epochs 2 --lr 1e-5 --grad-accum 2 > $O/math/8b-base-selfexp.train.log 2>&1 || die "weak self-exp SFT failed"
  sleep 15
  CUDA_VISIBLE_DEVICES=${G[1]} $PY scripts/rollout.py $EV --model $O/math/8b-base-selfexp/final \
    --model-id math-8b-base-selfexp --out $R/math-8b-base-selfexp.db > $R/math-8b-base-selfexp.log 2>&1 &
  pids+=($!)
else
  echo "  weak self-exp found no successes: nothing to train on (the arm stays at zero-shot)"
fi
for p in "${pids[@]}"; do wait $p || die "an eval failed"; done
for m in math-8b-base-eos math-8b-base-xfer math-8b-base-selfexp; do
  [ -f $R/$m.log ] && grep -h '^{"model' $R/$m.log
done
echo "=== $(date +%H:%M) done"
