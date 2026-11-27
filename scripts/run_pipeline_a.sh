#!/usr/bin/env bash
# Pipeline A: common agent SFT, then budget-matched self-exploration vs cross-model transfer.
#
#   1  8B-Base-init + common SFT (ToolACE, leak-free, full-param)          -> 8b-commonsft
#   1e eval 8b-commonsft on test (the new starting point both arms share)
#   2a self-exploration: 8b-commonsft samples train_full until it has spent the SAME env budget
#      the source spent (44,414 calls), then SFT on its own successes       -> 8b-selfexp
#   2b transfer: 8b-commonsft + SFT on the source's 223 successes           -> 8b-xfer
#   3  eval 2a and 2b on test (40 tasks x 8)
#
# Same init, same SFT recipe, same eval for every arm. Logs: results/pipeline_a/.
: "${XPER_HOME:?set XPER_HOME to the parent of models/ (see experiments/README.md)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
TR=$(dirname "$PY")/torchrun
O=${XPER_HOME}/models/xper
R=results/pipeline_a
BUDGET=44414
mkdir -p $R results/eval
fail() { echo "$(date +%H:%M) FAILED: $1 (see $2)"; tail -5 "$2"; exit 1; }
sft() {  # src init out gpus port log
  CUDA_VISIBLE_DEVICES=$4 $TR --nproc_per_node 4 --master_port $5 scripts/bc_train.py --full \
    --src "$1" --model "$2" --out "$3" --epochs 2 --lr 1e-5 --grad-accum 2 > "$6" 2>&1
}
evaluate() {  # model id gpu
  CUDA_VISIBLE_DEVICES=$3 $PY scripts/rollout.py --chatml --model "$1" --model-id "$2" --n 8 \
    --out results/eval/$2.db > results/eval/$2.log 2>&1
  grep -h '^{"' results/eval/$2.log
}

echo "=== $(date +%H:%M) 0: build common SFT data"
$PY scripts/prep_common_sft.py > $R/prep.log 2>&1 || fail prep $R/prep.log
head -1 $R/prep.log

echo "=== $(date +%H:%M) 1: common SFT (full-param, 8 GPUs)"
if [ ! -f $O/8b-commonsft/final/config.json ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $TR --nproc_per_node 8 --master_port 29641 scripts/bc_train.py --full \
    --src data/external/common_sft.jsonl --model $O/qwen3-8b-base-init-eos --out $O/8b-commonsft \
    --epochs 1 --lr 1e-5 --grad-accum 1 --max-len 8192 > $R/step1_commonsft.log 2>&1 \
    || fail "common SFT" $R/step1_commonsft.log
fi
grep -h '"saved"' $R/step1_commonsft.log | cut -c1-200

echo "=== $(date +%H:%M) 1e: eval 8b-commonsft"
evaluate $O/8b-commonsft/final 8b-commonsft 0

echo "=== $(date +%H:%M) 2a: self-exploration sampling until $BUDGET env calls"
spent=0; round=0
while [ "$spent" -lt "$BUDGET" ]; do
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i $PY scripts/rollout.py --chatml --model $O/8b-commonsft/final \
      --model-id 8b-selfexp --split train_full --stratify 600 --shard $i/8 --n 4 --seed $((100 + round)) \
      --env-workers 24 --gpu-util 0.8 --out $R/selfexp.r$round.shard$i.db \
      > $R/selfexp.r$round.shard$i.log 2>&1 &
  done
  wait
  spent=$($PY -c "
import glob
from xper.store import ExperienceStore
print(sum(ExperienceStore(p).env_call_total() for p in glob.glob('$R/selfexp.r*.shard*.db')))")
  echo "  round $round: cumulative env calls $spent"
  round=$((round + 1))
  [ $round -ge 8 ] && break
done
$PY - <<PYEOF
import glob, json
from xper.store import ExperienceStore
tot = succ = calls = 0; tasks = set()
for p in sorted(glob.glob("$R/selfexp.r*.shard*.db")):
    for _id, task, _m, r, _s, _n, ec in ExperienceStore(p).trajectories():
        tot += 1; calls += ec; succ += r >= 1
        if r >= 1: tasks.add(task)
print(json.dumps({"arm": "self-exploration", "episodes": tot, "env_calls": calls, "successes": succ,
                  "tasks_solved": len(tasks), "success_rate": round(succ / max(1, tot), 4)}))
PYEOF

echo "=== $(date +%H:%M) 2a/2b: SFT on own successes (GPUs 0-3) and on source successes (GPUs 4-7)"
sft "$R/selfexp.r*.shard*.db" $O/8b-commonsft/final $O/8b-selfexp 0,1,2,3 29642 $R/step2a_train.log &
P1=$!
sft 'results/src/4b-inst.shard*.db' $O/8b-commonsft/final $O/8b-xfer 4,5,6,7 29643 $R/step2b_train.log &
P2=$!
wait $P1 || echo "2a train FAILED (see $R/step2a_train.log; expected if self-exploration found no successes)"
wait $P2 || fail "2b train" $R/step2b_train.log

echo "=== $(date +%H:%M) 3: eval"
[ -f $O/8b-selfexp/final/config.json ] && evaluate $O/8b-selfexp/final 8b-selfexp 0 &
evaluate $O/8b-xfer/final 8b-xfer 4 &
wait
echo "=== $(date +%H:%M) done"
