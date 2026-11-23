#!/usr/bin/env bash
# Few-shot ICL baseline: the target reads the source's train_full successes in context, no training,
# no extra env calls beyond evaluation. Same examples (k, seed) across starts.
#   GPU 6: 8B-Instruct thinking-off (strong start, B2: 24.4% zero-shot)
#   GPU 7: Pipeline A weak start (common-SFT 8B-Base, 0% zero-shot)
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
H=${XPER_HOME}/models/hf
O=${XPER_HOME}/models/xper
SRC='results/src/4b-inst.shard*.db'
K=${K:-6}
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
for g in 6 7; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g already has ${mem}MiB in use"
done

echo "=== $(date +%H:%M) ICL k=$K"
CUDA_VISIBLE_DEVICES=6 $PY scripts/rollout.py --chatml --no-think --model $H/Qwen3-8B \
  --model-id 8b-instruct-icl$K --n 8 --icl-src "$SRC" --icl-k $K \
  --out results/eval/8b-instruct-icl$K.db > results/eval/8b-instruct-icl$K.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=7 $PY scripts/rollout.py --chatml --model $O/8b-commonsft/final \
  --model-id 8b-commonsft-icl$K --n 8 --icl-src "$SRC" --icl-k $K \
  --out results/eval/8b-commonsft-icl$K.db > results/eval/8b-commonsft-icl$K.log 2>&1 &
p2=$!
wait $p1 || die "instruct ICL failed"
wait $p2 || die "commonsft ICL failed"
for m in 8b-instruct-icl$K 8b-commonsft-icl$K; do grep -h -E '^ICL|^\{"model' results/eval/$m.log; done
echo "=== $(date +%H:%M) done"
