#!/usr/bin/env bash
# Math env G1 (headroom): zero-shot source and targets on the 300-problem test split, n=4 (Table 1).
# The source's train experience must help the target only if the target has room to improve
# and the source solves problems the target does not.
: "${XPER_HOME:?set XPER_HOME to the parent of models/ (see experiments/README.md)}"
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
H=${XPER_HOME}/models/hf
O=${XPER_HOME}/models/xper
read -ra G <<< "${GPUS:-0 1 6}"
mkdir -p results/math
die() { echo "$(date +%H:%M) ABORT: $1"; exit 1; }
for g in "${G[@]}"; do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$mem" -lt 1000 ] || die "GPU $g busy (${mem}MiB)"
done
common="--env math --chatml --split test --n 4 --max-steps 12 --max-tokens 2048 --env-workers 48"
run() {  # gpu model-id model extra
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/rollout.py $common $4 --model $3 --model-id $2 \
    --out results/math/$2.db > results/math/$2.log 2>&1
}
echo "=== $(date +%H:%M) math headroom on GPUs ${G[*]}"
run ${G[0]} math-4b-inst $H/Qwen3-4B-Instruct-2507 "" & p1=$!
run ${G[1]} math-8b-instruct $H/Qwen3-8B --no-think & p2=$!
# The Base model mostly runs to the token cap; the reported run covers the first 100 test problems (Table 1).
run ${G[2]} math-8b-base-eos $O/qwen3-8b-base-init-eos "--limit 100" & p3=$!
for p in $p1 $p2 $p3; do wait $p || die "a run failed"; done
for m in math-4b-inst math-8b-instruct math-8b-base-eos; do grep -h -E '^\{"(model|timing)' results/math/$m.log; done
echo "=== $(date +%H:%M) done"
