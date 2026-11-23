#!/usr/bin/env bash
# Format control: identical trajectories with permuted action order (state->action mapping destroyed).
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
cd "$(dirname "$0")/.."; export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
TR=$(dirname "$PY")/torchrun
O=${XPER_HOME}/models/xper
SRC='results/src/4b-inst.shard*.db'
job() {  # name gpus port eval_gpu
  CUDA_VISIBLE_DEVICES=$2 $TR --nproc_per_node 4 --master_port $3 scripts/bc_train.py --full \
    --src "$SRC" --model $O/qwen3-$1-base-init-eos --out $O/fmtctl-$1 --epochs 2 --lr 1e-5 --grad-accum 2 \
    --format-control > results/bc2/fmtctl-$1.log 2>&1 || { echo "$1 FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$4 $PY scripts/rollout.py --chatml --model $O/fmtctl-$1/final \
    --model-id fmtctl-$1 --n 8 --out results/eval/fmtctl-$1.db > results/eval/fmtctl-$1.log 2>&1
  grep -h '^{"' results/eval/fmtctl-$1.log
}
job 8b 0,1,2,3 29621 0 &
job 4b 4,5,6,7 29622 4 &
wait
