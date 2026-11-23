#!/usr/bin/env bash
# Full-param BC from eos-initialized Base checkpoints (scripts/prep_init.py), then eval on telecom test.
: "${XPER_HOME:?set XPER_HOME to the directory that contains XPER/, models/ and miniconda3/ (see README)}"
cd "$(dirname "$0")/.."; export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${XPER_PYTHON:-$XPER_HOME/miniconda3/envs/xper/bin/python}
TR=$(dirname "$PY")/torchrun
O=${XPER_HOME}/models/xper
SRC='results/src/4b-inst.shard*.db'
job() {  # name init gpus port eval_gpu
  CUDA_VISIBLE_DEVICES=$3 $TR --nproc_per_node 4 --master_port $4 scripts/bc_train.py --full \
    --src "$SRC" --model $O/$2 --out $O/bc3-$1 --epochs 2 --lr 1e-5 --grad-accum 2 \
    > results/bc2/bc3-$1.log 2>&1 || { echo "$1 train FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$5 $PY scripts/rollout.py --chatml --model $O/bc3-$1/final \
    --model-id bc3-$1 --n 8 --out results/eval/bc3-$1.db > results/eval/bc3-$1.log 2>&1
  grep -h '^{"' results/eval/bc3-$1.log
}
job 8b qwen3-8b-base-init-eos 0,1,2,3 29611 0 &
job 4b qwen3-4b-base-init-eos 4,5,6,7 29612 4 &
# untrained eos-init baselines on the remaining GPUs once training frees them is not needed: run now on GPU-share-free slots after training
wait
