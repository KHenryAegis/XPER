#!/usr/bin/env bash
# Source-data scaling in the discussion. Full-data baselines come from steps 03 and 08.
source "$(dirname "$0")/common.sh"
mkdir -p results/ablation
train_eval() {
  local name=$1 model=$2 cap=$3 draw=$4 seed=$5
  shift 5
  CUDA_VISIBLE_DEVICES=2,3,4,5 "$PY" -m torch.distributed.run --nproc_per_node 4 --master_port 29680 \
    scripts/bc_train.py --full --src "$SRC_TEL" --model "$model" --out "$O/$name" \
    --epochs 2 --lr 1e-5 --grad-accum 2 --max-trajs "$cap" --subsample-seed "$draw" --seed "$seed" \
    > "results/ablation/$name.train.log" 2>&1
  CUDA_VISIBLE_DEVICES=2 "$PY" scripts/rollout.py --chatml "$@" --model "$O/$name/final" \
    --model-id "$name" --n 8 --out "results/eval/$name.db" > "results/eval/$name.log" 2>&1
}
for spec in '25 56 10' '50 112 20' '75 167 30'; do
  read -r pct cap draw <<< "$spec"
  train_eval "8b-xfer-bare-p$pct" "$H/Qwen3-8B" "$cap" "$draw" 0 --no-think
  train_eval "8b-base-xfer-weak-p$pct" "$O/qwen3-8b-base-init-eos" "$cap" "$draw" 0
done
train_eval 8b-base-xfer-weak-p100s1 "$O/qwen3-8b-base-init-eos" 0 0 1
