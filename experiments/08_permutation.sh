#!/usr/bin/env bash
# Step 8: permutation control (Section 6.2). Delimiter-initialized Qwen3-8B-Base and Qwen3-4B-Base, no
# ToolACE warm-up: untrained, SFT on the 223 source trajectories with actions permuted inside each trajectory, and
# SFT on the real trajectories (2 epochs, lr 1e-5). Runs init-eos-{8b,4b}, fmtctl-{8b,4b}, bc3-{8b,4b}. Needs steps 0-1.
source "$(dirname "$0")/common.sh"
mkdir -p results/bc2

step "untrained starts"
pids=()
for m in 8b:0 4b:4; do
  size=${m%:*}; gpu=${m#*:}
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/rollout.py --chatml --model $O/qwen3-$size-base-init-eos \
    --model-id init-eos-$size --n 8 --out results/eval/init-eos-$size.db > results/eval/init-eos-$size.log 2>&1 &
  pids+=($!)
done
wait_all "${pids[@]}"

step "SFT on real trajectories (8B on GPUs 0-3, 4B on GPUs 4-7)"
bash scripts/bc3_full.sh

step "SFT on permuted trajectories"
bash scripts/fmtctl.sh

summary results/eval/{init-eos,bc3,fmtctl}-*.log
