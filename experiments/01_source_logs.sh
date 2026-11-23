#!/usr/bin/env bash
# Step 1: the source model explores once; its stores are the experience every reuse arm reads (Section 5.2).
#   telecom: Qwen3-4B-Instruct-2507, 4 episodes on each of 600 stratified training tasks, 8 GPUs
#            -> results/src/4b-inst.shard{0..7}.db   (2,400 episodes, 44,414 calls, 223 successes)
#   math:    1 episode on each of 2,000 training problems, 2 GPUs
#            -> results/math/src/4b-inst.shard{0,1}.db  (2,000 episodes, 10,403 calls, 452 successes)
source "$(dirname "$0")/common.sh"

step "telecom source log"
mkdir -p results/src
bash scripts/src_rollout.sh
$PY -B scripts/count_success.py "$SRC_TEL" | tee results/src/stats.json

step "math source log"
GPUS="0 1" bash scripts/run_math_source.sh
