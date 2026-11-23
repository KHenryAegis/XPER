#!/usr/bin/env bash
# Step 5 (CPU): exact-replay coverage of the target rollouts by the source's prefix trie (Section 3.1, Figure 2a).
#   training tasks -> results/coverage/{telecom,math}_train.json; test tasks are covered by step 3 (results/g1/driver.log)
# Needs the stores of steps 1-4.
source "$(dirname "$0")/common.sh"
mkdir -p results/coverage

step "telecom: source vs strong self-exploration (3 seeds), RFT (2), GRPO (3), weak self-exploration"
$PY -B scripts/coverage_multi.py --src "$SRC_TEL" --tgt \
  self="$SRC_TEL" \
  strong_selfexp='results/pipeline_b2s/selfexp.s40?.r*.shard*.db' \
  weak_selfexp='results/pipeline_a/selfexp.r*.shard*.db' \
  strong_grpo='results/grpo-instruct.s60?/it*.shard*.db' \
  strong_rft='results/rft-instruct.s50?/r*.shard*.db' > results/coverage/telecom_train.json

step "math: source vs strong and weak self-exploration"
$PY -B scripts/coverage_multi.py --src "$SRC_MATH" --tgt \
  strong_selfexp='results/math/selfexp/8b.shard*.db' \
  weak_selfexp='results/math/selfexp-weak/8b-base.shard*.db' > results/coverage/math_train.json
