#!/usr/bin/env bash
# Step 4: math with a Python tool, both targets.
#   Table 2   zero-shot of the source and both starts              math-4b-inst, math-8b-instruct, math-8b-base-eos
#   Table 6   strong: SFT on own successes / transfer SFT / ICL    math-8b-selfexp / math-8b-xfer / math-8b-icl4
#             weak:   SFT on own successes / transfer SFT          math-8b-base-selfexp / math-8b-base-xfer
#             strong: iterated RFT (2 seeds) / GRPO (3 seeds)      math-rft-instruct-s50{1,2} / math-grpo-instruct-s60{1,2,3}
# Needs steps 0-1 and data/external/math/train.jsonl. Evaluation logs: results/math/ and results/eval/.
source "$(dirname "$0")/common.sh"
STRONG=$H/Qwen3-8B

step "zero-shot (300 test problems x 4; the weak start on the first 100)"
GPUS="0 1 2" bash scripts/run_math_headroom.sh

step "strong target: self-exploration SFT, transfer SFT, ICL"
GPUS="0 1 2" bash scripts/run_math_pipeline.sh > results/math/pipeline.driver.log 2>&1

step "weak target: self-exploration SFT, transfer SFT"
GPUS="0 1 2" bash scripts/run_math_weak.sh > results/math/weak.driver.log 2>&1

step "strong target: iterated RFT (seeds 501, 502) and GRPO (seeds 601-603) at the source's 10,403 calls"
rft() {  # seed gpus port
  GPUS=$(comma "$2") PORT0=$3 START=$STRONG TAG=math-rft-instruct THINK=--no-think SEED=$1 STRAT=500 \
    BUDGET=10403 SPLIT=train ENVARGS="$MATH" EVALARGS="--n 4" \
    bash scripts/run_rft.sh > results/math-rft-instruct.s$1.driver.log 2>&1
}
grpo() {  # seed gpus port
  GPUS="$2" PORT0=$3 START=$STRONG TAG=math-grpo-instruct THINK=--no-think SEED=$1 TASKS=16 GROUP=8 \
    BUDGET=10403 SPLIT=train ENVARGS="$MATH" EVALARGS="--n 4" \
    bash scripts/run_grpo.sh > results/math-grpo-instruct.s$1.driver.log 2>&1
}
grpo 601 "$HALF_A" 29800 &
first=$!
grpo 602 "$HALF_B" 29950 &
wait_all "$first" "$!"
grpo 603 "$HALF_A" 29800 &
first=$!
rft 501 "$HALF_B" 29850 &
wait_all "$first" "$!"
rft 502 "$HALF_A" 29850

summary results/math/math-*.log results/eval/math-{rft,grpo}-instruct-s*.log
