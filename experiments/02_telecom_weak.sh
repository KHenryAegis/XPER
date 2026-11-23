#!/usr/bin/env bash
# Step 2: weak telecom target (Qwen3-8B-Base + delimiter init + one epoch of ToolACE SFT).
#   Table 2   zero-shot of the weak start                         8b-commonsft
#   Table 6   SFT on its own successes (1 round) / transfer SFT   8b-selfexp / 8b-xfer
#             iterated RFT (2 seeds) / GRPO (1 seed)              rft-commonsft-s50{1,2} / grpo-commonsft-s601
#   (transfer ICL of this start is run with the strong start's ICL in step 3)
# Needs steps 0-1 and data/external/toolace.json. Driver logs: results/pipeline_a/, results/<tag>.s<seed>.driver.log.
source "$(dirname "$0")/common.sh"

step "weak start, budget-matched self-exploration and transfer SFT (8 GPUs)"
bash scripts/run_pipeline_a.sh            # also runs scripts/prep_common_sft.py -> data/external/common_sft.jsonl
WEAK=$O/8b-commonsft/final

step "iterated RFT, seeds 501 and 502 (the 44,414-call budget ends both after round 6)"
rft() {  # seed gpus port
  GPUS=$(comma "$2") PORT0=$3 START=$WEAK TAG=rft-commonsft THINK="" SEED=$1 MAX_ROUNDS=40 \
    bash scripts/run_rft.sh > results/rft-commonsft.s$1.driver.log 2>&1
}
rft 501 "$HALF_A" 29700 &
first=$!
rft 502 "$HALF_B" 29900 &
wait_all "$first" "$!"

step "GRPO, seed 601: 64 tasks x 8 samples per iteration"
GPUS="$HALF_A" START=$WEAK TAG=grpo-commonsft THINK="" SEED=601 TASKS=64 GROUP=8 LR=1e-6 MAX_IT=150 \
  bash scripts/run_grpo.sh > results/grpo-commonsft.s601.driver.log 2>&1

summary results/eval/{8b-commonsft,8b-selfexp,8b-xfer,rft-commonsft-s501,rft-commonsft-s502,grpo-commonsft-s601}.log
