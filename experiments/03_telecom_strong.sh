#!/usr/bin/env bash
# Step 3: strong telecom target (Qwen3-8B, thinking off) and transfer ICL of both telecom starts.
#   Table 2   zero-shot of the source and the strong start          qwen3-4b-inst, qwen3-8b-nothink (results/g1/)
#   Table 6/7 SFT on own successes / transfer SFT, 3 seeds          8b-instruct-{selfexp,xfer}-s40{1,2,3}-nothink
#             transfer ICL k=6, 3 draws; k=3 and k=12; weak start   8b-instruct-icl6{,-s1,-s2}, -icl3-s0, -icl12-s0,
#                                                                   8b-commonsft-icl6
#             ICL on the transfer-SFT checkpoints                   8b-instruct-xfer-s40{1,2,3}-icl6
#             iterated RFT (2 seeds) / GRPO (3 seeds)               rft-instruct-s50{1,2} / grpo-instruct-s60{1,2,3}
#   Table 1   GRPO from the transfer-SFT checkpoint (2 seeds)       grpo-xferstart-s60{1,2}
#   Sec. 5.1  no test fault combination in the training pool, the source successes or any ICL draw
# Needs steps 0-2 (the weak-start ICL arm reads models/xper/8b-commonsft).
source "$(dirname "$0")/common.sh"
STRONG=$H/Qwen3-8B

step "zero-shot source and strong start, plus the test-task coverage of Figure 2(a)"
mkdir -p results/g1
bash scripts/g1.sh 2>&1 | tee results/g1/driver.log

step "self-exploration SFT vs transfer SFT, seeds 401-403"
bash scripts/run_b2_seeds.sh                # GPUs 2-7, as in the paper
bash scripts/eval_b2seeds_nothink.sh       # evaluate every checkpoint as it was trained (thinking off)

step "transfer ICL"
bash scripts/run_icl_baseline.sh           # k=6, draw 0: strong start (GPU 6) and weak start (GPU 7)
GPUS="$HALF_A" bash scripts/run_icl_sweep.sh "6:1" "6:2" "3:0" "12:0"
GPUS="$HALF_A" bash scripts/run_icl_on_xfer.sh
for d in 6:0 6:1 6:2 3:0 12:0; do $PY -B scripts/check_overlap.py --icl $d; done > results/eval/overlap_check.txt

step "iterated RFT, seeds 501 and 502"
rft() {  # seed gpus port
  GPUS=$(comma "$2") PORT0=$3 START=$STRONG TAG=rft-instruct THINK=--no-think SEED=$1 \
    bash scripts/run_rft.sh > results/rft-instruct.s$1.driver.log 2>&1
}
rft 501 "$HALF_A" 29700 &
first=$!
rft 502 "$HALF_B" 29900 &
wait_all "$first" "$!"

step "GRPO from the zero-shot start (seeds 601-603) and from transfer SFT (Table 1, seeds 601-602)"
# The paper started from the best transfer-SFT seed, s401 (38.4% pass@1); set XFER_START for another run.
XFER_START=${XFER_START:-$O/8b-instruct-xfer-s401/final}
grpo() {  # tag start seed gpus port
  GPUS="$4" PORT0=$5 START=$2 TAG=$1 THINK=--no-think SEED=$3 TASKS=16 GROUP=8 LR=1e-6 \
    bash scripts/run_grpo.sh > results/$1.s$3.driver.log 2>&1
}
grpo grpo-instruct $STRONG 601 "$HALF_A" 29800 &
first=$!
grpo grpo-instruct $STRONG 602 "$HALF_B" 29950 &
wait_all "$first" "$!"
grpo grpo-instruct $STRONG 603 "$HALF_A" 29800 &
first=$!
grpo grpo-xferstart $XFER_START 601 "$HALF_B" 29950 &
wait_all "$first" "$!"
grpo grpo-xferstart $XFER_START 602 "$HALF_A" 29800

summary results/g1/*.log results/eval/8b-instruct-*.log results/eval/8b-commonsft-icl6.log \
  results/eval/{rft-instruct-s50?,grpo-instruct-s60?,grpo-xferstart-s60?}.log
