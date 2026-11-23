#!/usr/bin/env bash
# Step 0: inputs and initial checkpoints. No GPU needed.
#   - tau2-bench at the paper's commit (third_party/tau2-bench)
#   - check of every external input against data/datasets.json
#   - delimiter initialization of Qwen3-8B-Base and Qwen3-4B-Base (Section 4.5; scripts/prep_init.py)
source "$(dirname "$0")/common.sh"

step "tau2-bench"
bash tools/fetch_tau2.sh

step "external inputs (data/datasets.json)"
$PY -B tools/check_inputs.py

step "delimiter-initialized Base checkpoints"
for size in 8B 4B; do
  out=$O/qwen3-$(echo $size | tr B b)-base-init-eos
  [ -f "$out/xper_init.json" ] || $PY -B scripts/prep_init.py --base "$H/Qwen3-$size-Base" --out "$out" --mode eos
done
