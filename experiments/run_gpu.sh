#!/usr/bin/env bash
# Run the paper's training and evaluation cohorts sequentially on an eight-GPU Linux host.
set -euo pipefail
cd "$(dirname "$0")/.."
steps=(
  00_setup.sh
  01_source_logs.sh
  02_telecom_weak.sh
  03_telecom_strong.sh
  04_math.sh
  05_exact_replay.sh
  06_injection_cohort.sh
  07_alternatives_probe.sh
  08_permutation.sh
  08_source_scaling.sh
)
if [[ ${1:-} == --list && $# == 1 ]]; then
  printf 'experiments/%s\n' "${steps[@]}"
  exit 0
fi
if (( $# )); then echo "Usage: bash experiments/run_gpu.sh [--list]" >&2; exit 2; fi
: "${XPER_HOME:?set XPER_HOME to the parent of models/}"
export XPER_PYTHON=${XPER_PYTHON:-$(command -v python)}
for step in "${steps[@]}"; do
  printf '\nRunning experiments/%s\n' "$step"
  bash "experiments/$step"
done
"$XPER_PYTHON" -B tools/export_eval_summaries.py --require-paper-runs --out outputs/gpu/eval_summaries.csv
"$XPER_PYTHON" -B scripts/analyze_alternatives.py --probe-only --out outputs/gpu
