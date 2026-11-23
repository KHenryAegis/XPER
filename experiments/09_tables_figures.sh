#!/usr/bin/env bash
# Recompute the displayed statistics from the episode, round and coverage records.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${XPER_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B scripts/reproduce.py
"$PY" -B tools/verify_table3.py
"$PY" -B scripts/compose_table3.py --rows outputs/table3_lower_rows.tex
for ref in data/results/tables/*.tex data/results/table10_probe.csv; do
  cmp "$ref" "outputs/${ref#data/results/}"
done
for plot in make_plots observations teaser overview; do
  "$PY" -B "scripts/figures/$plot.py"
done
echo "Tables and figures are in outputs/."
