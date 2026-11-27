#!/usr/bin/env bash
# Step 6: matched cold-start injection cohorts (Section 5.4, Section 6.1).
#   Table 3 (upper), Table 4, Figure 4  telecom caps 0/5/20/100, seeds 601/602          td{0,5,20,100}.s60{1,2}
#   Table 3, Table 5                    telecom information controls; math none/real   sc-tel-{obs_mask,schema_random}.s60{1,2},
#                                                                                      sc-math-{none,real}.s60{1,2}
#   Table 3                             third rollout seed of the four main arms       td{0,100}.s603, sc-math-{none,real}.s603
#   Section 6.1                         uncapped mismatch / shuffle controls           tp-{mismatch,shuffle}.s60{1,2}
# Every arm: weak start, 80 source-solved tasks, 16 tasks x 8 samples, lr 1e-6, 44,414 (telecom) or 10,403 (math)
# target calls, injection into all-failure groups only. Each controller writes an immutable job manifest, runs its
# jobs through scripts/gpu_queue.py on the free GPUs of this host (4 per run), verifies every run and blocks until
# the cohort is done. Status: python -B scripts/<controller>.py status. Needs steps 0-2.
source "$(dirname "$0")/common.sh"

step "telecom dose cohort (logs/queue/tel_dose_20261112/)"
$PY -B scripts/tel_dose.py run

step "information controls and the math cohort (logs/queue/strengthen_20261113/)"
$PY -B scripts/strengthen.py run

step "third rollout seed (logs/queue/third_seed_20261115/)"
$PY -B scripts/third_seed.py run

step "uncapped protocol controls (logs/queue/)"
$PY -B scripts/jobs_tel_protocol.py
bash scripts/run_tel_protocol.sh --exit-when-done

summary results/eval/{td,sc-,tp-}*.log
