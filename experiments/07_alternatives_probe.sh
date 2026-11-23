#!/usr/bin/env bash
# Step 7: alternatives to finite injection on the same stored successes, and the reuse probe.
#   Table 3 (lower)  no injection at 88.8k calls (E2), ReLIFT-style (E4a), LUFFY-style (E4b), self-replay (E5),
#                    seeds 601-603, same weak start, pool and recipe as step 6
#   Table 10         target and source episodes (k=4) on 40 held-out validation tasks per environment (E3)
# scripts/jobs_spec.py writes the jobs of the reported runs; the queue runs them on the free GPUs of this host and
# exits when all are done. Needs steps 0-2.
source "$(dirname "$0")/common.sh"

$PY -B scripts/jobs_spec.py
$PY -B scripts/gpu_queue.py --jobs logs/queue/spec_20261117/jobs.jsonl --exit-when-done

summary results/eval/e[245]*.log results/probe/*.log
