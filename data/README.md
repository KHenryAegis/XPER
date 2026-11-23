# Experiment data

## Included records

| Path | Contents and provenance |
|---|---|
| `records/eval/<run>.jsonl.gz` | The evaluation JSONL files emitted by `scripts/rollout.py`, compressed for distribution. Each line gives a task, sample index, reward, steps, environment calls and invalid-format count. The run-to-log mapping is in `results/eval_summaries.csv`. |
| `records/probe/` | Six validation JSONL files in the same format. Each holds four episodes on 40 held-out tasks for one model and environment. |
| `records/rounds.json.gz` | Per-iteration fields from the cold-start cohort audit and the GRPO driver logs, keyed by run. Includes mixed-group counts, and for injection runs, exposure counts and target training success. |
| `records/coverage_*.json` | Test coverage from the G1 driver and training coverage from step 05. Depth keys begin at zero. |
| `records/telecom_exploration.csv.gz` | Per-task episode and success counts from the eight telecom source shards and the three strong-target self-exploration seeds. |
| `records/failed_shard.json` | Recorded call bounds for the resampled math shard in seed 603. |
| `splits/` | Source-explored, source-solved and held-out probe task IDs. The benchmark defines the official telecom test split. |
| `source/source_successes_telecom.jsonl` | The 223 successful source trajectories, including their simulated actions and observations. |
| `results/` | Reference evaluation summaries, per-round curves, per-run charges, generated cold-start tables and the reuse probe. `table3_runs.csv` records partial and retried runs. Step 09 derives Table 3 per-task counts and lower-block rows in `outputs/`. |

Telecom evaluations normally sample eight episodes per task. Math evaluations sample four. Pass@1 is the mean success rate per task. Pass@n counts tasks solved in at least one of their n attempts. Bootstrap intervals resample shared tasks and condition on the observed policies.

## External inputs for GPU runs

| Input | Expected location | Used for |
|---|---|---|
| τ²-bench at `5ade54e282a4555e4d6c52d6b9b861ea8872afa6` | `third_party/tau2-bench/`; installed by step 00 | Telecom tasks and tools |
| DAPO-Math integer problems packaged by rStar2-Agent | `data/external/math/train.jsonl` | Math training and test split |
| ToolACE conversations | `data/external/toolace.json` | Step 02 generates `data/external/common_sft.jsonl` for the weak telecom initialization |
| Qwen3-4B-Instruct-2507, Qwen3-8B, Qwen3-8B-Base, Qwen3-4B-Base | One directory per checkpoint in `$XPER_HOME/models/hf/` | Source, strong target and delimiter-initialized controls |

See [datasets.json](datasets.json) for input sizes, SHA-256 values and model identifiers. Run `python tools/check_inputs.py` to inspect local inputs.
