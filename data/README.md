# Experiment data

## Included records

| Path | Contents and provenance |
|---|---|
| `records/eval/<run>.jsonl.gz` | The evaluation JSONL files emitted by `scripts/rollout.py`, compressed for distribution. Each line gives a task, sample index, reward, steps, environment calls and invalid-format count. The run-to-log mapping is in `results/eval_summaries.csv`. |
| `records/probe/` | Six validation JSONL files in the same format. Each holds four episodes on 40 held-out tasks for one model and environment. |
| `records/rounds.json.gz` | Per-iteration cold-start and GRPO results, keyed by run. Includes calls, injection and mixed-group counts, and the full group-type counts for the transfer-SFT comparison and its cap-0 control. |
| `records/reuse_comparisons.json` | SFT input sizes and token counts, ICL draws and prompt checks. Used with the episode and round records to produce `outputs/reuse_comparisons.json`. |
| `records/coverage_*.json` | Test coverage from the G1 driver and training coverage from step 05. Depth keys begin at zero. |
| `records/telecom_exploration.csv.gz` | Per-task episode and success counts from the eight telecom source shards and the three strong-target self-exploration seeds. |
| `records/failed_shard.json` | Recorded call bounds for the resampled math shard in seed 603. |
| `splits/` | Source-explored, source-solved and held-out probe task IDs. The benchmark defines the official telecom test split. |
| `source/{telecom,math}/4b-inst.shard*.db.gz` | The exact source stores used in the paper, including unsuccessful rollouts and their simulated actions and observations. Step 01 restores the stores while preserving their shard and trajectory order. |
| `results/` | Reference evaluation summaries, per-round curves, per-run charges, generated cold-start tables and the reuse probe. `table3_runs.csv` records partial and retried runs. The CPU command `python scripts/reproduce.py` recomputes the Table 2–4 scores and paired comparisons in `outputs/tables/coldstart.csv`. |

Telecom evaluations normally sample eight episodes per task. Math evaluations sample four. Pass@1 is the mean success rate per task. Pass@n counts tasks solved in at least one of their n attempts. Bootstrap intervals resample shared tasks and condition on the observed policies.

## External inputs for GPU runs

| Input | Expected location | Used for |
|---|---|---|
| τ²-bench at `5ade54e282a4555e4d6c52d6b9b861ea8872afa6` | `third_party/tau2-bench/`; installed by step 00 | Telecom tasks and tools |
| DAPO-Math integer problems packaged by rStar2-Agent | `data/external/math/train.jsonl` | Math training and test split |
| ToolACE conversations | `data/external/toolace.json` | Step 02 generates `data/external/common_sft.jsonl` for the weak telecom initialization |
| Qwen3-4B-Instruct-2507, Qwen3-8B, Qwen3-8B-Base, Qwen3-4B-Base | One directory per checkpoint in `$XPER_HOME/models/hf/` | Source, strong target and delimiter-initialized controls |
| Hermes-3-Llama-3.1-8B | `$XPER_HOME/models/hf/Hermes-3-Llama-3.1-8B` | Cross-family offline comparison |

See [datasets.json](datasets.json) for input sizes, SHA-256 values and model identifiers. Run `python tools/check_inputs.py` to inspect local inputs.
