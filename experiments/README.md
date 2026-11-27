# Reproducing the XPER experiments

Run all commands from the repository root. The paper used one Linux node with eight NVIDIA H20 GPUs (96 GB each), Python 3.12, PyTorch 2.8.0, CUDA 12.8 and vLLM 0.11.0. Several drivers select devices 0–7 explicitly. Reserve the node for the run. The math sandbox uses Linux user namespaces and libseccomp.

## Prepare the environment and inputs

Use a fresh checkout and a separate checkpoint directory for each full rerun. The drivers use fixed output names for each arm and seed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export XPER_PYTHON="$(command -v python)"
export XPER_HOME=/path/to/experiment-home
mkdir -p "$XPER_HOME/models/hf" "$XPER_HOME/models/xper" data/external/math
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
"$XPER_PYTHON" -B tools/fetch_inputs.py
nvidia-smi -L
```

The fetch command downloads the four Qwen3 checkpoints and the Hermes 3 Llama 3.1 checkpoint under `$XPER_HOME/models/hf/`. It also fetches ToolACE when needed. The math split ships in the repository as `data/external/math/train.jsonl`. Step 00 installs the pinned τ²-bench checkout, verifies the inputs and initializes the Base checkpoints. Model revisions and input SHA-256 values are in [data/datasets.json](../data/datasets.json).

## Run on GPUs

```bash
bash experiments/run_gpu.sh
```

The launcher runs steps 00–08 and 10 in order and stops if a step fails. Step 01 restores the original source stores so every reuse arm sees the recorded trajectories in the same order. To rerun source collection, use `bash experiments/01_source_logs.sh` after step 00 in a clean results directory. To recompute the paper’s statistics from the released records, run `python scripts/reproduce.py` on CPU.

| Step | Script | Paper results | Hardware | Needs |
|---|---|---|---|---|
| 0 | `00_setup.sh` | τ²-bench checkout, input check, delimiter-initialized Base checkpoints | CPU | `data/external/` files |
| 1 | `01_source_logs.sh` | the stored source logs (§5.2) | 8 GPUs | 0 |
| 2 | `02_telecom_weak.sh` | Table 2 and Table 6, weak telecom start | 8 GPUs | 0, 1 |
| 3 | `03_telecom_strong.sh` | §3.3 (GRPO from transfer SFT); Table 2 (source, strong telecom); Tables 6–7, strong telecom; transfer ICL of both telecom starts; §5.1 overlap check | 8 GPUs | 0–2 |
| 4 | `04_math.sh` | Table 2 and Table 6, math | 8 GPUs | 0, 1 |
| 5 | `05_exact_replay.sh` | §3.1 and Figure 2(a), training tasks | CPU | 1–4 |
| 6 | `06_injection_cohort.sh` | Table 3 (upper block), Tables 4–5, Figure 4, §6.1 controls | 8 GPUs | 0–2 |
| 7 | `07_alternatives_probe.sh` | Table 3 (lower block), Table 8 | 8 GPUs | 0–2 |
| 8 | `08_permutation.sh` | §6.2, permutation control | 8 GPUs | 0, 1 |
| 8 | `08_source_scaling.sh` | §7.3, source-data scaling | 4 GPUs (devices 2–5) | 0–1; full-data reference points from 3 and 8 (permutation) |
| 10 | `10_reuse_comparisons.sh` | Table 3 (transfer SFT → GRPO row), Table 7 (mobile-data ICL row), Table 6 (Hermes-3 row), §6.1 (matched-cohort math transfer SFT) | Up to 8 GPUs | 0–2. Steps 3 and 6 supply the reference evaluations for a live comparison |

The §7.1 decision rule reads off Table 6. Result file names follow an earlier draft's table numbers and do not always match the paper: `table10_probe.csv` holds Table 8, while `table3_*` happens to hold Table 3. The numerical inputs to Figures 2(b,c) are in `outputs/observations.json` after CPU recomputation.

Step 10 accepts an individual experiment selector.

```bash
bash experiments/10_reuse_comparisons.sh A     # weak telecom Transfer SFT followed by GRPO, seeds 601–603
bash experiments/10_reuse_comparisons.sh B     # mobile-data-only six-example ICL, draws 0–2
bash experiments/10_reuse_comparisons.sh C     # Hermes zero-shot, six-example ICL, transfer SFT
bash experiments/10_reuse_comparisons.sh math  # same-cohort math transfer SFT
```

A uses the step 02 `8b-xfer/final` checkpoint and the 44,414-call stopping threshold. B derives its demonstration filter from the 24 source-solved mobile-data tasks. C verifies that the two ChatML delimiters each occupy one dedicated token and that Hermes uses the end delimiter as EOS. C uses the same 223 telecom source successes and two-epoch SFT recipe as step 02. The math comparison uses the step 00 delimiter initializer, three SFT ranks and strict tool limits for the held-out evaluation. B, C and math need steps 00–01. A also needs step 02.

## Read your results

Each evaluation writes a SQLite store, episode JSONL and a log in `results/eval/`, `results/math/` or `results/g1/`. Training histories and checkpoints are under `results/`, `logs/queue/` and `$XPER_HOME/models/xper/`. Step 05 writes training replay coverage to `results/coverage/`. Step 07 writes validation episodes to `results/probe/`.

After the final experiment, the launcher exports the rerun evaluation logs to `outputs/gpu/eval_summaries.csv`. The six validation stores are available in `results/probe/`. For a partial run, export the completed evaluations with the following command.

```bash
python tools/export_eval_summaries.py --root "$PWD" --out outputs/gpu/partial_evals.csv
```

## Where each number comes from

Every evaluation ends its log `results/{eval,math,g1}/<run>.log` with one JSON line (pass@1, pass@n, calls,
format-error rate); `data/results/eval_summaries.csv` collects these lines for 104 held-out evaluations, including seven scaling evaluations. Means and SDs are
over the listed seeds.

| Paper item | Runs |
|---|---|
| Table 2, telecom | `qwen3-4b-inst` (source), `8b-instruct-start-nothink` (strong), `8b-commonsft` (weak) |
| Table 2, math | `math-4b-inst`, `math-8b-instruct`, `math-8b-base-eos` (only the weak-start pilot uses 100 problems) |
| Table 6, telecom weak | zero-shot `8b-commonsft`; SFT `8b-selfexp`; RFT `rft-commonsft-s50{1,2}`; GRPO `grpo-commonsft-s601`; transfer SFT `8b-xfer`; ICL `8b-commonsft-icl6` |
| Table 6, telecom strong | zero-shot `8b-instruct-start-nothink`; SFT `8b-instruct-selfexp-s40{1,2,3}-nothink`; RFT `rft-instruct-s50{1,2}`; GRPO `grpo-instruct-s60{1,2,3}`; transfer SFT `8b-instruct-xfer-s40{1,2,3}-nothink`; ICL `8b-instruct-icl6`, `-icl6-s1`, `-icl6-s2` |
| Table 6, math weak | zero-shot `math-8b-base-eos`; SFT `math-8b-base-selfexp`; transfer SFT `math-8b-base-xfer` |
| Table 6, math strong | zero-shot `math-8b-instruct`; SFT `math-8b-selfexp`; RFT `math-rft-instruct-s50{1,2}`; GRPO `math-grpo-instruct-s60{1,2,3}`; transfer SFT `math-8b-xfer`; ICL `math-8b-icl4` |
| §6.3 text | ICL k=3 and k=12: `8b-instruct-icl3-s0`, `8b-instruct-icl12-s0`; ICL on transfer SFT: `8b-instruct-xfer-s40{1,2,3}-icl6` |
| §3.3 (GRPO from transfer SFT) | zero-shot start: `grpo-instruct-s60{1,2,3}`; transfer-SFT start (`8b-instruct-xfer-s401`): `grpo-xferstart-s60{1,2}` |
| Table 7 | source exploration and success splits, weak `td{0,100}-s60{1,2,3}` and the strong-telecom runs of Table 6, grouped by ticket family in `outputs/tables/table7_families.csv` after CPU recomputation |
| §6.2, permutation control | `init-eos-{8b,4b}`, `fmtctl-{8b,4b}` (permuted), `bc3-{8b,4b}` (real) |
| Tables 3–5, Figure 4 | `td{0,5,20,100}-s60{1,2}`, `td{0,100}-s603`, `sc-tel-{obs_mask,schema_random}-s60{1,2}`, `sc-math-{none,real}-s60{1,2,3}` |
| Table 3 (lower), Table 8 | `e2-tel*`, `e4a-tel*`, `e4b-tel*`, `e5-tel*`, `results/probe/e3-*` |
| §7.3, source-data scaling | `8b-xfer-bare-p{25,50,75}`, `8b-base-xfer-weak-p{25,50,75,100s1}`. Full-data strong baseline uses three transfer seeds. Full-data weak baseline combines `bc3-8b` and `8b-base-xfer-weak-p100s1`. |
| §6.1, uncapped controls | `tp-{mismatch,shuffle}-s60{1,2}` |
| Table 3, transfer SFT → GRPO | `xs-grpo-tel-s60{1,2,3}`. The recorded start is `8b-xfer-rebuilt`. A fresh run uses the `8b-xfer` evaluation from step 02. |
| Table 7 mobile-data ICL; Table 6 Hermes-3; §6.1 matched math | `8b-instruct-icl6-mobile-s{0,1,2}`, `hermes8b-{start,icl6,xfer}`, `sc-math-xfer` |

## Conventions

- **Evaluation:** temperature 1.0; telecom: 40 test tasks × 8 samples; math: 300 test problems × 4 samples.
- **Budgets:** in target environment calls. Self-exploration arms are matched to the source's 44,414 (telecom) and
  10,403 (math) calls. The cohorts of steps 6–7 charge the same target budgets.
- **Seeds:** 401–403 for self-exploration and transfer SFT, 501–502 for iterated RFT, 601–603 for GRPO and the
  cohorts, 20261117 for the probe split and draws.
- **Outputs:** rollout stores in `results/<tag>.s<seed>/` and `results/pipeline_*/`; evaluations in
  `results/eval/`, `results/math/`, `results/g1/`; queue logs in `logs/queue/`; checkpoints in `$XPER_HOME/models/xper/`.

## Differences from the original runs

- **GPU sets.** The paper's runs used the GPUs that were free on shared 8-GPU nodes. These scripts fix the GPU
  sets to the two halves of one node (`HALF_A`, `HALF_B` in `common.sh`). Where a driver shards sampling over
  its GPUs, the shard count changes which sampling seed each episode gets, not the procedure. The drivers whose
  shard count defines a dataset keep the paper's count: the telecom source log (8), the self-exploration SFT seeds
  (6), and the math pipelines (2 or 3).
- **Retries.** Several runs of step 7 failed on shared GPUs and were rerun under new tags (`e4a-tel-r2`,
  `e2-tel-r3`, ...). `scripts/jobs_spec.py` writes the attempt the paper reports for each seed, so a rerun produces
  the reported tags. A fresh run of E2 seed 602 may finish; the paper reports two completed seeds.
- **One repaired shard.** In math real-source seed 603, one final-round shard failed to persist a lone Unicode
  surrogate (§5.4). It was resampled once with the same policy and seeds by a one-off script that is not part of
  this release; the encoding fix is in the released code (`tests/test_unicode_actions.py`), and the compact records include its call bounds. Table 3 charges the conservative 11,236 calls (†).

## Measured run times (NVIDIA H20, 96 GB)

| Run | GPUs | Time |
|---|---|---|
| Cohort GRPO, weak telecom start, 44.4k calls (steps 6, 7) | 4 | 10.5–11.5 h |
| Same, 88.8k calls (E2, step 7) | 4 | about 19 h |
| GRPO, strong telecom start or transfer-SFT start, 44.4k calls | 4–5 | about 3.5 h |
| Iterated RFT, telecom | 4–5 | 1.5–1.7 h |
| GRPO, math strong, 10.4k calls | 4 | about 4 h |
| Iterated RFT, math strong | 4 | about 1.7 h |

## Coverage of the bundled records

The included records support independent recalculation of held-out scores. The released source stores include their full simulated actions and observations. The target records retain episode outcomes and counts. Figure 2(a) can be redrawn from the included depth summaries. Rerun the relevant target rollout stages to rebuild the complete replay comparison and recount target-side retries and tool errors. The package records 10,820 successful-rollout calls and an upper bound of 416 additional calls for the failed math shard. Fresh GPU runs can vary with the hardware, kernels, and checkpoint reconstruction. The CPU command writes the Table 3–5 statistics to `outputs/tables/coldstart.csv` and the Table 8 probe to `outputs/table10_probe.csv`.
