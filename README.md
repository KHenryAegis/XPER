# XPER

Code and data for *Explore Once, Reuse Across Models? When Stored Agent Experience Pays Off for a New Model*. XPER measures how a new model uses a source agent’s stored trajectories on τ²-bench telecom tickets and integer-answer math problems with a Python tool.

## Reproduce the experiments on GPUs

Run the commands below from the checkout that contains `experiments/`. The paper used one Linux node with eight NVIDIA H20 GPUs (96 GB each), Python 3.12, PyTorch 2.8.0, CUDA 12.8 and vLLM 0.11.0. Several drivers select devices 0–7 explicitly. Reserve the node for the run. The math sandbox uses Linux user namespaces and libseccomp. A four-GPU weak-telecom GRPO run with a budget of about 44,400 target calls took 10.5–11.5 hours per seed.

### 1. Prepare the environment and inputs

Use a fresh checkout and a separate checkpoint directory for each full rerun. The drivers use fixed output names for each arm and seed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export XPER_PYTHON="$(command -v python)"
export XPER_HOME=/path/to/experiment-home
mkdir -p "$XPER_HOME/models/hf" "$XPER_HOME/models/xper" data/external/math
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
nvidia-smi -L
```

Place each Hugging Face checkpoint under `$XPER_HOME/models/hf/`, using its model name as the directory name. The four required checkpoints are `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-8B-Base` and `Qwen/Qwen3-4B-Base`. Place the rStar2-Agent integer-answer math file at `data/external/math/train.jsonl` and ToolACE at `data/external/toolace.json`. Step 00 installs the pinned τ²-bench checkout, checks the inputs and initializes the Base checkpoints. Dataset checksums and model index hashes are in [data/datasets.json](data/datasets.json).

### 2. Run the experiments

```bash
bash experiments/run_gpu.sh
```

The launcher executes the steps below in order and stops when a step returns an error. To see the execution order without starting jobs, run `bash experiments/run_gpu.sh --list`. To reproduce an individual experiment, run its script after its prerequisites.

| Command | Experiment and paper result | Prerequisites |
|---|---|---|
| `bash experiments/00_setup.sh` | Benchmark setup and delimiter initialization | Environment, datasets and four models |
| `bash experiments/01_source_logs.sh` | Collect the telecom and math source trajectories used by later arms | 00 |
| `bash experiments/02_telecom_weak.sh` | Weak telecom initialization, self-exploration and transfer SFT (Tables 2 and 6) | 00–01 |
| `bash experiments/03_telecom_strong.sh` | Strong telecom SFT, ICL, RFT and GRPO; weak-target ICL and test replay coverage (Tables 1, 2, 6 and 7; Figure 2) | 00–02 |
| `bash experiments/04_math.sh` | Weak and strong math starts, self-exploration and offline reuse (Tables 2 and 6) | 00–01 |
| `bash experiments/05_exact_replay.sh` | Compute prefix coverage from newly collected training stores (Figure 2a) | 01–04 |
| `bash experiments/06_injection_cohort.sh` | Injection caps, matched information controls and third-seed runs (Tables 3–5, weak rows of Table 7, Figure 4) | 00–02 |
| `bash experiments/07_alternatives_probe.sh` | Double-budget GRPO, interleaved SFT, mixed-policy training, self-replay and validation probe (Tables 3 and 10) | 00–02 |
| `bash experiments/08_permutation.sh` | Real vs. permuted trajectories on both Base sizes (Table 8) | 00–01 |
| `bash experiments/08_source_scaling.sh` | Source-data scaling at 56, 112 and 167 trajectories, plus the second full-data weak seed | 00–01; full-data comparisons use 03 and 08 permutation |

Table 9 summarizes the measured offline arms of Table 6. Figures 1 and 3 are diagrams. The [experiment index](experiments/README.md) maps the table cells to run IDs, budgets and seeds.

### 3. Read your run's results

Each evaluation writes a SQLite store, episode JSONL and a log in `results/eval/`, `results/math/` or `results/g1/`. Training histories and checkpoints are under `results/`, `logs/queue/` and `$XPER_HOME/models/xper/`. Step 05 writes training replay coverage to `results/coverage/`. Step 07 writes validation episodes to `results/probe/`.

After the final experiment, the launcher exports **your evaluation logs** to `outputs/gpu/eval_summaries.csv` and recomputes the validation probe in `outputs/gpu/table10_probe.csv` from your six probe stores. The completeness check uses the published run IDs; the scores come from your logs. For a partial run, export the completed evaluations with:

```bash
python tools/export_eval_summaries.py --root "$PWD" --out outputs/gpu/partial_evals.csv
```

## Optional: inspect the published runs

From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
bash experiments/09_tables_figures.sh
```

This command uses the included records and writes to `outputs/`. It needs neither model weights nor a dataset download. It checks 46,560 held-out episodes from 93 evaluations against the original log summaries, including task and sample counts, pass@1, pass@n, environment calls and format errors. It also reruns the Table 3 paired task bootstrap and the Table 10 decision probe. A failed reference comparison returns a nonzero exit status.

| Paper result | Inputs and what is computed | Output under `outputs/` |
|---|---|---|
| Tables 1–2 | Episode outcomes for each initialization and per-iteration GRPO group counts | `tables/table1_starts.csv`, `tables/table2_starts.csv` |
| Tables 3–5 | Paired evaluation tasks, training histories and recorded budget charges. The main comparison uses three policies. The dose and information controls use the same two policies as their controls. Generated LaTeX is compared to the included reference. | `tables/coldstart_{results,dose,controls}.tex` |
| Table 6 | Per-seed success and task coverage for each measured offline and online arm. SD is across training seeds. | `tables/table6_reuse.csv` |
| Table 7 | Source training split and successes, plus held-out outcomes for the weak and strong targets grouped by ticket family | `tables/table7_families.csv` |
| Tables 8–9 | Permutation control, and gains over the starting model for the settings in the decision guide | `tables/table8_permutation.csv`, `tables/table9_gains.csv` |
| Table 10 | Six validation sets, each with 40 tasks × 4 episodes. Resampling uses the paper’s fixed seed and 2,000 draws. The output is compared with the reference CSV. | `table10_probe.csv` |
| Figures 1–4 | Figures 1 and 3 are diagrams. Figure 2(a) uses archived prefix-coverage summaries. Figures 2(b,c) and 4 use evaluation outcomes and training rounds. | Six PDFs in `figures/`, including the three panels of Figure 2 |
| Source-data scaling and task overlap | Transfer SFT at 56, 112, 167 and 223 trajectories, the source and target success rates on 493 shared telecom tasks, and math tasks solved only by the source | `tables/source_scaling.csv`, `task_overlap.json` |

The weak math starting-model evaluation in Tables 2 and 6 covers the original 100-problem pilot. The weak math injection cohort uses a separate 300-problem evaluation. In Table 3 the double-budget arm has two completed seeds. Its unfinished third seed contributes a recorded cost and has no measured test accuracy.

### What needs the original trajectories?

The included records support independent recalculation of held-out scores. They retain episode outcomes and counts; they omit most individual tool requests and responses. Figure 2(a) can be redrawn from the included depth summaries. Rebuilding its prefix trie requires complete source and target stores. The same limitation applies to auditing every training and retry call against SQLite and recounting Python exceptions and timeouts in the strong-math discussion. The package includes the recorded charge for the failed math shard: 10,820 successful-rollout calls and an upper bound of 416 additional calls.

## Tests and license

```bash
python -m pip install pytest jsonschema
python -m pytest -rs tests
```

Tests that need CUDA, external datasets, the benchmark checkout or Linux kernel features report their skip reason. The project code is distributed under the [MIT License](LICENSE). Third-party datasets, benchmark code and model weights retain their respective licenses.
