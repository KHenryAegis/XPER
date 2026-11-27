"""Recompute the paper's measurements from the released episode and round records."""
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from source_records import published_successes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORDS = DATA / "records"
REF = DATA / "results"
OUT = ROOT / "outputs"


BOOT_SEED = 20261115
DRAWS = 20000

def bootstrap_arms(arrays, draws=DRAWS, seed=BOOT_SEED):
    """arrays[arm, seed, task]; shared task indices preserve pairing and clusters."""
    a = np.asarray(arrays, dtype=float)
    if a.ndim != 3 or not np.isfinite(a).all() or min(a.shape) == 0:
        raise ValueError("expected finite arm x seed x task data")
    per_task = a.mean(axis=1)
    rng = np.random.default_rng(seed)
    out = np.empty((a.shape[0], draws))
    for start in range(0, draws, 512):
        ids = rng.integers(0, a.shape[2], size=(min(512, draws - start), a.shape[2]))
        out[:, start:start + len(ids)] = per_task[:, ids].mean(axis=2)
    return out


def ci(values):
    return np.quantile(values, [.025, .975]).tolist()


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


SEEDS = (601, 602, 603)
A = [f"xs-grpo-tel-s{seed}" for seed in SEEDS]
B = [f"8b-instruct-icl6-mobile-s{seed}" for seed in (0, 1, 2)]
MIXED = ["8b-instruct-icl6", "8b-instruct-icl6-s1", "8b-instruct-icl6-s2"]
C = {"zero-shot": "hermes8b-start", "ICL": "hermes8b-icl6", "SFT": "hermes8b-xfer"}
MATH = ["sc-math-xfer", *(f"sc-math-real-s{seed}" for seed in SEEDS)]
FAMILIES = {"mms_issue": 16, "mobile_data_issue": 9, "service_issue": 15}


def comparison(rates, candidates, references, family=None):
    names = [*candidates, *references]
    ids = set(rates[names[0]])
    if any(set(rates[name]) != ids for name in names):
        raise ValueError(f"unpaired task sets for {names}")
    if family is not None:
        ids = {task for task in ids if task.startswith(f"[{family}]")}
        if len(ids) != FAMILIES[family]:
            raise ValueError(f"wrong task count for {family}")
    ids = sorted(ids)
    estimates = np.stack([[mean(rates[name][task] for name in names) for task in ids]
                          for names in (candidates, references)])
    boot = bootstrap_arms(estimates[:, None, :])
    return dict(tasks=len(ids), candidate=float(estimates[0].mean()), reference=float(estimates[1].mean()),
                delta=float((estimates[0] - estimates[1]).mean()), paired_95_ci=ci(boot[0] - boot[1]))


def write_summary(out, rates, stats, rounds, metadata, start):
    refs = [f"td100-s{seed}" for seed in SEEDS]
    cap0 = [f"td0-s{seed}" for seed in SEEDS]
    run_stats = []
    for name, control in zip(A, cap0):
        history = rounds[name]
        calls = history[-1]["env_calls"]
        if calls < 44414 or (len(history) > 1 and history[-2]["env_calls"] >= 44414):
            raise ValueError(f"wrong budget boundary for {name}")
        if any(row["injected"] for row in history):
            raise ValueError(f"unexpected injection for {name}")
        for rows in (history, rounds[control]):
            if any(row["all_success_groups"] + row["all_fail_groups"] + row["mixed_groups"] != row["groups"]
                   for row in rows):
                raise ValueError(f"inconsistent group counts for {name} or {control}")
        run_stats.append(dict(run=name, eval=stats[name], target_calls=calls, rounds=len(history),
                              grpo_supervised_tokens=sum(row["supervised_tokens"] for row in history),
                              mean_mixed_groups=mean(row["mixed_groups"] for row in history),
                              matched_no_injection_mean_mixed_groups=mean(row["mixed_groups"] for row in rounds[control])))
    offline_tokens = metadata["transfer_sft"][start]["supervised_tokens"]
    start_rate = mean(rates[start].values())
    source_rate = mean(rates["qwen3-4b-inst"].values())
    zero = C["zero-shot"]
    scores = {arm: mean(rates[name].values()) for arm, name in C.items()}
    predicted = "SFT" if stats[zero][2] >= 20 else "ICL" if scores["zero-shot"] < source_rate else "zero-shot"
    result = {
        "A": dict(start_pass1=start_rate, runs=run_stats, seed_sd_pct=stdev(stats[name][0] for name in A),
                  vs_injection=comparison(rates, A, refs), vs_no_injection=comparison(rates, A, cap0),
                  vs_offline_start=comparison(rates, A, [start]), transfer_sft_tokens_per_pass=offline_tokens,
                  mean_two_pass_sft_plus_grpo_tokens=2 * offline_tokens + mean(r["grpo_supervised_tokens"] for r in run_stats),
                  mean_injection_supervised_tokens=mean(metadata["capped_injection_supervised_tokens"].values())),
        "B": dict(overall=comparison(rates, B, MIXED),
                  families={f: comparison(rates, B, MIXED, family=f) for f in FAMILIES},
                  evaluations={name: stats[name] for name in B}),
        "C": dict(evaluations={arm: stats[name] for arm, name in C.items()}, predicted=predicted,
                  best_point_estimate=[arm for arm, score in scores.items() if score == max(scores.values())],
                  icl_vs_zero=comparison(rates, [C["ICL"]], [zero]),
                  sft_vs_zero=comparison(rates, [C["SFT"]], [zero]),
                  icl_vs_sft=comparison(rates, [C["ICL"]], [C["SFT"]])),
        "math": dict(transfer_sft_vs_injection=comparison(rates, MATH[:1], MATH[1:]),
                     transfer_sft_training_target_calls=0),
        "protocol_metadata": metadata,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote the reuse comparisons to {out}")


FMT_THRESHOLD = 0.20  # fixed before the validation results were read
TEST_BEST = {"telecom weak": "transfer SFT", "telecom strong": "transfer ICL",
             "math weak": "transfer SFT", "math strong": "no reuse"}


def decide(tgt, src):
    """tgt/src: lists of episodes; the paper's rule with the pre-registered format threshold."""
    bad = sum(e[2] for e in tgt) / max(1, sum(e[1] + e[2] for e in tgt))
    if bad >= FMT_THRESHOLD:
        return "transfer SFT"
    rate = lambda eps: sum(e[0] for e in eps) / len(eps)
    return "transfer ICL" if rate(tgt) < rate(src) else "no reuse"


def probe_rows(episodes, draws=2000):
    rows, extra = [], {}
    for env in ("telecom", "math"):
        S = episodes(f"e3-{'tel' if env == 'telecom' else 'math'}-src")
        for start in ("weak", "strong"):
            T = episodes(f"e3-{'tel' if env == 'telecom' else 'math'}-{start}")
            tasks = sorted(T)
            assert sorted(S) == tasks and len(tasks) == 40 and all(len(T[t]) == len(S[t]) == 4 for t in tasks)
            allT, allS = [e for t in tasks for e in T[t]], [e for t in tasks for e in S[t]]
            full = decide(allT, allS)
            rate = lambda eps: sum(e[0] for e in eps) / len(eps)
            setting = f"{env} {start}"
            row = dict(setting=setting, validation_gap=round(100 * (rate(allT) - rate(allS)), 2),
                       format_error_rate=round(sum(e[2] for e in allT) / sum(e[1] + e[2] for e in allT), 4),
                       decision=full, matches_test_best=full == TEST_BEST[setting],
                       target_calls=sum(e[3] for e in allT), source_calls=sum(e[3] for e in allS),
                       validation_task_ids_file="data/splits/val_probe_tasks.json")
            rng = random.Random(BOOT_SEED)
            extra[setting] = {"test_best": TEST_BEST[setting], "target_pass1": round(100 * rate(allT), 2),
                              "source_pass1": round(100 * rate(allS), 2)}
            for k in (1, 2, 4):
                # tab:probe method: k of the 4 episodes per task (both models, without replacement), tasks fixed
                agree = sum(decide([e for t in tasks for e in rng.sample(T[t], k)],
                                   [e for t in tasks for e in rng.sample(S[t], k)]) == full for _ in range(draws))
                row[f"agree_k{k}"] = round(agree / draws, 4)
                # stricter variant (summary only): tasks resampled with replacement as well
                agree = 0
                for _ in range(draws):
                    drawn = [rng.choice(tasks) for _ in tasks]
                    agree += decide([e for t in drawn for e in rng.sample(T[t], k)],
                                    [e for t in drawn for e in rng.sample(S[t], k)]) == full
                extra[setting][f"agree_task_bootstrap_k{k}"] = round(agree / draws, 4)
            rows.append(row)
    return rows, extra


def episodes(group, run, tasks, samples):
    by_task = defaultdict(dict)
    with gzip.open(RECORDS / group / (run + ".jsonl.gz"), "rt") as f:
        for line in f:
            row = json.loads(line)
            task, k = row["task"], row["k"]
            if k in by_task[task] or row["reward"] not in (0, 1):
                raise ValueError(f"{run}: duplicate attempt or invalid reward for {task}")
            by_task[task][k] = row
    if len(by_task) != tasks or any(set(v) != set(range(samples)) for v in by_task.values()):
        raise ValueError(f"{run}: incomplete evaluation")
    return {t: [v[i] for i in range(samples)] for t, v in sorted(by_task.items())}


def write_table(name, rows, fields=None):
    write_csv(OUT / "tables" / (name + ".csv"), rows, fields or list(rows[0]))


def load_evals():
    rates, stats = {}, {}
    with (REF / "eval_summaries.csv").open() as f:
        for ref in csv.DictReader(f):
            run = ref["run"]
            if run in rates:
                raise ValueError(f"duplicate run in evaluation manifest: {run}")
            per = episodes("eval", run, int(ref["tasks"]), int(ref["samples"]))
            rates[run] = {t: mean(e["reward"] for e in v) for t, v in per.items()}
            values = list(rates[run].values())
            rows = [e for group in per.values() for e in group]
            bad = sum(e["bad_format"] for e in rows)
            steps = sum(e["steps"] for e in rows)
            p1, pn = mean(values), mean(x > 0 for x in values)
            fmt = bad / max(1, steps + bad)
            stats[run] = (100*p1, 100*pn, 100*fmt)
            # rollout.py rounds probabilities to four decimal places in its final log line.
            pct = lambda x: round(100 * round(x, 4), 2)
            measured = dict(pass1_pct=pct(p1), passn_pct=pct(pn), format_error_pct=pct(fmt),
                            mixed_tasks=sum(0 < x < 1 for x in values), env_calls=sum(e["env_calls"] for e in rows),
                            mean_steps=round(steps / len(rows), 2))
            for key, val in measured.items():
                if val != float(ref[key]):
                    raise ValueError(f"{run}: episode {key}={val}, log {ref[key]}")
    files = {p.name.removesuffix(".jsonl.gz") for p in (RECORDS / "eval").glob("*.jsonl.gz")}
    if files != set(rates):
        raise ValueError(f"unmatched evaluation records: {sorted(files ^ set(rates))}")
    print(f"Verified episode outcomes, calls and format errors for {len(stats)} evaluations.")
    return rates, stats


def seeds(prefix, nums, suffix=""):
    return [f"{prefix}-s{i}{suffix}" for i in nums]


SFT = range(401, 404)
RFT = range(501, 503)
RL = range(601, 604)
STRONG = [
    ["8b-instruct-start-nothink"], seeds("8b-instruct-selfexp", SFT, "-nothink"),
    seeds("rft-instruct", RFT), seeds("grpo-instruct", RL),
    seeds("8b-instruct-xfer", SFT, "-nothink"), ["8b-instruct-icl6", "8b-instruct-icl6-s1", "8b-instruct-icl6-s2"],
]
METHODS = ["Zero-shot", "Self-SFT", "Iterated RFT", "GRPO", "Transfer SFT", "Transfer ICL"]
SETTINGS = {
    "Telecom weak": [["8b-commonsft"], ["8b-selfexp"], seeds("rft-commonsft", RFT), ["grpo-commonsft-s601"],
                     ["8b-xfer"], ["8b-commonsft-icl6"]],
    "Telecom strong": STRONG,
    "Math weak": [["math-8b-base-eos"], ["math-8b-base-selfexp"], [], [], ["math-8b-base-xfer"], []],
    "Math strong": [["math-8b-instruct"], ["math-8b-selfexp"], seeds("math-rft-instruct", RFT),
                    seeds("math-grpo-instruct", RL), ["math-8b-xfer"], ["math-8b-icl4"]],
}


def tables(stats, rounds):
    starts = ["qwen3-4b-inst", STRONG[0][0], "8b-commonsft", "math-4b-inst", "math-8b-instruct", "math-8b-base-eos"]
    write_table("table2_starts", [dict(run=r, pass1=stats[r][0], passn=stats[r][1], format_error=stats[r][2]) for r in starts])
    grid, gains = [], []
    for setting, groups in SETTINGS.items():
        p1 = {}
        for method, names in zip(METHODS, groups):
            if not names:
                continue
            values = [stats[r][0] for r in names]
            p1[method] = mean(values)
            grid.append(dict(setting=setting, method=method, runs=len(names), pass1=mean(values),
                             sd=stdev(values) if len(values)>1 else "", passn=mean(stats[r][1] for r in names)))
        for method in ("Transfer SFT", "Transfer ICL", "GRPO"):
            if method in p1:
                gains.append(dict(setting=setting, method=method, gain_pp=p1[method]-p1["Zero-shot"]))
    write_table("table6_reuse", grid)
    write_table("table9_gains", gains)
    write_table("table8_permutation", [dict(size=size, untrained=stats["init-eos-"+size][0],
                permuted=stats["fmtctl-"+size][0], real=stats["bc3-"+size][0]) for size in ("8b","4b")])
    groups, gaps, initializations = [], [], []
    for start, names in [("Weak", ["grpo-commonsft-s601"]), ("Zero-shot", STRONG[3]),
                         ("Transfer SFT", seeds("grpo-xferstart", (601,602)))]:
        histories = [rounds[r] for r in names]
        mixed = mean(mean(h["mixed_groups"] for h in v) for v in histories)
        totals = {h["groups"] for v in histories for h in v}
        if len(totals) != 1:
            raise ValueError(f"group size changed for {start}")
        groups.append([round(mixed,1), totals.pop()])
        if start != "Weak":
            base = STRONG[0 if start=="Zero-shot" else 4][0]
            initializations.append(dict(start=start, initial_pass1=stats[base][0],
                final_pass1=mean(stats[r][0] for r in names), sd=stdev(stats[r][0] for r in names),
                final_pass8=mean(stats[r][1] for r in names), mixed=mixed))
            gaps.append(round(stats[base][1]-stats[base][0],1))
    gaps.append(round(mean(stats[r][1]-stats[r][0] for r in seeds("grpo-xferstart",(601,602))),1))
    write_table("table1_starts", initializations)
    (OUT / "observations.json").write_text(json.dumps(dict(groups=groups,gaps=gaps),indent=2)+"\n")
    scaling = []
    for setting, prefix, full in [("Strong","8b-xfer-bare",STRONG[4]),
                                  ("Delimiter-initialized","8b-base-xfer-weak",["bc3-8b","8b-base-xfer-weak-p100s1"])]:
        for n, names in [(56,[prefix+"-p25"]),(112,[prefix+"-p50"]),(167,[prefix+"-p75"]),(223,full)]:
            scaling.append(dict(start=setting, trajectories=n, runs=len(names), pass1=mean(stats[r][0] for r in names),
                                pass8=mean(stats[r][1] for r in names)))
    write_table("source_scaling", scaling)


def families(rates):
    split = json.loads((DATA / "splits/telecom_extra_splits.json").read_text())
    solved = published_successes("telecom")
    tasks = set(rates["qwen3-4b-inst"])
    labels = {"mms_issue":"MMS","mobile_data_issue":"Mobile data","service_issue":"Service"}
    def family(t):
        return labels[t[1:t.index("]")]]
    if set(solved) != set(split["train_src_solved"]) or tasks.intersection(split["train_src_explored"]):
        raise ValueError("training and held-out task sets are inconsistent")
    rows = []
    for label, ids in [("Source training tasks explored",split["train_src_explored"]),("Source training tasks solved",set(solved)),
                       ("Successful source trajectories",solved),("Held-out test tasks",tasks)]:
        rows.append({"row":label,**Counter(family(t) for t in ids)})
    for label, names in [("Weak target, no injection",seeds("td0",RL)),("Weak target, capped injection",seeds("td100",RL)),
                         ("Source, zero-shot",["qwen3-4b-inst"]),*zip(METHODS,STRONG),
                         ("Transfer ICL (mobile-data only)", B)]:
        if any(set(rates[r]) != tasks for r in names):
            raise ValueError(f"{label}: test tasks differ")
        places = 2 if label.startswith("Weak") else 1
        rows.append({"row":label,**{f:f"{100*mean(rates[r][t] for r in names for t in tasks if family(t)==f):.{places}f}"
                                      for f in labels.values()}})
    strong_runs = dict(zip(METHODS, STRONG))
    contrast = {"row": "Transfer ICL minus GRPO (pp)"}
    interval = {"row": "Paired 95% task-bootstrap interval (pp)"}
    for name in labels.values():
        ids = sorted(t for t in tasks if family(t) == name)
        arms = np.array([[[100 * rates[run][t] for t in ids] for run in strong_runs[method]]
                         for method in ("Transfer ICL", "GRPO")])
        icl, grpo = bootstrap_arms(arms)
        low, high = ci(icl - grpo)
        contrast[name] = f"{arms[0].mean() - arms[1].mean():+.1f}"
        interval[name] = f"[{low:+.1f}, {high:+.1f}]"
    rows.extend([contrast, interval])
    write_table("table7_families", rows, ["row", *labels.values()])


def coldstart(rates, stats, rounds, metadata, run_charges):
    """Match each comparison to the same completed rollout seeds on both arms."""
    failed = json.loads((RECORDS / "failed_shard.json").read_text())
    rows = []

    def add(table, label, names, references=(), experiment=None):
        scores = [stats[name] for name in names]
        row = dict(table=table, arm=label, runs=" / ".join(names),
                   pass1_pct=mean(s[0] for s in scores), sd_pct=stdev(s[0] for s in scores),
                   passn_pct=mean(s[1] for s in scores), per_seed_pass1_pct=[s[0] for s in scores])
        if references:
            result = comparison(rates, names, references)
            row.update(delta_pp=100 * result["delta"],
                       ci95_pp=[100 * bound for bound in result["paired_95_ci"]])
        if experiment:
            charges = [r for r in run_charges if r["exp_id"] == experiment]
            if [int(r["seed"]) for r in charges] != list(RL):
                raise ValueError(f"incomplete charge records for {experiment}")
            row.update(charge_seeds=[int(r["seed"]) for r in charges],
                       target_calls=[int(r["target_calls_charged"]) for r in charges],
                       exposures=[int(r["injections_or_exposures"]) for r in charges])
        else:
            calls = [rounds[name][-1]["env_calls"] for name in names]
            if "sc-math-real-s603" in names:
                i = names.index("sc-math-real-s603")
                if calls[i] != failed["successful_rollout_calls"]:
                    raise ValueError("failed shard disagrees with recorded charges")
                calls[i] = failed["total_calls_conservatively_charged"]
            exposures = [sum(r["injected"] for r in rounds[name]) for name in names]
            if names == A:
                sft = metadata["transfer_sft"]["8b-xfer-rebuilt"]
                exposures = [int(sft["epochs"] * sft["trajectories"])] * len(names)
            row.update(charge_seeds=[int(name.rsplit("-s", 1)[1]) for name in names],
                       target_calls=calls, exposures=exposures)
        rows.append(row)

    for env, base, real in [("Telecom", "td0", "td100"), ("Math", "sc-math-none", "sc-math-real")]:
        controls = seeds(base, RL)
        add(2, env + " no injection", controls)
        add(2, env + " real source", seeds(real, RL), controls)
    # The double-budget comparison excludes the unfinished seed 602 on both arms.
    add(2, "No injection, 88.8k", ["e2-tel-r3-s601", "e2-tel-s603"], seeds("td100", (601, 603)), experiment="E2")
    for label, names, experiment in [
        ("Transfer SFT then GRPO", A, None),
        ("LUFFY-style", ["e4b-tel-r2-s601", "e4b-tel-s602", "e4b-tel-r2-s603"], "E4b"),
        ("ReLIFT-style", ["e4a-tel-s601", "e4a-tel-r2-s602", "e4a-tel-s603"], "E4a"),
        ("Self-replay", ["e5-tel-s601", "e5-tel-r2-s602", "e5-tel-r2-s603"], "E5"),
    ]:
        add(2, label, names, seeds("td100", RL), experiment)
    two = (601, 602)
    for cap in (0, 5, 20, 100):
        add(3, f"Cap {cap}", seeds(f"td{cap}", two))
    controls = seeds("td0", two)
    add(4, "No injection", controls)
    for label, prefix in [("Real source", "td100"), ("Masked observations", "sc-tel-obs_mask"),
                          ("Random schema", "sc-tel-schema_random")]:
        add(4, label, seeds(prefix, two), controls)
    write_table("coldstart", rows, ["table", "arm", "runs", "pass1_pct", "sd_pct", "passn_pct",
                                    "per_seed_pass1_pct", "delta_pp", "ci95_pp", "charge_seeds", "target_calls", "exposures"])


def overlap(rates):
    rows = list(csv.DictReader(gzip.open(RECORDS / "telecom_exploration.csv.gz","rt")))
    shared = [r for r in rows if int(r["source_episodes"]) and int(r["target_episodes"])]
    totals = {f:sum(int(r[f]) for r in shared) for f in ("source_successes","target_successes","source_episodes","target_episodes")}
    source, target = rates["math-4b-inst"],rates["math-8b-instruct"]
    if set(source) != set(target):
        raise ValueError("math test split differs")
    summary = dict(telecom_shared_tasks=len(shared),**totals,
                   source_pct=100*totals["source_successes"]/totals["source_episodes"],
                   target_pct=100*totals["target_successes"]/totals["target_episodes"],
                   math_source_only_tasks=sum(source[t]>0 and target[t]==0 for t in source))
    (OUT / "task_overlap.json").write_text(json.dumps(summary,indent=2)+"\n")


def main():
    (OUT / "tables").mkdir(parents=True,exist_ok=True)
    rates, stats = load_evals()
    with gzip.open(RECORDS / "rounds.json.gz","rt") as f:
        rounds = json.load(f)
    tables(stats,rounds)
    families(rates)
    reuse_metadata = json.loads((RECORDS / "reuse_comparisons.json").read_text())
    with (REF / "table3_runs.csv").open() as f:
        run_charges = list(csv.DictReader(f))
    coldstart(rates, stats, rounds, reuse_metadata, run_charges)
    reuse_metadata["capped_injection_supervised_tokens"] = {
        f"td100-s{row['seed']}": int(row["supervised_tokens_completed"])
        for row in run_charges
        if (row["exp_id"], row["env"], row["arm"]) == ("REF", "telecom", "Real source")
    }
    write_summary(OUT / "reuse_comparisons.json", rates, stats, rounds, reuse_metadata,
                                start="8b-xfer-rebuilt")
    overlap(rates)
    split = json.loads((DATA / "splits/val_probe_tasks.json").read_text())

    def probe(name):
        per = episodes("probe", name, 40, 4)
        env = "telecom" if name.startswith("e3-tel-") else "math"
        if set(per) != set(split[env]):
            raise ValueError(f"{name}: unexpected validation split")
        return {t:[(e["reward"],e["steps"],e["bad_format"],e["env_calls"]) for e in v] for t,v in per.items()}
    rows,_ = probe_rows(episodes=probe)
    with (REF / "table10_probe.csv").open() as f:
        fields = next(csv.reader(f))
    write_csv(OUT / "table10_probe.csv", rows, fields)


if __name__ == "__main__":
    main()
