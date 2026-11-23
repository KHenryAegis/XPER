"""Recompute the paper's measurements from the released episode and round records."""
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

import analyze_coldstart as core
import analyze_paper as paper
from analyze_alternatives import NEW, alternative_rows, probe_rows, write_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORDS = DATA / "records"
REF = DATA / "results"
OUT = ROOT / "outputs"


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
    solved = [json.loads(l)["task_id"] for l in (DATA / "source/source_successes_telecom.jsonl").read_text().splitlines()]
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
                         ("Source, zero-shot",["qwen3-4b-inst"]),*zip(METHODS,STRONG)]:
        if any(set(rates[r]) != tasks for r in names):
            raise ValueError(f"{label}: test tasks differ")
        places = 2 if label.startswith("Weak") else 1
        rows.append({"row":label,**{f:f"{100*mean(rates[r][t] for r in names for t in tasks if family(t)==f):.{places}f}"
                                      for f in labels.values()}})
    write_table("table7_families",rows,["row",*labels.values()])


def injection(rates, rounds):
    arrays, arms, matched = {}, {}, {}
    accounting = json.loads((RECORDS / "failed_shard.json").read_text())
    sets = {}
    for key, (env,label,tag) in core.ARMS.items():
        runs, vals = [], []
        for seed in RL if key in paper.MAIN else (601,602):
            name = f"{tag}-s{seed}"
            ids = sorted(rates[name])
            if sets.setdefault(env,ids) != ids:
                raise ValueError(f"{name}: unpaired task set")
            v = np.array([rates[name][t] for t in ids])
            vals.append(v)
            h = rounds[name]
            run = dict(pass1=float(v.mean()), passk=float((v>0).mean()), history=h,
                       source_injections=sum(x["injected"] for x in h), target_calls=h[-1]["env_calls"])
            if name=="sc-math-real-s603":
                if run["target_calls"] != accounting["successful_rollout_calls"]:
                    raise ValueError("failed-shard accounting does not match round records")
                run.update(budget_accounting=accounting,target_calls_charged=accounting["total_calls_conservatively_charged"])
            runs.append(run)
        arrays[key] = np.array(vals)
        arms[key] = dict(environment=env,label=label,runs=runs,**paper.summarize(runs))
        matched[key] = paper.summarize(runs[:2])

    def compare(key, ref, n):
        a,b = arrays[key][:n],arrays[ref][:n]
        boot = core.bootstrap_arms(np.stack([a,b]))
        return dict(mean_difference=float((a-b).mean()),task_bootstrap_ci95=core.ci(boot[0]-boot[1]))
    main = {env:compare(prefix+"_real",prefix+"_none",3) for env,prefix in [("telecom","tel"),("math","math")]}
    controls = {k+"_vs_tel_none":compare(k,"tel_none",2) for k in ("tel_real","tel_mask","tel_schema")}
    data = dict(arms=arms, main_comparisons=main, matched_two_seed=dict(arms=matched,comparisons=controls))
    paper.write_tables(data)
    paper.plot(data)


def export_task_counts(rates):
    """Export the Table 3 task counts from the same episodes used for scoring."""
    runs = [("REF", env, label, seed, f"{tag}-s{seed}")
            for key, (env, label, tag) in core.ARMS.items() if key in paper.MAIN
            for seed in RL]
    for exp, label, _, spec in NEW:
        runs.extend((exp, "telecom", label, seed, run)
                    for seed, (run, status, _) in spec.items() if status != "partial")
    fields = ["exp_id", "env", "arm", "seed", "task_id", "n_success", "n_samples"]
    with (OUT / "table3_per_task.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for exp, env, label, seed, run in runs:
            samples = 8 if env == "telecom" else 4
            for task, rate in rates[run].items():
                writer.writerow([exp, env, label, seed, task, round(rate * samples), samples])


def alternatives(rates, stats):
    def ordered(name):
        tasks = sorted(rates[name])
        return tasks, np.array([rates[name][t] for t in tasks])
    refs = {(key, seed):ordered(f"{tag}-s{seed}") for key,tag in [("tel_none","td0"),("tel_real","td100")]
            for seed in RL}
    names = {(exp,seed):name for exp,_,_,spec in NEW for seed,(name,status,_) in spec.items() if status != "partial"}
    values = {k:ordered(v) for k,v in names.items()}
    if any(ids != refs["tel_real",601][0] for ids,_ in values.values()):
        raise ValueError("unpaired tasks across alternative arms")
    with (REF / "table3_runs.csv").open() as f:
        runs = [r for r in csv.DictReader(f) if r["exp_id"] != "REF"]
    for r in runs:
        r["seed"] = int(r["seed"])
        for field in ("target_calls_charged","injections_or_exposures"):
            r[field] = int(r[field])
        key = (r["exp_id"],r["seed"])
        if key in names:
            r["test_pass1"], r["test_passn"] = [round(x,4) for x in stats[names[key]][:2]]
    alternative_rows(OUT,runs,refs,values)


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
    export_task_counts(rates)
    injection(rates,rounds)
    alternatives(rates,stats)
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
