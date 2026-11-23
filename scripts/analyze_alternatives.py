"""Evidence of the alternative arms and the reuse probe: Table 3 (lower block) and Table 10.

Inputs are opened read-only: eval DBs (results/eval), training stores and logs of the spec runs
(results/<tag>.s<seed>, logs/queue/spec_20261117), the audited reference cohort
(results/coldstart_analysis_20261116.json) and the E3 probe DBs (results/probe).
Writes to --out: table3_runs.csv, table3_per_task.csv, table3_curves.csv, table3_lower_rows.tex, table10_probe.csv,
alternatives_stats.json and alternatives_input_hashes.csv. Run IDs: E2, E3, E4a, E4b, E5
and REF, the reference arms of the injection cohort (Table 3, upper block).

Run: python -B scripts/analyze_alternatives.py          # writes outputs/
"""
import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_coldstart import BOOT_SEED, DRAWS, ITER, bootstrap_arms, ci, digest, load_eval, readonly

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "logs/queue/spec_20261117"
SRC_TEL_CALLS = 44414  # source collection cost of the stored telecom log (never charged to target budgets)
TEL = ("telecom", 40, 8)

# exp_id, arm label, tag, {seed: (run name, status, earlier failed attempts as (results dir, log))}
NEW = [
    ("E2", "No injection, total-cost budget 88.8k", "e2-tel", {
        601: ("e2-tel-r3-s601", "retried", [("e2-tel.s601", "e2-tel-s601"), ("e2-tel-r2.s601", "e2-tel-s601-r2")]),
        602: ("e2-tel-s602", "partial", []),
        603: ("e2-tel-s603", "complete", [])}),
    ("E4a", "ReLIFT-style interleaved SFT", "e4a-tel", {
        601: ("e4a-tel-s601", "complete", []),
        602: ("e4a-tel-r2-s602", "retried", [("e4a-tel.s602", "e4a-tel-s602")]),
        603: ("e4a-tel-s603", "complete", [])}),
    ("E4b", "LUFFY-style mixed-policy", "e4b-tel", {
        601: ("e4b-tel-r2-s601", "retried", [("e4b-tel.s601", "e4b-tel-s601")]),
        602: ("e4b-tel-s602", "complete", []),
        603: ("e4b-tel-r2-s603", "retried", [("e4b-tel.s603", "e4b-tel-s603")])}),
    ("E5", "Self-replay (own successes, cap 100)", "e5-tel", {
        601: ("e5-tel-s601", "complete", []),
        602: ("e5-tel-r2-s602", "retried", [("e5-tel.s602", "e5-tel-s602")]),
        603: ("e5-tel-r2-s603", "retried", [("e5-tel.s603", "e5-tel-s603")])}),
]
REF = {"tel_none": "td0", "tel_real": "td100", "math_none": "sc-math-none", "math_real": "sc-math-real"}
HASHED = []


def run_dir(name):  # run name "<tag>-s601" -> results/<tag>.s601
    tag, seed = name.rsplit("-s", 1)
    return ROOT / "results" / f"{tag}.s{seed}"


def store_calls(directory):
    """Env calls recorded in a run's rollout stores (all completed shards, including an aborted round)."""
    total = 0
    for db in sorted(directory.glob("it*.shard*.db")):
        with readonly(db) as c:
            total += c.execute("SELECT COALESCE(SUM(env_calls), 0) FROM trajectories").fetchone()[0]
        HASHED.append(db)
    return total


def history(log):
    rows, relift = [], {}
    text = log.read_text()
    for i, stats, calls in ITER.findall(text):
        rows.append(dict(round=int(i), calls=int(calls), **json.loads(stats)))
    for i, stats in re.findall(r"^\s*it (\d+) relift: (\{.*\})$", text, re.M):
        relift[int(i)] = json.loads(stats)["exposures"]
    if [r["round"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"missing rounds in {log}")
    return rows, relift


def supervised_tokens(directory):
    n = 0
    for path in list(directory.glob("it*.train.log")) + list(directory.glob("it*.sft.log")):
        for line in path.read_text().splitlines():
            if line.startswith('{"saved"'):
                n += json.loads(line)["supervised_tokens"]
    return n


def new_runs():
    """One row per seed of every new arm, with conservative accounting of failed attempts."""
    rows, curves, rates = [], [], {}
    for exp, label, tag, seeds in NEW:
        for seed, (name, status, failed) in seeds.items():
            directory = run_dir(name)
            log = Q / f"{name.replace(f'-r2-s{seed}', f'-s{seed}-r2').replace(f'-r3-s{seed}', f'-s{seed}-r3')}.log"
            hist, relift = history(log)
            HASHED.extend([log] + sorted(directory.glob("it*.log")))
            cum, inj = 0, []
            for r in hist:
                if exp == "E4a":
                    cum = relift.get(r["round"], cum)
                else:
                    cum += r["injected"] - r["injected_own"] if exp == "E4b" else r["injected_own"]
                inj.append(cum)
                curves.append(dict(exp_id=exp, env="telecom", arm=label, seed=seed, round=r["round"],
                                   cumulative_target_calls=r["calls"], train_success=r["mean_reward"],
                                   mixed_groups=r["mixed_groups"], groups=r["groups"], cumulative_injections=cum))
            notes = []
            for fdir, flog in failed:
                fh, _ = history(Q / f"{flog}.log")
                abort = re.findall(r"ABORT: (.*)", (Q / f"{flog}.log").read_text())
                notes.append(f"earlier attempt {flog} aborted ({abort[-1] if abort else 'no abort line'}) after "
                             f"{len(fh)} complete rounds; {store_calls(ROOT / 'results' / fdir)} target calls recorded "
                             f"in its stores (charged in addition, not counted in target_calls_charged)")
            if exp == "E4b":
                notes.append("exposure count is uncapped by design (official LUFFY recipe): 1 source success per group")
            if exp == "E5":
                possible = sum(r.get("own_possible", 0) for r in hist)
                notes.append(f"all-fail groups {sum(r.get('all_fail_groups', 0) for r in hist)}; "
                             f"groups with an own earlier success available {possible}")
            row = dict(exp_id=exp, env="telecom", start="Qwen3-8B-Base + ToolACE SFT (weak)", arm=label, seed=seed,
                       n_eval_samples="", test_pass1="", test_passn="",
                       target_calls_charged=store_calls(directory),
                       source_calls_charged=SRC_TEL_CALLS if exp in ("E4a", "E4b") else 0,
                       injections_or_exposures=inj[-1], rounds=len(hist),
                       supervised_tokens_completed=supervised_tokens(directory), status=status, notes="; ".join(notes))
            db = ROOT / "results/eval" / f"{name}.db"
            if status == "partial":
                row["notes"] = (f"stopped at round {len(hist)} by a rollout failure (GPU taken by another process); "
                                f"not retried, the spec minimum of 2 complete E2 seeds is met by 601 and 603")
            else:
                ids, rate, cov = load_eval(db, TEL[1], TEL[2], name)
                HASHED.append(db)
                rates[(exp, seed)] = (ids, rate)
                row.update(n_eval_samples=TEL[1] * TEL[2], test_pass1=round(100 * rate.mean(), 4),
                           test_passn=round(100 * cov.mean(), 4))
            rows.append(row)
    return rows, curves, rates


def reference_runs():
    d = json.loads((ROOT / "results/coldstart_analysis_20261116.json").read_text())
    rows, curves, rates = [], [], {}
    for key, tag in REF.items():
        arm = d["arms"][key]
        env, n, k = arm["environment"], arm["tasks"], arm["samples"]
        for r in arm["runs"]:
            db = ROOT / "results/eval" / f"{r['name']}.db"
            ids, rate, cov = load_eval(db, n, k, r["name"])
            HASHED.append(db)
            assert abs(rate.mean() - r["pass1"]) < 1e-9
            rates[(key, r["seed"])] = (ids, rate)
            real = key.endswith("real")
            rows.append(dict(exp_id="REF", env=env, start="weak", arm=arm["label"], seed=r["seed"],
                             test_pass1=round(100 * rate.mean(), 4), test_passn=round(100 * cov.mean(), 4),
                             n_eval_samples=n * k, target_calls_charged=r["target_calls"],
                             source_calls_charged=(SRC_TEL_CALLS if env == "telecom" else 10403) if real else 0,
                             injections_or_exposures=r["source_injections"], rounds=len(r["history"]),
                             supervised_tokens_completed=r["dataset_supervised_tokens"], status="complete",
                             notes="existing paper cohort (tab:injection), audited in results/coldstart_analysis_20261116.json"
                                   + ("; " + d["failure_accounting_note"] if r["name"] == "sc-math-real-s603" else "")))
            cum = 0
            for h in r["history"]:
                cum += h["injected"]
                curves.append(dict(exp_id="REF", env=env, arm=arm["label"], seed=r["seed"], round=h["iteration"],
                                   cumulative_target_calls=h["env_calls"], train_success=h["mean_reward"],
                                   mixed_groups=h["mixed_groups"], groups=h["groups"], cumulative_injections=cum))
    return rows, curves, rates


FMT_THRESHOLD = 0.20  # fixed before the validation results were read
TEST_BEST = {"telecom weak": "transfer SFT", "telecom strong": "transfer ICL",
             "math weak": "transfer SFT", "math strong": "no reuse"}


def probe_episodes(name):
    db = ROOT / "results/probe" / f"{name}.db"
    HASHED.append(db)
    per = {}
    with readonly(db) as c:
        for task, succ, steps, calls, meta in c.execute(
                "SELECT task_id, success, n_steps, env_calls, meta FROM trajectories"):
            per.setdefault(task, []).append((succ, steps, json.loads(meta)["bad_format"], calls))
    return per


def decide(tgt, src):
    """tgt/src: lists of episodes; the paper's rule with the pre-registered format threshold."""
    bad = sum(e[2] for e in tgt) / max(1, sum(e[1] + e[2] for e in tgt))
    if bad >= FMT_THRESHOLD:
        return "transfer SFT"
    rate = lambda eps: sum(e[0] for e in eps) / len(eps)
    return "transfer ICL" if rate(tgt) < rate(src) else "no reuse"


def probe_rows(draws=2000, episodes=probe_episodes):
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


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def alternative_rows(out, new, ref_rates, new_rates):
    # paired telecom comparisons: seed-averaged per-task success, shared task draws across arms
    arms = {"none": [ref_rates[("tel_none", s)][1] for s in (601, 602, 603)],
            "real": [ref_rates[("tel_real", s)][1] for s in (601, 602, 603)]}
    for exp, *_ in NEW:
        arms[exp] = [v[1] for (e, s), v in sorted(new_rates.items()) if e == exp]
    keys = list(arms)
    boot = bootstrap_arms(np.stack([np.mean(arms[k], axis=0)[None, :] for k in keys]))
    pct = lambda x: round(100 * float(x), 2)
    comp = {}
    for a, b in [("real", "none")] + [(e, "real") for e, *_ in NEW] + [("E2", "none")]:
        d = boot[keys.index(a)] - boot[keys.index(b)]
        lo, hi = ci(d)
        comp[f"{a}_vs_{b}"] = dict(mean_difference=pct(np.mean(arms[a]) - np.mean(arms[b])),
                                   ci95=[pct(lo), pct(hi)], seeds=[len(arms[a]), len(arms[b])])
    summary = {}
    for exp, label, tag, seeds in NEW:
        rs = [r for r in new if r["exp_id"] == exp]
        p1 = [r["test_pass1"] for r in rs if r["test_pass1"] != ""]
        summary[exp] = dict(label=label, per_seed_pass1={r["seed"]: r["test_pass1"] for r in rs},
                            mean_pass1=round(float(np.mean(p1)), 2), sd_pass1=round(float(np.std(p1, ddof=1)), 2),
                            mean_passn=round(float(np.mean([r["test_passn"] for r in rs if r["test_passn"] != ""])), 2),
                            target_calls={r["seed"]: r["target_calls_charged"] for r in rs},
                            exposures={r["seed"]: r["injections_or_exposures"] for r in rs},
                            status={r["seed"]: r["status"] for r in rs})
    lines = ["% Generated by scripts/analyze_alternatives.py. Delta: paired 95% task-bootstrap interval "
             f"({DRAWS} draws, seed {BOOT_SEED}) against real-source injection (cap 100), in points."]
    for exp, *_ in NEW:
        s, c = summary[exp], comp[f"{exp}_vs_real"]
        calls = " / ".join(f"{s['target_calls'][x] / 1000:.2f}" + ("$^\\ddagger$" if s["status"][x] == "partial" else "")
                           for x in (601, 602, 603))
        inj = " / ".join(str(s["exposures"][x]) for x in (601, 602, 603))
        lines.append(f"Telecom & {s['label']} & \\ms{{{s['mean_pass1']:.2f}}}{{{s['sd_pass1']:.2f}}} & "
                     f"{s['mean_passn']:.2f} & {c['mean_difference']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}] & "
                     f"{calls} & {inj} \\\\")
    lines.append("% $\\ddagger$: E2 seed 602 stopped early (partial, no test evaluation); E2 mean/SD over seeds 601 and 603.")
    (out / "table3_lower_rows.tex").write_text("\n".join(lines) + "\n")
    return comp, summary


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    new, new_curves, new_rates = new_runs()
    ref, ref_curves, ref_rates = reference_runs()
    rates = {**new_rates, **ref_rates}
    results = ref + new
    write_csv(out / "table3_runs.csv", results, ["exp_id", "env", "start", "arm", "seed", "test_pass1", "test_passn",
              "n_eval_samples", "target_calls_charged", "source_calls_charged", "injections_or_exposures", "rounds",
              "supervised_tokens_completed", "status", "notes"])
    write_csv(out / "table3_curves.csv", ref_curves + new_curves, ["exp_id", "env", "arm", "seed", "round",
              "cumulative_target_calls", "train_success", "mixed_groups", "groups", "cumulative_injections"])
    per_task = []
    for r in results:
        if r["test_pass1"] == "":
            continue
        key = (r["exp_id"], r["seed"]) if r["exp_id"] != "REF" else \
            (next(k for k, a in REF.items() if (k, r["seed"]) in ref_rates and
                  json.loads((ROOT / "results/coldstart_analysis_20261116.json").read_text())["arms"][k]["label"] == r["arm"]
                  and k.startswith("tel" if r["env"] == "telecom" else "math")), r["seed"])
        ids, rate = rates[key]
        k = 8 if r["env"] == "telecom" else 4
        per_task += [dict(exp_id=r["exp_id"], env=r["env"], arm=r["arm"], seed=r["seed"], task_id=t,
                          n_success=int(round(x * k)), n_samples=k) for t, x in zip(ids, rate)]
    write_csv(out / "table3_per_task.csv", per_task, ["exp_id", "env", "arm", "seed", "task_id", "n_success", "n_samples"])
    probe, probe_extra = probe_rows()
    write_csv(out / "table10_probe.csv", probe, ["setting", "validation_gap", "format_error_rate", "decision",
              "matches_test_best", "agree_k1", "agree_k2", "agree_k4", "target_calls", "source_calls",
              "validation_task_ids_file"])

    comp, summary = alternative_rows(out, new, ref_rates, new_rates)
    hashes = []
    for p in sorted(set(HASHED) | {ROOT / "results/coldstart_analysis_20261116.json", ROOT / "data/splits/val_probe_tasks.json"}):
        hashes.append(dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=digest(p)))
    write_csv(out / "alternatives_input_hashes.csv", hashes, ["path", "bytes", "sha256"])
    (out / "alternatives_stats.json").write_text(json.dumps(dict(bootstrap=dict(draws=DRAWS, seed=BOOT_SEED, unit="test task",
        pairing="shared draws across arms; per-task success averaged over each arm's seeds"),
        comparisons=comp, arms=summary, probe=probe_extra, probe_format_threshold=FMT_THRESHOLD), indent=1))
    print(json.dumps(dict(comparisons=comp, probe=probe, probe_extra=probe_extra, files=len(hashes)), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--probe-only", action="store_true", help="compute Table 10 from results/probe/")
    args = p.parse_args()
    if args.probe_only:
        rows, _ = probe_rows()
        args.out.mkdir(parents=True, exist_ok=True)
        write_csv(args.out / "table10_probe.csv", rows, list(rows[0]))
    else:
        main(args.out)
