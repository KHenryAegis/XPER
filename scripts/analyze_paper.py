"""Cold-start injection results from the 24 finished cohort runs, on CPU only: Table 3 (upper block), Tables 4-5
and Figure 4.

    python -B scripts/analyze_paper.py

Main real/none comparisons use seeds 601/602/603. Dose and information controls use their matched 601/602
cohorts; no unpaired third seed is pooled. The audited per-run collector is scripts/analyze_coldstart.py.
Writes results/coldstart_analysis_20261116.json (machine-readable audit with input SHA-256),
outputs/tables/coldstart_{results,dose,controls}.tex and outputs/figures/coldstart_dynamics.pdf.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import analyze_coldstart as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/coldstart_analysis_20261116.json"
MAIN = ("tel_none", "tel_real", "math_none", "math_real")


def summarize(runs):
    rates = np.array([r["pass1"] for r in runs])
    return dict(mean_pass1=float(rates.mean()), sd_pass1=float(rates.std(ddof=1)),
                mean_passk=float(np.mean([r["passk"] for r in runs])))


def collect():
    data = core.collect()
    data["matched_two_seed"] = {
        "arms": {k: {f: copy.deepcopy(a[f]) for f in ("mean_pass1", "sd_pass1", "mean_passk", "task_bootstrap_ci95_pass1")}
                 for k, a in data["arms"].items()},
        "comparisons": copy.deepcopy(data["comparisons"]),
    }
    del data["comparisons"]  # final artifact exposes explicitly scoped comparisons only
    report_path = ROOT / "results/third_seed_20261115_report.json"
    report = json.loads(report_path.read_text())
    if not report["complete"] or not report["terminal"] or report.get("math_start_cleanup") != "done":
        raise ValueError("third-seed cohort incomplete")
    data["input_sha256"][str(report_path.relative_to(ROOT))] = core.digest(report_path)
    for key in MAIN:
        arm = data["arms"][key]
        name = core.ARMS[key][2] + "-s603"
        rec = report["jobs"][name]
        if rec["status"] != "ok" or rec["errors"] or rec["cleanup"] != "done":
            raise ValueError(f"unverified third seed: {name}")
        path = ROOT / "results/eval" / f"{name}.db"
        ids, rates, covered = core.load_eval(path, arm["tasks"], arm["samples"], name)
        if ids != data["task_ids"][arm["environment"]] or rates.mean() != rec["eval"]["db_pass1"]:
            raise ValueError(f"task/outcome mismatch: {name}")
        base = ROOT / "logs/queue/third_seed_20261115"
        logs = [base / f"{name}.log"] + sorted(base.glob(f"recover*-{name}.log"))
        history = core.merge_history(logs)
        tag = name.rsplit("-s", 1)[0]
        directory = ROOT / "results" / f"{tag}.s603"
        core.audit_training(directory, history, rec)
        injected = sum(r["injected"] for r in history)
        if injected != rec["injected_total"]:
            raise ValueError("dose mismatch")
        updates = []
        for p in directory.glob("it*.train.log"):
            updates += [json.loads(line) for line in p.read_text().splitlines() if line.startswith('{"saved"')]
        run = dict(name=name, seed=603, pass1=float(rates.mean()), passk=float(covered.mean()),
                   target_calls=rec["training"]["env_calls"], source_injections=injected,
                   target_episodes=rec["training"]["episodes"], updates=len(updates),
                   dataset_supervised_tokens=sum(x["supervised_tokens"] for x in updates),
                   history=history, **core.after_injection(history))
        if "budget_accounting" in rec:
            accounting = copy.deepcopy(rec["budget_accounting"])
            if accounting["failed_calls_exact"] or accounting["failed_calls_upper_bound"] != 416:
                raise ValueError("unexpected failure accounting")
            archive = directory / "failed-it16-shard1-20261116"
            for old_path, expected in accounting["artifacts"].items():
                saved = archive / Path(old_path).name
                if core.digest(saved) != expected:
                    raise ValueError("failed-shard evidence changed")
                data["input_sha256"][str(saved.relative_to(ROOT))] = expected
            run["budget_accounting"] = accounting
        run["target_calls_charged"] = run.get("budget_accounting", {}).get(
            "total_calls_conservatively_charged", run["target_calls"])
        arm["runs"].append(run)
        arm.update(summarize(arm["runs"]))
        for p in [path, *logs]:
            data["input_sha256"][str(p.relative_to(ROOT))] = core.digest(p)
        print(f"audited {name}: test={rates.mean():.6f}, successful calls={run['target_calls']}, charged={run['target_calls_charged']}", flush=True)
    data["main_comparisons"] = {}
    for env, keys in (("telecom", ("tel_real", "tel_none")), ("math", ("math_real", "math_none"))):
        arrays = []
        for key in keys:
            a = data["arms"][key]
            seed_arrays = []
            for r in a["runs"]:
                _, rates, _ = core.load_eval(ROOT / "results/eval" / (r["name"] + ".db"), a["tasks"], a["samples"], r["name"])
                seed_arrays.append(rates)
            arrays.append(seed_arrays)
        arrays = np.array(arrays)
        boot = core.bootstrap_arms(arrays)
        for i, key in enumerate(keys):
            data["arms"][key]["task_bootstrap_ci95_pass1"] = core.ci(boot[i])
        delta = arrays[0] - arrays[1]
        data["main_comparisons"][env] = dict(mean_difference=float(delta.mean()),
            task_bootstrap_ci95=core.ci(boot[0]-boot[1]), seed_differences=delta.mean(axis=1).tolist(), seeds=[601, 602, 603])
    data.update(schema_version=2,
        interval_method="Paired task-cluster percentile bootstrap, 20,000 draws, analysis seed 20261115. Main effects use three fixed policies; matched ablations use two.",
        failure_accounting_note="Math real s603 repeats one serialization-failed shard. Successful calls 10820; total known lower bound 10822, conservatively charged upper bound 11236. No exact failed-call claim.")
    data["limitations"][0] = "Intervals condition on observed policies (three main, two ablation seeds), not future training seeds."
    data["input_sha256"][str(Path(__file__).relative_to(ROOT))] = core.digest(Path(__file__))
    return data


def view(data, key, n=None):
    a = data["arms"][key]
    return data["matched_two_seed"]["arms"][key] if n == 2 else a


def write_tables(data):
    out = ROOT / "outputs/tables"
    out.mkdir(parents=True, exist_ok=True)
    pp, interval = core.pp, core.interval
    lines = [r"% Generated by scripts/analyze_paper.py.", r"\begin{table*}[t]", r"\centering\footnotesize",
        r"\caption{Main cold-start results over rollout seeds 601/602/603. Test percentages are mean $\pm$ sample SD; $n=8$ telecom and $n=4$ math. Differences have paired 95\% task-bootstrap intervals conditional on these policies. Target calls (thousands) and injection exposures follow seed order. Source collection is additional. $\dagger$: a conservative failed-shard charge, detailed in the text.}",
        r"\label{tab:injection}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"Env & Injection & pass@1 & pass@$n$ & $\Delta$ vs. none [95\% CI], pp & Target calls (k) & Injections \\", r"\midrule"]
    for key in MAIN:
        a = data["arms"][key]
        c = data["main_comparisons"][a["environment"]] if key.endswith("real") else None
        diff = "--" if c is None else f"{pp(c['mean_difference'])} {interval(c['task_bootstrap_ci95'])}"
        costs = " / ".join(f"{r.get('target_calls_charged',r['target_calls'])/1000:.2f}" + (r"$\dagger$" if "budget_accounting" in r else "") for r in a["runs"])
        lines.append(f"{'Telecom' if key.startswith('tel') else 'Math'} & {a['label']} & "
            + rf"\ms{{{pp(a['mean_pass1'])}}}{{{pp(a['sd_pass1'])}}} & {pp(a['mean_passk'])} & {diff} & {costs} & "
            + " / ".join(str(r["source_injections"]) for r in a["runs"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (out / "coldstart_results.tex").write_text("\n".join(lines) + "\n")
    lines = [r"% Generated by scripts/analyze_paper.py.", r"\begin{table}[t]", r"\centering\footnotesize",
        r"\caption{Matched telecom information controls, seeds 601/602 only. All percentages use this two-seed cohort, including its shared real/no-injection rows. Intervals condition on the observed policies.}",
        r"\label{tab:injection-controls}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
        r"Injection & pass@1 & $\Delta$ vs. none [95\% CI], pp \\", r"\midrule"]
    for key in ("tel_none", "tel_real", "tel_mask", "tel_schema"):
        a = view(data, key, 2)
        c = data["matched_two_seed"]["comparisons"].get(key + "_vs_tel_none")
        diff = "--" if c is None else f"{pp(c['mean_difference'])} {interval(c['task_bootstrap_ci95'])}"
        lines.append(data["arms"][key]["label"] + " & " + rf"\ms{{{pp(a['mean_pass1'])}}}{{{pp(a['sd_pass1'])}}} & {diff} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "coldstart_controls.tex").write_text("\n".join(lines) + "\n")
    # The dose experiment retains its original paired two-seed comparison.
    lines = [r"% Generated by scripts/analyze_paper.py.", r"\begin{table}[t]", r"\centering",
        r"\caption{Telecom dose experiment, seeds 601/602. The cap-0 and cap-100 rows are the same two-seed runs used for the information controls; seed603 is confined to the main comparison.}",
        r"\label{tab:injection-dose}", r"\begin{tabular}{@{}rrrrr@{}}", r"\toprule",
        r"Cap & Seed 601 & Seed 602 & Mean & Actual injections \\", r"\midrule"]
    for cap, key in ((0,"tel_none"),(5,"tel_d5"),(20,"tel_d20"),(100,"tel_real")):
        a = data["arms"][key];v = view(data,key,2);runs=a["runs"][:2]
        lines.append(f"{cap} & {pp(runs[0]['pass1'])} & {pp(runs[1]['pass1'])} & {pp(v['mean_pass1'])} & "
                     + " / ".join(str(r["source_injections"]) for r in runs) + r" \\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (out / "coldstart_dose.tex").write_text("\n".join(lines)+"\n")



def plot(data):
    os.environ.setdefault("MPLCONFIGDIR","/tmp/xper-paper-analysis-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":7,"axes.labelsize":7,"axes.titlesize":8,
        "legend.fontsize":6,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"lines.linewidth":1.1})
    panels=[("Telecom main (3 seeds)",["tel_none","tel_real"],None),
            ("Math main (3 seeds)",["math_none","math_real"],None),
            ("Telecom dose (2 seeds)",["tel_none","tel_d5","tel_d20","tel_real"],2),
            ("Telecom controls (2 seeds)",["tel_none","tel_real","tel_mask","tel_schema"],2)]
    colors={"tel_none":"#737373","math_none":"#737373","tel_d5":"#7570b3","tel_d20":"#e6ab02",
            "tel_real":"#d95f02","math_real":"#d95f02","tel_mask":"#1b9e77","tel_schema":"#377eb8"}
    fig,axes=plt.subplots(2,3,figsize=(7,3.7))
    curves=[panels[2],panels[3],panels[1]]
    for col,(title,keys,n) in enumerate(curves):
        top,bottom=axes[:,col]
        for key in keys:
            for i,r in enumerate(data['arms'][key]['runs'][:n]):
                h=r['history'];style=['-','--',':'][i]
                # The failed work happened during round16; s603 uses its
                # conservative charge thereafter, not a fabricated exact cost.
                x=[(v['env_calls']+(r['budget_accounting']['failed_calls_upper_bound'] if 'budget_accounting' in r and v['iteration']>=r['budget_accounting']['iteration'] else 0))/1000 for v in h]
                top.plot(x,[100*v['target_success_rate'] for v in h],ls=style,color=colors[key],label=data['arms'][key]['label'] if i==0 else None)
                bottom.step(x,np.cumsum([v['injected'] for v in h]),where='post',color=colors[key],ls=style)
        top.set_title(title);top.set_ylim(0,100 if col<2 else 50);bottom.set_ylim(-3,105)
        bottom.set_xlabel('target environment calls (k)');top.legend(frameon=False,loc='upper left');top.grid(axis='y',alpha=.15)
        if col==0:top.set_ylabel('target training success (%)');bottom.set_ylabel('cumulative source injections')
    fig.tight_layout(pad=.7);(ROOT/'outputs/figures').mkdir(parents=True,exist_ok=True);fig.savefig(ROOT/'outputs/figures/coldstart_dynamics.pdf',bbox_inches='tight');plt.close(fig)


if __name__ == '__main__':
    data=collect()
    OUT.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    write_tables(data);plot(data)
    print('wrote',OUT.relative_to(ROOT),'outputs/tables/coldstart_{results,dose,controls}.tex and outputs/figures/coldstart_dynamics.pdf',flush=True)
