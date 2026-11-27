"""Export the run-level outcome of every evaluation the paper reports to CSV.

    python -B tools/export_eval_summaries.py [--root DIR] [--out FILE]

Each evaluation log written by scripts/rollout.py ends with one JSON summary line (pass@1, pass@n, tasks, samples,
environment calls, mean steps, format-error rate). This tool reads those lines from ROOT/results (default: this
repository after running the evaluations) for the runs listed in REPORTED, grouped by the paper item they
support. It only opens the logs named after those runs (results/{eval,math,g1}/<run>.log).
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# paper item -> run-name patterns (regular expressions matched against the summary's "model" field)
REPORTED = {
    "Table 1 (zero-shot starts)": [r"qwen3-4b-inst", r"qwen3-8b-nothink", r"8b-instruct-start-nothink", r"8b-commonsft",
                                   r"math-4b-inst", r"math-8b-instruct", r"math-8b-base-eos"],
    "Table 2 / Figure 4 (cold-start injection)": [r"td(0|100)-s60[123]", r"sc-math-(none|real)-s60[123]",
                                                  r"sc-math-headroom"],
    "Table 2 lower block (alternatives)": [r"e2-tel(-r[23])?-s60[123]", r"e4a-tel(-r2)?-s60[123]",
                                           r"e4b-tel(-r2)?-s60[123]", r"e5-tel(-r2)?-s60[123]"],
    "Tables 3-4 (dose and information controls)": [r"td(5|20)-s60[12]", r"sc-tel-(obs_mask|schema_random)-s60[12]"],
    "Section 6.1 (uncapped exploratory controls)": [r"tp-(mismatch|shuffle)-s60[12]"],
    "Section 3.3 and Tables 5-6 (offline reuse and self-exploration)": [
        r"8b-commonsft-icl6", r"8b-xfer", r"8b-selfexp", r"rft-commonsft-s50[12]", r"grpo-commonsft-s601",
        r"8b-instruct-icl(3|6|12)(-s[0-2])?", r"8b-instruct-xfer-s40[123]-(nothink|icl6)",
        r"8b-instruct-selfexp-s40[123]-nothink", r"rft-instruct-s50[12]", r"grpo-instruct-s60[123]",
        r"grpo-xferstart-s60[12]", r"math-8b-(base-)?(selfexp|xfer)", r"math-8b-icl4",
        r"math-rft-instruct-s50[12]", r"math-grpo-instruct-s60[123]"],
    "Section 7.3 (source-data scaling)": [r"8b-xfer-bare-p(25|50|75)", r"8b-base-xfer-weak-p(25|50|75|100s1)"],
    "Section 6.2 (permutation control)": [r"init-eos-(4|8)b", r"fmtctl-(4|8)b", r"bc3-(4|8)b"],
    "Table 2 (transfer SFT to GRPO)": [r"xs-grpo-tel-s60[123]", r"8b-xfer-rebuilt"],
    "Table 6 (mobile-data ICL)": [r"8b-instruct-icl6-mobile-s[0-2]"],
    "Table 5 (cross-family offline comparison)": [r"hermes8b-(start|icl6|xfer)"],
    "Section 6.1 (matched-cohort math transfer SFT)": [r"sc-math-xfer"],
}
FIELDS = ["paper_item", "run", "log", "split", "tasks", "samples", "pass1_pct", "passn_pct", "mixed_tasks",
          "env_calls", "mean_steps", "format_error_pct"]


def reported(name):
    return any(re.fullmatch(p, name) for pats in REPORTED.values() for p in pats)


def summaries(root):
    for path in sorted(p for d in ("eval", "math", "g1") for p in (root / "results" / d).glob("*.log")):
        if not reported(path.stem):
            continue
        for line in path.open(errors="replace"):
            if line.startswith('{"model"'):
                yield path.relative_to(root), json.loads(line)
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="tree that contains results/ (default: this repository)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/eval_summaries.csv")
    ap.add_argument("--require-paper-runs", action="store_true",
                    help="require an evaluation log for every run ID in the paper cohort")
    a = ap.parse_args()
    rows, seen = [], set()
    for log, s in summaries(a.root.resolve()):
        item = next((k for k, pats in REPORTED.items() if any(re.fullmatch(p, s["model"]) for p in pats)), None)
        if item is None or s["model"] in seen:
            continue
        seen.add(s["model"])
        n = s.get("n")
        rows.append(dict(paper_item=item, run=s["model"], log=str(log), split=s.get("split"), tasks=s.get("tasks"),
                         samples=n, pass1_pct=round(100 * s["pass@1"], 2),
                         passn_pct=round(100 * s.get(f"pass@{n}", float("nan")), 2),
                         mixed_tasks=s.get("mixed_tasks"), env_calls=s.get("env_calls"), mean_steps=s.get("mean_steps"),
                         format_error_pct=round(100 * s.get("bad_format_rate", float("nan")), 2)))
    rows.sort(key=lambda r: (list(REPORTED).index(r["paper_item"]), r["run"]))
    if a.require_paper_runs:
        with (ROOT / "data/results/eval_summaries.csv").open() as f:
            expected = {row["run"] for row in csv.DictReader(f)}
        # A fresh run already evaluates its transfer-SFT start in step 02.
        if "8b-xfer" in seen:
            expected.discard("8b-xfer-rebuilt")
        missing_runs = sorted(expected - seen)
        if missing_runs:
            failed = ", ".join(missing_runs)
            raise SystemExit(f"Missing {len(missing_runs)} paper evaluation logs: {failed}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    missing = [p for pats in REPORTED.values() for p in pats
               if not any(re.fullmatch(p, r["run"]) for r in rows)]
    print(f"wrote {len(rows)} runs to {a.out}" + (f"; patterns with no run: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
