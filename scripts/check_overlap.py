"""Near-duplicate check between telecom test tasks and train data.

A telecom task id is "[family]fault1|fault2|...[PERSONA:x]". test ids are excluded from train_full
exactly, but the same fault set can recur under another persona. Report, per test task, whether its
fault set (persona dropped) appears in train_full, among the 223 source successes, and among the
ICL examples for a given (k, seed).
"""

import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xper.icl import pick_examples
from xper.store import ExperienceStore
from xper.tau import split


def key(tid: str) -> tuple[str, frozenset]:
    fam = tid.split("]")[0] + "]"
    body = re.sub(r"\[PERSONA:[^\]]*\]", "", tid[len(fam):])
    return fam, frozenset(f for f in body.split("|") if f)


p = argparse.ArgumentParser()
p.add_argument("--src", default="results/src/4b-inst.shard*.db")
p.add_argument("--icl", default="6:0", help="k:seed of the ICL draw to check")
a = p.parse_args()

test = split("telecom", "test")
train = split("telecom", "train_full")
src = set()
for path in sorted(glob.glob(a.src)):
    for _t, tid, _m, reward, *_ in ExperienceStore(path).trajectories():
        if reward >= 1.0:
            src.add(tid)
k, s = map(int, a.icl.split(":"))
icl = {tid for tid, _ in pick_examples(a.src, k, s, exclude=set(test))}

tk, srck, iclk = {key(t) for t in train}, {key(t) for t in src}, {key(t) for t in icl}
print(f"test {len(test)} | train_full {len(train)} ({len(tk)} fault sets) | source successes "
      f"{len(src)} tasks ({len(srck)} fault sets) | ICL {a.icl}: {len(icl)} tasks")
print(f"exact test id in train_full: {len(set(test) & set(train))}")
rows = [(t, key(t) in tk, key(t) in srck, key(t) in iclk) for t in test]
print(f"test fault set also in train_full: {sum(r[1] for r in rows)}/40, in source successes: "
      f"{sum(r[2] for r in rows)}/40, in ICL examples: {sum(r[3] for r in rows)}/40")
for t, a_, b, c in rows:
    if b or c:
        print(f"  {'SRC' if b else '   '} {'ICL' if c else '   '} {t[:110]}")
