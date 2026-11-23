"""Few-shot in-context baseline: show source-agent successes in the prompt, no training.

Uses the same experience the transfer arm trains on (source successes on train_full), so the
comparison is "learn from the experience by SFT" vs "read the experience in context". Examples
are chosen once and shared by every task, so the long prefix is cached by vLLM.
"""

from __future__ import annotations

import glob
import json
import random

from .actions import Action
from .store import ExperienceStore


def pick_examples(src_glob: str, k: int, seed: int, exclude: set[str],
                  keep: set[str] | None = None) -> list[tuple[str, list]]:
    """k successful trajectories, round-robin over issue families, never from ``exclude``."""
    by_fam: dict[str, list[tuple[str, list]]] = {}
    for path in sorted(glob.glob(src_glob)):
        st = ExperienceStore(path)
        for traj_id, task_id, _m, reward, *_ in st.trajectories():
            if reward < 1.0 or (keep is not None and task_id not in keep):
                continue
            if task_id in exclude:
                raise SystemExit(f"refusing to use a test task as an example: {task_id} ({path})")
            # telecom ids carry an issue family "[fam]..."; math ids do not -> one pool, uniform draw
            fam = task_id.split("]")[0] if task_id.startswith("[") else "all"
            by_fam.setdefault(fam, []).append((task_id, list(st.steps(traj_id))))
    rng = random.Random(seed)
    for v in by_fam.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < k and any(i < len(v) for v in by_fam.values()):
        for fam in sorted(by_fam):
            if i < len(by_fam[fam]) and len(out) < k:
                out.append(by_fam[fam][i])
        i += 1
    return out


def examples_block(examples: list[tuple[str, list]], obs_chars: int) -> str:
    """Render examples as action/observation transcripts. Telecom tickets are omitted (only five
    ticket texts; the hidden state is what differs); math problems are shown (they all differ)."""
    is_math = bool(examples) and examples[0][0].startswith("math-")
    parts = []
    for n, (task_id, rows) in enumerate(examples, 1):
        lines = [f"<example {n}>"]
        if is_math:
            from .mathenv import problems
            lines.append("Problem: " + problems()[task_id]["question"])
        for row in rows:
            act = Action.from_json(row[3])
            lines.append(act.to_text())
            obs = (row[4] or "").strip()
            if act.tool not in ("done", "submit"):
                lines.append("Observation:\n" + (obs[:obs_chars] + (" ..." if len(obs) > obs_chars else "")))
        lines.append(f"</example {n}>")
        parts.append("\n".join(lines))
    head = ("Transcripts of an agent that solved other problems with the same tools:" if is_math else
            "Transcripts of an agent that resolved earlier tickets of this kind "
            "(different customers and device states):")
    return "<examples>\n" + head + "\n" + "\n".join(parts) + "\n</examples>"


def with_examples(system: str, block: str) -> str:
    """Insert the block before the task (<ticket> / <problem>) so the shared prefix is cached."""
    for tag in ("<ticket>", "<problem>", "<database>"):
        i = system.rfind(tag)
        if i >= 0:
            return system[:i] + block + "\n" + system[i:]
    return system + "\n" + block
