"""GRPO plumbing: advantages from saved conversations, and weighted training data loading."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

needs_tau2 = pytest.mark.skipif(importlib.util.find_spec("tau2") is None, reason="needs tau2-bench (tools/fetch_tau2.sh)")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def _conv(task, reward, k):
    return {"task": task, "k": k, "reward": reward,
            "messages": [{"role": "system", "content": f"sys {task}"},
                         {"role": "assistant", "content": f"Action:\n{{\"tool\": \"done\", \"arguments\": {{}}}} {k}"}]}


def test_advantages(tmp_path):
    convs = tmp_path / "c.jsonl"
    rows = [_conv("a", r, k) for k, r in enumerate([1, 0, 0, 0])]  # mixed: kept
    rows += [_conv("b", 0, k) for k in range(4)]                    # all failed: dropped
    rows += [_conv("c", 1, k) for k in range(4)]                    # all solved: dropped
    convs.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "t.jsonl"
    res = subprocess.run([sys.executable, str(ROOT / "scripts/grpo_advantages.py"), "--convs", str(convs),
                          "--out", str(out)], capture_output=True, text=True, check=True)
    stats = json.loads(res.stdout)
    assert stats == {"groups": 3, "mixed_groups": 1, "episodes": 12, "injected": 0, "injected_own": 0,
                     "train_seqs": 4, "mean_reward": 0.4167}
    ws = [json.loads(l)["weight"] for l in out.read_text().splitlines()]
    assert abs(sum(ws)) < 1e-6 and ws[0] > 0 and all(w < 0 for w in ws[1:])
    assert abs(ws[0] - 3 ** 0.5) < 1e-3  # (1 - .25) / sqrt(.1875)


def test_weighted_jsonl_loader(tmp_path):
    from bc_train import load_jsonl_conversations
    f = tmp_path / "w.jsonl"
    f.write_text(json.dumps({"task": "t1", "weight": -0.5, "messages": [{"role": "system", "content": "s"}]}) + "\n"
                 + json.dumps({"messages": [{"role": "system", "content": "s"}]}))
    convs, tasks, weights, off = load_jsonl_conversations(str(f))
    assert len(convs) == 2 and tasks[0] == "t1" and tasks[1].startswith("jsonl:") and weights == [-0.5, 1.0]


def _src_store(path, tasks):
    from xper.actions import Action
    from xper.store import ExperienceStore, Step
    st = ExperienceStore(str(path))
    for t in tasks:
        steps = [Step(Action("tool", "lookup", {"q": f"{t}-{i}"}), f"obs {t} {i}") for i in range(3)]
        steps.append(Step(Action("tool", "done", {}), ""))
        st.add_trajectory(t, steps, 1.0, True, "src")
    st.close()


def _inject(tmp_path, control, monkeypatch, extra=(), rewards=None):
    import bc_train
    from xper import tau
    monkeypatch.setattr(tau, "system_prompt", lambda tid, domain="telecom": f"sys {tid}")
    src = tmp_path / "s.db"
    _src_store(src, ["a", "b", "c"])
    convs = tmp_path / "c.jsonl"
    convs.write_text("\n".join(json.dumps(_conv(t, (rewards or {}).get(t, [0]*4)[k], k))
                               for t in "ab" for k in range(4)))
    out = tmp_path / "t.jsonl"
    sys.argv = ["grpo_advantages.py", "--convs", str(convs), "--out", str(out), "--inject-src", str(src),
                "--inject-control", control, "--seed", "3", *extra]
    import runpy
    runpy.run_path(str(ROOT / "scripts/grpo_advantages.py"), run_name="__main__")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    return [r for r in rows if r["weight"] > 0]


@needs_tau2
def test_inject_mismatch_uses_other_task_content(tmp_path, monkeypatch):
    for r in _inject(tmp_path, "mismatch", monkeypatch):
        task = r["task"]
        assert r["messages"][0]["content"] == f"sys {task}"          # own prompt
        body = " ".join(m["content"] for m in r["messages"][1:])
        assert f"{task}-0" not in body and "-0" in body               # another task's actions


@needs_tau2
def test_inject_shuffle_keeps_done_last(tmp_path, monkeypatch):
    for r in _inject(tmp_path, "shuffle", monkeypatch):
        acts = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        assert '"done"' in acts[-1] and len(acts) == 4
        assert sorted(acts[:-1]) == sorted(a for a in acts[:-1])
        assert all(r["task"] in a for a in acts[:-1])                 # same task's actions


@needs_tau2
def test_source_dose_is_rationed_across_rounds(tmp_path, monkeypatch):
    # A three-trajectory experiment uses 2 + 1 + 0 across successive rounds,
    # rather than injecting three trajectories afresh on every invocation.
    for control in ("none", "mismatch", "shuffle"):
        used = 0
        counts = []
        for _ in range(3):
            rows = _inject(tmp_path, control, monkeypatch,
                           ["--max-src-injections", str(3-used)])
            counts.append(len(rows))
            used += len(rows)
        assert counts == [2, 1, 0] and used == 3


@needs_tau2
def test_zero_dose_preserves_on_policy_success(tmp_path, monkeypatch):
    rows = _inject(tmp_path, "none", monkeypatch, ["--max-src-injections", "0"],
                   rewards={"a": [1, 0, 0, 0]})
    assert len(rows) == 1 and rows[0]["task"] == "a"


def _run_adv(tmp_path, rows, extra=()):
    convs = tmp_path / "c.jsonl"
    convs.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "t.jsonl"
    res = subprocess.run([sys.executable, str(ROOT / "scripts/grpo_advantages.py"), "--convs", str(convs),
                          "--out", str(out), *extra], capture_output=True, text=True, check=True)
    return json.loads(res.stdout), [json.loads(l) for l in out.read_text().splitlines()]


def test_no_std_advantage(tmp_path):
    _, rows = _run_adv(tmp_path, [_conv("a", r, k) for k, r in enumerate([1, 0, 0, 0])], ["--no-std"])
    assert [round(r["weight"], 4) for r in rows] == [0.75, -0.25, -0.25, -0.25]


def test_self_replay_only_is_capped(tmp_path):
    # earlier round: own successes on a and b; this round: a, b, c all fail
    old = tmp_path / "old.convs.jsonl"
    old.write_text("\n".join(json.dumps(_conv(t, 1, 0)) for t in "ab"))
    rows = [_conv(t, 0, k) for t in "abc" for k in range(4)]
    stats, out = _run_adv(tmp_path, rows, ["--inject-convs", str(old), "--max-own-injections", "1"])
    assert stats["all_fail_groups"] == 3 and stats["own_possible"] == 2
    assert stats["injected"] == stats["injected_own"] == 1
    assert sum(r["weight"] > 0 for r in out) == 1


@needs_tau2
def test_relift_buffer_caps_exposures(tmp_path, monkeypatch):
    import runpy
    from xper import tau
    monkeypatch.setattr(tau, "system_prompt", lambda tid, domain="telecom": f"sys {tid}")
    src = tmp_path / "s.db"
    _src_store(src, ["a", "b", "c"])
    convs = tmp_path / "c.jsonl"
    convs.write_text("\n".join(json.dumps(_conv(t, 0, k)) for t in "abd" for k in range(2)))  # d: no source
    state, out = tmp_path / "state.json", tmp_path / "sft.jsonl"
    seqs = []
    for i in range(4):
        out.unlink(missing_ok=True)
        sys.argv = ["relift_buffer.py", "--convs", str(convs), "--src", str(src), "--state", str(state),
                    "--out", str(out), "--size", "3", "--cap", "5", "--seed", str(i)]
        runpy.run_path(str(ROOT / "scripts/relift_buffer.py"), run_name="__main__")
        seqs.append(len(out.read_text().splitlines()) if out.exists() else 0)
    # 2 buffered per round; step of 3 when full, last step shrunk to the remaining cap of 2
    assert seqs == [0, 3, 2, 0] and json.loads(state.read_text())["exposures"] == 5
