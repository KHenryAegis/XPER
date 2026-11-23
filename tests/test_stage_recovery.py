import argparse
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import uuid

import pytest

needs_proc = pytest.mark.skipif(not Path("/proc").exists(), reason="needs Linux /proc")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage_guard
from recover_strengthen import guard_driver, plan, shard_status


def args(tmp_path, kind, code):
    return argparse.Namespace(gpus="0", kind=kind, log=str(tmp_path / "stage.log"), poll=0,
                              command=[sys.executable, "-c", code])


@needs_proc
def test_update_oom_waits_and_retries_from_same_command(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_guard, "idle", lambda gpus: True)
    monkeypatch.setattr(stage_guard.signal, "signal", lambda *a: None)
    counter = tmp_path / "count"
    code = ("from pathlib import Path; import sys; "
            f"p=Path({str(counter)!r}); n=int(p.read_text())+1 if p.exists() else 1; "
            "p.write_text(str(n)); print('CUDA out of memory' if n==1 else 'saved'); sys.exit(n==1)")
    a = args(tmp_path, "update", code)
    assert stage_guard.main(a) == 0
    assert counter.read_text() == "2"
    assert Path(a.log).read_text().strip() == "saved"
    assert "out of memory" in Path(a.log + ".oom-first").read_text()
    assert json.loads(Path(a.log + ".stage.json").read_text())["attempts"] == 2


@needs_proc
def test_rollout_after_reset_is_not_silently_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_guard, "idle", lambda gpus: True)
    monkeypatch.setattr(stage_guard.signal, "signal", lambda *a: None)
    a = args(tmp_path, "rollout", "print('reset 32 episodes in 1s'); print('CUDA out of memory'); raise SystemExit(1)")
    assert stage_guard.main(a) == 1
    assert not Path(a.log + ".oom-first").exists()


def test_outer_retry_rename_preserves_reset_evidence(tmp_path):
    result = tmp_path / "task.s603"
    result.mkdir()
    # An OOM can happen after reset but before the first episode is committed.
    with sqlite3.connect(result / "it0.shard0.db") as db:
        db.execute("CREATE TABLE trajectories(task_id TEXT,env_calls INT,model_id TEXT)")
    (result / "it0.shard0.try1.log").write_text("reset 32 episodes in 1s\nCUDA out of memory\n")
    assert shard_status(result, 0, 0) == 2


@needs_proc
def test_group_cleanup_requires_the_created_child_token():
    token = uuid.uuid4().hex
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             start_new_session=True, env=dict(os.environ, XPER_STAGE_OWNER=token))
    try:
        with pytest.raises(RuntimeError, match="unverified"):
            stage_guard.stop_owned_group(child.pid, "wrong-token")
        assert child.poll() is None
        stage_guard.stop_owned_group(child.pid, token)
        assert child.wait(timeout=5) == -15
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("phase", ["update", "rollout"])
def test_recovery_preserves_committed_calls_and_source_quota(tmp_path, monkeypatch, phase):
    import strengthen as s
    name = "sc-tel-obs_mask-s601"
    result = tmp_path / "results/sc-tel-obs_mask.s601"
    result.mkdir(parents=True)
    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(s, "MODELS", tmp_path / "models")
    (result / "owner.json").write_text(json.dumps(dict(batch=s.BATCH, name=name)))
    policy = tmp_path / "models/committed/final"
    policy.mkdir(parents=True)
    (policy / "weights.safetensors").write_bytes(b"test fixture")
    (policy / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
    (policy.parent / "data_stats.json").write_text(json.dumps({"src": "round2"}))
    (result / "it2.train.log").write_text(json.dumps(dict(saved=str(policy), src="round2")))
    history = [dict(iteration=i, env_calls=100*(i+1), injected=n, injected_own=0)
               for i, n in enumerate([16, 16, 10])]
    if phase == "update":
        history.append(dict(iteration=3, env_calls=400, injected=6, injected_own=0))
    monkeypatch.setattr(s, "iterations", lambda _: history)
    for shard in (range(4) if phase == "update" else [3]):
        path = result / f"it3.shard{shard}.db"
        tasks = [f"task{shard}-{i//8}" for i in range(32)]
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE trajectories(task_id TEXT,env_calls INT,model_id TEXT)")
            db.executemany("INSERT INTO trajectories VALUES(?,?,?)", [(t, 1, "sc-tel-obs_mask-it3") for t in tasks])
        path.with_suffix(".convs.jsonl").write_text("\n".join(json.dumps({"task": t}) for t in tasks))
    r = plan(name)
    assert r["phase"] == phase and r["iteration"] == 3
    assert r["spent_prior"] == 300 and r["src_used_prior"] == 42
    assert r["shard_status"] == ([0, 0, 0, 0] if phase == "update" else [1, 1, 1, 0])
    original = (Path(__file__).resolve().parents[1] / "scripts/run_grpo.sh").read_text()
    driver = guard_driver(original, r, tmp_path / "plan.json")
    subprocess.run(["bash", "-n"], input=driver, text=True, check=True)
    assert "scripts/stage_guard.py" in driver and "src_used=42" in driver
