import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import strengthen as s


def line(iteration, calls, injected):
    stats = dict(groups=16, mixed_groups=16, episodes=128, injected=injected,
                 injected_own=0, train_seqs=128+injected, mean_reward=0.0)
    return f"  it {iteration}: {json.dumps(stats)} | cumulative env calls {calls}\n"


def test_reporting_merges_recovery_logs_without_duplicate_injections(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "BASE", tmp_path)
    (tmp_path / "task.log").write_text(line(0, 400, 16) + line(1, 1000, 14))
    (tmp_path / "recover1-task.log").write_text(line(1, 1000, 14) + line(2, 2000, 10))
    (tmp_path / "recover2-task.log").write_text(line(3, 3000, 0))
    rows = s.iterations("task")
    assert [r["iteration"] for r in rows] == [0, 1, 2, 3]
    assert rows[-1]["env_calls"] == 3000
    assert sum(r["injected"] for r in rows) == 40


def test_conflicting_recovery_rounds_require_an_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "BASE", tmp_path)
    (tmp_path / "task.log").write_text(line(0, 400, 16))
    (tmp_path / "recover1-task.log").write_text(line(0, 500, 16))
    with pytest.raises(ValueError, match="conflicting records"):
        s.iterations("task")


def test_failed_verification_prevents_checkpoint_cleanup(tmp_path, monkeypatch):
    model = tmp_path / "checkpoint"
    model.mkdir()
    weight = model / "weights.safetensors"
    weight.write_bytes(b"keep")
    def reject(_name):
        raise ValueError("eval incomplete")
    monkeypatch.setattr(s, "verify", reject)
    with pytest.raises(ValueError, match="eval incomplete"):
        s.finalize_verified_run("sc-math-real-s601")
    assert weight.read_bytes() == b"keep"
