"""Statistical and evidence-integrity checks for the paper analysis."""
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("coldstart_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_coldstart.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_task_clusters_remain_paired_between_identical_arms():
    a = np.array([[0, 0, 1, 1], [.125, .5, 1, 0]])
    boot = analysis.bootstrap_arms(np.stack([a, a]), draws=2000)
    np.testing.assert_array_equal(boot[0], boot[1])
    assert np.ptp(boot[0]) > 0


def test_repeating_same_policy_does_not_artificially_narrow_task_interval():
    # Two tasks, each with perfectly correlated samples: extra copies of the
    # same policy must not create more independent tasks or smaller intervals.
    single = np.array([[[0, 1]]])
    repeated = np.repeat(single, 8, axis=1)
    a = analysis.bootstrap_arms(single, draws=1000)
    b = analysis.bootstrap_arms(repeated, draws=1000)
    np.testing.assert_array_equal(a, b)
    assert set(a[0]) == {0, .5, 1}


def test_evaluation_rejects_duplicate_samples_with_correct_total_count(tmp_path):
    path = tmp_path / "eval.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE trajectories(task_id TEXT,reward REAL,success INT,model_id TEXT,meta TEXT)")
        db.executemany("INSERT INTO trajectories VALUES(?,?,?,?,?)", [
            ("a", 1, 1, "model", json.dumps({"k": 0})),
            ("a", 0, 0, "model", json.dumps({"k": 0})),
        ])
    with pytest.raises(ValueError, match="missing/duplicate"):
        analysis.load_eval(path, tasks=1, samples=2, model="model")


def test_recovery_merge_rejects_conflicting_completed_rounds(tmp_path):
    first, recovered = tmp_path / "original.log", tmp_path / "recovery.log"
    first.write_text('it 0: {"episodes": 128} | cumulative env calls 100\n')
    recovered.write_text(first.read_text())
    assert len(analysis.merge_history([first, recovered])) == 1
    recovered.write_text('it 0: {"episodes": 128} | cumulative env calls 101\n')
    with pytest.raises(ValueError, match="conflicting"):
        analysis.merge_history([first, recovered])


def test_post_injection_summary_excludes_the_injection_round():
    rows = [dict(iteration=i, injected=x, env_calls=10*(i+1), target_successes=s, episodes=8)
            for i, (x, s) in enumerate([(1, 0), (1, 1), (0, 4), (0, 8)])]
    result = analysis.after_injection(rows)
    assert result["post_injection_rounds"] == 2
    assert result["post_injection_calls"] == 20
    assert result["post_injection_target_success_rate"] == .75
