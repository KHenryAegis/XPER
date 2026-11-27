"""Checks for the reported task-bootstrap intervals and evaluation records."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reproduce as analysis


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


def test_evaluation_rejects_duplicate_samples_with_correct_total_count(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "RECORDS", tmp_path)
    (tmp_path / "eval").mkdir()
    with gzip.open(tmp_path / "eval/model.jsonl.gz", "wt") as stream:
        for reward in (1, 0):
            print(json.dumps({"task": "a", "k": 0, "reward": reward}), file=stream)
    with pytest.raises(ValueError, match="duplicate"):
        analysis.episodes("eval", "model", tasks=1, samples=2)
