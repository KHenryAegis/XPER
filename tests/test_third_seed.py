import json
from pathlib import Path
import shutil
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import strengthen
import third_seed


def test_new_cohort_does_not_mutate_historical_reporting():
    old = (strengthen.BATCH, strengthen.BASE, strengthen.MATH_START, strengthen.SPECS)
    s = third_seed.cohort()
    assert s.BATCH == "third_seed_20261115"
    assert len([v for v in s.SPECS.values() if v["kind"] == "train"]) == 4
    assert (strengthen.BATCH, strengthen.BASE, strengthen.MATH_START, strengthen.SPECS) == old


@pytest.mark.parametrize("name,math,inject,cap", [
    ("td100-s603", False, True, "100"), ("td0-s603", False, False, "0"),
    ("sc-math-real-s603", True, True, "100"), ("sc-math-none-s603", True, False, "100"),
])
def test_worker_keeps_reference_recipes(tmp_path, monkeypatch, name, math, inject, cap):
    s = third_seed.cohort()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(third_seed.ROOT / "scripts/run_grpo.sh", scripts / "run_grpo.sh")
    s.ROOT, s.BASE, s.CACHE = tmp_path, tmp_path / "logs", tmp_path / "cache"
    s.MODELS, s.MATH_START = tmp_path / "models", tmp_path / "models/math-start"
    s.BASE.mkdir()
    (tmp_path / "results").mkdir()
    (s.BASE / "launch.json").write_text(json.dumps(dict(host=socket.gethostname(), specs=s.SPECS)))
    captured = {}
    def fake_run(command, **kwargs):
        captured.update(kwargs["env"])
    monkeypatch.setattr(s.subprocess, "run", fake_run)
    monkeypatch.setattr(s, "verify", lambda _: dict(verified="test"))
    s.worker(name)
    assert captured["SEED"] == "603" and captured["TASKS"] == "16" and captured["GROUP"] == "8"
    assert captured["LR"] == "1e-6" and captured["MAX_IT"] == "400"
    assert captured["INJECT_MODE"] == "fail" and captured["INJECT_CONTROL"] == "none"
    assert bool(captured["INJECT"]) == inject and captured["INJECT_LIMIT"] == cap
    assert captured["BUDGET"] == ("10403" if math else "44414")
    assert captured["SPLIT"] == "train_src_solved" and captured["THINK"] == ""
    assert captured["XPER_STRICT_MATH"] == "1"
    assert ("--env math" in captured.get("ENVARGS", "")) == math
    tag, seed = name.rsplit("-s", 1)
    script = (tmp_path / "results" / f"{tag}.s{seed}" / "driver.sh").read_text()
    assert "stage_guard.py" in script
    assert "rm -rf --" in script  # old slot cleanup after the new policy is saved


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "logs/queue/strengthen_20261113").is_dir(),
                    reason="needs the data archive (logs/queue/strengthen_20261113)")
def test_rebuilt_math_start_must_match_saved_initializer(tmp_path):
    s = third_seed.cohort()
    s.MATH_START = tmp_path
    (tmp_path / "weights.safetensors").write_bytes(b"fixture")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}))
    previous = json.loads((third_seed.ROOT / "logs/queue/strengthen_20261113/sc-math-init.verified.json").read_text())
    meta = previous["initializer"]
    (tmp_path / "xper_init.json").write_text(json.dumps(meta))
    assert s.verify(third_seed.PREP)["matches_previous_initializer"]
    meta["mode"] = "mean"
    (tmp_path / "xper_init.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="differs"):
        s.verify(third_seed.PREP)
