"""Check the external inputs that are not shipped with the artifact (listed in data/datasets.json).

    python -B tools/check_inputs.py

Reports, without changing anything: the tau2-bench checkout and its commit, the math and ToolACE files (by
SHA-256), and the model indices and shard files under $XPER_HOME/models/hf.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def status(label, ok, detail):
    print(f"{'OK  ' if ok else 'MISS'} {label:34s} {detail}")


def main():
    spec = json.loads((ROOT / "data/datasets.json").read_text())
    ds = spec["datasets"]

    tau = ds["tau2-bench-telecom"]
    repo = ROOT / tau["local_path"]
    head = None
    if (repo / ".git").exists():
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    valid = head == tau["commit"]
    status("tau2-bench checkout", valid, f"HEAD {head or 'none'}; expected {tau['commit']} ({tau['obtain']})")

    files = [(ds[k]["local_path"], ds[k]["bytes"], ds[k]["sha256"], ds[k]["used_for"]) for k in ("dapo-math-integer", "toolace")]
    d = ds["toolace"]["derived"]
    if (ROOT / d["path"]).exists():
        files.append((d["path"], d["bytes"], d["sha256"], f"built by {d['built_by']}"))
    for rel, size, digest, why in files:
        path = ROOT / rel
        ok = path.is_file() and path.stat().st_size == size and sha256(path) == digest
        valid &= ok
        status(rel, ok, "sha256 matches" if ok else
               ("present but differs from the recorded SHA-256" if path.exists() else "not present") + f" ({why})")

    home = os.environ.get("XPER_HOME")
    for name, rec in spec["models"].items():
        if name.startswith("_"):
            continue
        path = Path(home) / "models/hf" / name if home else None
        index = path / "model.safetensors.index.json" if path else None
        ok = bool(index and index.is_file() and sha256(index) == rec["index_sha256"])
        if ok:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            ok = (path / "config.json").is_file() and all((path / shard).is_file() for shard in shards)
        valid &= ok
        status(f"model {name}", ok, f"{path if path else '$XPER_HOME not set'} ({rec['huggingface']}; GPU runs only)")

    return valid


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
