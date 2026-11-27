"""Fetch the model snapshots and ToolACE used in the paper."""
import json
import os
from pathlib import Path
import shutil
import urllib.request

from huggingface_hub import snapshot_download

from check_inputs import ROOT, sha256


def main():
    home = os.environ.get("XPER_HOME")
    if not home:
        raise SystemExit("Set XPER_HOME to the parent of models/")
    manifest = json.loads((ROOT / "data/datasets.json").read_text())
    ace = manifest["datasets"]["toolace"]
    data = ROOT / ace["local_path"]
    if not data.exists():
        data.parent.mkdir(parents=True, exist_ok=True)
        temp = data.with_suffix(".download")
        with urllib.request.urlopen(ace["url"], timeout=60) as src, temp.open("wb") as out:
            shutil.copyfileobj(src, out)
        if sha256(temp) != ace["sha256"]:
            raise ValueError("ToolACE download does not match the recorded SHA-256")
        temp.replace(data)
    if sha256(data) != ace["sha256"]:
        raise ValueError(f"Existing ToolACE input has the wrong SHA-256: {data}")
    for name, spec in manifest["models"].items():
        if name.startswith("_"):
            continue
        folder = Path(home) / "models/hf" / name
        index = folder / "model.safetensors.index.json"
        if index.is_file() and sha256(index) == spec["index_sha256"]:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if all((folder / shard).is_file() and (folder / shard).stat().st_size for shard in shards):
                print(f"Using {folder}")
                continue
        snapshot_download(repo_id=spec["huggingface"], revision=spec["revision"], local_dir=folder,
                          allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE*"])
        if sha256(index) != spec["index_sha256"]:
            raise ValueError(f"Model index does not match for {name}")


if __name__ == "__main__":
    main()
