"""Restore the published source stores, preserving shard order and trajectory IDs."""
import gzip
import hashlib
from pathlib import Path
import shutil
import sqlite3
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "telecom": ("results/src", 8, (2400, 223, 44414)),
    "math": ("results/math/src", 2, (2000, 452, 10403)),
}


def restore(environment, destination):
    _, count, expected = SOURCES[environment]
    destination.mkdir(parents=True, exist_ok=True)
    successful_tasks = []
    episodes = calls = 0
    with tempfile.TemporaryDirectory(dir=destination) as staging:
        staged = []
        for i in range(count):
            name = f"4b-inst.shard{i}.db"
            target = destination / name
            temp = Path(staging) / name
            with gzip.open(ROOT / "data/source" / environment / f"{name}.gz", "rb") as stream, temp.open("wb") as out:
                shutil.copyfileobj(stream, out)
            with sqlite3.connect(temp.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
                if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise ValueError(f"corrupt {environment} source shard {i}")
                n, spent = db.execute("SELECT COUNT(*), SUM(env_calls) FROM trajectories").fetchone()
                successful_tasks.extend(t for (t,) in db.execute("SELECT task_id FROM trajectories WHERE success = 1"))
                episodes += n
                calls += spent
            if target.exists():
                with target.open("rb") as existing, temp.open("rb") as restored:
                    if hashlib.file_digest(existing, "sha256").digest() != hashlib.file_digest(restored, "sha256").digest():
                        raise ValueError(f"{target} already contains different trajectories")
            else:
                staged.append((temp, target))
        if (episodes, len(successful_tasks), calls) != expected:
            raise ValueError(f"{environment}: source episode, success or call count differs")
        for temp, target in staged:
            temp.rename(target)
    return successful_tasks


def published_successes(environment):
    with tempfile.TemporaryDirectory() as folder:
        return restore(environment, Path(folder))


if __name__ == "__main__":
    for env, (folder, _, _) in SOURCES.items():
        successes = restore(env, ROOT / folder)
        print(f"Restored {env} source stores in {folder} ({len(successes)} successes)")
