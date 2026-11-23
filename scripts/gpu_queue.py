"""A small GPU job queue for a shared box. Run it on the GPU machine under nohup.

Jobs live in a JSONL file that may be appended to while the queue runs; each line is
  {"name": "...", "gpus": 1, "cmd": "bash ...", "after": ["other-name", ...]}
A job starts when every job in "after" finished successfully and enough GPUs are free. A GPU is free
when this queue has not handed it out and nvidia-smi shows no process and < FREE_MB used on it, so
other users' jobs are never preempted: the queue waits. The command gets CUDA_VISIBLE_DEVICES and
GPUS (space-separated ids). State: <jobs>.state.json; logs: logs/queue/<name>.log.

Several machines can serve one jobs file: give each queue its own --state and a shared --claim-dir. A job
is started only by the queue that atomically creates <claim-dir>/<name> (O_EXCL on the shared FS), so it
never runs twice. The state file records each job's process group ("pgid"), so exactly that job can be
stopped later (kill -- -<pgid>) without touching other users' processes.

Usage: nohup python scripts/gpu_queue.py --jobs logs/queue/jobs.jsonl > logs/queue/queue.log 2>&1 &
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

FREE_MB = 1000


def gpu_status():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    used = {int(i): int(m) for i, m in (l.split(",") for l in out.strip().splitlines())}
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout
    busy_uuid = set(procs.split()) if procs else set()
    busy = {int(i) for i, u in (l.split(", ") for l in uuids.strip().splitlines()) if u.strip() in busy_uuid}
    return {g for g, m in used.items() if m < FREE_MB and g not in busy}


def load_jobs(path):
    jobs = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            j = json.loads(line)
            jobs[j["name"]] = j
    return jobs


def main(a):
    logdir = Path(a.jobs).parent
    state_p = Path(a.state or a.jobs + ".state.json")
    claims = Path(a.claim_dir) if a.claim_dir else None
    if claims:
        claims.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    state = {k: v for k, v in state.items() if v.get("status") != "running"}  # restart: rerun
    running = {}  # name -> (Popen, gpus)
    while True:
        jobs = load_jobs(a.jobs)
        for name, (p, gpus) in list(running.items()):
            if p.poll() is not None:
                if a.done_dir:
                    Path(a.done_dir, f"{name}.{'ok' if p.returncode == 0 else 'fail'}").write_text(time.strftime("%m-%d %H:%M\n"))
                state[name] = {"status": "ok" if p.returncode == 0 else f"fail({p.returncode})",
                               "gpus": gpus, "end": time.strftime("%m-%d %H:%M")}
                print(f"[{time.strftime('%m-%d %H:%M')}] {name} -> {state[name]['status']}", flush=True)
                del running[name]
        mine = {g for _, gs in running.values() for g in gs}
        free = sorted(gpu_status() - mine)
        for name, j in jobs.items():
            if name in state or name in running:
                continue
            deps = j.get("after", [])
            done_dir = Path(a.done_dir) if a.done_dir else None  # cross-host deps: <done_dir>/<name>.ok|.fail
            ok = lambda d: state.get(d, {}).get("status") == "ok" or bool(done_dir and (done_dir / f"{d}.ok").exists())
            bad = lambda d: state.get(d, {}).get("status", "").startswith("fail") or bool(done_dir and (done_dir / f"{d}.fail").exists())
            if any(bad(d) for d in deps):
                state[name] = {"status": "skipped (dependency failed)"}
                print(f"[{time.strftime('%m-%d %H:%M')}] {name} skipped", flush=True)
                continue
            if not all(ok(d) for d in deps):
                continue
            n = j.get("gpus", 1)
            if len(free) < n:
                continue
            if claims:
                try:
                    fd = os.open(claims / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue  # claimed by another machine's queue
                os.write(fd, f"{socket.gethostname()} {a.host_label} {time.strftime('%m-%d %H:%M')}\n".encode())
                os.close(fd)
            gpus, free = free[:n], free[n:]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), GPUS=" ".join(map(str, gpus)))
            log = open(logdir / f"{name}.log", "a")
            p = subprocess.Popen(["bash", "-c", j["cmd"]], cwd=a.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            running[name] = (p, gpus)
            print(f"[{time.strftime('%m-%d %H:%M')}] start {name} on {a.host_label} GPUs {gpus} (pgid {p.pid})",
                  flush=True)
        state_p.write_text(json.dumps({**state, **{k: {"status": "running", "gpus": g, "pgid": pr.pid,
                                                       "host": a.host_label}
                                                   for k, (pr, g) in running.items()}}, indent=1))
        if a.exit_when_done and not running and all(n in state for n in jobs):
            failed = sorted(n for n in jobs if state[n]["status"] != "ok")
            print(f"[{time.strftime('%m-%d %H:%M')}] all {len(jobs)} jobs finished"
                  + (f"; not ok: {', '.join(failed)}" if failed else ""), flush=True)
            return 1 if failed else 0
        time.sleep(a.poll)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--cwd", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--state", default="", help="state file (default <jobs>.state.json); one per machine")
    p.add_argument("--claim-dir", default="", help="shared dir of per-job claim files (multi-machine)")
    p.add_argument("--host-label", default=socket.gethostname())
    p.add_argument("--done-dir", default="", help="shared dir of <job>.ok/.fail markers so 'after' works across hosts")
    p.add_argument("--exit-when-done", action="store_true",
                   help="exit once every job has finished (single host; non-zero if any job did not succeed)")
    sys.exit(main(p.parse_args()))
