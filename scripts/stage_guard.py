"""Wait for idle assigned GPUs and retry OOM without restarting an experiment.

Rollout retries are allowed only before episode reset, so no environment calls
are silently dropped. All subprocess groups signalled here were created here
and their members must carry this invocation's random ownership token.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
import uuid


def idle(gpus):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = {int(i): (u.strip(), int(m)) for i, u, m in (line.split(",") for line in out.splitlines())}
    procs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                     "--format=csv,noheader"], text=True)
    busy = set(procs.split())
    return all(g in rows and rows[g][1] < 1000 and rows[g][0] not in busy for g in gpus)


def retryable(kind, output):
    transient = any(x in output.lower() for x in (
        "out of memory", "free memory on device", "address already in use"))
    if kind in ("rollout", "eval") and re.search(r"^reset \d+ episodes", output, re.M):
        return False
    return transient


def stop_owned_group(pgid, token):
    members = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if os.getpgid(int(p.name)) != pgid:
                continue
            status = (p / "stat").read_text().split()
            if status[2] == "Z":
                continue
            env = (p / "environ").read_bytes().split(b"\0")
            if f"XPER_STAGE_OWNER={token}".encode() not in env:
                raise RuntimeError("refusing to signal a group with an unverified member")
            members.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    if members:
        os.killpg(pgid, signal.SIGTERM)


def main(a):
    gpus = [int(x) for x in re.split(r"[ ,]+", a.gpus.strip())]
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not cmd:
        raise ValueError("missing stage command")
    log = Path(a.log)
    state_path = log.with_name(log.name + ".stage.json")
    active = {"child": None, "token": None}
    def record(status, **extra):
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(status=status, kind=a.kind, gpus=gpus,
                                      guard_pid=os.getpid(), updated=time.time(), **extra)) + "\n")
        tmp.replace(state_path)
    def interrupted(sig, _frame):
        child = active["child"]
        if child is not None:
            stop_owned_group(child.pid, active["token"])
            child.wait(timeout=30)
        record("terminated", signal=sig)
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if log.exists() and os.environ.get("XPER_RECOVERY_ATTEMPT"):
        before = log.with_name(log.name + ".before-recovery")
        if not before.exists():
            shutil.copyfile(log, before)
    attempt = 0
    while True:
        print(f"stage {a.kind}: waiting for idle GPUs {gpus}", flush=True)
        record("waiting", attempts=attempt)
        stable = 0
        while stable < 2:
            stable = stable + 1 if idle(gpus) else 0
            if stable < 2:
                time.sleep(a.poll)
        attempt += 1
        args = list(cmd)
        if a.kind == "update" and "--master_port" in args:
            with socket.socket() as sock:
                sock.bind(("", 0))
                args[args.index("--master_port") + 1] = str(sock.getsockname()[1])
        token = uuid.uuid4().hex
        print(f"stage {a.kind}: attempt {attempt} on GPUs {gpus}", flush=True)
        with log.open("w") as output:
            child = subprocess.Popen(args, env=dict(os.environ, XPER_STAGE_OWNER=token),
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            active.update(child=child, token=token)
            record("running", attempt=attempt, pid=child.pid, pgid=child.pid, owner_token=token)
            rc = child.wait()
        stop_owned_group(child.pid, token)
        active.update(child=None, token=None)
        if rc == 0:
            record("done", attempts=attempt)
            return 0
        content = log.read_text(errors="replace")
        if not retryable(a.kind, content):
            print(f"stage {a.kind}: non-retryable failure {rc}; preserving {log}", flush=True)
            record("failed", attempts=attempt, returncode=rc)
            return rc if rc > 0 else 1
        first = log.with_name(log.name + ".oom-first")
        shutil.copyfile(log, first if not first.exists() else log.with_name(log.name + ".oom-last"))
        retry_path = log.with_name(log.name + ".retry.json")
        retry_path.write_text(json.dumps(dict(attempts=attempt, reason="transient resource failure",
                                          kind=a.kind, gpus=gpus)) + "\n")
        print(f"stage {a.kind}: transient resource failure; own group cleaned, waiting to retry", flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", required=True)
    p.add_argument("--kind", choices=("rollout", "update", "eval"), required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--poll", type=int, default=15)
    p.add_argument("command", nargs=argparse.REMAINDER)
    raise SystemExit(main(p.parse_args()))
