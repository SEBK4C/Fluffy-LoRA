#!/usr/bin/env python3
"""mine_ta_lib.py — shared helpers for the MINE-TEXTAUDIO lanes.

MINING-OPS contract pieces implemented here:
  - durable state file, atomic tmp+rename updates (§1)
  - dir-claim queue: claim = atomic mkdir under queue/.claims (§5)
  - teacher client for :9020 (llama-server /v1/embeddings, Qwen3-Emb-8B):
    batched, retry-with-wait — NEVER treats a down teacher as fatal, it
    waits and logs (the teacher must never be left down; it may be busy
    with judge/query-gen work).

All compute-path scripts import this and stay runnable standalone
(args: --chunk file, --out dir) per MINING-OPS §5.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

FLUFFY = "/pool-ssd/fluffy"
QUEUE = os.path.join(FLUFFY, "queue")
CLAIMS = os.path.join(QUEUE, ".claims")
LOGS = os.path.join(FLUFFY, "logs")
STATE_PATH = os.path.join(FLUFFY, "state", "mine-textaudio.json")
TEACHER_URL = os.environ.get("TEACHER_URL", "http://127.0.0.1:9020")
HOSTNAME = socket.gethostname()


def log(tag: str, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    subprocess.run(["logger", "-t", f"mine-ta-{tag}"], input=msg.encode(),
                   check=False)


# ------------------------------------------------------------- state file --

def update_state(source: str, **fields) -> None:
    """Atomic read-modify-write of the agent state file (tmp+rename),
    serialized by an fcntl lock so parallel workers don't lose updates."""
    lock_path = STATE_PATH + ".lock"
    with open(lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(STATE_PATH) as f:
                state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            state = {"agent": "MINE-TEXTAUDIO", "sources": {}}
        src = state["sources"].setdefault(source, {})
        src.update(fields)
        src["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = STATE_PATH + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, STATE_PATH)


def read_state(source: str) -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f).get("sources", {}).get(source, {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ------------------------------------------------------------ claim queue --

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, other uid


def claim(chunk_id: str) -> bool:
    """Atomic mkdir claim per MINING-OPS §5. True = ours.

    Same-host semantics: the claim dir carries an owner file {pid}. If the
    dir already exists, the claim is ours ONLY if the recorded pid is dead
    (a killed worker on THIS host — resuming our own work) or is us.
    A live sibling worker's claim is respected. Cross-host claims are never
    touched (stale-claim breaking is Opus-manager-only per MINING-OPS)."""
    os.makedirs(CLAIMS, exist_ok=True)
    path = os.path.join(CLAIMS, f"{chunk_id}__{HOSTNAME}")
    owner = os.path.join(path, "owner")
    try:
        os.mkdir(path)
    except FileExistsError:
        try:
            with open(owner) as f:
                rec = json.load(f)
            if rec["pid"] != os.getpid() and _pid_alive(rec["pid"]):
                return False  # live sibling worker owns it
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass  # torn owner file -> take over
    except OSError:
        return False
    # write/refresh ownership under a lock (two takeovers racing)
    with open(os.path.join(path, ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(owner) as f:
                rec = json.load(f)
            if rec["pid"] != os.getpid() and _pid_alive(rec["pid"]):
                return False  # lost the takeover race
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        tmp = owner + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"pid": os.getpid(), "t": time.time()}, f)
        os.replace(tmp, owner)
    with open(os.path.join(path, "claimed_at"), "w") as f:
        f.write(str(time.time()))
    return True


def claimed_elsewhere(chunk_id: str) -> bool:
    if not os.path.isdir(CLAIMS):
        return False
    for d in os.listdir(CLAIMS):
        if d.startswith(chunk_id + "__") and d != f"{chunk_id}__{HOSTNAME}":
            return True
    return False


def touch_claim(chunk_id: str) -> None:
    p = os.path.join(CLAIMS, f"{chunk_id}__{HOSTNAME}", "claimed_at")
    if os.path.isdir(os.path.dirname(p)):
        with open(p, "w") as f:
            f.write(str(time.time()))


# ---------------------------------------------------------------- teacher --

def teacher_up() -> bool:
    try:
        with urllib.request.urlopen(TEACHER_URL + "/health", timeout=5) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def embed(texts: list[str], tag: str = "embed", batch: int = 64,
          normalize: bool = True):
    """Embed texts via the :9020 teacher. Blocks-and-waits through teacher
    downtime (poll every 30 s, log every 5 min) — mining pauses, never
    fails, and never restarts the teacher itself (it is shared)."""
    import numpy as np
    out = []
    i = 0
    down_since = None
    while i < len(texts):
        chunk = texts[i:i + batch]
        body = json.dumps({"input": chunk, "model": "q"}).encode()
        req = urllib.request.Request(
            TEACHER_URL + "/v1/embeddings", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())["data"]
            if down_since is not None:
                log(tag, f"teacher back after {time.time()-down_since:.0f}s")
                down_since = None
            out.extend(d["embedding"] for d in data)
            i += batch
        except Exception as e:  # noqa: BLE001
            if down_since is None:
                down_since = time.time()
                log(tag, f"teacher unreachable ({e}); waiting (poll 30s)")
            elif time.time() - down_since > 300:
                log(tag, f"teacher still down {time.time()-down_since:.0f}s")
                down_since = time.time() - 1  # re-arm the 5-min logger
            time.sleep(30)
    arr = np.asarray(out, dtype=np.float32)
    if normalize:
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(n, 1e-8, None)
    return arr


# -------------------------------------------------------------------- CAS --

def cas_write(root: str, data: bytes, ext: str | None = None) -> str:
    """Write bytes into <root>/cas/sha256/<2>/<sha>; returns sha256."""
    sha = hashlib.sha256(data).hexdigest()
    d = os.path.join(root, "cas", "sha256", sha[:2])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, sha)
    if not os.path.exists(path):
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return sha


def cas_path(root: str, sha: str) -> str:
    return os.path.join(root, "cas", "sha256", sha[:2], sha)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


if __name__ == "__main__":
    print(json.dumps({"teacher_up": teacher_up(), "state": STATE_PATH,
                      "queue": QUEUE, "host": HOSTNAME}))
