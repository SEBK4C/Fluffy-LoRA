"""common.py — MINE-IMAGE shared lib: paths, queue claims, atomic state.

MINING-OPS contracts implemented here:
  §1 restartability — state file /pool-ssd/fluffy/state/mine-image.json,
     atomic tmp+rename updates under an flock; every stage records counts.
  §5 scale-out — dir-claim work queue: tasks are JSON files under
     /pool-ssd/fluffy/queue/image/<source>/, a worker claims one by
     mkdir /pool-ssd/fluffy/queue/.claims/<task>__<tag> (atomic; two-phase
     re-check with deterministic lowest-tag-wins tiebreak so two hosts can
     never both hold a chunk). Finished tasks are renamed into .done/.
"""
from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import os
import socket
import time

FLUFFY = "/pool-ssd/fluffy"
QUEUE = f"{FLUFFY}/queue"
CLAIMS = f"{QUEUE}/.claims"
DONE = f"{QUEUE}/.done"
STATE_PATH = f"{FLUFFY}/state/mine-image.json"
LOGS = f"{FLUFFY}/logs"

# ORCH ruling (T9 23:13Z 2026-07-12): frozen v2 stage-1 instruction string.
# Stamped VERBATIM on every exposure until MINE-TEXTAUDIO's instruction set
# freezes (quality bar §3.1) — exposures are regenerable, cards are not.
INSTRUCTION = "Retrieve the matching description."

SEED = 20260712
SHARD_SIZE = 8192
CHUNK_ITEMS = 8192          # encode chunk = one shard's worth (OPS §5)
K_MAX = 8                   # decision D
PERCPOS = 0.95              # decision H: ceiling = 0.95 x query positive sim
MINER = "vl-ann-v1"
ENCODER = "Qwen/Qwen3-VL-Embedding-2B"

SRC_ROOT = {
    "mmeb": f"{FLUFFY}/image-mmeb",
    "colpali": f"{FLUFFY}/image-colpali",
    "visrag": f"{FLUFFY}/image-visrag",
}


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: {msg}", flush=True)


def worker_tag(extra: str = "") -> str:
    t = f"{socket.gethostname()}-{os.getpid()}"
    return f"{t}-{extra}" if extra else t


def cas_path(root: str, sha: str) -> str:
    return os.path.join(root, "cas", "sha256", sha[:2], sha)


def cas_store(root: str, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    p = cas_path(root, sha)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.rename(tmp, p)
    return sha


def item_id(kind: str, text: str | None, sha: str | None) -> str:
    """Stable id for an encode item (dedups identical encode work)."""
    h = hashlib.sha256(f"{kind}|{sha or ''}|{text or ''}".encode())
    return h.hexdigest()[:32]


# --- queue ------------------------------------------------------------------

def task_dir(source: str) -> str:
    return f"{QUEUE}/image/{source}"


def enqueue(source: str, name: str, payload: dict) -> str:
    """Create task <name>.json atomically (O_EXCL; exists = already queued)."""
    d = task_dir(source)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    if os.path.exists(path) or os.path.exists(os.path.join(DONE, f"{name}.json")):
        return path
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    try:
        os.link(tmp, path)          # atomic create-if-absent
    except FileExistsError:
        pass
    finally:
        os.unlink(tmp)
    return path


def deps_met(payload: dict) -> bool:
    for dep in payload.get("after", []):
        if not os.path.exists(os.path.join(DONE, f"{dep}.json")):
            return False
    return True


def claim(task_path: str, tag: str) -> bool:
    """Two-phase atomic-mkdir claim (OPS §5). True = ours."""
    base = os.path.basename(task_path).removesuffix(".json")
    os.makedirs(CLAIMS, exist_ok=True)
    mine = os.path.join(CLAIMS, f"{base}__{tag}")
    if glob.glob(os.path.join(CLAIMS, f"{base}__*")):
        return False
    try:
        os.mkdir(mine)
    except FileExistsError:
        return False
    holders = sorted(glob.glob(os.path.join(CLAIMS, f"{base}__*")))
    if os.path.basename(holders[0]) != f"{base}__{tag}":
        os.rmdir(mine)              # lost the tiebreak
        return False
    heartbeat(task_path, tag)
    return True


def heartbeat(task_path: str, tag: str) -> None:
    base = os.path.basename(task_path).removesuffix(".json")
    hb = os.path.join(CLAIMS, f"{base}__{tag}", "hb")
    with open(hb, "w") as f:
        f.write(str(time.time()))


def release(task_path: str, tag: str) -> None:
    base = os.path.basename(task_path).removesuffix(".json")
    d = os.path.join(CLAIMS, f"{base}__{tag}")
    hb = os.path.join(d, "hb")
    if os.path.exists(hb):
        os.unlink(hb)
    if os.path.isdir(d):
        os.rmdir(d)


def complete(task_path: str, tag: str) -> None:
    os.makedirs(DONE, exist_ok=True)
    os.rename(task_path, os.path.join(DONE, os.path.basename(task_path)))
    release(task_path, tag)


def next_task(source: str, prefix: str, tag: str) -> tuple[str, dict] | None:
    """Claim the next unclaimed dep-satisfied task with this prefix."""
    for path in sorted(glob.glob(os.path.join(task_dir(source), f"{prefix}*.json"))):
        try:
            payload = json.load(open(path))
        except (json.JSONDecodeError, FileNotFoundError):
            continue                # mid-create or just completed
        if not deps_met(payload):
            continue
        if claim(path, tag):
            if not os.path.exists(path):        # completed during claim
                release(path, tag)
                continue
            return path, payload
    return None


# --- state (OPS §1) ----------------------------------------------------------

def update_state(fn) -> dict:
    """Read-modify-write mine-image.json under flock, atomic rename."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    lock = open(f"{STATE_PATH}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        state = {}
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                state = json.load(f)
        state.setdefault("agent", "mine-image")
        state.setdefault("sources", {})
        fn(state)
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = f"{STATE_PATH}.tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1, sort_keys=True)
        os.rename(tmp, STATE_PATH)
        return state
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def subset_state(state: dict, source: str, subset: str) -> dict:
    src = state["sources"].setdefault(source, {"subsets": {}})
    return src["subsets"].setdefault(subset, {})
