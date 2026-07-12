#!/usr/bin/env python3
"""encode_worker.py — PVE-side driver that feeds ONE remote GPU from the
dir-claim queue (OPS §5). The compute path stays standalone (encode_items.py
on the rig); this loop only claims, ships, invokes, retrieves, records.

  encode_worker.py --gpu 0

Rig connection comes from /pool-ssd/fluffy/rig.env (mode 600, OUTSIDE the
repo — tailnet names never in commits):
  RIG_SSH=user@host   RIG_KEY=/root/.ssh/...   RIG_BASE=/abs/miner2
  RIG_VENV=/abs/venv/bin/python3   RIG_MODEL=/abs/Qwen3-VL-Embedding-2B
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import log

SOURCES = ["mmeb", "colpali", "visrag"]   # claim priority order


def rig_env() -> dict:
    env = {}
    with open(f"{common.FLUFFY}/rig.env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    return env


class Rig:
    def __init__(self, e: dict, gpu: int):
        self.e = e
        self.gpu = gpu
        self.ssh_opts = ["-i", e["RIG_KEY"], "-o", "BatchMode=yes",
                         "-o", "ControlMaster=auto",
                         "-o", f"ControlPath=/tmp/fluffy-enc{gpu}-%r@%h",
                         "-o", "ControlPersist=600"]

    def run(self, cmd: str, timeout: int = 3600) -> None:
        subprocess.run(["ssh", *self.ssh_opts, self.e["RIG_SSH"], cmd],
                       check=True, timeout=timeout)

    def rsync(self, args: list[str], timeout: int = 3600) -> None:
        subprocess.run(["rsync", "-a", "--partial",
                        "-e", "ssh " + " ".join(self.ssh_opts), *args],
                       check=True, timeout=timeout)


def ship_and_encode(rig: Rig, task: dict) -> None:
    e = rig.e
    base = e["RIG_BASE"]
    name = f"{task['source']}__{task['subset']}__{task['chunk']}"
    items = [json.loads(l) for l in open(task["items"])]

    # 1. CAS files needed by this chunk (incremental mirror per source)
    #    (per-PID path: two workers may share a GPU — no shared tmp files)
    lst = f"/tmp/fluffy-enc{rig.gpu}-{os.getpid()}-files.txt"
    with open(lst, "w") as f:
        for it in items:
            if it.get("image"):
                sha = it["image"]
                f.write(f"sha256/{sha[:2]}/{sha}\n")
    cas_dst = f"{base}/cas-{task['source']}"
    rig.run(f"mkdir -p {cas_dst} {base}/chunks {base}/emb")
    rig.rsync([f"--files-from={lst}", task["cas_root"] + "/",
               f"{e['RIG_SSH']}:{cas_dst}/"])

    # 2. items + script
    rig.rsync([task["items"], f"{e['RIG_SSH']}:{base}/chunks/{name}.jsonl"])

    # 3. encode on the claimed GPU
    rig.run(
        f"cd {base} && CUDA_VISIBLE_DEVICES={rig.gpu} "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {e['RIG_VENV']} "
        f"encode_items.py --items chunks/{name}.jsonl --out emb/{name}.npy "
        f"--model {e['RIG_MODEL']} --media-root {cas_dst} --batch-image 32 "
        f">> {base}/logs/encode-gpu{rig.gpu}.log 2>&1")

    # 4. retrieve + verify
    out = task["out"]
    rig.rsync([f"{e['RIG_SSH']}:{base}/emb/{name}.npy", out + ".part"])
    rig.rsync([f"{e['RIG_SSH']}:{base}/emb/{name}.npy.done", out + ".done.part"])
    done = json.load(open(out + ".done.part"))
    if done["n"] != len(items):
        raise RuntimeError(f"{name}: row count {done['n']} != {len(items)}")
    os.rename(out + ".part", out)
    os.rename(out + ".done.part", out + ".done")
    rig.run(f"rm -f {base}/chunks/{name}.jsonl {base}/emb/{name}.npy "
            f"{base}/emb/{name}.npy.done")


def chunks_state(source: str, subset: str) -> tuple[int, int]:
    stg = os.path.join(common.SRC_ROOT[source], "staging", subset)
    total = len(glob.glob(os.path.join(stg, "items-*.jsonl")))
    done = len(glob.glob(os.path.join(stg, "emb-*.npy.done")))
    return done, total


def maybe_enqueue_minepack(source: str, subset: str) -> None:
    """All extract tasks done + all chunks encoded -> minepack task."""
    if source == "mmeb":
        marker = os.path.join(common.SRC_ROOT[source], "staging", subset,
                              "extract-done.json")
        if not os.path.exists(marker):
            return
        done, total = chunks_state(source, subset)
        if total and done == total:
            common.enqueue(source, f"minepack__{source}__{subset}",
                           {"task": "minepack", "source": source,
                            "subset": subset})
    else:  # page sources: one minepack for the whole source
        pending = [p for p in glob.glob(os.path.join(
            common.task_dir(source), "extract__*.json"))]
        if pending:
            return
        if not glob.glob(os.path.join(
                common.DONE, f"extract__{source}__*.json")):
            return
        done, total = chunks_state(source, "all")
        if total and done == total:
            common.enqueue(source, f"minepack__{source}",
                           {"task": "minepack", "source": source,
                            "subset": "all"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    tag = common.worker_tag(f"gpu{args.gpu}")
    rig = Rig(rig_env(), args.gpu)
    rig.run(f"mkdir -p {rig.e['RIG_BASE']}/logs")
    rig.rsync([os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "encode_items.py"),
               f"{rig.e['RIG_SSH']}:{rig.e['RIG_BASE']}/encode_items.py"])
    log(f"enc{args.gpu}", f"worker up, tag={tag}")

    fails: dict[str, int] = {}
    while True:
        got = None
        for source in SOURCES:
            got = common.next_task(source, "encode__", tag)
            if got:
                break
        if not got:
            if args.once:
                return
            time.sleep(20)
            continue
        path, task = got
        name = os.path.basename(path)
        if fails.get(name, 0) >= 3:
            common.release(path, tag)
            time.sleep(60)
            continue
        try:
            t0 = time.time()
            if not os.path.exists(task["out"] + ".done"):
                ship_and_encode(rig, task)
            n = sum(1 for _ in open(task["items"]))
            common.complete(path, tag)
            done, total = chunks_state(task["source"], task["subset"])

            def upd(state):
                ss = common.subset_state(state, task["source"], task["subset"])
                ss["chunks_done"] = done
            common.update_state(upd)
            log(f"enc{args.gpu}", f"{name}: {n} items in "
                f"{time.time() - t0:.0f}s ({done}/{total} chunks)")
            maybe_enqueue_minepack(task["source"], task["subset"])
        except Exception as ex:  # noqa: BLE001
            fails[name] = fails.get(name, 0) + 1
            log(f"enc{args.gpu}", f"FAIL {name} (try {fails[name]}): "
                f"{type(ex).__name__}: {str(ex)[:300]}")
            common.release(path, tag)
            time.sleep(30)
        if args.once:
            return


if __name__ == "__main__":
    main()
