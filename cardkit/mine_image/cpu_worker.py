#!/usr/bin/env python3
"""cpu_worker.py — PVE CPU worker: claims extract/minepack tasks from the
dir-claim queue and runs the standalone stage scripts as subprocesses.
Launch N of these under nohup (logs to /pool-ssd/fluffy/logs/).

  cpu_worker.py [--roles extract,minepack] [--once]

A heartbeat thread touches the claim dir every 60 s while a stage runs
(stale-claim detection stays meaningful on multi-hour subsets).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
from common import log

SOURCES = ["mmeb", "colpali", "visrag"]


def stage_cmd(payload: dict) -> list[str]:
    if payload["task"] == "extract":
        if payload["source"] == "mmeb":
            return [sys.executable, os.path.join(HERE, "extract_mmeb.py"),
                    "--subset", payload["subset"]]
        return [sys.executable, os.path.join(HERE, "extract_pages.py"),
                "--source", payload["source"], "--file", str(payload["file"])]
    return [sys.executable, os.path.join(HERE, "minepack.py"),
            "--source", payload["source"], "--subset", payload["subset"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="extract,minepack")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    prefixes = [f"{r}__" for r in args.roles.split(",")]
    tag = common.worker_tag("cpu")
    log("cpu", f"worker up, tag={tag}, roles={args.roles}")

    fails: dict[str, int] = {}
    while True:
        got = None
        for source in SOURCES:             # source-major: mmeb drains first
            for prefix in prefixes:
                got = common.next_task(source, prefix, tag)
                if got:
                    break
            if got:
                break
        if not got:
            if args.once:
                return
            time.sleep(30)
            continue
        path, payload = got
        name = os.path.basename(path)
        if fails.get(name, 0) >= 2:
            common.release(path, tag)
            time.sleep(120)
            continue
        stop = threading.Event()

        def beat():
            while not stop.wait(60):
                try:
                    common.heartbeat(path, tag)
                except FileNotFoundError:
                    return
        th = threading.Thread(target=beat, daemon=True)
        th.start()
        try:
            t0 = time.time()
            log("cpu", f"RUN {name}")
            subprocess.run(stage_cmd(payload), check=True, timeout=4 * 3600)
            common.complete(path, tag)
            log("cpu", f"DONE {name} in {time.time() - t0:.0f}s")
            # extract done may complete a subset whose last encode chunk
            # already returned — re-check minepack readiness
            if payload["task"] == "extract":
                from encode_worker import maybe_enqueue_minepack
                maybe_enqueue_minepack(payload["source"],
                                       payload.get("subset", "all"))
        except Exception as ex:  # noqa: BLE001
            fails[name] = fails.get(name, 0) + 1
            log("cpu", f"FAIL {name} (try {fails[name]}): "
                f"{type(ex).__name__}: {str(ex)[:300]}")
            common.release(path, tag)
            time.sleep(30)
        finally:
            stop.set()
        if args.once:
            return


if __name__ == "__main__":
    main()
