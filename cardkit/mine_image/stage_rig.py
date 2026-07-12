#!/usr/bin/env python3
"""stage_rig.py — store-2 sync (OPS §2): shards -> rig HDD + verify + gate.

  stage_rig.py --source mmeb [--target-sps 60]

Per source: rsync OUT_ROOT/shards/ (tars + .idx.json + MANIFEST + SHA256SUMS)
-> rig:/pool-5tb/fluffy/shards/image/<source>/, then rig-side
`sha256sum -c SHA256SUMS` in EVERY subset dir, then the ALIGN readback gate
(HDD this time — the warmup gate ran on SSD, noted in its record). Results
land in the state file; failures raise (nothing marked staged on a red).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import log

RIG_SHARDS = "/pool-5tb/fluffy/shards/image"
READBACK = "/mnt/proxmox/llm-serve/fluffy/readback_gate.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target-sps", type=int, default=60)
    args = ap.parse_args()
    source = args.source
    root = common.SRC_ROOT[source]
    sdir = os.path.join(root, "shards")
    if not os.path.isdir(sdir):
        raise SystemExit(f"{sdir}: nothing packed yet")

    e = {}
    with open(f"{common.FLUFFY}/rig.env") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                e[k] = v
    ssh = ["ssh", "-i", e["RIG_KEY"], "-o", "BatchMode=yes", e["RIG_SSH"]]
    dst = f"{RIG_SHARDS}/{source}"

    t0 = time.time()
    subprocess.run([*ssh, f"mkdir -p {dst}"], check=True)
    subprocess.run(["rsync", "-a", "--partial", "--info=stats2",
                    "-e", f"ssh -i {e['RIG_KEY']} -o BatchMode=yes",
                    sdir + "/", f"{e['RIG_SSH']}:{dst}/"],
                   check=True, timeout=6 * 3600)
    rsync_s = round(time.time() - t0, 1)
    log("stage", f"{source}: rsync done in {rsync_s}s")

    r = subprocess.run([*ssh,
        f"cd {dst} && for d in */; do (cd $d && sha256sum -c SHA256SUMS "
        f">/dev/null && echo OK $d || echo FAIL $d); done"],
        capture_output=True, text=True, check=True, timeout=3600)
    lines = r.stdout.strip().splitlines()
    fails = [l for l in lines if l.startswith("FAIL")]
    log("stage", f"{source}: sha -c on rig — {len(lines) - len(fails)} OK, "
        f"{len(fails)} FAIL")
    if fails:
        raise SystemExit(f"sha256sum -c FAILED on rig: {fails}")

    r = subprocess.run([*ssh,
        f"{e['RIG_VENV']} {READBACK} --dir {dst} --seconds 120 "
        f"--target {args.target_sps}"],
        capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:])
    gate_pass = r.returncode == 0
    if not gate_pass:
        print(r.stderr[-1000:])
        raise SystemExit(f"readback gate FAILED for {source}")

    def upd(state):
        src = state["sources"].setdefault(source, {"subsets": {}})
        src["staged_rig"] = {
            "dst": dst, "sha_check": f"{len(lines)} dirs OK",
            "readback": "PASS (HDD, 120s, 10x target "
                        f"{args.target_sps} sps)",
            "rsync_s": rsync_s,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    common.update_state(upd)
    log("stage", f"{source}: STAGED + VERIFIED + READBACK PASS")


if __name__ == "__main__":
    main()
