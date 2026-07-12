#!/usr/bin/env python3
"""mine_ta_crossdedup.py — MINING-OPS §3.4 cross-source near-dup sweep:
teacher-embedding cosine >= 0.95 across ALL text (and captions when
MINE-IMAGE contributes theirs) AFTER mining, not just within source.

Standalone; CPU works, a GPU (torch+cuda) makes 2-3M vectors take minutes.
Inputs: --spec JSON file listing sources:
  [{"name": "kalm-paq", "npz": ".../kalm-paq-000.emb.npz", "key": "p",
    "priority": 10}, ...]
  (glob patterns allowed in "npz"; "key" = array inside the npz/npy)
Priority: LOWER number wins (kept); higher-priority duplicates are dropped.
Output: --out drop list JSONL {source, index, dup_of_source, dup_of_index,
sim} — packers/relaunch mix consume it to skip dropped exposures.

Only CROSS-source pairs are reported (within-source handled by each
miner's own dedup + the TopK-PercPos ceiling).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mine_ta_lib as lib  # noqa: E402

THRESH = 0.95


def log(msg: str) -> None:
    lib.log("crossdedup", msg)


def load_spec(spec_path: str):
    import numpy as np
    spec = json.load(open(spec_path))
    names, owner, arrays = [], [], []
    for s in sorted(spec, key=lambda x: x.get("priority", 100)):
        paths = sorted(sum((glob.glob(p) for p in
                            ([s["npz"]] if isinstance(s["npz"], str)
                             else s["npz"])), []))
        parts = []
        for p in paths:
            if p.endswith(".npz"):
                with np.load(p) as z:
                    parts.append(z[s.get("key", "p")].astype(np.float16))
            else:
                parts.append(np.load(p).astype(np.float16))
        if not parts:
            log(f"WARN: no arrays for {s['name']}")
            continue
        a = np.vstack(parts)
        names.append(s["name"])
        owner.append((len(arrays), a.shape[0]))
        arrays.append(a)
        log(f"loaded {s['name']}: {a.shape[0]} x {a.shape[1]}")
    return names, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")  # cuda:0 on a rig slot
    ap.add_argument("--tile", type=int, default=16384)
    a = ap.parse_args()
    import numpy as np
    names, arrays = load_spec(a.spec)
    sizes = [x.shape[0] for x in arrays]
    starts = np.cumsum([0] + sizes)
    big = np.vstack(arrays).astype(np.float16)
    n = big.shape[0]
    src_of = np.zeros(n, dtype=np.int32)
    for i, (s0, sz) in enumerate(zip(starts, sizes)):
        src_of[s0:s0 + sz] = i
    log(f"total {n} vectors from {len(names)} sources")

    use_torch = a.device.startswith("cuda")
    if use_torch:
        import torch
        t_big = torch.from_numpy(big).to(a.device)
    drops = []
    T = a.tile
    for i0 in range(0, n, T):
        i1 = min(i0 + T, n)
        if use_torch:
            import torch
            with torch.no_grad():
                sims = (t_big[i0:i1].float() @ t_big.T.float()).cpu().numpy()
        else:
            sims = big[i0:i1].astype(np.float32) @ big.T.astype(np.float32)
        rows, cols = np.nonzero(sims >= THRESH)
        for r, c in zip(rows, cols):
            gi, gj = i0 + int(r), int(c)
            if gj >= gi:               # each unordered pair once
                continue
            si, sj = src_of[gi], src_of[gj]
            if si == sj:               # within-source: miner's own problem
                continue
            # sources are priority-ordered: LOWER global index wins
            drops.append({"source": names[si], "index": int(gi - starts[si]),
                          "dup_of_source": names[sj],
                          "dup_of_index": int(gj - starts[sj]),
                          "sim": round(float(sims[r, c]), 4)})
        if (i0 // T) % 8 == 0:
            log(f"tile {i1}/{n} — {len(drops)} cross-dups so far")
    with open(a.out, "w") as f:
        for d in drops:
            f.write(json.dumps(d) + "\n")
    log(f"DONE: {len(drops)} cross-source near-dups -> {a.out}")


if __name__ == "__main__":
    main()
