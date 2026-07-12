#!/usr/bin/env python3
"""export_text_embs.py — text-side (caption/query/answer) embedding export
for the joint cross-source near-dup sweep (mine_ta_crossdedup.py, §3.4).

  export_text_embs.py --source mmeb

Writes per source: staging/crossdedup/<source>-text-emb.npy (fp16, aligned)
+ <source>-text-map.jsonl {row, item_id, text_sha, subset} so drop-list rows
map back to items -> cards -> exposures at relaunch-mix time.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import log


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    source = args.source
    root = common.SRC_ROOT[source]
    outd = os.path.join(root, "staging", "crossdedup")
    os.makedirs(outd, exist_ok=True)

    rows, mats = [], []
    seen = set()
    for itf in sorted(glob.glob(os.path.join(root, "staging", "*",
                                             "items-*.jsonl"))):
        embf = itf.replace("items-", "emb-").replace(".jsonl", ".npy")
        if not os.path.exists(embf):
            continue
        subset = os.path.basename(os.path.dirname(itf))
        items = [json.loads(l) for l in open(itf)]
        m = np.load(embf)
        keep = [i for i, it in enumerate(items)
                if it["kind"] == "text" and it["id"] not in seen]
        for i in keep:
            seen.add(items[i]["id"])
            rows.append({"row": len(rows), "item_id": items[i]["id"],
                         "text_sha": hashlib.sha256(
                             items[i]["text"].encode()).hexdigest()[:16],
                         "subset": subset})
        mats.append(m[keep])
    if not mats:
        raise SystemExit(f"{source}: no encoded text items yet")
    emb = np.concatenate(mats, axis=0).astype(np.float16)
    np.save(os.path.join(outd, f"{source}-text-emb.npy"), emb)
    with open(os.path.join(outd, f"{source}-text-map.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log("export", f"{source}: {emb.shape[0]} text vectors -> {outd}")


if __name__ == "__main__":
    main()
