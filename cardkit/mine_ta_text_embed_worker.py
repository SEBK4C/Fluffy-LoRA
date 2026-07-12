#!/usr/bin/env python3
"""mine_ta_text_embed_worker.py — claim text queue chunks, embed both pair
sides via the :9020 teacher, write fp16 embedding arrays next to the chunk.

Standalone compute path (MINING-OPS §5): args --queue-dir/--out-dir, no
orchestration logic. Restartable: output .npz existing and loadable = chunk
done (claims + on-disk checks, never session memory). Blocks-and-waits
through teacher downtime (mine_ta_lib.embed) — NEVER kills or restarts the
teacher, it is shared with judge/query-gen work.

Loop: while (unclaimed chunk exists) or (EXTRACT-ALL-DONE not present):
claim -> embed queries + passages -> save <chunk>.emb.npz (fp16, L2-normed)
-> state update. Exits when the sentinel exists and nothing is left.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mine_ta_lib as lib  # noqa: E402


def log(msg: str) -> None:
    lib.log("text-embed", msg)


def chunk_done(out_dir: str, chunk_id: str) -> bool:
    p = os.path.join(out_dir, chunk_id + ".emb.npz")
    if not os.path.exists(p):
        return False
    try:
        import numpy as np
        with np.load(p) as z:
            return "q" in z and "p" in z
    except Exception:  # torn file from a kill: redo
        os.unlink(p)
        return False


def process(chunk_path: str, out_dir: str) -> None:
    import numpy as np
    with open(chunk_path) as f:
        chunk = json.load(f)
    cid = chunk["chunk_id"]
    pairs = chunk["pairs"]
    t0 = time.time()
    emb_q = lib.embed([p["query"] for p in pairs], tag="text-embed")
    lib.touch_claim(cid)
    emb_p = lib.embed([p["passage"] for p in pairs], tag="text-embed")
    arrays = {"q": emb_q.astype(np.float16), "p": emb_p.astype(np.float16)}
    if all("contra" in p for p in pairs):  # NLI triplets: embed the
        lib.touch_claim(cid)               # ground-truth contradiction too
        arrays["c"] = lib.embed([p["contra"] for p in pairs],
                                tag="text-embed").astype(np.float16)
    out = os.path.join(out_dir, cid + ".emb.npz")
    tmp = out + f".tmp.{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    dt = time.time() - t0
    log(f"{cid}: embedded {len(pairs)} pairs in {dt:.0f}s "
        f"({2*len(pairs)/dt:.1f} texts/s)")
    lib.update_state(f"text-kalm-{chunk['subset']}",
                     **{f"encoded_{cid}": len(pairs)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-dir", default=os.path.join(lib.QUEUE, "text",
                                                        "kalm"))
    ap.add_argument("--out-dir",
                    default="/pool-ssd/fluffy/mine-ta/text/emb")
    ap.add_argument("--chunk", help="process just this one chunk file")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.chunk:
        process(a.chunk, a.out_dir)
        return
    idle_logged = False
    while True:
        sentinel = os.path.exists(os.path.join(a.queue_dir,
                                               "EXTRACT-ALL-DONE"))
        todo = sorted(
            f for f in os.listdir(a.queue_dir)
            if f.endswith(".json") and not f.endswith(".tmp"))
        progressed = False
        for f in todo:
            cid = f[:-5]
            if chunk_done(a.out_dir, cid):
                continue
            if lib.claimed_elsewhere(cid):
                continue
            if not lib.claim(cid):
                continue
            if chunk_done(a.out_dir, cid):  # claimed by us in a past life
                continue
            process(os.path.join(a.queue_dir, f), a.out_dir)
            progressed = True
            idle_logged = False
        if not progressed:
            remaining = [f for f in todo
                         if not chunk_done(a.out_dir, f[:-5])]
            if sentinel and not remaining:
                log("queue drained + sentinel present — worker exits")
                return
            if not idle_logged:
                log(f"idle: {len(remaining)} chunks pending "
                    f"(claimed elsewhere or extractor still running)")
                idle_logged = True
            time.sleep(20)


if __name__ == "__main__":
    main()
