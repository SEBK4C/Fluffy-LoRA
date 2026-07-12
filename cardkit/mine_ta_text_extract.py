#!/usr/bin/env python3
"""mine_ta_text_extract.py — kalm_sample parquet -> cleaned, deduped,
deterministically-sampled text pairs -> dir-claim queue chunks.

Reads the CORPUS-ACQ snapshot (READ-ONLY pool). Emits
/pool-ssd/fluffy/queue/text/kalm/<subset>-<nnn>.json chunks of
CHUNK_SIZE pairs each, plus EXTRACT-DONE-<subset> markers so embed
workers can pipeline while extraction is still running.

Sampling is deterministic (seeded hash on (subset, row_index)):
row-group skipping keeps IO sane on the 60 GB falcon subset while
staying unbiased at row-group granularity.

Per-subset plan carries the task_type for the (draft) instruction set —
exposures ship task_type NOW + the current frozen instruction string
verbatim; restamping post-freeze is a map, not a re-mine.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mine_ta_lib as lib  # noqa: E402

SNAP = "/pool-6b/corpus-acq/work/kalm_sample/snapshot"
QDIR = os.path.join(lib.QUEUE, "text", "kalm")
SEED = 20260712
CHUNK_SIZE = 16384
Q_LEN = (8, 600)          # query char caps (truncate long queries at cap)
P_LEN = (80, 1200)        # passage char caps (truncate at word boundary)

# subset -> (extract_target, card-id abbrev, task_type)
PLAN = {
    "paq":                   (88000, "paq", "qa_passage"),
    "stackexchange":         (60000, "sx",  "qa_passage"),
    "stackoverflow":         (60000, "so",  "qa_passage"),
    "s2orc":                 (60000, "s2o", "title_doc"),
    "wikipedia":             (60000, "wik", "title_doc"),
    "falcon":                (50000, "fal", "web_query_doc"),
    "dbpedia-entity":        (44000, "dbp", "entity_desc"),
    "swim-ir-cross-lingual": (44000, "swx", "crosslingual"),
    "swim-ir-monolingual":   (44000, "swm", "qa_passage"),
    "codesearchnet":         (33000, "csn", "code_search"),
    "csl":                   (22000, "csl", "title_doc"),
    "big_patent":            (11000, "pat", "title_doc"),
}

WS = re.compile(r"\s+")


def log(msg: str) -> None:
    lib.log("text-extract", msg)


def clean(s: str, lo: int, hi: int, is_query: bool) -> str | None:
    s = WS.sub(" ", html.unescape(s)).strip()
    if len(s) < lo:
        return None
    if len(s) > hi:
        cut = s[:hi]
        sp = cut.rfind(" ")
        s = cut[:sp] if sp > hi * 0.7 else cut
    return s


def keep(subset: str, idx: int, frac: float) -> bool:
    h = hashlib.sha1(f"{SEED}:{subset}:{idx}".encode()).hexdigest()
    return int(h, 16) % 1_000_000 < int(frac * 1_000_000)


def extract_subset(subset: str) -> None:
    import pyarrow.parquet as pq
    target, abbr, task_type = PLAN[subset]
    done_marker = os.path.join(QDIR, f"EXTRACT-DONE-{subset}")
    if os.path.exists(done_marker):
        log(f"{subset}: already extracted — skip")
        return
    files = sorted(
        os.path.join(SNAP, subset, f)
        for f in os.listdir(os.path.join(SNAP, subset))
        if f.endswith(".parquet"))
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    frac = min(1.0, target * 1.3 / total_rows)
    # row-group skipping: keep within-rg sampling ~1-in-20 for diversity
    rg_step = max(1, int(1 / (frac * 20))) if frac < 0.05 else 1
    rg_frac = min(1.0, frac * rg_step)
    log(f"{subset}: rows={total_rows:,} target={target} frac={frac:.4f} "
        f"rg_step={rg_step} in-rg-frac={rg_frac:.3f}")

    pairs, seen_q, nchunk, gidx = [], set(), 0, 0
    seq = 0
    stats = {"read": 0, "kept": 0, "rej_len": 0, "rej_dupq": 0,
             "rej_nopos": 0}

    def flush(final: bool = False) -> None:
        nonlocal pairs, nchunk
        while len(pairs) >= CHUNK_SIZE or (final and pairs):
            batch, pairs = pairs[:CHUNK_SIZE], pairs[CHUNK_SIZE:]
            cid = f"kalm-{subset}-{nchunk:03d}"
            path = os.path.join(QDIR, cid + ".json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"chunk_id": cid, "subset": subset,
                           "task_type": task_type, "pairs": batch}, f,
                          ensure_ascii=False)
            os.replace(tmp, path)
            log(f"  wrote {cid} ({len(batch)} pairs)")
            nchunk += 1

    os.makedirs(QDIR, exist_ok=True)
    rg_index = 0
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rg in range(pf.metadata.num_row_groups):
            rg_index += 1
            if (rg_index - 1) % rg_step != 0:
                gidx += pf.metadata.row_group(rg).num_rows
                continue
            tbl = pf.read_row_group(rg, columns=["query", "pos",
                                                 "relevance"])
            for row in tbl.to_pylist():
                gidx += 1
                if stats["kept"] >= target:
                    break
                if not keep(subset, gidx, rg_frac):
                    continue
                stats["read"] += 1
                if not row["pos"]:
                    stats["rej_nopos"] += 1
                    continue
                q = clean(row["query"], Q_LEN[0], Q_LEN[1], True)
                p = clean(row["pos"][0], P_LEN[0], P_LEN[1], False)
                if q is None or p is None or q == p:
                    stats["rej_len"] += 1
                    continue
                qk = hashlib.sha256(q.lower().encode()).hexdigest()
                if qk in seen_q:
                    stats["rej_dupq"] += 1
                    continue
                seen_q.add(qk)
                pairs.append({
                    "qid": f"flf-kt-{abbr}-{seq:07d}", "query": q,
                    "passage": p,
                    "relevance": round(row["relevance"] or 0.0, 4)})
                seq += 1
                stats["kept"] += 1
            flush()
            if stats["kept"] >= target:
                break
        if stats["kept"] >= target:
            break
    flush(final=True)
    with open(done_marker, "w") as f:
        json.dump({"subset": subset, "chunks": nchunk, **stats}, f)
    lib.update_state(f"text-kalm-{subset}", extracted=True,
                     pairs=stats["kept"], chunks=nchunk,
                     task_type=task_type, stats=stats)
    log(f"{subset}: DONE kept={stats['kept']} chunks={nchunk} "
        f"rejects={ {k: v for k, v in stats.items() if k.startswith('rej')} }")


def main() -> None:
    subsets = sys.argv[1:] or list(PLAN)
    t0 = time.time()
    for s in subsets:
        extract_subset(s)
    # global sentinel: embed workers may exit when this exists and the
    # queue holds no unclaimed chunks. Only written when EVERY subset in
    # the PLAN has its own done marker (partial runs must not fire it).
    if all(os.path.exists(os.path.join(QDIR, f"EXTRACT-DONE-{s}"))
           for s in PLAN):
        with open(os.path.join(QDIR, "EXTRACT-ALL-DONE"), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log("EXTRACT-ALL-DONE sentinel written")
    log(f"extraction pass finished in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
