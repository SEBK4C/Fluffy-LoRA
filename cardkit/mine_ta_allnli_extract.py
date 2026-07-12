#!/usr/bin/env python3
"""mine_ta_allnli_extract.py — acquire + extract AllNLI (SNLI+MNLI)
triplets into the text queue (NLI-class gap fill, MINE-TA brief item 3).

Acquisition: sentence-transformers/all-nli, config "triplet"
(anchor premise, entailed hypothesis, contradiction hypothesis) — the
contradiction is a GROUND-TRUTH hard negative, carried per pair as
"contra" (mine_pack turns it into a text-contra self-view negative).
~55 MB parquet -> /pool-ssd/fluffy/mine-ta/acq/allnli (well under the
5 GB T9-first threshold; rights obvious: SNLI CC BY-SA 4.0, MNLI
OANC/mixed — rights_tier source_audit_required like the other web-derived
text until SIGNOFF-001).

Chunks are named kalm-allnli-* so the generic embed workers + mine_pack
pick them up unchanged (subset="allnli").
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mine_ta_lib as lib  # noqa: E402

ACQ = "/pool-ssd/fluffy/mine-ta/acq/allnli"
QDIR = os.path.join(lib.QUEUE, "text", "kalm")
SEED = 20260712
TARGET = 50000
CHUNK_SIZE = 16384
WS = re.compile(r"\s+")


def log(msg: str) -> None:
    lib.log("allnli", msg)


def clean(s: str) -> str | None:
    s = WS.sub(" ", html.unescape(s or "")).strip()
    return s if 8 <= len(s) <= 600 else None


def main() -> None:
    done_marker = os.path.join(QDIR, "EXTRACT-DONE-allnli")
    if os.path.exists(done_marker):
        log("already extracted — skip")
        return
    os.makedirs(ACQ, exist_ok=True)
    from huggingface_hub import hf_hub_download
    files = ["triplet/train-00000-of-00001.parquet"]
    paths = []
    for f in files:
        p = hf_hub_download("sentence-transformers/all-nli", f,
                            repo_type="dataset", local_dir=ACQ)
        paths.append(p)
        log(f"acquired {f} -> {p} "
            f"({os.path.getsize(p)/1e6:.1f} MB, sha256 "
            f"{lib.sha256_file(p)[:16]}...)")
    import pyarrow.parquet as pq
    rows_all = 0
    kept, seen, pairs = 0, set(), []
    nchunk = 0

    def flush(final=False):
        nonlocal pairs, nchunk
        while len(pairs) >= CHUNK_SIZE or (final and pairs):
            batch, pairs = pairs[:CHUNK_SIZE], pairs[CHUNK_SIZE:]
            cid = f"kalm-allnli-{nchunk:03d}"
            tmp = os.path.join(QDIR, cid + ".json.tmp")
            with open(tmp, "w") as f:
                json.dump({"chunk_id": cid, "subset": "allnli",
                           "task_type": "nli_entail", "pairs": batch}, f,
                          ensure_ascii=False)
            os.replace(tmp, os.path.join(QDIR, cid + ".json"))
            log(f"wrote {cid} ({len(batch)})")
            nchunk += 1

    for p in paths:
        pf = pq.ParquetFile(p)
        total = pf.metadata.num_rows
        frac = min(1.0, TARGET * 1.25 / total)
        for batch in pf.iter_batches(batch_size=8192):
            for i, row in enumerate(batch.to_pylist()):
                rows_all += 1
                if kept >= TARGET:
                    break
                h = hashlib.sha1(f"{SEED}:allnli:{rows_all}".encode())
                if int(h.hexdigest(), 16) % 1_000_000 > frac * 1_000_000:
                    continue
                a = clean(row["anchor"])
                pos = clean(row["positive"])
                neg = clean(row["negative"])
                if not (a and pos and neg) or a == pos:
                    continue
                k = hashlib.sha256(a.lower().encode()).hexdigest()
                if k in seen:
                    continue
                seen.add(k)
                pairs.append({"qid": f"flf-kt-nli-{kept:07d}", "query": a,
                              "passage": pos, "contra": neg,
                              "relevance": 1.0})
                kept += 1
            flush()
            if kept >= TARGET:
                break
    flush(final=True)
    with open(done_marker, "w") as f:
        json.dump({"subset": "allnli", "chunks": nchunk, "kept": kept}, f)
    lib.update_state("text-kalm-allnli", extracted=True, pairs=kept,
                     chunks=nchunk, task_type="nli_entail",
                     acquisition="sentence-transformers/all-nli triplet "
                                 "split, deterministic sample")
    log(f"DONE kept={kept} chunks={nchunk} (from {rows_all} rows scanned)")


if __name__ == "__main__":
    main()
