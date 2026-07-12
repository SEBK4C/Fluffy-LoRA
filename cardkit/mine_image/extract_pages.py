#!/usr/bin/env python3
"""extract_pages.py — one parquet file of a page source (ColPali / VisRAG)
-> pairs + CAS page images + encode chunks (standalone; CPU; idempotent).

  extract_pages.py --source colpali --file 3

Page images are recompressed to max-dim 1280 JPEG q85 when larger
("page-recompress-v1"): the training processor sees <=~900px anyway, and
raw ColPali pages (~445 KB avg) would blow shard budgets 9x via co-packed
negatives. rights.source_sha256 keeps the ORIGINAL bytes' sha for audit.

Guards: ColPali TEST split never read (train-*.parquet only) + eval-pin
query dedup-hashes and original-image shas excluded (image-eval-v1 ColPali
half). Query = first line, stripped (decision G boilerplate rule, byte-same
with build_frozen_evals.py). Cross-FILE dedup happens at minepack (files
extract in parallel).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
from common import log

SRC = {
    "colpali": "/pool-6b/corpus-acq/work/colpali/snapshot/data",
    "visrag": "/pool-6b/corpus-acq/work/visrag_indomain/snapshot/data",
}
MAX_DIM = 1280
JPEG_Q = 85


def load_guards(root: str) -> dict:
    p = os.path.join(root, "guards.json")
    if os.path.exists(p):
        g = json.load(open(p))
        return {"eval_qhash": set(g["eval_qhash"]),
                "eval_shas": set(g["eval_shas"])}
    return {"eval_qhash": set(), "eval_shas": set()}


def recompress(data: bytes):
    """-> (cas_bytes, transform_tag). Original bytes preserved when small."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        w, h = im.size
        if max(w, h) <= MAX_DIM and data[:2] == b"\xff\xd8":
            return data, None
        scale = MAX_DIM / max(w, h)
        if scale < 1.0:
            im = im.resize((max(1, round(w * scale)),
                            max(1, round(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=JPEG_Q)
        return buf.getvalue(), "page-recompress-v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["colpali", "visrag"])
    ap.add_argument("--file", type=int, required=True)
    args = ap.parse_args()
    source, fn = args.source, args.file
    root = common.SRC_ROOT[source]
    os.environ.setdefault("FLUFFY_CARDS_ROOT", root)
    import cardlib

    files = sorted(f for f in os.listdir(SRC[source])
                   if f.startswith("train-"))
    pf_name = files[fn]
    stg = os.path.join(root, "staging", "all")
    os.makedirs(stg, exist_ok=True)
    marker = os.path.join(stg, f"extract-f{fn:02d}-done.json")
    if os.path.exists(marker):
        log("pages", f"{source} f{fn:02d}: already extracted — re-enqueue only")
        enqueue_chunks(source, stg, fn)
        return

    import pyarrow.parquet as pq

    t0 = time.time()
    g = load_guards(root)
    st = collections.Counter()
    seen = set()
    pairs = []
    pf = pq.ParquetFile(os.path.join(SRC[source], pf_name))
    st["rows"] = pf.metadata.num_rows
    for batch in pf.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            q = (r["query"] or "").strip().split("\n")[0].strip()
            q = " ".join(q.split())
            if not (10 <= len(q) <= 500):
                st["drop_qlen"] += 1
                continue
            dh = cardlib.dedup_hash(q)
            if dh in g["eval_qhash"]:
                st["drop_eval_query"] += 1
                continue
            data = r["image"]["bytes"]
            if not data:
                st["drop_no_image"] += 1
                continue
            orig_sha = hashlib.sha256(data).hexdigest()
            if orig_sha in g["eval_shas"]:
                st["drop_eval_sha"] += 1
                continue
            key = (dh, orig_sha)
            if key in seen:
                st["drop_dup"] += 1
                continue
            seen.add(key)
            try:
                cas_bytes, transform = recompress(data)
            except Exception:  # noqa: BLE001
                st["drop_bad_image"] += 1
                continue
            sha = common.cas_store(root, cas_bytes)
            if transform:
                st["recompressed"] += 1
            native = (r.get("image_filename")
                      or f"{r.get('source', source)}#f{fn}r{len(pairs)}")
            pairs.append({
                "card_id": f"flf-{source[:4]}-f{fn:02d}-{len(pairs):06d}",
                "subset": "all", "kind": "docmatch",
                "anchor_text": q, "dedup_hash": dh, "native_id": native,
                "anchor": {"kind": "image", "text": None, "sha": sha,
                           "member": native},
                "positive": {"kind": "text", "text": q, "sha": None,
                             "member": None},
                "orig_sha256": orig_sha, "transform": transform,
            })
    st["kept"] = len(pairs)

    ppath = os.path.join(stg, f"pairs-f{fn:02d}.jsonl")
    with open(ppath + ".tmp", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    os.rename(ppath + ".tmp", ppath)

    items: dict[str, dict] = {}
    for p in pairs:
        for side in ("anchor", "positive"):
            s = p[side]
            iid = common.item_id(s["kind"], s["text"], s["sha"])
            items.setdefault(iid, {"id": iid, "kind": s["kind"],
                                   "text": s["text"], "image": s["sha"]})
    ids = sorted(items)
    n_chunks = 0
    for c0 in range(0, len(ids), common.CHUNK_ITEMS):
        chunk = ids[c0:c0 + common.CHUNK_ITEMS]
        path = os.path.join(stg, f"items-f{fn:02d}-{n_chunks:04d}.jsonl")
        with open(path + ".tmp", "w") as f:
            for iid in chunk:
                f.write(json.dumps(items[iid], ensure_ascii=False) + "\n")
        os.rename(path + ".tmp", path)
        n_chunks += 1
    st["items"], st["chunks"] = len(ids), n_chunks

    with open(marker + ".tmp", "w") as f:
        json.dump({"file": pf_name, "stats": dict(st),
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=1)
    os.rename(marker + ".tmp", marker)
    enqueue_chunks(source, stg, fn)

    def upd(state):
        ss = common.subset_state(state, source, "all")
        ss.setdefault("files", {})[f"f{fn:02d}"] = {
            "pairs": len(pairs), "items": len(ids), "stats": dict(st)}
        ss["extracted"] = sum(v["pairs"] for v in ss["files"].values())
        ss["chunks_total"] = len(
            __import__("glob").glob(os.path.join(stg, "items-*.jsonl")))
    common.update_state(upd)
    log("pages", f"{source} f{fn:02d}: DONE {len(pairs)} pairs "
        f"({round(time.time() - t0, 1)}s) stats={dict(st)}")


def enqueue_chunks(source: str, stg: str, fn: int) -> None:
    import glob as _g
    for path in sorted(_g.glob(os.path.join(stg, f"items-f{fn:02d}-*.jsonl"))):
        cid = os.path.basename(path).removeprefix("items-").removesuffix(".jsonl")
        common.enqueue(source, f"encode__{source}__all__{cid}", {
            "task": "encode", "source": source, "subset": "all", "chunk": cid,
            "items": path,
            "cas_root": os.path.join(common.SRC_ROOT[source], "cas"),
            "out": os.path.join(stg, f"emb-{cid}.npy"),
        })


if __name__ == "__main__":
    main()
