#!/usr/bin/env python3
"""extract_mmeb.py — one MMEB subset -> pairs.jsonl + CAS images + encode
chunks (standalone; CPU; idempotent).

  extract_mmeb.py --subset DocVQA

Contamination guards (belt + suspenders + braces):
  1. any image member containing "val2014" is dropped wholesale (the frozen
     image-eval-v1 MSCOCO half was sampled from val2014 members);
  2. members whose basename OR embedded 12-digit COCO id matches an eval pin
     are dropped (catches val2014 images renamed in COCO-2017 style, e.g.
     the MSCOCO grounding subset);
  3. after reading bytes, any image whose sha256 matches an eval-pin image
     OR a warmup-slice image is dropped (catches renames sha-exactly:
     A-OKVQA_image_N.jpg etc. are renamed COCO files).
Dedup guards: warmup 50k caption hashes (retrieval kinds), carry_text from
already-extracted sibling subsets (t2i after i2t), per-subset uniq policy.

Outputs under $OUT_ROOT/staging/<subset>/:
  pairs.jsonl, items-NNNN.jsonl (encode chunks), extract-done.json
and enqueues encode__mmeb__<subset>__NNNN queue tasks.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
from common import log
import mmeb_spec

SRC_MMEB = os.environ.get("SRC_MMEB",
                          "/pool-6b/corpus-acq/work/mmeb_train/snapshot")
OUT_ROOT = os.environ.get("OUT_ROOT", common.SRC_ROOT["mmeb"])
GUARDS = os.path.join(OUT_ROOT, "guards.json")

COCO_ID = re.compile(r"(\d{12})")


def load_guards() -> dict:
    g = json.load(open(GUARDS))
    return {"eval_shas": set(g["eval_shas"]),
            "eval_basenames": set(g["eval_basenames"]),
            "eval_coco_ids": set(g["eval_coco_ids"]),
            "warmup_shas": set(g["warmup_shas"]),
            "warmup_caps": set(g["warmup_caption_hashes"])}


def member_guarded(member: str, g: dict, st) -> bool:
    if "val2014" in member:
        st["drop_val2014"] += 1
        return True
    base = os.path.basename(member)
    if base in g["eval_basenames"]:
        st["drop_eval_basename"] += 1
        return True
    m = COCO_ID.search(base)
    if m and m.group(1) in g["eval_coco_ids"]:
        st["drop_eval_cocoid"] += 1
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    args = ap.parse_args()
    sub = args.subset
    spec = mmeb_spec.SUBSETS[sub]
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("FLUFFY_CARDS_ROOT", OUT_ROOT)
    import cardlib

    stg = os.path.join(OUT_ROOT, "staging", sub)
    done_marker = os.path.join(stg, "extract-done.json")
    if os.path.exists(done_marker):
        log("extract", f"{sub}: already extracted — re-enqueueing chunks only")
        enqueue_chunks(sub, stg)
        return
    os.makedirs(stg, exist_ok=True)

    g = load_guards()
    # carry_text: caption hashes from sibling subsets extracted first
    carry: set[str] = set(g["warmup_caps"]) if spec["kind"] == "retrieval" else set()
    for sib in spec.get("carry_text", []):
        with open(os.path.join(OUT_ROOT, "staging", sib, "pairs.jsonl")) as f:
            for line in f:
                carry.add(json.loads(line)["dedup_hash"])

    import pyarrow.parquet as pq
    from PIL import Image

    t0 = time.time()
    st = collections.Counter()
    rows = pq.ParquetFile(os.path.join(
        SRC_MMEB, sub, "train-00000-of-00001.parquet")).read().to_pylist()
    st["rows"] = len(rows)

    seen_uniq: set[str] = set()
    cands = []
    for r in rows:
        parsed = spec["parse"](r)
        if isinstance(parsed, str):
            st[parsed] += 1
            continue
        members = {}
        for side in ("anchor", "positive"):
            kind, member, _ = parsed[side]
            if kind in ("image", "imagetext"):
                member = member.removeprefix("images/")
                members[side] = member
        if any(member_guarded(m, g, st) for m in members.values()):
            continue
        dh = cardlib.dedup_hash(parsed["anchor_text"])
        text_unique = parsed["uniq"] == ("text",)   # retrieval-style kinds:
        # anchor text IS the card identity, deduped globally (warmup rule).
        # Composite kinds (cls/vqa/grounding/...) repeat texts by design and
        # dedup on their uniq-key instead.
        if text_unique and dh in carry:
            st["drop_carry_text"] += 1
            continue
        ukey = dh if "text" in parsed["uniq"] else ""
        if "anchor_image" in parsed["uniq"]:
            ukey += "|" + members.get("anchor", "")
        if "positive_image" in parsed["uniq"]:
            ukey += "|" + members.get("positive", "")
        if ukey in seen_uniq:
            st["drop_dup"] += 1
            continue
        seen_uniq.add(ukey)
        if text_unique:
            carry.add(dh)
        cands.append((parsed, members, dh))
    st["candidates"] = len(cands)

    cap = spec.get("cap")
    if cap:
        rng = random.Random(common.SEED)
        rng.shuffle(cands)
        per_cls = collections.Counter()
        kept = []
        for c in cands:
            key = c[0]["cls_key"]
            if per_cls[key] < cap:
                per_cls[key] += 1
                kept.append(c)
        st["drop_class_cap"] = len(cands) - len(kept)
        cands = kept

    # ---- single near-sequential zip pass over every needed member ----------
    need = sorted({m for _, members, _ in cands for m in members.values()})
    zf = zipfile.ZipFile(os.path.join(SRC_MMEB, "images_zip", f"{sub}.zip"))
    info = {i.filename: i for i in zf.infolist()}
    member_sha: dict[str, str | None] = {}
    order = sorted(need, key=lambda m: info[m].header_offset
                   if m in info else 1 << 62)
    for i, member in enumerate(order):
        if member not in info:
            member_sha[member] = None
            st["drop_missing_member"] += 1
            continue
        data = zf.read(member)
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            member_sha[member] = None
            st["drop_bad_image"] += 1
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in g["eval_shas"]:
            member_sha[member] = None
            st["drop_eval_sha"] += 1
            continue
        if sha in g["warmup_shas"]:
            member_sha[member] = None
            st["drop_warmup_sha"] += 1
            continue
        common.cas_store(OUT_ROOT, data)
        member_sha[member] = sha
        if (i + 1) % 10000 == 0:
            log("extract", f"{sub}: images {i + 1}/{len(order)}")
    zf.close()

    # ---- final pair records -------------------------------------------------
    pairs = []
    for parsed, members, dh in cands:
        sides = {}
        bad = False
        for side in ("anchor", "positive"):
            kind, member, text = parsed[side]
            sha = None
            if side in members:
                sha = member_sha.get(members[side])
                if sha is None:
                    bad = True
                    break
            sides[side] = {"kind": kind, "text": text, "sha": sha,
                           "member": members.get(side)}
        if bad:
            st["drop_image_failed"] += 1
            continue
        pairs.append({
            "card_id": f"flf-{spec['tag']}-{len(pairs):06d}",
            "subset": sub, "kind": spec["kind"],
            "anchor_text": parsed["anchor_text"], "dedup_hash": dh,
            "native_id": members.get("anchor") or members.get("positive"),
            "anchor": sides["anchor"], "positive": sides["positive"],
        })
    st["kept"] = len(pairs)

    tmp = os.path.join(stg, "pairs.jsonl.tmp")
    with open(tmp, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    os.rename(tmp, os.path.join(stg, "pairs.jsonl"))

    # ---- unique encode items -> chunks --------------------------------------
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
        path = os.path.join(stg, f"items-{n_chunks:04d}.jsonl")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for iid in chunk:
                f.write(json.dumps(items[iid], ensure_ascii=False) + "\n")
        os.rename(tmp, path)
        n_chunks += 1
    st["items"] = len(ids)
    st["chunks"] = n_chunks

    report = {"subset": sub, "stats": dict(st),
              "elapsed_s": round(time.time() - t0, 1),
              "done_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = done_marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=1)
    os.rename(tmp, done_marker)

    enqueue_chunks(sub, stg)

    def upd(state):
        ss = common.subset_state(state, "mmeb", sub)
        ss["extracted"] = len(pairs)
        ss["items"] = len(ids)
        ss["chunks_total"] = n_chunks
        ss["extract_stats"] = dict(st)
    common.update_state(upd)
    log("extract", f"{sub}: DONE {len(pairs)} pairs, {len(ids)} items, "
        f"{n_chunks} chunks, {round(time.time() - t0, 1)}s  stats={dict(st)}")


def enqueue_chunks(sub: str, stg: str) -> None:
    import glob as _g
    for path in sorted(_g.glob(os.path.join(stg, "items-*.jsonl"))):
        nn = os.path.basename(path)[6:10]
        common.enqueue("mmeb", f"encode__mmeb__{sub}__{nn}", {
            "task": "encode", "source": "mmeb", "subset": sub, "chunk": nn,
            "items": path, "cas_root": os.path.join(OUT_ROOT, "cas"),
            "out": os.path.join(stg, f"emb-{nn}.npy"),
        })


if __name__ == "__main__":
    main()
