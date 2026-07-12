#!/usr/bin/env python3
"""mine_ta_text_mine_pack.py — per-subset: teacher-band negatives ->
CARD-SPEC cards -> exposures -> validator gates -> shards_v2 WDS shards.

Standalone (MINING-OPS §5): `--subset X` works alone once that subset's
chunks + embeddings exist. Idempotent: a finished subset (shards dir +
REPORT.json + MANIFEST) is skipped unless --force.

Recipe (mirrors build_image_lane.py, text2text):
  - positive sim = q . p (teacher fp16, L2-normed)
  - judge-proxy filter (MINING-OPS §3.3 sample-based rule): drop the bottom
    decile of pos_sim per subset (calibrated floor recorded in REPORT;
    MLLM-judge sampling pass upgrades this when a judge GPU window exists)
  - ANN negatives: query -> passages of other pairs, within-subset,
    TopK-PercPos ceiling 0.95 x that query's positive sim, k=8, hardest
    first, band_rule recorded per negative (NO dataset-shipped negatives)
  - card: anchor_text = query; views.text = query, views.text-passage =
    passage; negatives.text -> other cards' text-passage views
  - exposure: lane text2text, anchor text -> positive text-passage,
    task_type stamped (instruction restamp post-freeze = map by task_type),
    instruction = current frozen string VERBATIM, pos_sim + kalm relevance
    carried as difficulty metadata (§3.6)
  - 250-sample validate_card CLI gate BEFORE bulk (>30% reject = abort)
  - shards: shards_v2 contract, .idx.json sidecars, MANIFEST + SHA256SUMS,
    re-hash + reader spot-check verification
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import mine_ta_lib as lib  # noqa: E402
import cardlib  # noqa: E402
import validate_card  # noqa: E402

QDIR = os.path.join(lib.QUEUE, "text", "kalm")
EMB_DIR = "/pool-ssd/fluffy/mine-ta/text/emb"
OUT_ROOT = "/pool-ssd/fluffy/mine-ta/text"
SEED = 20260712
K_MAX = 8
PERCPOS = 0.95
SHARD_SIZE = 8192
MINER = "qwen3emb8b-ann-v1"
INSTRUCTION = "Retrieve the matching description."  # frozen stage-1 string
DECILE_DROP = 0.10

LICENSES = {
    "paq": "CC BY-SA 3.0", "stackexchange": "CC BY-SA 4.0",
    "stackoverflow": "CC BY-SA 4.0", "s2orc": "ODC-By 1.0",
    "wikipedia": "CC BY-SA 3.0/4.0", "falcon": "ODC-By 1.0 (RefinedWeb)",
    "dbpedia-entity": "CC BY-SA 3.0",
    "swim-ir-cross-lingual": "CC BY-SA 4.0",
    "swim-ir-monolingual": "CC BY-SA 4.0",
    "codesearchnet": "MIT harness over permissive OSS",
    "csl": "Apache-2.0", "big_patent": "CC BY 4.0",
    "allnli": "SNLI CC BY-SA 4.0 + MultiNLI (OANC/mixed, "
              "see nyu-mll/multi_nli card)",
}


def log(msg: str) -> None:
    lib.log("text-mine", msg)


def load_subset(subset: str):
    import numpy as np
    chunks = sorted(f for f in os.listdir(QDIR)
                    if f.startswith(f"kalm-{subset}-") and f.endswith(".json"))
    if not chunks:
        raise SystemExit(f"{subset}: no chunks in queue")
    if not os.path.exists(os.path.join(QDIR, f"EXTRACT-DONE-{subset}")):
        raise SystemExit(f"{subset}: extraction not complete")
    pairs, eq, ep, ec = [], [], [], []
    task_type = None
    for c in chunks:
        cid = c[:-5]
        npz = os.path.join(EMB_DIR, cid + ".emb.npz")
        if not os.path.exists(npz):
            raise SystemExit(f"{subset}: missing embeddings for {cid}")
        with open(os.path.join(QDIR, c)) as f:
            ch = json.load(f)
        task_type = ch["task_type"]
        with np.load(npz) as z:
            q, p = z["q"].astype(np.float32), z["p"].astype(np.float32)
            if "c" in z:
                ec.append(z["c"].astype(np.float32))
        assert len(ch["pairs"]) == q.shape[0] == p.shape[0], f"{cid} drift"
        pairs.extend(ch["pairs"])
        eq.append(q)
        ep.append(p)
    emb_c = np.vstack(ec) if len(ec) == len(chunks) else None
    return pairs, np.vstack(eq), np.vstack(ep), emb_c, task_type


def mine_pack(subset: str, force: bool = False) -> None:
    import numpy as np
    sys.path.insert(0, os.path.dirname(HERE))
    import shards_v2

    src_name = f"text-kalm-{subset}"
    out_dir = os.path.join(OUT_ROOT, subset)
    shards_dir = os.path.join(out_dir, "shards")
    report_path = os.path.join(out_dir, "REPORT.json")
    if (not force and os.path.exists(report_path)
            and os.path.exists(os.path.join(shards_dir, "MANIFEST.jsonl"))):
        log(f"{subset}: already packed — skip")
        return
    os.makedirs(shards_dir, exist_ok=True)
    pairs, emb_q, emb_p, emb_c, task_type = load_subset(subset)
    n0 = len(pairs)
    log(f"{subset}: loaded {n0} pairs, dim {emb_q.shape[1]}, "
        f"task_type {task_type}")

    pos = np.einsum("ij,ij->i", emb_q, emb_p)
    floor = float(np.quantile(pos, DECILE_DROP))
    keep = pos >= floor
    idx_keep = np.flatnonzero(keep)
    pairs = [pairs[i] for i in idx_keep]
    emb_q, emb_p, pos = emb_q[idx_keep], emb_p[idx_keep], pos[idx_keep]
    contra_sim = (np.einsum("ij,ij->i", emb_q, emb_c[idx_keep])
                  if emb_c is not None else None)
    n = len(pairs)
    log(f"{subset}: judge-proxy decile filter: floor={floor:.4f}, "
        f"kept {n}/{n0}")

    cal = np.sort(pos[random.Random(SEED).sample(range(n), min(1000, n))])
    calib = {
        "n": len(cal), "band_rule": f"topk-percpos-{PERCPOS}",
        "positive_sim": {
            "min": float(cal[0]), "p5": float(np.percentile(cal, 5)),
            "median": float(np.percentile(cal, 50)),
            "p95": float(np.percentile(cal, 95)), "max": float(cal[-1]),
            "mean": float(cal.mean())},
        "judge_proxy_floor_p10": floor,
        "implied_median_ceiling": float(PERCPOS * np.percentile(cal, 50))}
    log(f"{subset}: calibration {json.dumps(calib['positive_sim'])}")

    # ---- ANN: query -> other pairs' passages, per-query ceiling ----
    negs_idx = np.zeros((n, K_MAX), dtype=np.int64)
    negs_sim = np.zeros((n, K_MAX), dtype=np.float32)
    negs_cnt = np.zeros(n, dtype=np.int64)
    blk = 2048
    for s in range(0, n, blk):
        e = min(s + blk, n)
        sims = emb_q[s:e] @ emb_p.T
        sims[np.arange(e - s), np.arange(s, e)] = -2.0  # mask own positive
        ceil = PERCPOS * pos[s:e]
        sims = np.where(sims < ceil[:, None], sims, -2.0)
        top = np.argpartition(-sims, K_MAX + 8, axis=1)[:, :K_MAX + 8]
        for r in range(e - s):
            cand = top[r][np.argsort(-sims[r, top[r]])]
            cand = [int(j) for j in cand if sims[r, j] > -1.5][:K_MAX]
            negs_cnt[s + r] = len(cand)
            negs_idx[s + r, :len(cand)] = cand
            negs_sim[s + r, :len(cand)] = sims[r, cand]
        if (s // blk) % 8 == 0:
            log(f"  ann {subset}: {e}/{n}")
    del sims

    # ---- cards ----
    band_tag = f"topk-percpos-{PERCPOS}"
    lic = LICENSES.get(subset, "see kalm_subset_licenses.yaml")
    # origin must be a top-level frozen-enum source (v1.1a additive
    # extension adds kalm/allnli); the subset rides in native_id
    origin_tag = "kalm" if subset != "allnli" else "allnli"
    cards = []
    for k, p in enumerate(pairs):
        ceil = round(float(PERCPOS * pos[k]), 4)
        nid = f"{subset}:{p['qid']}"
        views = {
            "text": {"content": [{"type": "text", "text": p["query"]}],
                     "source": "real", "origin": origin_tag,
                     "native_id": nid},
            "text-passage": {
                "content": [{"type": "text", "text": p["passage"]}],
                "source": "real", "origin": origin_tag,
                "native_id": nid},
        }
        negs = [{"card_id": pairs[j]["qid"], "view": "text-passage",
                 "sim": round(float(sm), 4), "miner": MINER,
                 "band_rule": f"{band_tag}:ceil={ceil}"}
                for j, sm in zip(negs_idx[k][:negs_cnt[k]],
                                 negs_sim[k][:negs_cnt[k]])]
        if "contra" in p:  # NLI ground-truth hard negative, self-view
            views["text-contra"] = {
                "content": [{"type": "text", "text": p["contra"]}],
                "source": "real", "origin": origin_tag,
                "native_id": nid}
            negs = [{"card_id": p["qid"], "view": "text-contra",
                     "sim": round(float(contra_sim[k]), 4),
                     "miner": "nli-contradiction",
                     "band_rule": "ground-truth"}] + negs[:K_MAX - 1]
        cards.append({
            "card_id": p["qid"],
            "anchor_text": p["query"],
            "views": views,
            "negatives": {"text": negs},
            "rights": {"tier": "source_audit_required", "license": lic,
                       "audit": "pending", "redistribution_ok": False},
            "dedup": {"protocol": cardlib.DEDUP_PROTOCOL,
                      "hash": cardlib.dedup_hash(p["query"])},
        })

    # ---- 250-sample CLI gate BEFORE bulk ----
    rng = random.Random(SEED)
    sample = rng.sample(cards, min(250, len(cards)))
    spath = os.path.join(out_dir, "sample-validate.jsonl")
    known = os.path.join(out_dir, "known-ids.txt")
    with open(spath, "w") as f:
        for c in sample:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(known, "w") as f:
        for c in cards:
            f.write(c["card_id"] + "\n")
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "validate_card.py"), spath,
         "--known", known],
        capture_output=True, text=True,
        env={**os.environ, "FLUFFY_CARDS_ROOT": out_dir})
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="")
        raise SystemExit(f"{subset}: 250-sample gate FAILED — stop + post "
                         f"to T9 per brief (>30% reject rule)")
    log(f"{subset}: 250-sample CLI gate PASS")

    # ---- bulk validate ----
    schema = json.load(open(os.path.join(HERE, "card.schema.json")))
    ids = [c["card_id"] for c in cards]
    dupes = [i for i, ct in collections.Counter(ids).items() if ct > 1]
    if dupes:
        raise SystemExit(f"{subset}: duplicate card_ids {dupes[:5]}")
    known_ids = set(ids)
    errs = []
    for i, c in enumerate(cards):
        errs.extend(validate_card.check_card(c, schema, known_ids))
        if (i + 1) % 20000 == 0:
            log(f"  bulk validate {subset}: {i+1}/{n}")
    if errs:
        for e in errs[:10]:
            print(f"  {e}")
        raise SystemExit(f"{subset}: bulk validation FAILED "
                         f"({len(errs)} errors)")
    log(f"{subset}: bulk validate {n}/{n} PASS")
    with open(os.path.join(out_dir, "cards-v2.jsonl"), "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lib.update_state(src_name, gated=True, cards=n,
                     gate="250-sample CLI PASS + bulk 100%")

    # ---- exposures ----
    exposures = []
    for k, c in enumerate(cards):
        exposures.append({
            "anchor": {"card": c["card_id"], "view": "text"},
            "positive": {"card": c["card_id"], "view": "text-passage"},
            "negatives": [{"card": g["card_id"], "view": g["view"]}
                          for g in c["negatives"]["text"]],
            "lane": "text2text", "instruction": INSTRUCTION,
            "task_type": task_type,
            "pos_sim": round(float(pos[k]), 4),
            "relevance": pairs[k]["relevance"]})
    with open(os.path.join(out_dir, "exposures-text2text.jsonl"), "w") as f:
        for e in exposures:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    hist = collections.Counter(len(e["negatives"]) for e in exposures)
    log(f"{subset}: {len(exposures)} exposures, neg histogram "
        f"{dict(sorted(hist.items()))}")

    # ---- pack (shards_v2 contract) ----
    cards_by_id = {c["card_id"]: c for c in cards}
    random.Random(SEED).shuffle(exposures)

    def entry(ref: dict, extra: dict | None = None) -> dict:
        card = cards_by_id[ref["card"]]
        return {"card": ref["card"], "view": ref["view"],
                "content": card["views"][ref["view"]]["content"],
                **(extra or {})}

    manifest = []
    for s0 in range(0, len(exposures), SHARD_SIZE):
        chunk = exposures[s0:s0 + SHARD_SIZE]
        name = f"text-kalm-{subset}-{s0 // SHARD_SIZE:06d}.tar"
        path = os.path.join(shards_dir, name)
        w = shards_v2.ShardWriter(path)
        for j, e in enumerate(chunk):
            key = f"{s0 + j:08d}"
            card_negs = cards_by_id[e["anchor"]["card"]]["negatives"]["text"]
            negs = []
            for ref, cn in zip(e["negatives"], card_negs):
                assert cn["card_id"] == ref["card"], f"{key} neg drift"
                negs.append(entry(ref, {"miner": cn["miner"],
                                        "sim": cn["sim"],
                                        "band_rule": cn["band_rule"]}))
            w.add(key, {"lane": e["lane"], "instruction": e["instruction"],
                        "task_type": e["task_type"],
                        "pos_sim": e["pos_sim"],
                        "relevance": e["relevance"],
                        "anchor": entry(e["anchor"]),
                        "positive": entry(e["positive"]),
                        "negatives": negs}, {})
        w.close()
        manifest.append({"shard": name, "idx": name + ".idx.json",
                         "sha256": lib.sha256_file(path),
                         "samples": len(chunk),
                         "bytes": os.path.getsize(path)})
        log(f"  packed {name}: {len(chunk)} samples, "
            f"{manifest[-1]['bytes']/1e6:.1f} MB")

    with open(os.path.join(shards_dir, "MANIFEST.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(shards_dir, "SHA256SUMS"), "w") as f:
        for m in manifest:
            f.write(f"{m['sha256']}  {m['shard']}\n")
            hidx = lib.sha256_file(os.path.join(shards_dir, m["idx"]))
            f.write(f"{hidx}  {m['idx']}\n")

    # ---- verify: re-hash + reader spot checks ----
    for m in manifest:
        path = os.path.join(shards_dir, m["shard"])
        assert lib.sha256_file(path) == m["sha256"], f"{m['shard']} re-hash"
    store = shards_v2.ExposureStore(
        [os.path.join(shards_dir, m["shard"]) for m in manifest])
    lane_keys = store.lanes["text2text"]
    assert len(lane_keys) == len(exposures), "reader/exposure count drift"
    for probe in (0, len(lane_keys) // 2, len(lane_keys) - 1):
        si, key = lane_keys[probe]
        smp, _media = store.get(si, key)
        assert smp["lane"] == "text2text" and smp["anchor"]["content"]
    log(f"{subset}: verification PASS ({len(manifest)} shards)")

    report = {"subset": subset, "source": src_name, "task_type": task_type,
              "pairs_in": n0, "pairs_kept": n, "exposures": len(exposures),
              "calibration": calib, "k_max": K_MAX, "miner": MINER,
              "negatives_histogram": {str(k): v for k, v
                                      in sorted(hist.items())},
              "instruction": INSTRUCTION,
              "shards": len(manifest),
              "bytes": sum(m["bytes"] for m in manifest),
              "license": lic, "rights_tier": "source_audit_required",
              "packed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime())}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    lib.update_state(src_name, packed=True, exposures=len(exposures),
                     shards=len(manifest),
                     gb=round(report["bytes"] / 1e9, 2),
                     last_shard=manifest[-1]["shard"])
    log(f"{subset}: DONE — {len(exposures)} exposures, "
        f"{len(manifest)} shards, {report['bytes']/1e9:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    mine_pack(a.subset, a.force)


if __name__ == "__main__":
    main()
