#!/usr/bin/env python3
"""minepack.py — negatives + cards + gates + exposures + shards for one
extracted+encoded unit (an MMEB subset, or a whole page source). Standalone
CPU stage (OPS §5):

  minepack.py --source mmeb --subset DocVQA
  minepack.py --source colpali            # subset 'all' (page sources)

Mining is the warmup recipe generalized (REUSE of build_image_lane.cmd_mine):
TopK-PercPos banding (ceiling = 0.95 x the query's own positive sim, decision
H), ANN top-k<=8 from WITHIN the unit, dataset-shipped negatives never used.
Every negative records sim + band_rule (difficulty metadata, quality bar §6).
Gates: 250-sample validate_card.py CLI run BEFORE bulk (>30% reject rule:
any CLI failure aborts), then bulk in-process validation of every card.
Pack: shards_v2 FORMAT CONTRACT + .idx.json sidecars + MANIFEST + SHA256SUMS
+ full re-hash + reader-path spot checks.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))                    # cardkit/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root

import common
from common import log
import mmeb_spec

CAL_N = 1000

SOURCE_META = {
    "colpali": dict(kind="docmatch", origin="colpali",
                    license="vidore/colpali_train_set constituents "
                            "(per-source audit)"),
    "visrag": dict(kind="docmatch", origin="visrag",
                   license="openbmb/VisRAG in-domain constituents "
                           "(per-source audit)"),
}


def load_unit(source: str, subset: str):
    """pairs + aligned item-embedding matrix for one minepack unit."""
    import numpy as np
    root = common.SRC_ROOT[source]
    stg = os.path.join(root, "staging", subset)
    pairs = []
    for pf in sorted(glob.glob(os.path.join(stg, "pairs*.jsonl"))):
        pairs += [json.loads(l) for l in open(pf)]
    ids, mats = [], []
    for itf in sorted(glob.glob(os.path.join(stg, "items-*.jsonl"))):
        embf = itf.replace("items-", "emb-").replace(".jsonl", ".npy")
        rows = [json.loads(l) for l in open(itf)]
        m = np.load(embf).astype(np.float32)
        assert m.shape[0] == len(rows), f"{embf}: emb/items mismatch"
        ids += [r["id"] for r in rows]
        mats.append(m)
    emb = np.concatenate(mats, axis=0) if mats else np.zeros((0, 2048), "f4")
    idx = {}
    for i, iid in enumerate(ids):
        idx.setdefault(iid, i)      # first occurrence wins (page-source dups)
    return root, stg, pairs, emb, idx


def dedup_pages(pairs: list[dict], stats) -> list[dict]:
    """Cross-file dedup for page sources (extract runs per-file in parallel,
    so global dedup happens here): unique (query-text-hash, page-sha)."""
    sys.path.insert(0, os.path.dirname(HERE))
    import cardlib
    seen, out = set(), []
    for p in pairs:
        key = (cardlib.dedup_hash(p["anchor_text"]),
               p["anchor"].get("sha") or p["positive"].get("sha"))
        if key in seen:
            stats["drop_crossfile_dup"] += 1
            continue
        seen.add(key)
        out.append(p)
    return out


def ann(q, c, pos_sim, own_cand, tag: str):
    """Top-K_MAX in-band negatives per query. own_cand[i] = candidate index
    of query i's own positive item (masked). Ceiling = PERCPOS * pos_sim[i]."""
    import numpy as np
    n, K = q.shape[0], common.K_MAX
    negs_idx = np.zeros((n, K), dtype=np.int64)
    negs_sim = np.zeros((n, K), dtype=np.float32)
    negs_cnt = np.zeros(n, dtype=np.int64)
    chunk = 2048
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = q[s:e] @ c.T
        sims[np.arange(e - s), own_cand[s:e]] = -2.0
        ceil = common.PERCPOS * pos_sim[s:e]
        sims_f = np.where(sims < ceil[:, None], sims, -2.0)
        kk = min(K + 8, sims_f.shape[1] - 1)
        top = np.argpartition(-sims_f, kk, axis=1)[:, :kk]
        for r in range(e - s):
            cand = top[r][np.argsort(-sims_f[r, top[r]])]
            cand = [int(j) for j in cand if sims_f[r, j] > -1.5][:K]
            negs_cnt[s + r] = len(cand)
            negs_idx[s + r, :len(cand)] = cand
            negs_sim[s + r, :len(cand)] = sims_f[r, cand]
        if s % (chunk * 16) == 0:
            log("mine", f"  ann {tag}: {e}/{n}")
    return negs_idx, negs_sim, negs_cnt


def side_view(kind_info: dict, side: str) -> str:
    return kind_info["anchor_view" if side == "anchor" else "positive_view"]


def content_of(rec: dict) -> list[dict]:
    out = []
    if rec["sha"]:
        out.append({"type": "image", "image": f"cas://{rec['sha']}"})
    if rec["text"]:
        out.append({"type": "text", "text": rec["text"]})
    return out


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--subset", default="all")
    args = ap.parse_args()
    source, sub = args.source, args.subset
    t0 = time.time()

    if source == "mmeb":
        spec = mmeb_spec.SUBSETS[sub]
        kind, origin, tag = spec["kind"], "mmeb", spec["tag"]
        license_ = f"{sub}/MMEB-train (per-source audit)"
    else:
        meta = SOURCE_META[source]
        kind, origin, tag = meta["kind"], meta["origin"], source[:4]
        license_ = meta["license"]
    ki = mmeb_spec.KIND[kind]

    root, stg, pairs, emb, idx = load_unit(source, sub)
    os.environ["FLUFFY_CARDS_ROOT"] = root
    import cardlib
    import validate_card
    import shards_v2

    stats = collections.Counter()
    if source != "mmeb":
        pairs = dedup_pages(pairs, stats)
    n = len(pairs)
    log("mine", f"{source}/{sub}: {n} pairs, kind={kind}, emb {emb.shape}")

    def item_row(rec):
        return idx[common.item_id(rec["kind"], rec["text"], rec["sha"])]

    a_rows = np.array([item_row(p["anchor"]) for p in pairs])
    p_rows = np.array([item_row(p["positive"]) for p in pairs])
    A, P = emb[a_rows], emb[p_rows]
    pos = np.einsum("ij,ij->i", A, P)

    cal_idx = random.Random(common.SEED).sample(range(n), min(CAL_N, n))
    cal = np.sort(pos[cal_idx])
    calib = {
        "n": len(cal_idx),
        "positive_sim": {k: float(v) for k, v in [
            ("min", cal[0]), ("p5", np.percentile(cal, 5)),
            ("p25", np.percentile(cal, 25)), ("median", np.percentile(cal, 50)),
            ("p75", np.percentile(cal, 75)), ("p95", np.percentile(cal, 95)),
            ("max", cal[-1]), ("mean", cal.mean())]},
        "band_rule": f"topk-percpos-{common.PERCPOS}",
        "implied_median_ceiling": float(common.PERCPOS * np.percentile(cal, 50)),
    }
    log("mine", f"calibration: {json.dumps(calib['positive_sim'])}")

    # ---- candidate spaces: unique items per role ----------------------------
    def unique_space(rows):
        uniq, inv = np.unique(rows, return_inverse=True)
        rep_card = {}
        for i, p in enumerate(pairs):
            rep_card.setdefault(int(rows[i]), i)
        return uniq, inv, rep_card

    # forward: anchors query the positive-item space
    pu, pinv, prep = unique_space(p_rows)
    fwd = ann(A, emb[pu], pos, pinv, "fwd")
    directions = {"fwd": (fwd, pu, prep, side_view(ki, "positive"))}
    if len(ki["lanes"]) == 2:       # bidirectional: positives query anchors
        au, ainv, arep = unique_space(a_rows)
        rev = ann(P, emb[au], pos, ainv, "rev")
        directions["rev"] = (rev, au, arep, side_view(ki, "anchor"))

    band_tag = f"topk-percpos-{common.PERCPOS}"
    cards, negs_by_dir = [], {d: [] for d in directions}
    for k, p in enumerate(pairs):
        ceil = round(float(common.PERCPOS * pos[k]), 4)
        br = f"{band_tag}:ceil={ceil}"
        views = {}
        for side in ("anchor", "positive"):
            vname = side_view(ki, side)
            views[vname] = {"content": content_of(p[side]), "source": "real",
                            "origin": origin, "native_id": p["native_id"]}
        negatives: dict[str, list] = {}
        for d, ((nix, nsm, ncn), uniq, rep, view) in directions.items():
            bucket = mmeb_spec.view_modality(view)
            refs = []
            for j, s in zip(nix[k][:ncn[k]], nsm[k][:ncn[k]]):
                other = pairs[rep[int(uniq[j])]]
                refs.append({"card_id": other["card_id"], "view": view,
                             "sim": round(float(s), 4),
                             "miner": common.MINER, "band_rule": br})
            negatives.setdefault(bucket, []).extend(refs)
            negs_by_dir[d].append(refs)
        sha = p["anchor"]["sha"] or p["positive"]["sha"]
        cards.append({
            "card_id": p["card_id"], "anchor_text": p["anchor_text"],
            "views": views, "negatives": negatives,
            "rights": {"tier": "source_audit_required", "license": license_,
                       "source_sha256": sha, "audit": "pending",
                       "redistribution_ok": False},
            "dedup": {"protocol": "anchor-sha256-v1",
                      "hash": cardlib.dedup_hash(p["anchor_text"])},
        })

    # ---- 250-sample validator CLI gate BEFORE bulk (brief hard rule) --------
    rng = random.Random(common.SEED)
    sample = rng.sample(cards, min(250, len(cards)))
    spath = os.path.join(stg, "sample-validate.jsonl")
    known = os.path.join(stg, "known-ids.txt")
    with open(spath, "w") as f:
        for c in sample:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(known, "w") as f:
        for c in cards:
            f.write(c["card_id"] + "\n")
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(HERE),
                                      "validate_card.py"), spath,
         "--known", known],
        capture_output=True, text=True,
        env={**os.environ, "FLUFFY_CARDS_ROOT": root})
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="")
        raise SystemExit(f"{source}/{sub}: sample gate FAILED — stop + post "
                         "to T9 per brief (>30% reject rule)")
    log("mine", "sample gate: 250 cards via validate_card.py CLI — PASS")

    schema = json.load(open(os.path.join(os.path.dirname(HERE),
                                         "card.schema.json")))
    ids = [c["card_id"] for c in cards]
    dupes = [i for i, c in collections.Counter(ids).items() if c > 1]
    if dupes:
        raise SystemExit(f"duplicate card_ids: {dupes[:10]}")
    known_ids = set(ids)
    errs = []
    for i, c in enumerate(cards):
        errs.extend(validate_card.check_card(c, schema, known_ids))
        if (i + 1) % 20000 == 0:
            log("mine", f"bulk validate: {i + 1}/{n}")
    if errs:
        for e in errs[:20]:
            print(f"  {e}")
        raise SystemExit(f"bulk validation FAILED: {len(errs)} error(s)")
    log("mine", f"bulk validate: {n}/{n} cards PASS")

    cards_path = os.path.join(stg, "cards-v2.jsonl")
    with open(cards_path + ".tmp", "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.rename(cards_path + ".tmp", cards_path)

    # ---- exposures (difficulty metadata rides on every negative) ------------
    exposures = []
    av, pv = side_view(ki, "anchor"), side_view(ki, "positive")
    lane_fwd = ki["lanes"][0]
    for k, p in enumerate(pairs):
        exposures.append({
            "anchor": {"card": p["card_id"], "view": av},
            "positive": {"card": p["card_id"], "view": pv},
            "negatives": [{"card": r["card_id"], "view": r["view"]}
                          for r in negs_by_dir["fwd"][k]],
            "lane": lane_fwd, "task": ki["task"], "source": source,
            "subset": sub, "instruction": common.INSTRUCTION})
        if "rev" in negs_by_dir:
            exposures.append({
                "anchor": {"card": p["card_id"], "view": pv},
                "positive": {"card": p["card_id"], "view": av},
                "negatives": [{"card": r["card_id"], "view": r["view"]}
                              for r in negs_by_dir["rev"][k]],
                "lane": ki["lanes"][1], "task": ki["task"], "source": source,
                "subset": sub, "instruction": common.INSTRUCTION})
    hist = collections.Counter(len(e["negatives"]) for e in exposures)
    log("mine", f"exposures: {len(exposures)}; neg hist "
        f"{dict(sorted(hist.items()))}")

    # ---- pack (shards_v2 FORMAT CONTRACT) ------------------------------------
    card_by_id = {c["card_id"]: c for c in cards}
    negref_by_card = {}
    for c in cards:
        flat = {}
        for bucket, refs in c["negatives"].items():
            for r in refs:
                flat[(r["card_id"], r["view"])] = r
        negref_by_card[c["card_id"]] = flat

    random.Random(common.SEED).shuffle(exposures)

    def entry(ref: dict, media: dict, extra: dict | None = None) -> dict:
        card = card_by_id[ref["card"]]
        content = []
        for item in card["views"][ref["view"]]["content"]:
            if item["type"] == "image":
                with open(common.cas_path(root, item["image"][6:]), "rb") as f:
                    data = f.read()
                mname = shards_v2.media_name(data, "jpg")
                media[mname] = data
                content.append({"type": "image", "image": f"member://{mname}"})
            else:
                content.append(item)
        return {"card": ref["card"], "view": ref["view"],
                "content": content, **(extra or {})}

    sdir = os.path.join(root, "shards", sub if source == "mmeb" else source)
    os.makedirs(sdir, exist_ok=True)
    prefix = f"{source}-{tag}" if source == "mmeb" else source
    manifest, n_media = [], 0
    for s0 in range(0, len(exposures), common.SHARD_SIZE):
        chunkx = exposures[s0:s0 + common.SHARD_SIZE]
        name = f"{prefix}-{s0 // common.SHARD_SIZE:06d}.tar"
        path = os.path.join(sdir, name)
        w = shards_v2.ShardWriter(path)
        for j, e in enumerate(chunkx):
            key = f"{s0 + j:08d}"
            media: dict[str, bytes] = {}
            anchor_negs = negref_by_card[e["anchor"]["card"]]
            negs = []
            for ref in e["negatives"]:
                nr = anchor_negs[(ref["card"], ref["view"])]
                negs.append(entry(ref, media,
                                  {"miner": nr["miner"], "sim": nr["sim"],
                                   "band_rule": nr["band_rule"]}))
            exposure = {"lane": e["lane"], "task": e["task"],
                        "source": e["source"], "subset": e["subset"],
                        "instruction": e["instruction"],
                        "anchor": entry(e["anchor"], media),
                        "positive": entry(e["positive"], media),
                        "negatives": negs}
            w.add(key, exposure, media)
            n_media += len(media)
        w.close()
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        manifest.append({"shard": name, "idx": name + ".idx.json",
                         "sha256": h.hexdigest(), "samples": len(chunkx),
                         "bytes": os.path.getsize(path),
                         "first_key": f"{s0:08d}",
                         "last_key": f"{s0 + len(chunkx) - 1:08d}"})
        log("pack", f"{name}: {len(chunkx)} samples, "
            f"{manifest[-1]['bytes'] / 1e6:.1f} MB")

    with open(os.path.join(sdir, "MANIFEST.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(sdir, "SHA256SUMS"), "w") as f:
        for m in manifest:
            f.write(f"{m['sha256']}  {m['shard']}\n")
            h = hashlib.sha256(open(os.path.join(
                sdir, m["idx"]), "rb").read()).hexdigest()
            f.write(f"{h}  {m['idx']}\n")

    # verify: re-hash + reader-path spot checks (warmup discipline)
    for m in manifest:
        h = hashlib.sha256()
        with open(os.path.join(sdir, m["shard"]), "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        assert h.hexdigest() == m["sha256"], f"{m['shard']}: sha drift"
    store = shards_v2.ExposureStore(
        [os.path.join(sdir, m["shard"]) for m in manifest])
    counts = store.counts()
    assert sum(counts.values()) == len(exposures), f"reader count {counts}"
    for i, m in enumerate(manifest):
        for key in (m["first_key"], m["last_key"]):
            e, media = store.get(i, key)
            for ref in [e["anchor"], e["positive"], *e["negatives"]]:
                mat = shards_v2.materialize(ref["content"], media)
                assert mat
    log("pack", f"verify: {len(manifest)} shards re-hashed + reader "
        f"spot-checks PASS; lanes {counts}")

    report = {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source, "subset": sub, "kind": kind,
        "task": ki["task"], "lanes": ki["lanes"],
        "instruction": common.INSTRUCTION,
        "encoder": common.ENCODER, "mining": {
            "calibration": calib, "k_max": common.K_MAX,
            "miner": common.MINER,
            "negatives_histogram": {str(k): v for k, v in sorted(hist.items())},
        },
        "gate": {"sample_250_cli": "PASS", "bulk": f"{n}/{n}"},
        "cards": n, "exposures": len(exposures),
        "shards": len(manifest), "shard_size": common.SHARD_SIZE,
        "shuffle_seed": common.SEED,
        "total_bytes": sum(m["bytes"] for m in manifest),
        "media_members": n_media, "stats_extra": dict(stats),
        "rights": "source_audit_required, audit=pending (SIGNOFF-001: "
                  "training OK, release gated)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    rpath = os.path.join(stg, "REPORT.json")
    with open(rpath + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
    os.rename(rpath + ".tmp", rpath)

    def upd(state):
        ss = common.subset_state(state, source, sub)
        ss["gated"] = {"sample_250_cli": "PASS", "bulk": f"{n}/{n}"}
        ss["mined"] = {"cards": n, "exposures": len(exposures),
                       "neg_hist": {str(k): v for k, v in sorted(hist.items())}}
        ss["packed"] = {"shards": len(manifest),
                        "bytes": sum(m["bytes"] for m in manifest)}
    common.update_state(upd)
    log("mine", f"{source}/{sub}: MINEPACK DONE in {time.time() - t0:.0f}s "
        f"— {n} cards, {len(exposures)} exposures, {len(manifest)} shards")


if __name__ == "__main__":
    main()
