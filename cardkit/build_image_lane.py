#!/usr/bin/env python3
"""build_image_lane.py — MMEB-train -> CARD-SPEC v1.1 image-lane warmup slice
(BUILDER-BRIEF item 4). Multi-stage; stages are separate subcommands so the
GPU step can run on another host with only staging/ + cas/ rsynced over.

  extract    CPU, PVE. MMEB parquets + images_zip -> pairs.jsonl + CAS images.
             Sources (real photos only): MSCOCO_i2t (train2014 ONLY —
             the frozen image-eval-v1 draws its MSCOCO half from val2014
             members inside this same subset, so ALL val2014 is excluded),
             VisualNews_i2t. One card per unique image; caption dedup by
             cardlib.dedup_hash; image dedup by sha256; PIL verify.
  encode     GPU (or CPU smoke). Qwen3-VL-Embedding-2B via sentence-
             transformers -> emb-text.npy + emb-image.npy (fp16, L2-normed).
  mine       CPU. Calibration (positive-sim distribution on a seeded ~1k
             sample) -> TopK-PercPos banding (negative ceiling = 0.95 x the
             query's OWN positive sim, NV-Retriever rule, decision H) ->
             ANN top-k (k<=8) hard negatives per direction from WITHIN the
             mined slice (never MMEB's bundled negatives — documented too
             easy) -> cards-v2.jsonl (validator sample gate FIRST, then
             bulk) -> exposures (lanes image2text + text2image).
  pack       CPU. WebDataset shards: sample JSON + referenced media
             co-packed per CARD-SPEC storage rules, MANIFEST.jsonl +
             SHA256SUMS, deterministic shuffle, full re-read verify.

Sample format (WDS, members share the zero-padded key):
  <key>.json           exposure + "resolved" {card/view -> content[]} +
                       "media" {cas://sha -> member name}
  <key>.<n>.jpg        each unique image referenced by the sample

Env (defaults = live paths):
  SRC_MMEB   /pool-6b/corpus-acq/work/mmeb_train/snapshot   (READ-ONLY)
  OUT_ROOT   /pool-ssd/fluffy/image-v001-warmup
  EVAL_JSONL /pool-ssd/fluffy-cards/eval/image-eval-v1.jsonl (exclusion list)
  N_PER_SUBSET 25000   SEED 20260712   SHARD_SIZE 8192   K_MAX 8
  MODEL_DIR  local snapshot dir of Qwen/Qwen3-VL-Embedding-2B (encode)
  DEVICE     cuda|cpu (encode)   BATCH 32   LIMIT (encode smoke, 0=all)
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SRC_MMEB = os.environ.get(
    "SRC_MMEB", "/pool-6b/corpus-acq/work/mmeb_train/snapshot")
OUT_ROOT = os.environ.get("OUT_ROOT", "/pool-ssd/fluffy/image-v001-warmup")
# cardlib binds FLUFFY_CARDS_ROOT at import time — force it BEFORE any
# cardlib import so cas:// refs resolve against THIS lane's media root.
os.environ["FLUFFY_CARDS_ROOT"] = OUT_ROOT
EVAL_JSONL = os.environ.get(
    "EVAL_JSONL", "/pool-ssd/fluffy-cards/eval/image-eval-v1.jsonl")
N_PER_SUBSET = int(os.environ.get("N_PER_SUBSET", "25000"))
SEED = int(os.environ.get("SEED", "20260712"))
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "8192"))
K_MAX = int(os.environ.get("K_MAX", "8"))
PERCPOS = 0.95           # decision H: ceiling = 0.95 x query's positive sim
MINER = "vl-ann-v1"
CAL_N = 1000             # calibration sample size
CAP_LEN = (20, 600)      # caption length filter (chars)

# subset -> (card prefix, i2t instruction, t2i instruction)
SUBSETS = {
    "MSCOCO_i2t": (
        "imgc",
        "Find an image caption describing the given everyday image.",
        "Find me an everyday image that matches the given caption."),
    "VisualNews_i2t": (
        "imgn",
        "Find a caption for the news in the given photo.",
        "Retrieve an image of this news caption."),
}

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def cas_path(sha: str) -> str:
    return os.path.join(OUT_ROOT, "cas", "sha256", sha[:2], sha)


def staging(name: str) -> str:
    return os.path.join(OUT_ROOT, "staging", name)


# --------------------------------------------------------------- extract ---

def eval_exclusion_basenames() -> set[str]:
    """Basenames of every image the frozen eval uses (belt) + we also drop
    ALL val2014 members (suspenders — the eval was sampled from them)."""
    out = set()
    with open(EVAL_JSONL) as f:
        for line in f:
            r = json.loads(line)
            nid = r.get("native_id", "")
            out.add(os.path.basename(nid))
    return out


def cmd_extract() -> None:
    import pyarrow.parquet as pq
    from PIL import Image

    os.makedirs(staging(""), exist_ok=True)
    excl = eval_exclusion_basenames()
    log(f"eval exclusion list: {len(excl)} basenames")

    seen_caption: set[str] = set()
    seen_sha: set[str] = set()
    pairs: list[dict] = []
    stats: dict[str, collections.Counter] = {}

    import cardlib
    for subset, (prefix, _, _) in SUBSETS.items():
        st = stats[subset] = collections.Counter()
        pf = pq.ParquetFile(os.path.join(
            SRC_MMEB, subset, "train-00000-of-00001.parquet"))
        rows = pf.read().to_pylist()
        st["rows"] = len(rows)

        cands = []
        for r in rows:
            member = r["qry_image_path"].removeprefix("images/")
            cap = " ".join(r["pos_text"].split()).strip()
            if "val2014" in member:
                st["drop_val2014"] += 1
                continue
            if os.path.basename(member) in excl:
                st["drop_eval"] += 1
                continue
            if not (CAP_LEN[0] <= len(cap) <= CAP_LEN[1]):
                st["drop_len"] += 1
                continue
            dh = cardlib.dedup_hash(cap)
            if dh in seen_caption:
                st["drop_dup_caption"] += 1
                continue
            seen_caption.add(dh)
            cands.append((member, cap, dh))
        st["candidates"] = len(cands)

        rng = random.Random(SEED)
        rng.shuffle(cands)
        want = min(N_PER_SUBSET, len(cands))
        sel = cands[: int(want * 1.05) + 64]  # buffer for image failures

        zf = zipfile.ZipFile(os.path.join(SRC_MMEB, "images_zip",
                                          f"{subset}.zip"))
        info = {i.filename: i for i in zf.infolist()}
        # read in archive order -> near-sequential I/O on the HDD pool
        sel.sort(key=lambda c: info[c[0]].header_offset
                 if c[0] in info else 1 << 62)
        n_kept = 0
        for member, cap, dh in sel:
            if n_kept >= want:
                break
            if member not in info:
                st["drop_missing_member"] += 1
                continue
            data = zf.read(member)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.verify()
                with Image.open(io.BytesIO(data)) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001
                st["drop_bad_image"] += 1
                continue
            sha = hashlib.sha256(data).hexdigest()
            if sha in seen_sha:
                st["drop_dup_image"] += 1
                continue
            seen_sha.add(sha)
            p = cas_path(sha)
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                tmp = p + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.rename(tmp, p)
            pairs.append({
                "card_id": f"flf-{prefix}-{n_kept:06d}",
                "subset": subset,
                "anchor_text": cap,
                "native_id": member,
                "image_sha256": sha,
                "image_bytes": len(data),
                "image_wh": [w, h],
                "dedup_hash": dh,
                "rights_tier": "source_audit_required",
            })
            n_kept += 1
            if n_kept % 5000 == 0:
                log(f"{subset}: staged {n_kept}/{want}")
        st["kept"] = n_kept
        log(f"{subset}: kept {n_kept}  stats={dict(st)}")

    with open(staging("pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(staging("extract-stats.json"), "w") as f:
        json.dump({k: dict(v) for k, v in stats.items()}, f, indent=2)
    tot = sum(p["image_bytes"] for p in pairs)
    log(f"extract DONE: {len(pairs)} pairs, {tot / 1e9:.2f} GB CAS images "
        f"-> {staging('pairs.jsonl')}")


# ---------------------------------------------------------------- encode ---

def cmd_encode() -> None:
    import numpy as np

    model_dir = os.environ.get("MODEL_DIR") or "Qwen/Qwen3-VL-Embedding-2B"
    device = os.environ.get("DEVICE", "cuda")
    batch = int(os.environ.get("BATCH", "32"))
    limit = int(os.environ.get("LIMIT", "0"))

    pairs = [json.loads(l) for l in open(staging("pairs.jsonl"))]
    if limit:
        pairs = pairs[:limit]
    log(f"encode: {len(pairs)} pairs, device={device}, batch={batch}")

    from sentence_transformers import SentenceTransformer
    import torch
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = SentenceTransformer(
        model_dir, device=device, trust_remote_code=True,
        model_kwargs={"dtype": dtype})
    log("model loaded")

    def run(items: list, tag: str) -> np.ndarray:
        out, t1 = [], time.time()
        for i in range(0, len(items), batch):
            emb = model.encode(items[i:i + batch], convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
            out.append(emb.astype(np.float16))
            done = min(i + batch, len(items))
            if done % (batch * 32) < batch or done == len(items):
                log(f"  {tag}: {done}/{len(items)} "
                    f"({done / (time.time() - t1):.1f}/s)")
        return np.concatenate(out, axis=0)

    texts = [p["anchor_text"] for p in pairs]
    emb_t = run(texts, "text")
    np.save(staging("emb-text.npy"), emb_t)

    images = [{"image": cas_path(p["image_sha256"])} for p in pairs]
    emb_i = run(images, "image")
    np.save(staging("emb-image.npy"), emb_i)

    meta = {"model": "Qwen/Qwen3-VL-Embedding-2B", "device": device,
            "dtype": str(dtype), "batch": batch, "n": len(pairs),
            "dim": int(emb_t.shape[1]),
            "elapsed_s": round(time.time() - t0, 1)}
    with open(staging("encode-meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"encode DONE: {meta}")


# ------------------------------------------------------------------ mine ---

def _views_for(p: dict) -> dict:
    return {
        "text": {"content": [{"type": "text", "text": p["anchor_text"]}],
                 "source": "real", "origin": "mmeb",
                 "native_id": p["native_id"]},
        "image": {"content": [{"type": "image",
                               "image": f"cas://{p['image_sha256']}"}],
                  "source": "real", "origin": "mmeb",
                  "native_id": p["native_id"]},
    }


def _license_for(subset: str) -> str:
    base = {"MSCOCO_i2t": "MSCOCO", "VisualNews_i2t": "VisualNews"}[subset]
    return f"{base}/MMEB-train (per-source audit)"


def cmd_mine() -> None:
    import numpy as np
    import validate_card

    pairs = [json.loads(l) for l in open(staging("pairs.jsonl"))]
    emb_t = np.load(staging("emb-text.npy")).astype(np.float32)
    emb_i = np.load(staging("emb-image.npy")).astype(np.float32)
    n = len(pairs)
    assert emb_t.shape[0] == n and emb_i.shape[0] == n, "emb/pairs mismatch"
    log(f"mine: {n} pairs, dim {emb_t.shape[1]}")

    pos = np.einsum("ij,ij->i", emb_i, emb_t)  # positive sims (i <-> i)
    cal_idx = random.Random(SEED).sample(range(n), min(CAL_N, n))
    cal = np.sort(pos[cal_idx])
    calib = {
        "n": len(cal_idx),
        "positive_sim": {
            "min": float(cal[0]), "p5": float(np.percentile(cal, 5)),
            "p25": float(np.percentile(cal, 25)),
            "median": float(np.percentile(cal, 50)),
            "p75": float(np.percentile(cal, 75)),
            "p95": float(np.percentile(cal, 95)), "max": float(cal[-1]),
            "mean": float(cal.mean())},
        "band_rule": f"topk-percpos-{PERCPOS}",
        "ceiling_note": "per-query: negative sim < "
                        f"{PERCPOS} x that query's positive sim",
        "implied_median_ceiling": float(PERCPOS * np.percentile(cal, 50)),
    }
    log(f"calibration: {json.dumps(calib['positive_sim'])}")

    def ann(q: np.ndarray, c: np.ndarray, tag: str) -> tuple:
        """top-K_MAX hard negatives per query under the per-query ceiling."""
        negs_idx = np.zeros((n, K_MAX), dtype=np.int64)
        negs_sim = np.zeros((n, K_MAX), dtype=np.float32)
        negs_cnt = np.zeros(n, dtype=np.int64)
        chunk = 2048
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            sims = q[s:e] @ c.T                      # (chunk, n)
            rows = np.arange(s, e)
            sims[np.arange(e - s), rows] = -2.0      # mask self
            ceil = PERCPOS * pos[s:e]
            sims_f = np.where(sims < ceil[:, None], sims, -2.0)
            top = np.argpartition(-sims_f, K_MAX, axis=1)[:, :K_MAX + 8]
            for r in range(e - s):
                cand = top[r][np.argsort(-sims_f[r, top[r]])]
                cand = [int(j) for j in cand if sims_f[r, j] > -1.5][:K_MAX]
                negs_cnt[s + r] = len(cand)
                negs_idx[s + r, :len(cand)] = cand
                negs_sim[s + r, :len(cand)] = sims_f[r, cand]
            if s % (chunk * 8) == 0:
                log(f"  ann {tag}: {e}/{n}")
        return negs_idx, negs_sim, negs_cnt

    # image2text: anchor=image, candidates=texts of other cards
    i2t_idx, i2t_sim, i2t_cnt = ann(emb_i, emb_t, "image->text")
    # text2image: anchor=text, candidates=images of other cards
    t2i_idx, t2i_sim, t2i_cnt = ann(emb_t, emb_i, "text->image")

    band_tag = f"topk-percpos-{PERCPOS}"
    cards = []
    for k, p in enumerate(pairs):
        ceil = round(float(PERCPOS * pos[k]), 4)
        br = f"{band_tag}:ceil={ceil}"
        card = {
            "card_id": p["card_id"],
            "anchor_text": p["anchor_text"],
            "views": _views_for(p),
            "negatives": {
                "text": [{"card_id": pairs[j]["card_id"], "view": "text",
                          "sim": round(float(s), 4), "miner": MINER,
                          "band_rule": br}
                         for j, s in zip(i2t_idx[k][:i2t_cnt[k]],
                                         i2t_sim[k][:i2t_cnt[k]])],
                "image": [{"card_id": pairs[j]["card_id"], "view": "image",
                           "sim": round(float(s), 4), "miner": MINER,
                           "band_rule": br}
                          for j, s in zip(t2i_idx[k][:t2i_cnt[k]],
                                          t2i_sim[k][:t2i_cnt[k]])],
            },
            "rights": {"tier": p["rights_tier"],
                       "license": _license_for(p["subset"]),
                       "source_sha256": p["image_sha256"],
                       "audit": "pending", "redistribution_ok": False},
            "dedup": {"protocol": "anchor-sha256-v1",
                      "hash": p["dedup_hash"]},
        }
        cards.append(card)
        if (k + 1) % 10000 == 0:
            log(f"cards built: {k + 1}/{n}")

    # ---- validator sample gate (>=250 through the CLI) BEFORE bulk ----
    rng = random.Random(SEED)
    sample = rng.sample(cards, 250)
    spath = os.path.join(OUT_ROOT, "sample-validate.jsonl")
    known = os.path.join(OUT_ROOT, "known-ids.txt")
    with open(spath, "w") as f:
        for c in sample:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(known, "w") as f:
        for c in cards:
            f.write(c["card_id"] + "\n")
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "validate_card.py"), spath,
         "--known", known],
        capture_output=True, text=True,
        env={**os.environ, "FLUFFY_CARDS_ROOT": OUT_ROOT})
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="")
        raise SystemExit("sample gate FAILED (>30% reject = stop & "
                         "investigate per brief) — aborting before bulk")
    log("sample gate: 250 cards through validate_card.py CLI — PASS")

    # ---- bulk validate (same checks; Counter replaces O(n^2) dup scan) ----
    schema = json.load(open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "card.schema.json")))
    ids = [c["card_id"] for c in cards]
    dupes = [i for i, c in collections.Counter(ids).items() if c > 1]
    if dupes:
        raise SystemExit(f"duplicate card_ids: {dupes[:10]}")
    known_ids = set(ids)
    errs = []
    for i, c in enumerate(cards):
        errs.extend(validate_card.check_card(c, schema, known_ids))
        if (i + 1) % 5000 == 0:
            log(f"bulk validate: {i + 1}/{n}")
    if errs:
        for e in errs[:20]:
            print(f"  {e}")
        raise SystemExit(f"bulk validation FAILED: {len(errs)} error(s)")
    log(f"bulk validate: {n}/{n} cards PASS")

    with open(os.path.join(OUT_ROOT, "cards-v2.jsonl"), "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- exposures: both lanes from every card ----
    inst = {p["card_id"]: SUBSETS[p["subset"]] for p in pairs}
    exposures = []
    for c in cards:
        cid = c["card_id"]
        _, i2t_inst, t2i_inst = inst[cid]
        exposures.append({
            "anchor": {"card": cid, "view": "image"},
            "positive": {"card": cid, "view": "text"},
            "negatives": [{"card": g["card_id"], "view": "text"}
                          for g in c["negatives"]["text"]],
            "lane": "image2text", "instruction": i2t_inst})
        exposures.append({
            "anchor": {"card": cid, "view": "text"},
            "positive": {"card": cid, "view": "image"},
            "negatives": [{"card": g["card_id"], "view": "image"}
                          for g in c["negatives"]["image"]],
            "lane": "text2image", "instruction": t2i_inst})
    for lane in ("image2text", "text2image"):
        path = os.path.join(OUT_ROOT, f"exposures-{lane}-v001.jsonl")
        with open(path, "w") as f:
            for e in exposures:
                if e["lane"] == lane:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
    hist = collections.Counter(len(e["negatives"]) for e in exposures)
    log(f"exposures: {len(exposures)} (2 lanes x {n}); "
        f"negatives histogram {dict(sorted(hist.items()))}")

    with open(staging("mine-report.json"), "w") as f:
        json.dump({"calibration": calib, "k_max": K_MAX, "miner": MINER,
                   "negatives_histogram": {str(k): v for k, v
                                           in sorted(hist.items())},
                   "cards": n, "exposures": len(exposures)}, f, indent=2)
    log("mine DONE")


# ------------------------------------------------------------------ pack ---

def cmd_pack() -> None:
    cards = {}
    for line in open(os.path.join(OUT_ROOT, "cards-v2.jsonl")):
        c = json.loads(line)
        cards[c["card_id"]] = c
    exposures = []
    for lane in ("image2text", "text2image"):
        path = os.path.join(OUT_ROOT, f"exposures-{lane}-v001.jsonl")
        exposures += [json.loads(l) for l in open(path)]
    log(f"pack: {len(exposures)} exposures over {len(cards)} cards")

    random.Random(SEED).shuffle(exposures)  # same-card runs would poison
    # in-batch negatives for sequential readers

    os.makedirs(os.path.join(OUT_ROOT, "shards"), exist_ok=True)
    manifest = []
    n_media_members = 0
    for s0 in range(0, len(exposures), SHARD_SIZE):
        chunk = exposures[s0:s0 + SHARD_SIZE]
        name = f"image-v001-{s0 // SHARD_SIZE:06d}.tar"
        path = os.path.join(OUT_ROOT, "shards", name)
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tf:
            for j, e in enumerate(chunk):
                key = f"{s0 + j:08d}"
                refs = [e["anchor"], e["positive"], *e["negatives"]]
                resolved, media = {}, {}
                for ref in refs:
                    c = cards[ref["card"]]
                    content = c["views"][ref["view"]]["content"]
                    resolved[f"{ref['card']}/{ref['view']}"] = content
                    for item in content:
                        if item["type"] == "image":
                            media[item["image"]] = None
                for m_i, cas_ref in enumerate(sorted(media)):
                    member = f"{key}.{m_i}.jpg"
                    media[cas_ref] = member
                    with open(cas_path(cas_ref[6:]), "rb") as f:
                        data = f.read()
                    ti = tarfile.TarInfo(name=member)
                    ti.size, ti.mtime, ti.uid, ti.gid = len(data), 0, 0, 0
                    tf.addfile(ti, io.BytesIO(data))
                    n_media_members += 1
                sample = {**e, "resolved": resolved, "media": media}
                data = json.dumps(sample, ensure_ascii=False).encode()
                ti = tarfile.TarInfo(name=f"{key}.json")
                ti.size, ti.mtime, ti.uid, ti.gid = len(data), 0, 0, 0
                tf.addfile(ti, io.BytesIO(data))
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        manifest.append({"shard": name, "sha256": h.hexdigest(),
                         "samples": len(chunk),
                         "bytes": os.path.getsize(path),
                         "first_key": f"{s0:08d}",
                         "last_key": f"{s0 + len(chunk) - 1:08d}"})
        log(f"packed {name}: {len(chunk)} samples, "
            f"{manifest[-1]['bytes'] / 1e6:.1f} MB")

    with open(os.path.join(OUT_ROOT, "shards", "MANIFEST.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(OUT_ROOT, "shards", "SHA256SUMS"), "w") as f:
        for m in manifest:
            f.write(f"{m['sha256']}  {m['shard']}\n")

    # verify: re-hash every tar + decode first/last sample + media present
    for m in manifest:
        path = os.path.join(OUT_ROOT, "shards", m["shard"])
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        assert h.hexdigest() == m["sha256"], f"{m['shard']}: sha mismatch"
        with tarfile.open(path) as tf:
            names = tf.getnames()
            jnames = sorted(x for x in names if x.endswith(".json"))
            assert len(jnames) == m["samples"], f"{m['shard']}: sample count"
            for member in (jnames[0], jnames[-1]):
                s = json.load(tf.extractfile(member))
                assert s["lane"] in ("image2text", "text2image")
                nameset = set(names)
                for ref in [s["anchor"], s["positive"], *s["negatives"]]:
                    key = f"{ref['card']}/{ref['view']}"
                    for item in s["resolved"][key]:
                        if item["type"] == "image":
                            assert s["media"][item["image"]] in nameset
                        else:
                            assert item["text"]
    log(f"verify: {len(manifest)} shard(s) re-hashed + spot-decoded — PASS")

    calib = json.load(open(staging("mine-report.json")))
    enc = json.load(open(staging("encode-meta.json")))
    stats = json.load(open(staging("extract-stats.json")))
    report = {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "src": "TIGER-Lab/MMEB-train @ 76dd0a4 (CORPUS-ACQ fetch)",
        "subsets": {s: SUBSETS[s][0] for s in SUBSETS},
        "extract_stats": stats,
        "eval_contamination_guard": "ALL val2014 members excluded + "
                                    "image-eval-v1 native_id basenames",
        "encoder": enc, "mining": calib,
        "cards": len(cards), "exposures": len(exposures),
        "lanes": ["image2text", "text2image"],
        "shards": len(manifest), "shard_size": SHARD_SIZE,
        "shuffle_seed": SEED,
        "total_bytes": sum(m["bytes"] for m in manifest),
        "media_members": n_media_members,
        "sample_format": "exposure + resolved{card/view: content[]} + "
                         "media{cas://sha: tar member}; images co-packed "
                         "per sample as <key>.<n>.jpg",
        "rights": "source_audit_required, audit=pending (SIGNOFF-001: "
                  "training OK, release gated)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_ROOT, "REPORT.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(json.dumps(report, indent=1))


if __name__ == "__main__":
    cmds = {"extract": cmd_extract, "encode": cmd_encode,
            "mine": cmd_mine, "pack": cmd_pack}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(2)
    cmds[sys.argv[1]]()
