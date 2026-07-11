#!/usr/bin/env python3
"""build_text_lane.py — v001 text corpus -> CARD-SPEC v1.1 text-lane cards,
exposures, and WebDataset shards (BUILDER-BRIEF item 3).

Pipeline (all CPU, all output on pool-ssd — NOTHING on the root fs):
  1. accepted-v001.jsonl (40,941 cards) -> card-v2 records, text views only
     (views.image / views.audio absent for this lane).
  2. Validator sample gate: >=200 random cards through cardkit's
     validate_card CLI BEFORE anything bulk ships.
  3. Bulk validation of every card (reuses validate_card.check_card; the
     CLI's O(n^2) duplicate scan is replaced by a Counter, same checks).
  4. Exposures, lane "text2text": anchor = canonical text view, one exposure
     per positive view. Positive derivation reproduces train.py v1
     load_pairs() EXACTLY, including the eval-task blacklist — the frozen
     G0 eval must stay uncontaminated. v1 count to match: 224,474.
  5. Negatives: the corpus' EXISTING teacher-band sims. v1 in-band rule
     (0.75 <= nm_sim <= 0.92) selects near-miss renditions of the same card
     (self-negatives naming a view, per spec). k=8 is the spec ceiling; the
     corpus ships at most ~3 in-band near-misses per card, so exposures
     carry 0-3 explicit negatives + in-batch at train time (decision D).
     TopK-PercPos (H) is a MINING-time rule for ANN negatives; it is not
     re-applied to curated near-misses (their sims exist only vs the
     canonical anchor). Recorded per negative via band_rule.
  6. WebDataset tar shards (deterministic shuffle, fixed seed) + sha256
     manifest per shard (TRAINING-CHECKLIST section C), verified by
     re-reading every tar after packing.

Shard sample format (one member per sample, "<key>.json"):
  the spec exposure object, plus a "resolved" map
  "<card_id>/<view>" -> content array, so samples are self-contained and
  the trainer never needs a side lookup into cards-v2.jsonl.

Env (defaults are the live paths):
  SRC_V001       accepted-v001.jsonl
  FL_BLACKLIST   eval-task-blacklist.json (train.py's FL_BLACKLIST)
  OUT_ROOT       output root (default /pool-ssd/fluffy/text-v001)
  SHARD_SIZE     samples per shard (default 8192)
  SEED           shuffle seed (default 20260712)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardlib  # noqa: E402
import validate_card  # noqa: E402

SRC_V001 = os.environ.get(
    "SRC_V001", "/pool-ssd/synth-forge/corpus/manifests/accepted-v001.jsonl")
BLACKLIST = os.environ.get(
    "FL_BLACKLIST", "/root/SYNTH-FORGE/state/eval-task-blacklist.json")
OUT_ROOT = os.environ.get("OUT_ROOT", "/pool-ssd/fluffy/text-v001")
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "8192"))
SEED = int(os.environ.get("SEED", "20260712"))

NM_BAND = (0.75, 0.92)  # v1 in-band near-miss rule (train.py load_pairs)
BAND_RULE = "v1-inband-0.75-0.92"
MINER = "teacher-band-v1"
LANE = "text2text"
INSTRUCTION = "Retrieve the matching description."  # spec exposure example
K_MAX = 8  # spec ceiling (decision D); availability may be lower

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def tview(text: str, native_id: str) -> dict:
    return {"content": [{"type": "text", "text": text}],
            "source": "synthetic", "origin": "v001", "native_id": native_id}


def convert_card(r: dict, blacklisted: bool) -> tuple[dict, list[str], list[str]]:
    """One v001 row -> (card, positive_view_names, negative_view_names).

    positive_view_names reproduces train.py load_pairs() positives 1:1
    (same texts, same order); negative_view_names reproduces its in-band
    near-miss filter 1:1.
    """
    nid = r["card_id"]
    cid = f"flf-{nid}"
    anchor = r["canonical_text"]
    views = {"text": tview(anchor, nid)}
    positives: list[str] = []

    for i, p in enumerate(r.get("paraphrases") or []):
        if isinstance(p, str):  # v1 condition, verbatim
            name = f"text-para-{i + 1}"
            views[name] = tview(p, nid)
            positives.append(name)
    if r.get("intent_text"):
        views["text-intent"] = tview(r["intent_text"], nid)
        positives.append("text-intent")
    for i, qa in enumerate((r.get("qa_pairs") or [])[:2]):
        if isinstance(qa, dict) and qa.get("q"):
            name = f"text-qa-{i + 1}"
            views[name] = tview(f"{qa['q']} {qa.get('a', '')}", nid)
            positives.append(name)

    for i, nm in enumerate(r.get("near_misses") or []):
        if isinstance(nm, str) and nm:
            views[f"text-nm-{i + 1}"] = tview(nm, nid)
    neg_views: list[tuple[str, float]] = []
    for i, (nm, s) in enumerate(zip(r.get("near_misses") or [],
                                    r.get("nm_sims") or [])):
        if isinstance(nm, str) and nm and NM_BAND[0] <= s <= NM_BAND[1]:
            neg_views.append((f"text-nm-{i + 1}", s))
    neg_views.sort(key=lambda t: -t[1])  # hardest first

    card = {
        "card_id": cid,
        "anchor_text": anchor,
        "views": views,
        "negatives": {"text": [
            {"card_id": cid, "view": name, "sim": round(s, 4),
             "miner": MINER, "band_rule": BAND_RULE}
            for name, s in neg_views]},
        "rights": {"tier": r["rights_tier"], "license": "self-synthetic",
                   "redistribution_ok": bool(r.get("redistribution_ok"))},
        "dedup": {"protocol": cardlib.DEDUP_PROTOCOL,
                  "hash": cardlib.dedup_hash(anchor)},
    }
    if not card["negatives"]["text"]:
        del card["negatives"]
    if blacklisted:
        card["notes"] = "eval-task-blacklist: card is excluded from exposures"
    return card, positives, [n for n, _ in neg_views]


def sample_gate(cards: list[dict], n: int = 250) -> None:
    """>=200 random cards through the real cardkit validator CLI."""
    rng = random.Random(SEED)
    sample = rng.sample(cards, n)
    path = os.path.join(OUT_ROOT, "sample-validate.jsonl")
    with open(path, "w") as f:
        for c in sample:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "validate_card.py"), path],
        capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, end="")
        raise SystemExit(f"sample validation FAILED ({n} cards) — aborting "
                         f"before bulk, see {path}")
    log(f"sample gate: {n} cards through validate_card.py CLI — PASS")


def bulk_validate(cards: list[dict]) -> int:
    """Every card through validate_card.check_card (same checks as the CLI;
    Counter replaces its quadratic duplicate scan)."""
    schema = json.load(open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "card.schema.json")))
    ids = [c["card_id"] for c in cards]
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate card_ids: {dupes[:10]}")
    known = set(ids)
    errs = []
    for i, c in enumerate(cards):
        errs.extend(validate_card.check_card(c, schema, known))
        if (i + 1) % 5000 == 0:
            log(f"bulk validate: {i + 1}/{len(cards)} "
                f"({(i + 1) / (time.time() - t0):.0f} cards/s cumulative)")
    if errs:
        for e in errs[:20]:
            print(f"  {e}")
        raise SystemExit(f"bulk validation FAILED: {len(errs)} error(s)")
    return len(cards)


def pack_shards(samples: list[dict]) -> list[dict]:
    """samples -> WebDataset tars + per-shard sha256 manifest entries."""
    manifest = []
    for s0 in range(0, len(samples), SHARD_SIZE):
        chunk = samples[s0:s0 + SHARD_SIZE]
        name = f"{LANE}-v001-{s0 // SHARD_SIZE:06d}.tar"
        path = os.path.join(OUT_ROOT, "shards", name)
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tf:
            for j, s in enumerate(chunk):
                data = json.dumps(s, ensure_ascii=False).encode()
                ti = tarfile.TarInfo(name=f"{s0 + j:08d}.json")
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
    return manifest


def verify_shards(manifest: list[dict]) -> None:
    """Re-read every tar: re-hash vs manifest, member count, first/last
    sample decode. This is the pre-rsync half of the section-C gate."""
    for m in manifest:
        path = os.path.join(OUT_ROOT, "shards", m["shard"])
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        assert h.hexdigest() == m["sha256"], f"{m['shard']}: sha mismatch"
        with tarfile.open(path) as tf:
            names = tf.getnames()
            assert len(names) == m["samples"], f"{m['shard']}: member count"
            for member in (names[0], names[-1]):
                s = json.load(tf.extractfile(member))
                assert s["lane"] == LANE and s["negatives"] is not None
                for ref in [s["anchor"], s["positive"], *s["negatives"]]:
                    key = f"{ref['card']}/{ref['view']}"
                    assert s["resolved"][key][0]["text"], f"{key} unresolved"
    log(f"verify: {len(manifest)} shard(s) re-hashed + spot-decoded — PASS")


def main() -> None:
    os.makedirs(os.path.join(OUT_ROOT, "shards"), exist_ok=True)
    bl = set(json.loads(open(BLACKLIST).read())["eval_task_ids"])
    log(f"blacklist: {len(bl)} eval task_ids")

    cards, pos_map, neg_map, n_black = [], {}, {}, 0
    with open(SRC_V001) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            blacklisted = r["task_id"] in bl  # train.py rule, verbatim
            n_black += blacklisted
            card, positives, negs = convert_card(r, blacklisted)
            cards.append(card)
            if not blacklisted:
                pos_map[card["card_id"]] = positives
                neg_map[card["card_id"]] = negs
            if len(cards) % 10000 == 0:
                log(f"converted {len(cards)} cards "
                    f"({len(cards) / (time.time() - t0):.0f}/s)")
    log(f"cards: {len(cards)} total, {n_black} blacklisted "
        f"(exposure-excluded), {len(pos_map)} exposure-eligible")

    sample_gate(cards)
    n_valid = bulk_validate(cards)
    log(f"bulk validate: {n_valid}/{len(cards)} cards PASS")

    cards_path = os.path.join(OUT_ROOT, "cards-v2.jsonl")
    with open(cards_path, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log(f"wrote {cards_path}")

    views_by_card = {c["card_id"]: c["views"] for c in cards}
    exposures = []
    for cid, positives in pos_map.items():
        negs = [{"card": cid, "view": v} for v in neg_map[cid][:K_MAX]]
        for pv in positives:
            exposures.append({"anchor": {"card": cid, "view": "text"},
                              "positive": {"card": cid, "view": pv},
                              "negatives": negs, "lane": LANE,
                              "instruction": INSTRUCTION})
    exp_path = os.path.join(OUT_ROOT, f"exposures-{LANE}-v001.jsonl")
    with open(exp_path, "w") as f:
        for e in exposures:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    nneg = collections.Counter(len(e["negatives"]) for e in exposures)
    log(f"exposures: {len(exposures)} (v1 target 224,474) -> {exp_path}; "
        f"negatives-per-exposure histogram {dict(sorted(nneg.items()))}")

    random.Random(SEED).shuffle(exposures)  # same-card runs would poison
    # in-batch negatives for any sequential reader
    samples = []
    for e in exposures:
        refs = [e["anchor"], e["positive"], *e["negatives"]]
        resolved = {f"{r['card']}/{r['view']}":
                    views_by_card[r["card"]][r["view"]]["content"]
                    for r in refs}
        samples.append({**e, "resolved": resolved})

    manifest = pack_shards(samples)
    man_path = os.path.join(OUT_ROOT, "shards", "MANIFEST.jsonl")
    with open(man_path, "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(OUT_ROOT, "shards", "SHA256SUMS"), "w") as f:
        for m in manifest:
            f.write(f"{m['sha256']}  {m['shard']}\n")
    verify_shards(manifest)

    report = {
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "src": SRC_V001, "blacklist": BLACKLIST,
        "blacklist_task_ids": len(bl),
        "cards_in": len(cards), "cards_out": len(cards),
        "cards_blacklisted": n_black,
        "exposures": len(exposures), "v1_pair_target": 224474,
        "negatives_per_exposure": {str(k): v
                                   for k, v in sorted(nneg.items())},
        "k_spec": K_MAX, "nm_band": list(NM_BAND), "band_rule": BAND_RULE,
        "lane": LANE, "instruction": INSTRUCTION,
        "shards": len(manifest),
        "shard_size": SHARD_SIZE, "shuffle_seed": SEED,
        "total_bytes": sum(m["bytes"] for m in manifest),
        "sample_format": "spec exposure + resolved: {card_id/view: content[]}",
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT_ROOT, "REPORT.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
