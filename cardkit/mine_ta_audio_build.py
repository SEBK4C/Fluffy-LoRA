#!/usr/bin/env python3
"""mine_ta_audio_build.py — real-audio pairs.jsonl -> CARD-SPEC cards ->
speech<->transcript / sound<->label exposures -> gated -> shards_v2 WDS.

Standalone (MINING-OPS §5): `--source librispeech|mls|fsd50k|tts-v001`.
Idempotent per source (REPORT.json + MANIFEST present = skip).

Contrast structure (CARD-SPEC + checklist §E):
  audio2text: anchor=audio, positive=transcript/labels, k=8 TEXT negatives
      from the teacher band (ceiling 0.95 — self-sim positives; the
      NV-Retriever ceiling kills near-dup false negatives). FSD50K text
      negatives additionally require label-set DISJOINTNESS (ground truth).
  text2audio: anchor=text, positive=audio, 2 AUDIO negatives:
      speech: same-voice-different-text (anti TTS/voice-shortcut rule),
      mined within the shard (speaker-grouped shard assignment makes
      same-speaker clips shard-local -> zero media duplication);
      fsd50k: disjoint-label clips ranked by label-text sim, within shard.
      In-batch covers the rest (k=8 is a ceiling, decision D).

Packing: cards are assigned to shards by SPEAKER GROUP (speech) so each
wav is stored once per shard; exposures shuffled within shard (seeded).
The trainer's manifest-index shuffle handles global ordering.

tts-v001: joins the 15,080 gated TTS views (audio-views-v001.jsonl) onto
v001 cards; the G0 eval-task blacklist (69 task_ids) is applied — nothing
v001-derived ships without it (G0-blacklist rule).
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

ROOT = "/pool-ssd/fluffy/mine-ta"
AUDIO_ROOT = os.path.join(ROOT, "audio")
SEED = 20260712
K_TEXT = 8
K_AUDIO = 1
PERCPOS = 0.95
SHARD_SIZE = 8192
# Media are stored PER SAMPLE in shards_v2 tars (no cross-sample dedup),
# so every t2a exposure ships 1+K_AUDIO wav copies. Full both-lane packing
# would be ~208 GB for LibriSpeech alone. Design: audio2text for ALL cards
# (1 wav each = the bulk alignment signal), text2audio for a deterministic
# 20% subset carrying the same-voice-diff-text anti-shortcut negative.
# In-batch negatives cover the rest (k is a ceiling, decision D).
T2A_FRAC = 0.20
INSTRUCTION = "Retrieve the matching description."  # frozen stage-1 string
V001_CARDS = "/pool-ssd/synth-forge/corpus/manifests/accepted-v001.jsonl"
V001_BLACKLIST = "/root/SYNTH-FORGE/state/eval-task-blacklist.json"
V001_AUDIO_VIEWS = "/pool-ssd/fluffy-cards/bulk/audio-views-v001.jsonl"
V001_CAS_ROOT = "/pool-ssd/fluffy-cards"


def log(msg: str) -> None:
    lib.log("audio-build", msg)


# --------------------------------------------------------- source loaders --

def load_speech(source: str):
    """librispeech | mls (all langs merged). Returns rows with speaker."""
    if source == "librispeech":
        paths = [os.path.join(AUDIO_ROOT, "librispeech", "pairs.jsonl")]
        prefix, origin = "als", "librispeech"
    else:
        base = AUDIO_ROOT
        paths = sorted(
            os.path.join(base, d, "pairs.jsonl") for d in os.listdir(base)
            if d.startswith("mls-")
            and os.path.exists(os.path.join(base, d, "pairs.jsonl")))
        prefix, origin = "amls", "mls"
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # dedup by native_id (LAST wins — torn/duplicate appends from killed
    # extractors), then deterministic order + ids
    rows = list({r["native_id"]: r for r in rows}.values())
    rows.sort(key=lambda r: r["native_id"])
    for i, r in enumerate(rows):
        r["card_id"] = f"flf-{prefix}-{i:07d}"
        r["origin"] = origin
        r["rights"] = {"tier": "commercial_after_attribution",
                       "license": "CC BY 4.0", "audit": "clear",
                       "redistribution_ok": False}
    return rows


def load_fsd50k():
    rows = []
    with open(os.path.join(AUDIO_ROOT, "fsd50k", "pairs.jsonl")) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    lic_map = {}
    lp = os.path.join(AUDIO_ROOT, "fsd50k", "licenses.json")
    if os.path.exists(lp):
        lic_map = json.load(open(lp))
    rows = list({r["native_id"]: r for r in rows}.values())
    rows.sort(key=lambda r: r["native_id"])
    for i, r in enumerate(rows):
        r["card_id"] = f"flf-afsd-{i:07d}"
        r["origin"] = "fsd50k"
        r["speaker"] = None
        clip = os.path.basename(r["native_id"])
        lic = (lic_map.get(clip) or "").lower()
        if "zero" in lic or "publicdomain/zero" in lic:
            tier, lname, audit = "commercial", "CC0 1.0", "clear"
        elif "by-nc" in lic or "sampling" in lic:
            tier, lname, audit = "research_only", lic, "pending"
        elif "/by/" in lic or lic.startswith("http://creativecommons.org/licenses/by"):
            tier, lname, audit = ("commercial_after_attribution",
                                  "CC BY", "clear")
        else:
            tier, lname, audit = ("source_audit_required", lic or "unknown",
                                  "pending")
        r["rights"] = {"tier": tier, "license": lname, "audit": audit,
                       "redistribution_ok": False}
    return rows


def load_tts_v001():
    """Join gated TTS views onto v001 card texts; APPLY THE G0 BLACKLIST."""
    bl_tasks = set(json.load(open(V001_BLACKLIST))["eval_task_ids"])
    texts, task_of = {}, {}
    with open(V001_CARDS) as f:
        for line in f:
            c = json.loads(line)
            texts[c["card_id"]] = c["canonical_text"]
            task_of[c["card_id"]] = c["task_id"]
    views = {}
    with open(V001_AUDIO_VIEWS) as f:
        for line in f:
            v = json.loads(line)
            # LAST row per card_id wins (overlength-correction rows append)
            views[v["v001_card_id"]] = v["view"]
    rows, n_bl = [], 0
    for cid, view in sorted(views.items()):
        if task_of.get(cid, "") in bl_tasks:
            n_bl += 1
            continue
        if not view.get("gate", {}).get("pass"):
            continue  # only gated views ride along
        txt = texts.get(cid, "")
        if not txt:
            continue
        rows.append({"card_id": f"flf-{cid}",  # matches text-v001 ids
                     "native_id": cid, "text": txt,
                     "view_obj": view, "origin": "v001-tts",
                     "speaker": (view.get("gen", {}) or {}).get("voice"),
                     "lang": "en", "duration_s": None,
                     "rights": {"tier": "commercial", "audit": "clear",
                                "license": "self-synthetic text + "
                                           "Supertonic-3/Kokoro TTS",
                                "redistribution_ok": True}})
    log(f"tts-v001: {len(rows)} joined views ({n_bl} G0-blacklisted "
        f"cards excluded)")
    return rows


# ------------------------------------------------------------------ build --

def build(source: str, force: bool = False) -> None:
    import numpy as np
    import shards_v2

    src_name = f"audio-{source}"
    out_dir = os.path.join(ROOT, "audio-lanes", source)
    shards_dir = os.path.join(out_dir, "shards")
    report_path = os.path.join(out_dir, "REPORT.json")
    if (not force and os.path.exists(report_path)
            and os.path.exists(os.path.join(shards_dir, "MANIFEST.jsonl"))):
        log(f"{source}: already packed — skip")
        return
    os.makedirs(shards_dir, exist_ok=True)

    if source in ("librispeech", "mls"):
        rows = load_speech(source)
        task_type = "speech_transcript"
    elif source == "fsd50k":
        rows = load_fsd50k()
        task_type = "sound_label"
    elif source == "tts-v001":
        rows = load_tts_v001()
        task_type = "speech_transcript"
    else:
        raise SystemExit(f"unknown source {source}")
    n = len(rows)
    log(f"{source}: {n} rows, task_type {task_type}")

    # ---- teacher embeddings of the text side (slice-resumable) ----
    emb_path = os.path.join(out_dir, "emb-text.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path).astype(np.float32)
        assert emb.shape[0] == n, "stale emb cache — rerun with --force"
        log(f"{source}: text embeddings loaded from cache")
    else:
        parts_dir = os.path.join(out_dir, "emb-parts")
        os.makedirs(parts_dir, exist_ok=True)
        SLICE = 8192
        parts = []
        for s in range(0, n, SLICE):
            pp = os.path.join(parts_dir, f"part-{s:08d}.npy")
            if os.path.exists(pp):
                try:
                    a = np.load(pp)
                    if a.shape[0] == min(SLICE, n - s):
                        parts.append(a)
                        continue
                except Exception:  # torn part from a kill -> redo
                    pass
            a = lib.embed([r["text"] for r in rows[s:s + SLICE]],
                          tag="audio-build").astype(np.float16)
            np.save(pp + f".tmp.{os.getpid()}.npy", a)
            os.replace(pp + f".tmp.{os.getpid()}.npy", pp)
            parts.append(a)
            log(f"{source}: embedded slice {s}-{s+len(a)}/{n}")
            lib.update_state(src_name, encoded_upto=s + len(a))
        emb = np.vstack(parts).astype(np.float32)
        np.save(emb_path + f".tmp.{os.getpid()}.npy", emb.astype(np.float16))
        os.replace(emb_path + f".tmp.{os.getpid()}.npy", emb_path)
        log(f"{source}: embedded {n} texts (cache written)")
    lib.update_state(src_name, encoded=True, rows=n)

    # ---- global TEXT negatives (teacher band, ceiling 0.95) ----
    label_sets = None
    if source == "fsd50k":
        label_sets = [frozenset(r.get("labels", [])) for r in rows]
    negs_idx = np.zeros((n, K_TEXT), dtype=np.int64)
    negs_sim = np.zeros((n, K_TEXT), dtype=np.float32)
    negs_cnt = np.zeros(n, dtype=np.int64)
    blk = 2048
    kw = K_TEXT + 24  # wider pool: disjointness/self filters eat candidates
    for s in range(0, n, blk):
        e = min(s + blk, n)
        sims = emb[s:e] @ emb.T
        sims[np.arange(e - s), np.arange(s, e)] = -2.0
        sims = np.where(sims < PERCPOS, sims, -2.0)  # ceiling: 0.95 x self
        top = np.argpartition(-sims, kw, axis=1)[:, :kw]
        for r in range(e - s):
            cand = top[r][np.argsort(-sims[r, top[r]])]
            out = []
            for j in cand:
                j = int(j)
                if sims[r, j] <= -1.5:
                    break
                if label_sets is not None and label_sets[s + r] & label_sets[j]:
                    continue  # ground-truth overlap -> false negative
                out.append(j)
                if len(out) == K_TEXT:
                    break
            negs_cnt[s + r] = len(out)
            negs_idx[s + r, :len(out)] = out
            negs_sim[s + r, :len(out)] = [float(sims[r, j]) for j in out]
        if (s // blk) % 8 == 0:
            log(f"  ann {source}: {e}/{n}")

    # ---- cards ----
    band_rule = f"topk-percpos-{PERCPOS}:ceil={PERCPOS}"
    cards = []
    for k, r in enumerate(rows):
        if source == "tts-v001":
            audio_view = r["view_obj"]
        else:
            audio_view = {
                "content": [{"type": "audio",
                             "audio": f"cas://{r['sha256']}"}],
                "source": "real", "origin": r["origin"],
                "native_id": r["native_id"]}
        card = {
            "card_id": r["card_id"],
            "anchor_text": r["text"],
            "views": {
                "text": {"content": [{"type": "text", "text": r["text"]}],
                         "source": "real" if source != "tts-v001"
                                   else "synthetic",
                         "origin": r["origin"],
                         "native_id": r["native_id"]},
                "audio": audio_view,
            },
            "negatives": {
                "text": [{"card_id": rows[j]["card_id"], "view": "text",
                          "sim": round(float(sm), 4),
                          "miner": "qwen3emb8b-ann-v1",
                          "band_rule": band_rule}
                         for j, sm in zip(negs_idx[k][:negs_cnt[k]],
                                          negs_sim[k][:negs_cnt[k]])]},
            "rights": r["rights"],
            "dedup": {"protocol": cardlib.DEDUP_PROTOCOL,
                      "hash": cardlib.dedup_hash(r["text"])},
        }
        cards.append(card)

    # ---- 250-sample CLI gate BEFORE bulk (media checks resolve via CAS) --
    cas_root = V001_CAS_ROOT if source == "tts-v001" else ROOT
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
    rr = subprocess.run(
        [sys.executable, os.path.join(HERE, "validate_card.py"), spath,
         "--known", known],
        capture_output=True, text=True,
        env={**os.environ, "FLUFFY_CARDS_ROOT": cas_root})
    print(rr.stdout, end="")
    if rr.returncode != 0:
        print(rr.stderr, end="")
        raise SystemExit(f"{source}: 250-sample gate FAILED — stop + post")
    log(f"{source}: 250-sample CLI gate PASS")

    # ---- bulk validate ----
    schema = json.load(open(os.path.join(HERE, "card.schema.json")))
    known_ids = set(c["card_id"] for c in cards)
    os.environ["FLUFFY_CARDS_ROOT"] = cas_root
    errs = []
    for i, c in enumerate(cards):
        errs.extend(validate_card.check_card(c, schema, known_ids))
        if (i + 1) % 20000 == 0:
            log(f"  bulk validate: {i+1}/{n} ({len(errs)} errs)")
    if errs:
        for e in errs[:10]:
            print(f"  {e}")
        raise SystemExit(f"{source}: bulk validation FAILED ({len(errs)})")
    log(f"{source}: bulk validate {n}/{n} PASS")
    with open(os.path.join(out_dir, "cards-v2.jsonl"), "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lib.update_state(src_name, gated=True, cards=n)

    # ---- t2a subset selection (deterministic) ----
    def t2a_selected(card_id: str) -> bool:
        h = hashlib.sha1(f"{SEED}:t2a:{card_id}".encode()).hexdigest()
        return int(h, 16) % 100 < int(T2A_FRAC * 100)

    # ---- shard assignment: speaker-grouped (speech) / seeded (fsd) ----
    idx_by_speaker: dict = collections.defaultdict(list)
    for k, r in enumerate(rows):
        idx_by_speaker[r.get("speaker") or f"solo-{k}"].append(k)
    speakers = sorted(idx_by_speaker)
    random.Random(SEED).shuffle(speakers)
    cards_per_shard = int(SHARD_SIZE / (1 + T2A_FRAC))
    shard_of = {}
    cur, cur_shard = 0, 0
    for spk in speakers:
        for k in idx_by_speaker[spk]:
            shard_of[k] = cur_shard
            cur += 1
            if cur >= cards_per_shard:
                cur, cur_shard = 0, cur_shard + 1
    n_shards = cur_shard + (1 if cur else 0)

    # ---- within-shard AUDIO negatives (same-voice-diff-text priority) ----
    shard_members: dict = collections.defaultdict(list)
    for k, sh in shard_of.items():
        shard_members[sh].append(k)
    audio_negs: dict = {}
    for sh, members in shard_members.items():
        by_spk = collections.defaultdict(list)
        for k in members:
            by_spk[rows[k].get("speaker") or f"solo-{k}"].append(k)
        m_emb = emb[members]
        m_index = {k: i for i, k in enumerate(members)}
        for k in members:
            if not t2a_selected(rows[k]["card_id"]):
                continue  # audio negatives only needed for t2a exposures
            spk = rows[k].get("speaker") or f"solo-{k}"
            same = [j for j in by_spk[spk] if j != k]
            picks = []
            if same:
                sims_same = emb[same] @ emb[k]
                order = np.argsort(-sims_same)
                for o in order:
                    if sims_same[o] < PERCPOS and (label_sets is None
                            or not (label_sets[k] & label_sets[same[o]])):
                        picks.append((same[o], float(sims_same[o]),
                                      "same-voice-diff-text"))
                    if len(picks) == K_AUDIO:
                        break
            if len(picks) < K_AUDIO:
                sims_all = m_emb @ emb[k]
                sims_all[m_index[k]] = -2.0
                order = np.argsort(-sims_all)
                for o in order:
                    j = members[int(o)]
                    if j == k or any(p[0] == j for p in picks):
                        continue
                    if sims_all[o] <= -1.5 or sims_all[o] >= PERCPOS:
                        continue
                    if label_sets is not None and (label_sets[k]
                                                   & label_sets[j]):
                        continue
                    picks.append((j, float(sims_all[o]),
                                  "in-shard-band"))
                    if len(picks) == K_AUDIO:
                        break
            audio_negs[k] = picks

    # ---- exposures ----
    def a_negs(k):
        return [{"card": rows[j]["card_id"], "view": "audio",
                 "miner": miner, "sim": round(sm, 4),
                 "band_rule": band_rule}
                for j, sm, miner in audio_negs.get(k, [])]

    exposures_by_shard: dict = collections.defaultdict(list)
    n_t2a = 0
    for k, c in enumerate(cards):
        sh = shard_of[k]
        meta = {"task_type": task_type,
                "duration_s": rows[k].get("duration_s"),
                "lang": rows[k].get("lang")}
        exposures_by_shard[sh].append({
            "anchor": {"card": c["card_id"], "view": "audio"},
            "positive": {"card": c["card_id"], "view": "text"},
            "negatives": [{"card": g["card_id"], "view": "text",
                           "miner": g["miner"], "sim": g["sim"],
                           "band_rule": g["band_rule"]}
                          for g in c["negatives"]["text"]],
            "lane": "audio2text", "instruction": INSTRUCTION, **meta})
        if t2a_selected(c["card_id"]):
            n_t2a += 1
            exposures_by_shard[sh].append({
                "anchor": {"card": c["card_id"], "view": "text"},
                "positive": {"card": c["card_id"], "view": "audio"},
                "negatives": a_negs(k),
                "lane": "text2audio", "instruction": INSTRUCTION, **meta})

    # ---- pack ----
    cards_by_id = {c["card_id"]: c for c in cards}
    rows_by_id = {r["card_id"]: r for r in rows}

    def entry(ref: dict, media: dict, extra: dict | None = None) -> dict:
        card = cards_by_id[ref["card"]]
        content = []
        for item in card["views"][ref["view"]]["content"]:
            if item["type"] == "audio":
                sha = item["audio"][6:]
                with open(lib.cas_path(cas_root, sha), "rb") as f:
                    data = f.read()
                mname = shards_v2.media_name(data, "wav")
                media[mname] = data
                content.append({"type": "audio",
                                "audio": f"member://{mname}"})
            else:
                content.append(item)
        return {"card": ref["card"], "view": ref["view"],
                "content": content, **(extra or {})}

    manifest = []
    total_exp = 0
    for sh in range(n_shards):
        chunk = exposures_by_shard[sh]
        random.Random(SEED + sh).shuffle(chunk)
        name = f"audio-{source}-{sh:06d}.tar"
        path = os.path.join(shards_dir, name)
        w = shards_v2.ShardWriter(path)
        for j, e in enumerate(chunk):
            key = f"{total_exp + j:08d}"
            media: dict = {}
            negs = [entry({"card": g["card"], "view": g["view"]}, media,
                          {"miner": g.get("miner"), "sim": g.get("sim"),
                           "band_rule": g.get("band_rule")})
                    for g in e["negatives"]]
            w.add(key, {"lane": e["lane"], "instruction": e["instruction"],
                        "task_type": e["task_type"],
                        "duration_s": e["duration_s"], "lang": e["lang"],
                        "anchor": entry(e["anchor"], media),
                        "positive": entry(e["positive"], media),
                        "negatives": negs}, media)
        w.close()
        total_exp += len(chunk)
        manifest.append({"shard": name, "idx": name + ".idx.json",
                         "sha256": lib.sha256_file(path),
                         "samples": len(chunk),
                         "bytes": os.path.getsize(path)})
        log(f"  packed {name}: {len(chunk)} samples "
            f"{manifest[-1]['bytes']/1e6:.0f} MB")
        lib.update_state(src_name, last_shard=name, packed_shards=sh + 1)

    with open(os.path.join(shards_dir, "MANIFEST.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(shards_dir, "SHA256SUMS"), "w") as f:
        for m in manifest:
            f.write(f"{m['sha256']}  {m['shard']}\n")
            f.write(f"{lib.sha256_file(os.path.join(shards_dir, m['idx']))}"
                    f"  {m['idx']}\n")

    # verify: re-hash + reader spot-check both lanes
    for m in manifest:
        assert lib.sha256_file(os.path.join(shards_dir, m["shard"])) == \
            m["sha256"], f"{m['shard']} re-hash mismatch"
    store = shards_v2.ExposureStore(
        [os.path.join(shards_dir, m["shard"]) for m in manifest])
    for lane in ("audio2text", "text2audio"):
        keys = store.lanes[lane]
        for probe in (0, len(keys) // 2, len(keys) - 1):
            si, key = keys[probe]
            smp, media = store.get(si, key)
            refs = [it["audio"][9:] for part in
                    ([smp["anchor"], smp["positive"]] + smp["negatives"])
                    for it in part["content"] if it["type"] == "audio"]
            assert all(rf in media for rf in refs), f"{lane} media missing"
    log(f"{source}: verification PASS ({len(manifest)} shards, "
        f"{total_exp} exposures)")

    neg_hist = collections.Counter(
        len(e["negatives"]) for ch in exposures_by_shard.values()
        for e in ch)
    langs = collections.Counter(r.get("lang") for r in rows)
    report = {
        "source": src_name, "task_type": task_type, "cards": n,
        "exposures": total_exp,
        "lanes": {"audio2text": n, "text2audio": n_t2a},
        "t2a_frac": T2A_FRAC,
        "audio_kind": "tts" if source == "tts-v001" else "real",
        "langs": dict(langs),
        "negatives_histogram": {str(k): v for k, v
                                in sorted(neg_hist.items())},
        "k_text": K_TEXT, "k_audio": K_AUDIO,
        "band_rule": band_rule, "instruction": INSTRUCTION,
        "shards": len(manifest),
        "bytes": sum(m["bytes"] for m in manifest),
        "packing": "speaker-grouped shards; audio negatives shard-local",
        "packed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    lib.update_state(src_name, packed=True, exposures=total_exp,
                     shards=len(manifest),
                     gb=round(report["bytes"] / 1e9, 2))
    log(f"{source}: DONE — {total_exp} exposures, {len(manifest)} shards, "
        f"{report['bytes']/1e9:.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["librispeech", "mls", "fsd50k", "tts-v001"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.source, a.force)


if __name__ == "__main__":
    main()
