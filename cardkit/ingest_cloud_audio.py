#!/usr/bin/env python3
"""ingest_cloud_audio.py — pull cloud-gated audio shards, finish the A3 gate.

Downloads audio/*.tar from the dataset repo, and for each clip: CAS-put the
16 kHz WAV, compute teacher round-trip sim (the tailnet-only half of the
gate), and append a manifest row identical to bulk_audio.py's. Idempotent:
already-ingested card_ids and already-processed tars are skipped.

Run repeatedly while the cloud job streams shards.
"""
from __future__ import annotations

import io
import json
import os
import tarfile

import cardlib
from build_golden import WER_MAX, TTS_SIM_MIN

OUT = os.path.join(cardlib.ROOT, "bulk", "audio-v001.jsonl")
STATE = os.path.join(cardlib.ROOT, "bulk", "ingested_tars.txt")


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    repo = os.environ.get("OUT_REPO", "SEBK4C/fluffy-noisy-tier")
    done_tars = set()
    if os.path.exists(STATE):
        done_tars = {l.strip() for l in open(STATE) if l.strip()}
    done_cards = set()
    if os.path.exists(OUT):
        done_cards = {json.loads(l)["card_id"] for l in open(OUT)
                      if l.strip() and "cas" in l}

    tars = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
            if f.startswith("audio/") and f.endswith(".tar")
            and f not in done_tars]
    print(f"{len(tars)} new tars")
    n_new = n_pass = 0
    for name in sorted(tars):
        path = hf_hub_download(repo, name, repo_type="dataset")
        clips: dict[str, dict] = {}
        with tarfile.open(path) as tf:
            for m in tf:
                cid, ext = m.name.rsplit(".", 1)
                clips.setdefault(cid, {})[ext] = tf.extractfile(m).read()
        rows = []
        for cid, parts in clips.items():
            if cid in done_cards or "wav" not in parts or "json" not in parts:
                continue
            meta = json.loads(parts["json"])
            sha = cardlib.cas_put(parts["wav"])
            rows.append((cid, sha, meta))
        # teacher sim locally (tailnet-only half of the A3 gate);
        # ref texts join on card_id from the tasks file
        texts = {t["card_id"]: t["text"] for t in TASKS}
        for cid, sha, meta in rows:
            ref = texts.get(cid)
            if ref is None:
                continue
            sim = cardlib.cos(*cardlib.teacher_embed(
                [ref, meta.get("transcript") or " "]))
            row = {"card_id": cid, "cas": sha, "voice": meta["voice"],
                   "wer": meta["wer"], "sim": round(float(sim), 4),
                   "pass": meta["wer"] <= WER_MAX and sim >= TTS_SIM_MIN,
                   "gen": meta["gen"], "src": "cloud"}
            with open(OUT, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done_cards.add(cid)
            n_new += 1
            n_pass += row["pass"]
        with open(STATE, "a") as f:
            f.write(name + "\n")
        print(f"{name}: +{len(rows)} clips")
    print(f"ingested {n_new} new, {n_pass} passed "
          f"({n_pass / max(n_new, 1):.0%})")


TASKS = [json.loads(l) for l in open(
    os.path.join(cardlib.ROOT, "bulk", "cloud_audio_tasks.jsonl"))
    if l.strip()]

if __name__ == "__main__":
    main()
