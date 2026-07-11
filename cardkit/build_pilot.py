#!/usr/bin/env python3
"""build_pilot.py — 200-card pilot, stratified across §E sources.

Unlike build_golden (curated: reject-and-advance), the pilot generates for
the first N candidates per source and records the FULL gate-value
distribution — that is what derives thresholds. Views that fail their gate
are dropped from the card (never shipped failing); the card survives if at
least one view remains.

Strata: v001 80 (en) · mmeb 30 · colpali 30 · librispeech 30 · fsd50k 30.
ColPali queries get the proposed boilerplate-strip rule (first line only);
applied instances are counted in the report.

Output: $FLUFFY_CARDS_ROOT/pilot/cards.jsonl + pilot_report.json
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import zipfile

import numpy as np
import soundfile as sf

import cardlib
from cardlib import cas_put, cas_ref

from build_golden import SRC, WER_MAX, RENDER_SIM_MIN, card_base, text_view

VOICES = ["af_heart", "af_bella", "am_michael", "bm_george"]
report: dict = {"strata": {}, "tts": [], "render": [],
                "colpali_boilerplate_stripped": 0}


def tts_view(text: str, voice: str, origin: str, native_id: str) -> dict | None:
    wav = cardlib.tts(text, voice)
    sha = cas_put(wav)
    m = cardlib.asr_wer(cardlib.cas_path(sha), text)
    sim = cardlib.cos(*cardlib.teacher_embed([text, m["transcript"] or " "]))
    report["tts"].append({"origin": origin, "native_id": native_id,
                          "voice": voice, "wer": m["asr_wer"],
                          "sim": round(sim, 4)})
    if m["asr_wer"] > WER_MAX:
        return None
    return {"content": [{"type": "audio", "audio": cas_ref(sha)}],
            "source": "tts", "origin": origin, "native_id": native_id,
            "gen": {"model": "kokoro-82m-gguf", "version": "tts.cpp-ape",
                    "voice": voice},
            "gate": {"asr_wer": m["asr_wer"], "asr_model": m["asr_model"],
                     "roundtrip_sim": round(sim, 4), "pass": True}}


def render_view(text: str, origin: str, native_id: str) -> dict | None:
    png = cardlib.render_text_card(text)
    sha = cas_put(png)
    m = cardlib.rendered_roundtrip(cardlib.cas_path(sha), text)
    report["render"].append({"origin": origin, "native_id": native_id,
                             "sim": m["roundtrip_sim"]})
    if m["roundtrip_sim"] < RENDER_SIM_MIN:
        return None
    return {"content": [{"type": "image", "image": cas_ref(sha)}],
            "source": "rendered", "origin": origin, "native_id": native_id,
            "gen": {"model": "pil-typographic-card", "version": "v1"},
            "gate": {"roundtrip_sim": m["roundtrip_sim"], "ocr": m["ocr"],
                     "pass": True}}


def v001(n: int) -> list[dict]:
    out = []
    with open(SRC["v001"]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("lang") != "en" or not (60 <= len(r["canonical_text"]) <= 350):
                continue
            i = len(out)
            c = card_base(f"flf-p{i + 1:03d}", r["canonical_text"],
                          {"tier": "commercial", "license": "self-synthetic",
                           "redistribution_ok": True})
            c["views"]["text"] = text_view(r["canonical_text"], "v001",
                                           source="synthetic",
                                           native_id=r["card_id"])
            v = tts_view(r["canonical_text"], VOICES[i % len(VOICES)],
                         "v001", r["card_id"])
            if v:
                c["views"]["audio"] = v
            if i % 4 == 0:  # renders on a quarter of the stratum
                v = render_view(r["canonical_text"], "v001", r["card_id"])
                if v:
                    c["views"]["image"] = v
            out.append(c)
            if len(out) == n:
                break
    return out


def mmeb(n: int, start: int) -> list[dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(os.path.join(
        SRC["mmeb"], "MSCOCO_i2t", "train-00000-of-00001.parquet"))
    z = zipfile.ZipFile(os.path.join(SRC["mmeb"], "images_zip", "MSCOCO_i2t.zip"))
    out = []
    for batch in pf.iter_batches(batch_size=128):
        for r in batch.to_pylist():
            cap = r["pos_text"].strip()
            if not (30 <= len(cap) <= 250):
                continue
            member = r["qry_image_path"].removeprefix("images/")
            i = len(out)
            c = card_base(f"flf-p{start + i:03d}", cap,
                          {"tier": "source_audit_required", "audit": "pending",
                           "license": "MSCOCO/MMEB-train (per-source audit)",
                           "redistribution_ok": False})
            c["views"]["text"] = text_view(cap, "mmeb", native_id=member)
            c["views"]["image"] = {
                "content": [{"type": "image", "image": cas_ref(cas_put(z.read(member)))}],
                "source": "real", "origin": "mmeb", "native_id": member}
            v = tts_view(cap, VOICES[i % len(VOICES)], "mmeb", member)
            if v:
                c["views"]["audio"] = v
            out.append(c)
            if len(out) == n:
                return out
    return out


BOILER = re.compile(
    r"\n.*(answer|response|respond|provide|output|reply)[^\n]*$",
    re.IGNORECASE | re.DOTALL)


def colpali(n: int, start: int) -> list[dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(SRC["colpali"])
    out = []
    for batch in pf.iter_batches(batch_size=128):
        for r in batch.to_pylist():
            q = r["query"].strip()
            q2 = BOILER.sub("", q).strip()
            if q2 != q:
                report["colpali_boilerplate_stripped"] += 1
            if not (20 <= len(q2) <= 250):
                continue
            i = len(out)
            c = card_base(f"flf-p{start + i:03d}", q2,
                          {"tier": "source_audit_required", "audit": "pending",
                           "license": "vidore/colpali_train_set (per-source audit)",
                           "redistribution_ok": False})
            c["views"]["text"] = text_view(q2, "colpali",
                                           native_id=r["image_filename"])
            c["views"]["image"] = {
                "content": [{"type": "image",
                             "image": cas_ref(cas_put(r["image"]["bytes"]))}],
                "source": "real", "origin": "colpali",
                "native_id": r["image_filename"]}
            v = tts_view(q2, VOICES[i % len(VOICES)], "colpali",
                         r["image_filename"])
            if v:
                c["views"]["audio"] = v
            out.append(c)
            if len(out) == n:
                return out
    return out


def librispeech(n: int, start: int) -> list[dict]:
    done: list[tuple[str, bytes, str]] = []
    pend_flac: dict[str, tuple[str, bytes]] = {}
    pend_trans: dict[str, dict[str, str]] = {}
    speakers: set[str] = set()
    with tarfile.open(SRC["librispeech"], "r:gz") as tf:
        for m in tf:
            if len(done) >= n:
                break
            parts = m.name.split("/")
            if len(parts) < 4:
                continue
            spk, chap = parts[-3], parts[-2]
            if spk in speakers:
                continue
            key = f"{spk}/{chap}"
            if m.name.endswith(".flac") and key not in pend_flac:
                utt = parts[-1].removesuffix(".flac")
                data = tf.extractfile(m).read()
                if key in pend_trans and utt in pend_trans[key]:
                    done.append((utt, data, pend_trans[key][utt]))
                    speakers.add(spk)
                else:
                    pend_flac[key] = (utt, data)
            elif m.name.endswith(".trans.txt"):
                txt = tf.extractfile(m).read().decode()
                trans = dict(l.split(" ", 1) for l in txt.splitlines()
                             if " " in l)
                if key in pend_flac:
                    utt, data = pend_flac.pop(key)
                    if utt in trans:
                        done.append((utt, data, trans[utt]))
                        speakers.add(spk)
                else:
                    pend_trans[key] = trans
    out = []
    for i, (utt, flac, transcript) in enumerate(done):
        data, sr = sf.read(io.BytesIO(flac))
        if len(data) / sr > cardlib.MAX_AUDIO_S:
            continue
        wav = cardlib.to_wav16k(np.asarray(data), sr)
        text = transcript.strip().capitalize()
        c = card_base(f"flf-p{start + len(out):03d}", text,
                      {"tier": "commercial_after_attribution",
                       "license": "CC-BY-4.0 (LibriSpeech)",
                       "redistribution_ok": False})
        c["views"]["audio"] = {
            "content": [{"type": "audio", "audio": cas_ref(cas_put(wav))}],
            "source": "real", "origin": "librispeech", "native_id": utt}
        c["views"]["text"] = text_view(text, "librispeech", native_id=utt)
        v = render_view(text, "librispeech", utt)
        if v:
            c["views"]["image"] = v
        out.append(c)
    return out


def fsd50k(n: int, start: int) -> list[dict]:
    gt = subprocess.run(
        ["7z", "e", "-so",
         os.path.join(SRC["fsd50k"], "FSD50K.ground_truth.zip"),
         "FSD50K.ground_truth/dev.csv"],
        capture_output=True, check=True).stdout.decode()
    rows = list(csv.DictReader(io.StringIO(gt)))
    seen, picks = set(), []
    for r in rows:
        lead = r["labels"].split(",")[0]
        if lead not in seen:
            seen.add(lead)
            picks.append(r)
        if len(picks) == n * 2:  # headroom for >30s clips
            break
    out = []
    for r in picks:
        if len(out) == n:
            break
        member = f"FSD50K.dev_audio/{r['fname']}.wav"
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                ["7z", "e", "-y", f"-o{td}",
                 os.path.join(SRC["fsd50k"], "FSD50K.dev_audio.zip"), member],
                capture_output=True, check=True)
            data, sr = sf.read(os.path.join(td, f"{r['fname']}.wav"))
        if len(data) / sr > cardlib.MAX_AUDIO_S:
            continue
        wav = cardlib.to_wav16k(np.asarray(data), sr)
        labels = r["labels"].replace("_", " ").split(",")
        text = f"Environmental sound recording: {', '.join(labels)}."
        c = card_base(f"flf-p{start + len(out):03d}", text,
                      {"tier": "source_audit_required", "audit": "pending",
                       "license": "FSD50K per-clip CC (audit clip license)",
                       "redistribution_ok": False})
        c["views"]["audio"] = {
            "content": [{"type": "audio", "audio": cas_ref(cas_put(wav))}],
            "source": "real", "origin": "fsd50k", "native_id": r["fname"]}
        c["views"]["text"] = text_view(text, "fsd50k", native_id=r["fname"])
        out.append(c)
    return out


def mine_negatives(cards: list[dict], k: int = 4) -> None:
    E = cardlib.teacher_embed([c["anchor_text"] for c in cards])
    sims = E @ E.T
    np.fill_diagonal(sims, -1)
    for i, c in enumerate(cards):
        order = np.argsort(-sims[i])[:k]
        c["negatives"] = {"text": [
            {"card_id": cards[j]["card_id"],
             "sim": round(float(sims[i][j]), 4),
             "miner": "teacher-knn-pilot-v1"} for j in order]}


def main() -> None:
    strata = []
    strata += v001(80)
    strata += mmeb(30, start=81)
    strata += colpali(30, start=111)
    strata += librispeech(30, start=141)
    strata += fsd50k(30, start=171)
    mine_negatives(strata)

    outdir = os.path.join(cardlib.ROOT, "pilot")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "cards.jsonl"), "w") as f:
        for c in strata:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    per = {}
    for o in ("v001", "mmeb", "colpali", "librispeech", "fsd50k"):
        per[o] = sum(1 for c in strata
                     if any(v.get("origin") == o for v in c["views"].values()))
    wers = [t["wer"] for t in report["tts"]]
    report["strata"] = per
    report["summary"] = {
        "cards": len(strata),
        "tts_attempts": len(report["tts"]),
        "tts_pass_at": {str(th): sum(1 for w in wers if w <= th) / len(wers)
                        for th in (0.05, 0.08, 0.10, 0.15, 0.20)},
        "render_attempts": len(report["render"]),
        "render_pass": sum(1 for r in report["render"]
                           if r["sim"] >= RENDER_SIM_MIN),
    }
    with open(os.path.join(outdir, "pilot_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report["summary"], indent=1))
    print("strata:", per)


if __name__ == "__main__":
    main()
