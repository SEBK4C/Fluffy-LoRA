#!/usr/bin/env python3
"""build_golden.py — hand-built golden cards for CARD-SPEC v0.2.

15 cards, 3 per §E source (v001, MMEB, ColPali, LibriSpeech, FSD50K):
  flf-g001  v001, fully tri-modal with generated fill (TTS + rendered image,
            plus an alt-voice audio rendition)
  flf-g002  v001, carries the interleaved view (c1-permute recipe)
  flf-g003  v001, text + TTS audio (same voice as g001 -> anti-shortcut
            same-voice-diff-text negative)
  g004-006  MMEB MSCOCO_i2t: real photo + caption + TTS caption
  g007-009  ColPali: real page + query + TTS query
  g010-012  LibriSpeech dev-clean: REAL speech + transcript + rendered image
  g013-015  FSD50K dev: REAL env sound + label text

Every generated view gets its gate run FOR REAL (whisper WER / OCR+teacher
round-trip). Negatives mined among the goldens with the live teacher.
Source paths are env-overridable; defaults match the build host's mounts.

Output: $FLUFFY_CARDS_ROOT/golden/cards.jsonl + gate_report.json
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import numpy as np
import soundfile as sf

import cardlib
from cardlib import cas_put, cas_ref

SRC = {
    "v001": os.environ.get(
        "SRC_V001", "/pool-ssd/synth-forge/corpus/manifests/accepted-v001.jsonl"),
    "mmeb": os.environ.get(
        "SRC_MMEB", "/pool-6b/corpus-acq/work/mmeb_train/snapshot"),
    "colpali": os.environ.get(
        "SRC_COLPALI",
        "/pool-6b/corpus-acq/work/colpali/snapshot/data/train-00000-of-00082.parquet"),
    "librispeech": os.environ.get(
        "SRC_LIBRISPEECH",
        "/pool-6b/corpus-acq/work/librispeech/dev-clean.tar.gz"),
    "fsd50k": os.environ.get("SRC_FSD50K", "/pool-6b/corpus-acq/work/fsd50k"),
}

# FROZEN v1.0 gates (DECISIONS-CARDSPEC.md: A3 + C)
WER_MAX = 0.15
TTS_SIM_MIN = 0.90
RENDER_SIM_MIN = 0.80

report: dict = {"thresholds": {"asr_wer_max": WER_MAX,
                               "render_roundtrip_sim_min": RENDER_SIM_MIN},
                "gates": []}


def gate_tts(text: str, voice: str) -> tuple[str, dict, dict]:
    """Synthesize, store, gate (whisper WER + teacher round-trip sim)."""
    wav = cardlib.tts(text, voice)
    sha = cas_put(wav)
    m = cardlib.asr_wer(cardlib.cas_path(sha), text)
    sim = cardlib.cos(*cardlib.teacher_embed([text, m["transcript"] or " "]))
    gate = {"asr_wer": m["asr_wer"], "asr_model": m["asr_model"],
            "roundtrip_sim": round(sim, 4),
            "pass": m["asr_wer"] <= WER_MAX and sim >= TTS_SIM_MIN}
    report["gates"].append({"kind": "tts", "voice": voice,
                            "wer": m["asr_wer"], "sim": round(sim, 4),
                            "pass": gate["pass"], "text": text[:80]})
    return sha, {"model": "kokoro-82m-gguf", "version": "tts.cpp-ape",
                 "voice": voice}, gate


def gate_render(text: str) -> tuple[str, dict, dict]:
    png = cardlib.render_text_card(text)
    sha = cas_put(png)
    m = cardlib.rendered_roundtrip(cardlib.cas_path(sha), text)
    gate = {"roundtrip_sim": m["roundtrip_sim"], "ocr": m["ocr"],
            "pass": m["roundtrip_sim"] >= RENDER_SIM_MIN}
    report["gates"].append({"kind": "render", "sim": m["roundtrip_sim"],
                            "pass": gate["pass"], "text": text[:80]})
    return sha, {"model": "pil-typographic-card", "version": "v1"}, gate


def put_audio_16k(data: np.ndarray, sr: int) -> tuple[str, float]:
    wav = cardlib.to_wav16k(data, sr)
    sha = cas_put(wav)
    return sha, cardlib.wav_info(cardlib.cas_path(sha))["duration_s"]


def text_view(text: str, origin: str, source: str = "real", **kw) -> dict:
    return {"content": [{"type": "text", "text": text}],
            "source": source, "origin": origin, **kw}


def card_base(cid: str, anchor: str, rights: dict) -> dict:
    return {"card_id": cid, "anchor_text": anchor, "views": {},
            "rights": rights,
            "dedup": {"protocol": cardlib.DEDUP_PROTOCOL,
                      "hash": cardlib.dedup_hash(anchor)}}


# --- sources ---------------------------------------------------------------

def reject(src: str, native_id: str, why: str, **kw) -> None:
    report.setdefault("rejects", []).append(
        {"source": src, "native_id": native_id, "why": why, **kw})


def v001_cards() -> list[dict]:
    """First 3 en candidates whose TTS gate passes — reject-and-advance,
    exactly what the bulk generator will do."""
    picks = []
    with open(SRC["v001"]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("lang") == "en"
                    and 80 <= len(r["canonical_text"]) <= 300):
                continue
            sha, gen, gate = gate_tts(r["canonical_text"], "af_heart")
            if gate["pass"]:
                picks.append((r, sha, gen, gate))
            else:
                reject("v001", r["card_id"], "tts gate", wer=gate["asr_wer"])
            if len(picks) == 3:
                break
    rights = {"tier": "commercial", "license": "self-synthetic",
              "redistribution_ok": True}
    out = []
    for cid, (r, sha, gen, gate) in zip(
            ["flf-g001", "flf-g002", "flf-g003"], picks):
        c = card_base(cid, r["canonical_text"], dict(rights))
        c["views"]["text"] = text_view(r["canonical_text"], "v001",
                                       source="synthetic",
                                       native_id=r["card_id"])
        c["views"]["audio"] = {"content": [{"type": "audio", "audio": cas_ref(sha)}],
                               "source": "tts", "origin": "v001",
                               "native_id": r["card_id"], "gen": gen, "gate": gate}
        if cid in ("flf-g001", "flf-g002"):
            isha, igen, igate = gate_render(r["canonical_text"])
            c["views"]["image"] = {"content": [{"type": "image", "image": cas_ref(isha)}],
                                   "source": "rendered", "origin": "v001",
                                   "native_id": r["card_id"], "gen": igen,
                                   "gate": igate}
        if cid == "flf-g001":
            for alt_voice in ("bm_george", "am_michael", "af_bella"):
                asha, agen, agate = gate_tts(r["canonical_text"], alt_voice)
                if agate["pass"]:
                    c["views"]["audio-alt-voice"] = {
                        "content": [{"type": "audio", "audio": cas_ref(asha)}],
                        "source": "tts", "origin": "v001",
                        "native_id": r["card_id"], "gen": agen, "gate": agate}
                    break
                reject("v001", r["card_id"], f"alt-voice {alt_voice} tts gate",
                       wer=agate["asr_wer"])
            c["negatives"] = {"audio": [{"card_id": "flf-g003", "view": "audio",
                                         "miner": "same-voice-diff-text"}]}
        if cid == "flf-g002":
            c["interleaved"] = [{"recipe": "c1-permute", "content": [
                {"type": "image",
                 "image": c["views"]["image"]["content"][0]["image"]},
                {"type": "text", "text": r["canonical_text"]},
                {"type": "audio",
                 "audio": c["views"]["audio"]["content"][0]["audio"]}]}]
        out.append(c)
    return out


def mmeb_cards() -> list[dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(os.path.join(
        SRC["mmeb"], "MSCOCO_i2t", "train-00000-of-00001.parquet"))
    rows = next(pf.iter_batches(batch_size=64)).to_pylist()
    rows = [r for r in rows if 40 <= len(r["pos_text"]) <= 200]
    z = zipfile.ZipFile(os.path.join(SRC["mmeb"], "images_zip", "MSCOCO_i2t.zip"))
    picks = []
    for r in rows:
        asha, gen, gate = gate_tts(r["pos_text"].strip(), "af_bella")
        if gate["pass"]:
            picks.append((r, asha, gen, gate))
        else:
            reject("mmeb", r["qry_image_path"], "tts gate",
                   wer=gate["asr_wer"])
        if len(picks) == 3:
            break
    out = []
    for cid, (r, asha, gen, gate) in zip(
            ["flf-g004", "flf-g005", "flf-g006"], picks):
        caption = r["pos_text"].strip()
        member = r["qry_image_path"].removeprefix("images/")
        sha = cas_put(z.read(member))
        c = card_base(cid, caption,
                      {"tier": "source_audit_required", "audit": "pending",
                       "license": "MSCOCO/MMEB-train (per-source audit)",
                       "redistribution_ok": False})
        c["views"]["text"] = text_view(caption, "mmeb", native_id=member)
        c["views"]["image"] = {"content": [{"type": "image", "image": cas_ref(sha)}],
                               "source": "real", "origin": "mmeb",
                               "native_id": member}
        c["views"]["audio"] = {"content": [{"type": "audio", "audio": cas_ref(asha)}],
                               "source": "tts", "origin": "mmeb",
                               "native_id": member, "gen": gen, "gate": gate}
        out.append(c)
    return out


def colpali_cards() -> list[dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(SRC["colpali"])
    rows = next(pf.iter_batches(batch_size=64)).to_pylist()
    rows = [r for r in rows if 30 <= len(r["query"]) <= 200]
    picks = []
    for r in rows:
        asha, gen, gate = gate_tts(r["query"].strip(), "am_michael")
        if gate["pass"]:
            picks.append((r, asha, gen, gate))
        else:
            reject("colpali", r["image_filename"], "tts gate",
                   wer=gate["asr_wer"])
        if len(picks) == 3:
            break
    out = []
    for cid, (r, asha, gen, gate) in zip(
            ["flf-g007", "flf-g008", "flf-g009"], picks):
        query = r["query"].strip()
        sha = cas_put(r["image"]["bytes"])
        c = card_base(cid, query,
                      {"tier": "source_audit_required", "audit": "pending",
                       "license": "vidore/colpali_train_set (per-source audit)",
                       "redistribution_ok": False})
        c["views"]["text"] = text_view(query, "colpali",
                                       native_id=r["image_filename"])
        c["views"]["image"] = {"content": [{"type": "image", "image": cas_ref(sha)}],
                               "source": "real", "origin": "colpali",
                               "native_id": r["image_filename"]}
        c["views"]["audio"] = {"content": [{"type": "audio", "audio": cas_ref(asha)}],
                               "source": "tts", "origin": "colpali",
                               "native_id": r["image_filename"],
                               "gen": gen, "gate": gate}
        out.append(c)
    return out


def librispeech_cards() -> list[dict]:
    """Stream dev-clean.tar.gz; one utterance from each of 3 speakers."""
    done: list[tuple[str, bytes, str]] = []  # (utt_id, flac bytes, transcript)
    pend_flac: dict[str, tuple[str, bytes]] = {}   # chapter -> first flac
    pend_trans: dict[str, dict[str, str]] = {}     # chapter -> {utt: text}
    speakers: set[str] = set()
    with tarfile.open(SRC["librispeech"], "r:gz") as tf:
        for m in tf:
            if len(done) >= 3:
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
                trans = dict(l.split(" ", 1) for l in txt.splitlines() if " " in l)
                if key in pend_flac:
                    utt, data = pend_flac.pop(key)
                    if utt in trans:
                        done.append((utt, data, trans[utt]))
                        speakers.add(spk)
                else:
                    pend_trans[key] = trans
    out = []
    for cid, (utt, flac, transcript) in zip(
            ["flf-g010", "flf-g011", "flf-g012"], done):
        data, sr = sf.read(io.BytesIO(flac))
        sha, dur = put_audio_16k(np.asarray(data), sr)
        text = transcript.strip().capitalize()
        c = card_base(cid, text,
                      {"tier": "commercial_after_attribution",
                       "license": "CC-BY-4.0 (LibriSpeech)",
                       "redistribution_ok": False})
        c["views"]["audio"] = {"content": [{"type": "audio", "audio": cas_ref(sha)}],
                               "source": "real", "origin": "librispeech",
                               "native_id": utt}
        c["views"]["text"] = text_view(text, "librispeech", native_id=utt)
        isha, gen, gate = gate_render(text)
        if gate["pass"]:
            c["views"]["image"] = {
                "content": [{"type": "image", "image": cas_ref(isha)}],
                "source": "rendered", "origin": "librispeech",
                "native_id": utt, "gen": gen, "gate": gate}
        else:
            reject("librispeech", utt, "render gate",
                   sim=gate["roundtrip_sim"])
        out.append(c)
    return out


def fsd50k_cards() -> list[dict]:
    gt = subprocess.run(
        ["7z", "e", "-so", os.path.join(SRC["fsd50k"], "FSD50K.ground_truth.zip"),
         "FSD50K.ground_truth/dev.csv"],
        capture_output=True, check=True).stdout.decode()
    rows = list(csv.DictReader(io.StringIO(gt)))
    # diverse labels: first row per distinct leading label, skipping guitars
    seen, picks = set(), []
    for r in rows:
        lead = r["labels"].split(",")[0]
        if lead not in seen and lead not in ("Electric_guitar",):
            seen.add(lead)
            picks.append(r)
    out, i = [], 0
    for r in picks:
        if len(out) == 3:
            break
        member = f"FSD50K.dev_audio/{r['fname']}.wav"
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(
                ["7z", "e", "-y", f"-o{td}",
                 os.path.join(SRC["fsd50k"], "FSD50K.dev_audio.zip"), member],
                capture_output=True, check=True)
            wav_path = os.path.join(td, f"{r['fname']}.wav")
            data, sr = sf.read(wav_path)
        if len(data) / sr > cardlib.MAX_AUDIO_S:
            continue  # golden set only takes clips inside the cap
        sha, dur = put_audio_16k(np.asarray(data), sr)
        labels = r["labels"].replace("_", " ").split(",")
        text = f"Environmental sound recording: {', '.join(labels)}."
        cid = f"flf-g{13 + len(out):03d}"
        c = card_base(cid, text,
                      {"tier": "source_audit_required", "audit": "pending",
                       "license": "FSD50K per-clip CC (audit clip license)",
                       "redistribution_ok": False})
        c["views"]["audio"] = {"content": [{"type": "audio", "audio": cas_ref(sha)}],
                               "source": "real", "origin": "fsd50k",
                               "native_id": r["fname"]}
        c["views"]["text"] = text_view(text, "fsd50k", native_id=r["fname"])
        out.append(c)
    return out


# --- negatives: teacher kNN over the golden anchors ------------------------

def mine_negatives(cards: list[dict]) -> None:
    anchors = [c["anchor_text"] for c in cards]
    E = cardlib.teacher_embed(anchors)
    sims = E @ E.T
    np.fill_diagonal(sims, -1)
    for i, c in enumerate(cards):
        order = np.argsort(-sims[i])
        negs = [{"card_id": cards[j]["card_id"], "sim": round(float(sims[i][j]), 4),
                 "miner": "teacher-knn-golden-v1"} for j in order[:2]]
        c.setdefault("negatives", {}).setdefault("text", []).extend(negs)
        img_j = next((j for j in order
                      if "image" in cards[j]["views"]), None)
        if img_j is not None and "image" in c["views"]:
            c["negatives"].setdefault("image", []).append(
                {"card_id": cards[img_j]["card_id"],
                 "sim": round(float(sims[i][img_j]), 4),
                 "miner": "teacher-knn-golden-ximg-v1"})


def main() -> None:
    cards = (v001_cards() + mmeb_cards() + colpali_cards()
             + librispeech_cards() + fsd50k_cards())
    mine_negatives(cards)
    outdir = os.path.join(cardlib.ROOT, "golden")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "cards.jsonl"), "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_gate = len(report["gates"])
    n_pass = sum(1 for g in report["gates"] if g["pass"])
    report["summary"] = {"cards": len(cards), "gates_run": n_gate,
                         "gates_passed": n_pass}
    with open(os.path.join(outdir, "gate_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"{len(cards)} cards -> {outdir}/cards.jsonl")
    print(f"gates: {n_pass}/{n_gate} passed")
    for g in report["gates"]:
        val = g.get("wer", g.get("sim"))
        print(f"  {g['kind']:7s} {'PASS' if g['pass'] else 'FAIL':4s} "
              f"{val:.4f}  {g['text'][:60]}")


if __name__ == "__main__":
    main()
