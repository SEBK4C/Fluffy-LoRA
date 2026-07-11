#!/usr/bin/env python3
"""build_frozen_evals.py — per-lane FROZEN eval sets (§F + CARD-SPEC eval
alignment). Real media on at least one side of EVERY pair; generated media
never on the eval side. Teacher (3080 Ti, :9020) used for near-dup filtering
inside the eval sets. Byte-frozen: manifest sha256 pins in eval/*.freeze.

Image lane (gates the swap): 250 ColPali TEST-split pages + 250 MSCOCO
val2014 photos, each with its real text.
Audio lane: 250 LibriSpeech test-clean utterances (real speech) + up to 100
FSD50K eval-split clips (real env sound, <= 30 s).

Output: $FLUFFY_CARDS_ROOT/eval/{image-eval-v1.jsonl,audio-eval-v1.jsonl,*.freeze}
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile

import numpy as np
import soundfile as sf

import cardlib
from build_golden import SRC

DUP_SIM = 0.95


def dedup_by_teacher(items: list[dict], key: str = "text") -> list[dict]:
    """Drop items whose text embeds > DUP_SIM against an earlier keeper."""
    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []
    for i in range(0, len(items), 64):
        batch = items[i:i + 64]
        vecs = cardlib.teacher_embed([b[key] for b in batch])
        for item, v in zip(batch, vecs):
            if kept_vecs and float(np.max(np.stack(kept_vecs) @ v)) > DUP_SIM:
                continue
            kept.append(item)
            kept_vecs.append(v)
    return kept


def image_lane() -> list[dict]:
    import pyarrow.parquet as pq
    import zipfile

    rows = []
    pf = pq.ParquetFile(os.path.dirname(SRC["colpali"]) +
                        "/test-00000-of-00001.parquet")
    for batch in pf.iter_batches(batch_size=256):
        for r in batch.to_pylist():
            q = r["query"].strip().split("\n")[0].strip()
            if 20 <= len(q) <= 300:
                rows.append({"text": q, "img_bytes": r["image"]["bytes"],
                             "source": "colpali-test",
                             "native_id": r["image_filename"]})
        if len(rows) >= 400:
            break
    rows = dedup_by_teacher(rows)[:250]

    coco = []
    pf = pq.ParquetFile(os.path.join(
        SRC["mmeb"], "MSCOCO_i2t", "train-00000-of-00001.parquet"))
    z = zipfile.ZipFile(os.path.join(SRC["mmeb"], "images_zip",
                                     "MSCOCO_i2t.zip"))
    for batch in pf.iter_batches(batch_size=512):
        for r in batch.to_pylist():
            if "val2014" not in r["qry_image_path"]:
                continue
            cap = r["pos_text"].strip()
            if 25 <= len(cap) <= 250:
                member = r["qry_image_path"].removeprefix("images/")
                coco.append({"text": cap, "zip_member": member,
                             "source": "mscoco-val2014", "native_id": member})
        if len(coco) >= 400:
            break
    coco = dedup_by_teacher(coco)[:250]

    out = []
    for r in rows:
        sha = cardlib.cas_put(r["img_bytes"])
        out.append({"text": r["text"], "image": cardlib.cas_ref(sha),
                    "source": r["source"], "native_id": r["native_id"]})
    for r in coco:
        sha = cardlib.cas_put(z.read(r["zip_member"]))
        out.append({"text": r["text"], "image": cardlib.cas_ref(sha),
                    "source": r["source"], "native_id": r["native_id"]})
    return out


def audio_lane() -> list[dict]:
    out = []
    ls_path = SRC["librispeech"].replace("dev-clean", "test-clean")
    per_speaker: dict[str, int] = {}
    with tarfile.open(ls_path, "r:gz") as tf:
        trans: dict[str, dict[str, str]] = {}
        pend: dict[str, list[tuple[str, bytes]]] = {}
        for m in tf:
            if len(out) >= 250:
                break
            parts = m.name.split("/")
            if len(parts) < 4:
                continue
            spk, key = parts[-3], f"{parts[-3]}/{parts[-2]}"
            if per_speaker.get(spk, 0) >= 8:
                continue
            if m.name.endswith(".trans.txt"):
                txt = tf.extractfile(m).read().decode()
                trans[key] = dict(l.split(" ", 1)
                                  for l in txt.splitlines() if " " in l)
                for utt, data in pend.pop(key, []):
                    if utt in trans[key] and per_speaker.get(spk, 0) < 8:
                        data_np, sr = sf.read(io.BytesIO(data))
                        if len(data_np) / sr > cardlib.MAX_AUDIO_S:
                            continue
                        sha = cardlib.cas_put(
                            cardlib.to_wav16k(np.asarray(data_np), sr))
                        out.append({"text": trans[key][utt].strip().capitalize(),
                                    "audio": cardlib.cas_ref(sha),
                                    "source": "librispeech-test-clean",
                                    "native_id": utt})
                        per_speaker[spk] = per_speaker.get(spk, 0) + 1
            elif m.name.endswith(".flac"):
                utt = parts[-1].removesuffix(".flac")
                data = tf.extractfile(m).read()
                if key in trans:
                    if utt in trans[key]:
                        data_np, sr = sf.read(io.BytesIO(data))
                        if len(data_np) / sr <= cardlib.MAX_AUDIO_S:
                            sha = cardlib.cas_put(
                                cardlib.to_wav16k(np.asarray(data_np), sr))
                            out.append({"text": trans[key][utt].strip().capitalize(),
                                        "audio": cardlib.cas_ref(sha),
                                        "source": "librispeech-test-clean",
                                        "native_id": utt})
                            per_speaker[spk] = per_speaker.get(spk, 0) + 1
                else:
                    pend.setdefault(key, []).append((utt, data))

    gt = subprocess.run(
        ["7z", "e", "-so",
         os.path.join(SRC["fsd50k"], "FSD50K.ground_truth.zip"),
         "FSD50K.ground_truth/eval.csv"],
        capture_output=True, check=True).stdout.decode()
    seen_lead = set()
    n_fsd = 0
    for r in csv.DictReader(io.StringIO(gt)):
        if n_fsd >= 100:
            break
        lead = r["labels"].split(",")[0]
        if lead in seen_lead:
            continue
        member = f"FSD50K.eval_audio/{r['fname']}.wav"
        try:
            with tempfile.TemporaryDirectory() as td:
                subprocess.run(["7z", "e", "-y", f"-o{td}",
                                os.path.join(SRC["fsd50k"],
                                             "FSD50K.eval_audio.zip"), member],
                               capture_output=True, check=True)
                data, sr = sf.read(os.path.join(td, f"{r['fname']}.wav"))
        except Exception:  # noqa: BLE001
            continue
        if len(data) / sr > cardlib.MAX_AUDIO_S:
            continue
        seen_lead.add(lead)
        sha = cardlib.cas_put(cardlib.to_wav16k(np.asarray(data), sr))
        labels = r["labels"].replace("_", " ").split(",")
        out.append({"text": f"Environmental sound recording: "
                            f"{', '.join(labels)}.",
                    "audio": cardlib.cas_ref(sha),
                    "source": "fsd50k-eval", "native_id": r["fname"]})
        n_fsd += 1
    return out


def freeze(name: str, items: list[dict]) -> None:
    outdir = os.path.join(cardlib.ROOT, "eval")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.jsonl")
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path.replace(".jsonl", ".freeze"), "w") as f:
        f.write(f"{sha}  {name}.jsonl  n={len(items)}\n")
    print(f"{name}: {len(items)} pairs, frozen sha256={sha[:16]}…")


def main() -> None:
    freeze("image-eval-v1", image_lane())
    freeze("audio-eval-v1", audio_lane())


if __name__ == "__main__":
    main()
