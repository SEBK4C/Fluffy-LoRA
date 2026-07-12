#!/usr/bin/env python3
"""mine_ta_audio_extract.py — real-audio sources -> 16 kHz mono s16 WAV CAS
+ pairs.jsonl (speech<->transcript / sound<->labels ground truth).

Standalone compute-path script (MINING-OPS §5): claims its source chunk in
the dir-claim queue, restartable at every stage (on-disk artifact checks,
never chat-session memory). Run under nohup; logs to stdout + journald.

Sources (all read from the READ-ONLY /pool-6b/corpus-acq pool):
  librispeech --parts train-clean-100,train-clean-360[,train-other-500]
      untar -> walk flac (already 16k mono) -> wav CAS -> transcript pairs.
      dev-*/test-* are NEVER extracted (frozen audio-eval-v1 contamination).
  mls --lang german|dutch|french|spanish|italian|portuguese|polish --take N
      ONE streaming pass over mls_<lang>_opus.tar.gz with a deterministic
      per-utterance hash predicate (seeded) -> sampled opus -> wav CAS.
      train/ split only.
  fsd50k
      7z-extract dev_audio (multipart) + ground_truth -> wav CAS + labels.
      eval split NEVER extracted (frozen audio-eval-v1 used eval clips).

Output per source: /pool-ssd/fluffy/mine-ta/audio/<source>/pairs.jsonl
  {"native_id", "sha256", "duration_s", "text", "speaker", "lang",
   "subset", "labels"?}
Audio caps enforced HERE: 1.0s <= duration <= 30.0s, 16 kHz mono s16 WAV.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mine_ta_lib as lib  # noqa: E402

ROOT = "/pool-ssd/fluffy/mine-ta"
AUDIO_ROOT = os.path.join(ROOT, "audio")
RAW = os.path.join(ROOT, "raw")
SRC_LS = "/pool-6b/corpus-acq/work/librispeech"
SRC_MLS = "/pool-6b/corpus-acq/work/mls_non_en"
SRC_FSD = "/pool-6b/corpus-acq/work/fsd50k"
SEED = 20260712
MIN_S, MAX_S = 1.0, 30.0


def log(msg: str) -> None:
    lib.log("audio-extract", msg)


def done_ids(pairs_path: str) -> set:
    out = set()
    if os.path.exists(pairs_path):
        with open(pairs_path) as f:
            for line in f:
                try:
                    out.add(json.loads(line)["native_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # torn tail line from a kill -9: re-done
    return out


def wav_cas_from_file(src_path: str) -> tuple | None:
    """Decode any audio file -> 16k mono s16 wav bytes -> CAS.
    Returns (sha256, duration_s) or None if outside the duration caps.

    NOTE: ffmpeg must write to a real FILE — on a pipe it cannot seek back
    to patch the RIFF/data sizes and leaves 0xFFFFFFFF placeholders that
    break every header-trusting reader downstream (found by the fsd50k
    250-sample gate; repaired fleet-wide by mine_ta_fix_wav_headers.py)."""
    tmp = os.path.join(RAW, f".conv-{os.getpid()}.wav")
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src_path,
         "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
         "-bitexact", "-f", "wav", tmp],
        capture_output=True)
    if p.returncode != 0 or not os.path.exists(tmp):
        return None
    with open(tmp, "rb") as f:
        data = f.read()
    os.unlink(tmp)
    if len(data) < 1000:
        return None
    n_samples = (len(data) - 44) / 2.0
    dur = n_samples / 16000.0
    if not (MIN_S <= dur <= MAX_S):
        return None
    sha = lib.cas_write(ROOT, data, "wav")
    return sha, round(dur, 2)


def _work_one(job: tuple) -> dict | None:
    """(native_id, src_path, text, speaker, lang, subset, labels)"""
    native_id, src_path, text, speaker, lang, subset, labels = job
    try:
        r = wav_cas_from_file(src_path)
    except Exception:  # noqa: BLE001
        return None
    if r is None:
        return None
    sha, dur = r
    row = {"native_id": native_id, "sha256": sha, "duration_s": dur,
           "text": text, "speaker": speaker, "lang": lang, "subset": subset}
    if labels is not None:
        row["labels"] = labels
    return row


def run_pool(jobs: list, pairs_path: str, workers: int, tag: str) -> int:
    wrote = 0
    t0 = time.time()
    with open(pairs_path, "a") as out, mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_work_one, jobs,
                                                    chunksize=16)):
            if row is not None:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1
            if (i + 1) % 2000 == 0:
                out.flush()
                rate = (i + 1) / (time.time() - t0)
                log(f"{tag}: {i+1}/{len(jobs)} ({rate:.1f}/s, kept {wrote})")
                lib.touch_claim(tag)
    return wrote


# ------------------------------------------------------------ librispeech --

def cmd_librispeech(parts: list[str], workers: int) -> None:
    src_name = "audio-librispeech"
    os.makedirs(os.path.join(AUDIO_ROOT, "librispeech"), exist_ok=True)
    pairs_path = os.path.join(AUDIO_ROOT, "librispeech", "pairs.jsonl")
    for part in parts:
        if part.startswith(("dev-", "test-")):
            raise SystemExit(f"CONTAMINATION GUARD: refusing eval split {part}")
    for part in parts:
        chunk_id = f"audio-librispeech-{part}"
        if lib.claimed_elsewhere(chunk_id):
            log(f"{chunk_id} claimed elsewhere — skipping")
            continue
        if not lib.claim(chunk_id):
            continue
        dest = os.path.join(RAW, "librispeech")
        marker = os.path.join(dest, f".untar-done-{part}")
        if not os.path.exists(marker):
            log(f"untar {part} ...")
            os.makedirs(dest, exist_ok=True)
            subprocess.run(["tar", "-xzf",
                            os.path.join(SRC_LS, f"{part}.tar.gz"),
                            "-C", dest], check=True)
            open(marker, "w").close()
            log(f"untar {part} done")
        base = os.path.join(dest, "LibriSpeech", part)
        done = done_ids(pairs_path)
        jobs = []
        for spk in sorted(os.listdir(base)):
            for chap in sorted(os.listdir(os.path.join(base, spk))):
                d = os.path.join(base, spk, chap)
                trans = {}
                tpath = os.path.join(d, f"{spk}-{chap}.trans.txt")
                if not os.path.exists(tpath):
                    continue
                for line in open(tpath):
                    uid, _, txt = line.strip().partition(" ")
                    trans[uid] = txt
                for f in sorted(os.listdir(d)):
                    if not f.endswith(".flac"):
                        continue
                    uid = f[:-5]
                    nid = f"librispeech/{part}/{uid}"
                    if nid in done or uid not in trans:
                        continue
                    jobs.append((nid, os.path.join(d, f), trans[uid],
                                 f"ls-{spk}", "en", part, None))
        log(f"{part}: {len(jobs)} utterances to convert "
            f"({len(done)} already done)")
        wrote = run_pool(jobs, pairs_path, workers, chunk_id)
        log(f"{part}: DONE wrote {wrote}")
        lib.update_state(src_name, **{f"extracted_{part}": True})
    total = sum(1 for _ in open(pairs_path))
    lib.update_state(src_name, extracted=True, pairs=total,
                     pairs_path=pairs_path)
    log(f"librispeech extraction COMPLETE: {total} pairs")


# -------------------------------------------------------------------- mls --

def cmd_mls(lang: str, take: int, workers: int) -> None:
    src_name = f"audio-mls-{lang}"
    chunk_id = src_name
    if lib.claimed_elsewhere(chunk_id):
        log(f"{chunk_id} claimed elsewhere — skipping")
        return
    lib.claim(chunk_id)
    outdir = os.path.join(AUDIO_ROOT, f"mls-{lang}")
    os.makedirs(outdir, exist_ok=True)
    pairs_path = os.path.join(outdir, "pairs.jsonl")
    tmp_opus = os.path.join(RAW, f"mls-{lang}-opus")
    os.makedirs(tmp_opus, exist_ok=True)
    est = {"german": 476000, "dutch": 374000, "french": 258000,
           "spanish": 220000, "italian": 60000, "portuguese": 37000,
           "polish": 25000}[lang]
    rate = min(1.0, take * 1.15 / est)  # slight overdraw; caps trim later
    thresh = int(rate * 10_000)

    def sampled(utt: str) -> bool:
        h = hashlib.sha1(f"{SEED}:{utt}".encode()).hexdigest()
        return int(h, 16) % 10_000 < thresh

    stream_marker = os.path.join(tmp_opus, ".stream-done")
    transcripts: dict[str, str] = {}
    tpath = os.path.join(tmp_opus, "transcripts.txt")
    if not os.path.exists(stream_marker):
        log(f"mls-{lang}: streaming pass over tar (sample rate "
            f"{rate:.3f}, est {est}) ...")
        n_seen = n_kept = 0
        with tarfile.open(os.path.join(SRC_MLS, f"mls_{lang}_opus.tar.gz"),
                          mode="r|gz") as tf:
            for m in tf:
                name = m.name
                if name.endswith("transcripts.txt") and "/train/" in name:
                    with open(tpath, "wb") as f:
                        f.write(tf.extractfile(m).read())
                    log(f"mls-{lang}: got train transcripts")
                elif name.endswith(".opus") and "/train/audio/" in name:
                    n_seen += 1
                    utt = os.path.basename(name)[:-5]
                    if sampled(utt):
                        opath = os.path.join(tmp_opus, utt + ".opus")
                        if not os.path.exists(opath):
                            with open(opath, "wb") as f:
                                f.write(tf.extractfile(m).read())
                        n_kept += 1
                    if n_seen % 50000 == 0:
                        log(f"mls-{lang}: streamed {n_seen}, kept {n_kept}")
                        lib.touch_claim(chunk_id)
        open(stream_marker, "w").close()
        log(f"mls-{lang}: stream done — {n_seen} seen, {n_kept} sampled")
    if not os.path.exists(tpath):
        raise SystemExit(f"mls-{lang}: transcripts.txt missing after stream")
    for line in open(tpath):
        uid, _, txt = line.rstrip("\n").partition("\t")
        transcripts[uid] = txt
    done = done_ids(pairs_path)
    jobs = []
    for f in sorted(os.listdir(tmp_opus)):
        if not f.endswith(".opus"):
            continue
        utt = f[:-5]
        nid = f"mls-{lang}/train/{utt}"
        if nid in done or utt not in transcripts:
            continue
        spk = utt.split("_")[0]
        jobs.append((nid, os.path.join(tmp_opus, f), transcripts[utt],
                     f"mls{lang[:2]}-{spk}", lang, "train", None))
    log(f"mls-{lang}: {len(jobs)} to convert ({len(done)} done)")
    wrote = run_pool(jobs, pairs_path, workers, chunk_id)
    total = sum(1 for _ in open(pairs_path))
    lib.update_state(src_name, extracted=True, pairs=total,
                     pairs_path=pairs_path)
    log(f"mls-{lang}: DONE wrote {wrote}, total {total}")


# ----------------------------------------------------------------- fsd50k --

def cmd_fsd50k(workers: int) -> None:
    src_name = "audio-fsd50k"
    chunk_id = src_name
    if lib.claimed_elsewhere(chunk_id):
        log(f"{chunk_id} claimed elsewhere — skipping")
        return
    lib.claim(chunk_id)
    outdir = os.path.join(AUDIO_ROOT, "fsd50k")
    os.makedirs(outdir, exist_ok=True)
    pairs_path = os.path.join(outdir, "pairs.jsonl")
    dest = os.path.join(RAW, "fsd50k")
    os.makedirs(dest, exist_ok=True)
    # NOTE: dev split ONLY — audio-eval-v1 froze 102 FSD50K *eval* clips.
    marker = os.path.join(dest, ".extract-done")
    if not os.path.exists(marker):
        log("fsd50k: 7z-extracting dev_audio (multipart) + metadata ...")
        for z in ("FSD50K.dev_audio.zip", "FSD50K.ground_truth.zip",
                  "FSD50K.metadata.zip"):
            subprocess.run(["7z", "x", "-y", os.path.join(SRC_FSD, z),
                            f"-o{dest}"], check=True,
                           stdout=subprocess.DEVNULL)
        open(marker, "w").close()
        log("fsd50k: extraction done")
    import csv
    gt = os.path.join(dest, "FSD50K.ground_truth", "dev.csv")
    # per-clip licenses (rights rule from the CORPUS-ACQ registry: CC0/CC-BY
    # allowlist; CC-BY-NC / Sampling+ kept research_only)
    lic_path = os.path.join(dest, "FSD50K.metadata",
                            "collection", "collection_dev.csv")
    licenses: dict[str, str] = {}
    dev_info = os.path.join(dest, "FSD50K.metadata", "dev_clips_info_FSD50K.json")
    if os.path.exists(dev_info):
        info = json.load(open(dev_info))
        licenses = {k: v.get("license", "") for k, v in info.items()}
    done = done_ids(pairs_path)
    jobs = []
    with open(gt) as f:
        for row in csv.DictReader(f):
            fname = row["fname"]
            nid = f"fsd50k/dev/{fname}"
            if nid in done:
                continue
            labels = [l.replace("_", " ") for l in row["labels"].split(",")]
            wav = os.path.join(dest, "FSD50K.dev_audio", fname + ".wav")
            if not os.path.exists(wav):
                continue
            jobs.append((nid, wav, ", ".join(labels), None, "env",
                         row.get("split", "train"), labels))
    log(f"fsd50k: {len(jobs)} dev clips to convert ({len(done)} done)")
    wrote = run_pool(jobs, pairs_path, workers, chunk_id)
    # attach licenses in a sidecar (native_id -> license URL)
    with open(os.path.join(outdir, "licenses.json"), "w") as f:
        json.dump(licenses, f)
    total = sum(1 for _ in open(pairs_path))
    lib.update_state(src_name, extracted=True, pairs=total,
                     pairs_path=pairs_path)
    log(f"fsd50k: DONE wrote {wrote}, total {total}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("librispeech")
    p.add_argument("--parts", default="train-clean-100,train-clean-360")
    p.add_argument("--workers", type=int, default=10)
    p = sub.add_parser("mls")
    p.add_argument("--lang", required=True)
    p.add_argument("--take", type=int, default=12000)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("fsd50k")
    p.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "librispeech":
        cmd_librispeech(a.parts.split(","), a.workers)
    elif a.cmd == "mls":
        cmd_mls(a.lang, a.take, a.workers)
    elif a.cmd == "fsd50k":
        cmd_fsd50k(a.workers)


if __name__ == "__main__":
    main()
