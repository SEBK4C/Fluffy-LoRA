#!/usr/bin/env python3
"""mine_ta_fix_wav_headers.py — repair CAS wavs written via `ffmpeg ... -f
wav pipe:1`: on a pipe ffmpeg cannot seek back, so RIFF/data sizes are
0xFFFFFFFF placeholders (a LIST metadata chunk also rides along).
cardlib.wav_info / shards_v2.wav_to_float32 read the header sizes ->
everything downstream breaks (observed: "134217.7s exceeds 30.0s cap" =
INT32_MAX frames).

Repair per referenced file: parse chunks leniently, extract the true PCM
bytes, rebuild a canonical 44-byte-header 16k mono s16 WAV, write to CAS
under its NEW sha256, remove the old file, rewrite pairs.jsonl with the
new sha + recomputed duration. Deterministic + idempotent (already-sane
files are left alone; reruns converge).

Usage: mine_ta_fix_wav_headers.py <pairs.jsonl> [more.jsonl ...]
"""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import struct
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mine_ta_lib as lib  # noqa: E402

ROOT = "/pool-ssd/fluffy/mine-ta"


def log(msg: str) -> None:
    lib.log("wav-fix", msg)


def parse_pcm(d: bytes):
    """Return (rate, channels, sampwidth, pcm_bytes) from a possibly
    size-corrupt RIFF file."""
    assert d[:4] == b"RIFF" and d[8:12] == b"WAVE", "not RIFF/WAVE"
    off = 12
    fmt = None
    while off + 8 <= len(d):
        cid = d[off:off + 4]
        sz = struct.unpack("<I", d[off + 4:off + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", d[off + 8:off + 24])
            off += 8 + sz + (sz & 1)
        elif cid == b"data":
            pcm = d[off + 8:]          # size field untrustworthy: take rest
            real = min(sz, len(pcm)) if sz != 0xFFFFFFFF else len(pcm)
            return fmt[2], fmt[1], fmt[5] // 8, pcm[:real]
        else:
            if sz == 0xFFFFFFFF:
                raise ValueError(f"corrupt non-data chunk size {cid}")
            off += 8 + sz + (sz & 1)
    raise ValueError("no data chunk")


def fix_one(sha: str):
    """Returns (old_sha, new_sha, duration_s) or None if already sane."""
    path = lib.cas_path(ROOT, sha)
    d = open(path, "rb").read()
    riff_sz = struct.unpack("<I", d[4:8])[0]
    if riff_sz == len(d) - 8 and d[12:16] == b"fmt ":
        try:  # canonical + consistent -> verify data size and skip
            with wave.open(io.BytesIO(d)) as w:
                return None
        except Exception:  # noqa: BLE001
            pass
    rate, ch, sw, pcm = parse_pcm(d)
    assert (rate, ch, sw) == (16000, 1, 2), f"{sha}: {rate}/{ch}/{sw}"
    if len(pcm) % 2:
        pcm = pcm[:-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    data = buf.getvalue()
    new_sha = lib.cas_write(ROOT, data, "wav")
    if new_sha != sha:
        os.unlink(path)
    return sha, new_sha, round(len(pcm) / 2 / 16000.0, 3)


def main() -> None:
    for pairs_path in sys.argv[1:]:
        rows = []
        with open(pairs_path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        shas = sorted({r["sha256"] for r in rows})
        log(f"{pairs_path}: {len(rows)} rows, {len(shas)} unique wavs")
        remap, durs = {}, {}
        with mp.Pool(12) as pool:
            for i, res in enumerate(pool.imap_unordered(fix_one, shas,
                                                        chunksize=64)):
                if res:
                    old, new, dur = res
                    remap[old] = new
                    durs[new] = dur
                if (i + 1) % 20000 == 0:
                    log(f"  {i+1}/{len(shas)} repaired")
        tmp = pairs_path + ".fixed"
        n_re = 0
        with open(tmp, "w") as f:
            for r in rows:
                if r["sha256"] in remap:
                    r["sha256"] = remap[r["sha256"]]
                    r["duration_s"] = durs[r["sha256"]]
                    n_re += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, pairs_path)
        log(f"{pairs_path}: DONE — {n_re}/{len(rows)} rows re-sha'd "
            f"({len(remap)} files rebuilt)")


if __name__ == "__main__":
    main()
