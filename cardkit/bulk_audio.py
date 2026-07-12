#!/usr/bin/env python3
"""bulk_audio.py — bulk gated audio views for v001 en cards (audio lane).

Sebastian-authorized 9h sprint 2026-07-12. Supertonic-3 primary generator,
frozen A3 gate per clip (WER <= 0.15 AND teacher sim >= 0.90). Gate-passed
clips land in CAS; every attempt (pass or fail) is a manifest line, so the
run is resumable and the reject rate is measurable.

Output: $FLUFFY_CARDS_ROOT/bulk/audio-v001.jsonl
Workers: multiprocessing, each with its own Supertonic + whisper instance.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
WER_MAX, SIM_MIN = 0.15, 0.90
SRC = "/pool-ssd/synth-forge/corpus/manifests/accepted-v001.jsonl"

_ctx: dict = {}


def init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    from faster_whisper import WhisperModel
    from supertonic import TTS

    _ctx["tts"] = TTS(intra_op_num_threads=3)
    _ctx["whisper"] = WhisperModel("small", device="cpu",
                                   compute_type="int8", cpu_threads=3)
    _ctx["styles"] = {v: _ctx["tts"].get_voice_style(v) for v in VOICES}


def process(task: tuple[str, str, str]) -> dict:
    import numpy as np

    import cardlib

    card_id, text, voice = task
    t0 = time.time()
    try:
        wav, _ = _ctx["tts"].synthesize(text, voice_style=_ctx["styles"][voice])
        wav16 = cardlib.to_wav16k(np.asarray(wav).squeeze(),
                                  _ctx["tts"].sample_rate)
        dur = (len(wav16) - 44) / 2 / cardlib.SR
        if dur > cardlib.MAX_AUDIO_S:
            # spec hard cap: >750 audio tokens breaks the tower budget
            return {"card_id": card_id, "voice": voice, "pass": False,
                    "correction": "overlength", "duration_s": round(dur, 1)}
        sha = cardlib.cas_put(wav16)
        segs, _ = _ctx["whisper"].transcribe(cardlib.cas_path(sha))
        hyp = " ".join(s.text for s in segs).strip()
        import jiwer
        norm = jiwer.Compose([jiwer.ToLowerCase(),
                              jiwer.SubstituteRegexes({r"[-–—]": " "}),
                              jiwer.RemovePunctuation(),
                              jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                              jiwer.ReduceToListOfListOfWords()])
        wer = jiwer.wer(text, hyp, reference_transform=norm,
                        hypothesis_transform=norm)
        sim = cardlib.cos(*cardlib.teacher_embed([text, hyp or " "]))
        return {"card_id": card_id, "cas": sha, "voice": voice,
                "wer": round(wer, 4), "sim": round(sim, 4),
                "pass": wer <= WER_MAX and sim >= SIM_MIN,
                "gen": {"model": "supertonic-3", "voice": voice},
                "secs": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001 — one bad clip must not kill hour 3
        return {"card_id": card_id, "error": f"{type(e).__name__}: {e}"[:200],
                "voice": voice, "pass": False}


def main() -> None:
    import cardlib

    outdir = os.path.join(cardlib.ROOT, "bulk")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "audio-v001.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            # error rows (teacher outage, transient failures) are NOT done —
            # they retry on the next run. Rows that reached CAS are done;
            # overlength verdicts are deterministic, retrying is a loop.
            done = {r["card_id"] for r in map(json.loads,
                    (l for l in f if l.strip()))
                    if "cas" in r or r.get("correction") == "overlength"}
        print(f"resume: {len(done)} already processed (errors will retry)")

    tasks = []
    with open(SRC) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("lang") != "en" or r["card_id"] in done:
                continue
            if not isinstance(r.get("canonical_text"), str):
                continue
            if not (30 <= len(r["canonical_text"]) <= 500):
                continue
            tasks.append((r["card_id"], r["canonical_text"],
                          VOICES[len(tasks) % len(VOICES)]))
    take = int(os.environ.get("TAKE", 0))
    if take:
        tasks = tasks[:take]  # front slice; a cloud job owns the rest
    workers = int(os.environ.get("WORKERS", 6))
    print(f"{len(tasks)} clips to generate, {workers} workers")

    t0, n_pass, n_all = time.time(), 0, 0
    with mp.Pool(workers, initializer=init_worker) as pool, \
            open(out_path, "a") as out:
        for res in pool.imap_unordered(process, tasks, chunksize=4):
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            n_all += 1
            n_pass += bool(res.get("pass"))
            if n_all % 100 == 0:
                rate = n_all / (time.time() - t0)
                print(f"{n_all}/{len(tasks)}  pass={n_pass/n_all:.0%}  "
                      f"{rate:.2f} clips/s  eta {(len(tasks)-n_all)/rate/3600:.1f}h",
                      flush=True)
    print(f"DONE {n_all} clips, {n_pass} passed ({n_pass/max(n_all,1):.0%})")


if __name__ == "__main__":
    main()
