# /// script
# requires-python = ">=3.11"
# dependencies = ["supertonic", "faster-whisper", "jiwer", "soxr", "numpy",
#                 "soundfile", "huggingface_hub"]
# ///
"""cloud_audio_job.py — Supertonic synth + GPU-whisper WER on HF Jobs.

Half of the A3 gate (WER) runs here; the teacher round-trip sim runs at
local ingest (the teacher is tailnet-only). Emits tars of 16 kHz mono WAV +
per-clip JSON {card_id, voice, wer, transcript}; local ingest_cloud_audio.py
adds teacher sim + CAS + the final gate verdict.

Env: OUT_REPO, TASKS (path in repo), START, COUNT, SHARD, RUN_TAG.
"""
import io
import json
import os
import tarfile
import time
import wave

import numpy as np
import soxr
from huggingface_hub import HfApi, hf_hub_download

OUT_REPO = os.environ["OUT_REPO"]
TASKS = os.environ.get("TASKS", "audio_tasks.jsonl")
START = int(os.environ.get("START", 0))
COUNT = int(os.environ.get("COUNT", 100))
SHARD = int(os.environ.get("SHARD", 1000))
RUN_TAG = os.environ.get("RUN_TAG", "aud")
VOICES = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
SR = 16000

api = HfApi()
tasks_path = hf_hub_download(OUT_REPO, TASKS, repo_type="dataset")
tasks = [json.loads(l) for l in open(tasks_path) if l.strip()]
work = tasks[START:START + COUNT]
print(f"{len(work)} clips [{START}:{START + COUNT}]")

from faster_whisper import WhisperModel
from supertonic import TTS

tts = TTS()
styles = {v: tts.get_voice_style(v) for v in VOICES}
whisper = WhisperModel("small", device="cuda", compute_type="float16")

import jiwer

norm = jiwer.Compose([jiwer.ToLowerCase(),
                      jiwer.SubstituteRegexes({r"[-–—]": " "}),
                      jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(),
                      jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])


def wav16k(data: np.ndarray, sr: int) -> bytes:
    data = np.asarray(data, dtype=np.float32).squeeze()
    if sr != SR:
        data = soxr.resample(data, sr, SR)
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


t0, done, shard_idx = time.time(), 0, START // SHARD
buf = io.BytesIO()
tar = tarfile.open(fileobj=buf, mode="w")


def flush():
    global tar, buf, shard_idx
    tar.close()
    if buf.tell() > 0 and done > 0:
        buf.seek(0)
        api.upload_file(path_or_fileobj=buf,
                        path_in_repo=f"audio/{RUN_TAG}-{shard_idx:05d}.tar",
                        repo_id=OUT_REPO, repo_type="dataset")
        print(f"uploaded audio/{RUN_TAG}-{shard_idx:05d}.tar")
    shard_idx += 1
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w")


for idx, t in enumerate(work):
    voice = t.get("voice") or VOICES[(START + idx) % len(VOICES)]
    try:
        wav, _ = tts.synthesize(t["text"], voice_style=styles[voice])
        wb = wav16k(wav, tts.sample_rate)
        segs, _ = whisper.transcribe(io.BytesIO(wb))
        hyp = " ".join(s.text for s in segs).strip()
        wer = jiwer.wer(t["text"], hyp, reference_transform=norm,
                        hypothesis_transform=norm)
        meta = {"card_id": t["card_id"], "voice": voice,
                "wer": round(wer, 4), "transcript": hyp,
                "gen": {"model": "supertonic-3", "voice": voice,
                        "version": "hf-jobs"}}
        for ext, data in (("wav", wb),
                          ("json", json.dumps(meta, ensure_ascii=False).encode())):
            ti = tarfile.TarInfo(f"{t['card_id']}.{ext}")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
        done += 1
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {t['card_id']}: {type(e).__name__}: {e}"[:150], flush=True)
    if done and done % SHARD == 0:
        flush()
    if done and done % 200 == 0:
        r = done / (time.time() - t0)
        print(f"{done}/{len(work)} {r:.2f} clips/s eta "
              f"{(len(work) - done) / r / 60:.0f}m", flush=True)

flush()
print("SUMMARY", json.dumps({"count": done,
                             "elapsed_s": round(time.time() - t0, 1)}))
