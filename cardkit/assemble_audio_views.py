#!/usr/bin/env python3
"""assemble_audio_views.py — bulk manifest -> CARD-SPEC audio view objects.

Takes the last row per card_id from bulk/audio-v001.jsonl, keeps passes,
and emits audio-views-v001.jsonl: {v001_card_id, view} where view is the
exact CARD-SPEC v1.1 views["audio"] object (content + source + origin +
native_id + gen + gate). The builder merges these into cards / exposures
for the day-2 audio-lane entry. Idempotent full rewrite; run any time.
"""
from __future__ import annotations

import json
import os

import cardlib


def main() -> None:
    src = os.path.join(cardlib.ROOT, "bulk", "audio-v001.jsonl")
    out = os.path.join(cardlib.ROOT, "bulk", "audio-views-v001.jsonl")
    last: dict[str, dict] = {}
    for line in open(src):
        line = line.strip()
        if line:
            r = json.loads(line)
            last[r["card_id"]] = r
    n = 0
    with open(out + ".tmp", "w") as f:
        for cid, r in last.items():
            if not r.get("pass") or "cas" not in r:
                continue
            view = {"content": [{"type": "audio",
                                 "audio": cardlib.cas_ref(r["cas"])}],
                    "source": "tts", "origin": "v001", "native_id": cid,
                    "gen": r["gen"],
                    "gate": {"asr_wer": r["wer"],
                             "asr_model": "faster-whisper-small-int8",
                             "roundtrip_sim": r["sim"], "pass": True}}
            f.write(json.dumps({"v001_card_id": cid, "view": view},
                               ensure_ascii=False) + "\n")
            n += 1
    os.rename(out + ".tmp", out)
    print(f"{n} audio views -> {out}")


if __name__ == "__main__":
    main()
