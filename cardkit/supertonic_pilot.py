#!/usr/bin/env python3
"""supertonic_pilot.py — v1.1 item 1: Supertonic-3 pass-rate check.

Runs the PRIMARY TTS generator over the SAME stratified texts as the
Phase 4 pilot's TTS lane (the 140 v001/mmeb/colpali anchor texts) and
scores each clip against the UNCHANGED frozen A3 gate
(WER <= 0.15 AND teacher sim >= 0.90). Measurement only — the pilot
manifest is not mutated; clips land in CAS with supertonic provenance.

Output: $FLUFFY_CARDS_ROOT/pilot/supertonic_pilot.json
"""
from __future__ import annotations

import json
import os

import cardlib
from build_golden import WER_MAX, TTS_SIM_MIN

VOICES = ["F1", "F3", "M1", "M4"]  # round-robin, mirrors the Kokoro run


def main() -> None:
    src = os.path.join(cardlib.ROOT, "pilot", "cards.jsonl")
    texts = []
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        origin = next(iter(c["views"].values()))["origin"]
        if origin in ("v001", "mmeb", "colpali"):
            texts.append((c["card_id"], origin, c["anchor_text"]))

    results = []
    for i, (cid, origin, text) in enumerate(texts):
        voice = VOICES[i % len(VOICES)]
        wav = cardlib.tts_supertonic(text, voice)
        sha = cardlib.cas_put(wav)
        m = cardlib.asr_wer(cardlib.cas_path(sha), text)
        sim = cardlib.cos(*cardlib.teacher_embed(
            [text, m["transcript"] or " "]))
        results.append({"card_id": cid, "origin": origin, "voice": voice,
                        "wer": m["asr_wer"], "sim": round(sim, 4),
                        "cas": sha,
                        "pass": m["asr_wer"] <= WER_MAX and sim >= TTS_SIM_MIN})

    per: dict[str, list] = {}
    for r in results:
        per.setdefault(r["origin"], []).append(r)
    summary = {"generator": "supertonic-3", "gate": "A3 (frozen)",
               "attempts": len(results),
               "pass": sum(r["pass"] for r in results),
               "per_origin": {o: {"n": len(rs),
                                  "pass": sum(r["pass"] for r in rs)}
                              for o, rs in per.items()}}
    wers = sorted(r["wer"] for r in results)
    sims = sorted(r["sim"] for r in results)
    n = len(results)
    summary["wer_median"] = wers[n // 2]
    summary["wer_p90"] = wers[int(0.9 * n)]
    summary["sim_median"] = sims[n // 2]
    summary["sim_p10"] = sims[int(0.1 * n)]

    out = os.path.join(cardlib.ROOT, "pilot", "supertonic_pilot.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "clips": results}, f, indent=2,
                  ensure_ascii=False)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
