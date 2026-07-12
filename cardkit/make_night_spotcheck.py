#!/usr/bin/env python3
"""make_night_spotcheck.py — eyeball sample of the 9h-sprint production.

Sections: gated audio (passes across voices + gate-FAIL examples + one
overlength reject), noisy-tier images (incl. one CLIP-truncated pair), and
MINER warmup-shard cards. Self-contained HTML, media as data URIs.

Usage: make_night_spotcheck.py <noisy_tar> <miner_tar> <out.html>
"""
from __future__ import annotations

import base64
import html
import json
import random
import sys
import tarfile
import wave

import cardlib


def b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def cas_bytes(sha: str) -> bytes:
    return open(cardlib.cas_path(sha), "rb").read()


def audio_section() -> list[str]:
    last: dict[str, dict] = {}
    for line in open(f"{cardlib.ROOT}/bulk/audio-v001.jsonl"):
        line = line.strip()
        if line:
            r = json.loads(line)
            last[r["card_id"]] = r
    texts = {}
    for line in open("/pool-ssd/synth-forge/corpus/manifests/accepted-v001.jsonl"):
        line = line.strip()
        if line:
            c = json.loads(line)
            texts[c["card_id"]] = c.get("canonical_text", "")
    rows = list(last.values())
    random.seed(20260712)
    passes = [r for r in rows if r.get("pass")]
    by_voice: dict[str, list] = {}
    for r in passes:
        by_voice.setdefault(r["voice"], []).append(r)
    sample = [random.choice(v) for _, v in sorted(by_voice.items())][:6]
    fails_wer = [r for r in rows if "cas" in r and not r.get("pass")
                 and r.get("wer", 0) > 0.15][:1]
    fails_sim = [r for r in rows if "cas" in r and not r.get("pass")
                 and r.get("wer", 1) <= 0.15 and r.get("sim", 1) < 0.90][:1]
    over = [r for r in rows if r.get("correction") == "overlength"
            and "cas" in last.get(r["card_id"], {})]
    # overlength correction rows lack cas; find an early stored one
    over_stored = []
    seen_cids = {r["card_id"] for r in rows if r.get("correction") == "overlength"}
    for line in open(f"{cardlib.ROOT}/bulk/audio-v001.jsonl"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("card_id") in seen_cids and "cas" in r:
            try:
                with wave.open(cardlib.cas_path(r["cas"])) as w:
                    if w.getnframes() / w.getframerate() > 30.05:
                        over_stored.append(r)
                        break
            except OSError:
                pass

    parts = ["<h2>Gated audio (Supertonic-3, frozen A3 gate + 30 s cap)</h2>"]
    for label, rs in [("PASS", sample), ("REJECT — WER", fails_wer),
                      ("REJECT — teacher sim", fails_sim),
                      ("REJECT — overlength (pre-fix, corrected)", over_stored)]:
        for r in rs:
            gate = (f"voice={r['voice']} wer={r.get('wer', '—')} "
                    f"sim={r.get('sim', '—')}")
            parts.append(
                f"<div class='view'><b>{label}</b> "
                f"<span class='gate'>{html.escape(gate)}</span>"
                f"<p class='txt'>{html.escape(texts.get(r['card_id'], '?')[:300])}</p>"
                f"<audio controls src='{b64(cas_bytes(r['cas']), 'audio/wav')}'>"
                f"</audio></div>")
    return parts


def tar_section(tar_path: str, title: str, n: int, trunc_pids=frozenset()) -> list[str]:
    groups: dict[str, dict[str, bytes]] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            key = m.name.split(".")[0]
            groups.setdefault(key, {})[m.name.split(".")[-1]] = \
                tf.extractfile(m).read()
    random.seed(20260712)
    keys = [k for k in groups if "json" in groups[k]]
    picks = random.sample(keys, min(n, len(keys)))
    if trunc_pids:
        tk = [k for k in keys if k in trunc_pids]
        if tk:
            picks = picks[:-1] + [tk[0]]
    parts = [f"<h2>{html.escape(title)}</h2>"]
    for k in picks:
        g = groups[k]
        meta = json.loads(g["json"])
        text = meta.get("text") or meta.get("caption") or \
            meta.get("anchor_text") or json.dumps(meta)[:200]
        flag = " · <b>CLIP-77 TRUNCATED (weak pair, FLUX re-run queued)</b>" \
            if k in trunc_pids else ""
        img = next((v for kk, v in g.items() if kk == "jpg"), None)
        parts.append(f"<div class='view'><span class='gate'>{html.escape(k)}"
                     f"{flag}</span><p class='txt'>{html.escape(str(text)[:300])}"
                     f"</p>")
        if img:
            parts.append(f"<img src='{b64(img, 'image/jpeg')}'>")
        parts.append("</div>")
    return parts


def main() -> None:
    noisy_tar, miner_tar, out = sys.argv[1], sys.argv[2], sys.argv[3]
    trunc = {json.loads(l)["pid"] for l in
             open("/pool-ssd/fluffy-cards/noisy/clip_trunc.jsonl") if l.strip()}
    parts = ["""<!doctype html><meta charset='utf-8'>
<title>Fluffy-LoRA night-sprint spot-check</title><style>
body{font:15px/1.5 system-ui;max-width:900px;margin:2em auto;padding:0 1em}
.view{border:1px solid #ccc;border-radius:8px;padding:1em;margin:1em 0}
.gate{font-family:monospace;font-size:.85em;color:#444}
img{max-width:100%;max-height:400px}.txt{margin:.4em 0}
h2{margin-top:1.6em}</style>
<h1>Night-sprint spot-check — 2026-07-12</h1>
<p>~134k assets · $9.60 cloud · pass/fail examples included deliberately.</p>"""]
    parts += audio_section()
    parts += tar_section(noisy_tar, "Noisy tier (SDXL-Lightning 768px, warmup-only)",
                         8, trunc)
    parts += tar_section(miner_tar, "MINER gated cards (validator 50k/50k PASS)", 4)
    with open(out, "w") as f:
        f.write("\n".join(parts))
    import os
    print(f"{out} ({os.path.getsize(out) // 1024} KiB)")


if __name__ == "__main__":
    main()
