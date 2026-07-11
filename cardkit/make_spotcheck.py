#!/usr/bin/env python3
"""make_spotcheck.py — stratified eyeball sample as one self-contained HTML.

Samples 3 cards per source stratum (first / middle / last) from a manifest,
embeds every referenced media file as a data URI, and shows gate values next
to each view. Output goes NEXT TO the manifest (never committed — media of
gated-rights sources).

Usage: make_spotcheck.py cards.jsonl [out.html]
"""
from __future__ import annotations

import base64
import html
import json
import os
import sys

import cardlib


def data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".wav": "audio/wav"}


def media_tag(item: dict) -> str:
    kind = item["type"]
    if kind == "text":
        return f"<p class='txt'>{html.escape(item['text'])}</p>"
    path = cardlib.resolve_cas(item[kind])
    if kind == "image":
        head = open(path, "rb").read(4)
        mime = "image/png" if head[:2] == b"\x89P" else "image/jpeg"
        return f"<img src='{data_uri(path, mime)}'>"
    return f"<audio controls src='{data_uri(path, 'audio/wav')}'></audio>"


def main() -> None:
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(src), "spotcheck.html")
    cards = [json.loads(l) for l in open(src) if l.strip()]

    strata: dict[str, list[dict]] = {}
    for c in cards:
        origin = next(iter(c["views"].values()))["origin"]
        strata.setdefault(origin, []).append(c)
    sample = []
    for origin, cs in strata.items():
        idx = sorted({0, len(cs) // 2, len(cs) - 1})
        sample += [cs[i] for i in idx]

    parts = ["""<!doctype html><meta charset='utf-8'>
<title>Fluffy-LoRA pilot spot-check</title><style>
body{font:15px/1.5 system-ui;max-width:900px;margin:2em auto;padding:0 1em}
.card{border:1px solid #ccc;border-radius:8px;padding:1em;margin:1.5em 0}
.card h2{margin:0 0 .3em;font-size:1.05em}
.meta{color:#666;font-size:.85em}
.view{margin:.8em 0;padding:.6em;background:#f6f6f6;border-radius:6px}
.gate{font-family:monospace;font-size:.8em;color:#444}
img{max-width:100%;max-height:420px}
.txt{margin:.2em 0}
.neg{font-size:.85em;color:#555}
</style><h1>Pilot spot-check — eyeball sign-off gates bulk mining</h1>"""]
    for c in sample:
        origin = next(iter(c["views"].values()))["origin"]
        parts.append(f"<div class='card'><h2>{c['card_id']} "
                     f"<span class='meta'>[{origin} · rights: "
                     f"{c['rights']['tier']}]</span></h2>")
        parts.append(f"<p><b>anchor:</b> {html.escape(c['anchor_text'])}</p>")
        for vn, v in c["views"].items():
            gate = v.get("gate")
            gate_s = (" · gate: " + " ".join(
                f"{k}={vv}" for k, vv in gate.items()) if gate else "")
            parts.append(f"<div class='view'><b>{vn}</b> "
                         f"<span class='gate'>source={v['source']}"
                         f"{html.escape(gate_s)}</span><br>")
            parts += [media_tag(i) for i in v["content"]]
            parts.append("</div>")
        for il in c.get("interleaved", []):
            parts.append(f"<div class='view'><b>interleaved</b> "
                         f"<span class='gate'>recipe={il['recipe']}"
                         f" (order: image→text→audio)</span><br>")
            parts += [media_tag(i) for i in il["content"]]
            parts.append("</div>")
        negs = c.get("negatives", {})
        if negs:
            lines = [f"{m}: " + ", ".join(
                f"{n['card_id']}{'(' + str(n.get('sim')) + ')' if n.get('sim') is not None else ''}"
                for n in ns) for m, ns in negs.items()]
            parts.append(f"<p class='neg'><b>negatives</b> — "
                         f"{html.escape(' · '.join(lines))}</p>")
        parts.append("</div>")
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print(f"{len(sample)} cards -> {out} "
          f"({os.path.getsize(out) // 1024} KiB)")


if __name__ == "__main__":
    main()
