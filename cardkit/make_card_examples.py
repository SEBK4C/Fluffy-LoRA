#!/usr/bin/env python3
"""make_card_examples.py — self-contained HTML of complete CARD-SPEC cards.

Renders full cards (anchor, every view with playable media + gate values,
interleaved views, negatives with sims, rights, provenance) plus the raw
card JSON in a collapsible block. Media embedded as data URIs.

Usage: make_card_examples.py out.html card_id[,card_id...] manifest [manifest...]
"""
from __future__ import annotations

import base64
import html
import json
import os
import sys

import cardlib

MIME = {"image": None, "audio": "audio/wav"}


def data_uri(sha: str) -> str:
    path = cardlib.cas_path(sha)
    data = open(path, "rb").read()
    if data[:2] == b"\x89P":
        mime = "image/png"
    elif data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    else:
        mime = "audio/wav"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def content_html(content: list[dict]) -> str:
    out = []
    for item in content:
        if item["type"] == "text":
            out.append(f"<p class='txt'>{html.escape(item['text'])}</p>")
        elif item["type"] == "image":
            out.append(f"<img src='{data_uri(item['image'][6:])}'>")
        else:
            out.append(f"<audio controls src='{data_uri(item['audio'][6:])}'>"
                       f"</audio>")
    return "\n".join(out)


def card_html(card: dict) -> str:
    parts = [f"<div class='card'><h2>{card['card_id']} "
             f"<span class='meta'>rights: {card['rights']['tier']}"
             f"{' · ' + card['rights'].get('license', '') if card['rights'].get('license') else ''}"
             f"</span></h2>",
             f"<p><b>anchor_text</b> — {html.escape(card['anchor_text'])}</p>"]
    for vn, v in card["views"].items():
        gate = v.get("gate")
        badge = (" · gate: " + " ".join(f"{k}={vv}" for k, vv in gate.items()
                                        if k != "pass")
                 + (" ✓" if gate.get("pass") else " ✗")) if gate else ""
        gen = f" · gen: {v['gen']['model']}" + \
              (f"/{v['gen'].get('voice')}" if v.get("gen", {}).get("voice")
               else "") if v.get("gen") else ""
        parts.append(f"<div class='view'><b>views.{vn}</b> "
                     f"<span class='gate'>source={v['source']} "
                     f"origin={v['origin']}{html.escape(gen)}"
                     f"{html.escape(badge)}</span><br>"
                     f"{content_html(v['content'])}</div>")
    for il in card.get("interleaved", []):
        parts.append(f"<div class='view'><b>interleaved</b> "
                     f"<span class='gate'>recipe={il['recipe']} · hard rule: "
                     f"image→text→audio</span><br>"
                     f"{content_html(il['content'])}</div>")
    negs = card.get("negatives", {})
    if negs:
        lines = []
        for m, ns in negs.items():
            lines.append(f"{m}: " + ", ".join(
                f"{n['card_id']}"
                + (f".{n['view']}" if n.get('view') else "")
                + (f" (sim {n['sim']})" if n.get('sim') is not None else "")
                + f" [{n['miner']}]" for n in ns))
        parts.append(f"<p class='neg'><b>negatives</b> — "
                     f"{html.escape(' · '.join(lines))}</p>")
    parts.append(f"<details><summary>raw card JSON</summary><pre>"
                 f"{html.escape(json.dumps(card, indent=2, ensure_ascii=False))}"
                 f"</pre></details></div>")
    return "\n".join(parts)


def main() -> None:
    out_path, want = sys.argv[1], set(sys.argv[2].split(","))
    cards = []
    for manifest in sys.argv[3:]:
        for line in open(manifest):
            line = line.strip()
            if line:
                c = json.loads(line)
                if c["card_id"] in want:
                    cards.append(c)
    parts = ["""<!doctype html><meta charset='utf-8'>
<title>CARD-SPEC card examples</title><style>
body{font:15px/1.5 system-ui;max-width:960px;margin:2em auto;padding:0 1em}
.card{border:1px solid #bbb;border-radius:10px;padding:1.2em;margin:2em 0}
.card h2{margin:0 0 .5em;font-size:1.1em}
.meta{color:#666;font-size:.8em;font-weight:normal}
.view{margin:.9em 0;padding:.7em;background:#f6f6f6;border-radius:6px}
.gate{font-family:monospace;font-size:.8em;color:#444}
img{max-width:100%;max-height:420px;display:block;margin:.4em 0}
.txt{margin:.3em 0}.neg{font-size:.85em;color:#555}
pre{overflow-x:auto;font-size:.75em;background:#f0f0f0;padding:.8em;border-radius:6px}
details{margin-top:.8em}</style>
<h1>CARD-SPEC v1.1 — card examples</h1>
<p>One card = one semantic anchor with gated renditions in each modality,
stored in gemma-4's native chat-template format. Every generated view
carries generator provenance and a passing gate; negatives reference real
cards; interleaved views follow the image→text→audio hard rule.</p>"""]
    parts += [card_html(c) for c in cards]
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"{len(cards)} cards -> {out_path} "
          f"({os.path.getsize(out_path) // 1024} KiB)")


if __name__ == "__main__":
    main()
