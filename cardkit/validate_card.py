#!/usr/bin/env python3
"""validate_card.py — CARD-SPEC v0.2 validator.

Checks every card in a JSONL manifest against:
  1. the JSON Schema (cardkit/card.schema.json)
  2. referential integrity: cas:// refs resolve to real CAS files;
     negatives point at cards that exist (self-refs must name a view)
  3. media rules: audio is 16 kHz mono WAV <= 30 s; images open cleanly
  4. gate discipline: any generated view (rendered/tts/genai/captioned/asr)
     must carry gate.pass == true — failed gates never ship
  5. dedup: hash matches the protocol's recomputation
  6. interleaved modality order: image -> text -> audio (v1.0 hard rule,
     Gemma 4 pretraining convention)
  7. image-captioned renditions carry gen.layout with caption_frac inside
     the normative 10-20% band (v1.1 §E0.1 layout rules)

Usage: validate_card.py cards.jsonl [more.jsonl ...] [--known ids.txt]
Exit 0 = green. Deps: jsonschema, PIL, cardlib (numpy).
"""
from __future__ import annotations

import json
import sys

import jsonschema

import cardlib

GENERATED = {"rendered", "tts", "genai", "captioned", "asr"}


def iter_cards(path: str):
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield n, json.loads(line)


def media_refs(card: dict):
    """(view_name, item) for every cas:// item in views + interleaved."""
    for vname, view in card.get("views", {}).items():
        for item in view["content"]:
            yield vname, item
    for il in card.get("interleaved", []):
        for item in il["content"]:
            yield f"interleaved:{il['recipe']}", item


def check_card(card: dict, schema, known_ids: set[str]) -> list[str]:
    errs = []
    cid = card.get("card_id", "?")
    try:
        jsonschema.validate(card, schema)
    except jsonschema.ValidationError as e:
        return [f"{cid}: schema: {e.message} (at {'/'.join(map(str, e.path))})"]

    for where, item in media_refs(card):
        ref = item.get(item["type"])
        if not isinstance(ref, str) or not ref.startswith("cas://"):
            if item["type"] == "text":
                continue
            errs.append(f"{cid}: {where}: non-CAS media ref {ref!r}")
            continue
        path = cardlib.resolve_cas(ref)
        try:
            if item["type"] == "audio":
                info = cardlib.wav_info(path)
                if info["sr"] != cardlib.SR or info["channels"] != 1:
                    errs.append(f"{cid}: {where}: audio must be 16kHz mono, "
                                f"got {info['sr']}Hz/{info['channels']}ch")
                if info["duration_s"] > cardlib.MAX_AUDIO_S + 0.05:
                    errs.append(f"{cid}: {where}: audio {info['duration_s']:.1f}s "
                                f"exceeds {cardlib.MAX_AUDIO_S}s cap")
            elif item["type"] == "image":
                from PIL import Image
                with Image.open(path) as im:
                    im.verify()
        except FileNotFoundError:
            errs.append(f"{cid}: {where}: unresolved {ref}")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{cid}: {where}: unreadable {ref}: {e}")

    for view_name, view in card.get("views", {}).items():
        if view["source"] in GENERATED and not view.get("gate", {}).get("pass"):
            errs.append(f"{cid}: views.{view_name}: generated view without "
                        f"passing gate")
        if view_name == "image-captioned":
            frac = view.get("gen", {}).get("layout", {}).get("caption_frac")
            if frac is None:
                errs.append(f"{cid}: views.image-captioned: missing "
                            f"gen.layout.caption_frac")
            elif not (cardlib.CAPTION_FRAC_MIN <= frac
                      <= cardlib.CAPTION_FRAC_MAX):
                errs.append(f"{cid}: views.image-captioned: caption_frac "
                            f"{frac} outside normative "
                            f"[{cardlib.CAPTION_FRAC_MIN}, "
                            f"{cardlib.CAPTION_FRAC_MAX}]")

    MOD_ORDER = {"image": 0, "text": 1, "audio": 2}
    for il in card.get("interleaved", []):
        ranks = [MOD_ORDER[item["type"]] for item in il["content"]]
        if ranks != sorted(ranks):
            errs.append(f"{cid}: interleaved:{il['recipe']}: modality order "
                        f"must be image -> text -> audio")

    for modality, negs in card.get("negatives", {}).items():
        for neg in negs:
            if neg["card_id"] == cid and "view" not in neg:
                errs.append(f"{cid}: negatives.{modality}: self-negative "
                            f"must name a view")
            elif neg["card_id"] != cid and neg["card_id"] not in known_ids:
                errs.append(f"{cid}: negatives.{modality}: unknown card "
                            f"{neg['card_id']}")

    if card["dedup"]["protocol"] == cardlib.DEDUP_PROTOCOL:
        want = cardlib.dedup_hash(card["anchor_text"])
        if card["dedup"]["hash"] != want:
            errs.append(f"{cid}: dedup hash mismatch")
    return errs


def main(argv: list[str]) -> int:
    files, known = [], set()
    args = iter(argv)
    for a in args:
        if a == "--known":
            known |= {l.strip() for l in open(next(args)) if l.strip()}
        else:
            files.append(a)
    if not files:
        print(__doc__)
        return 2

    import os
    schema = json.load(open(os.path.join(os.path.dirname(__file__),
                                         "card.schema.json")))
    cards = [(f, n, c) for f in files for n, c in iter_cards(f)]
    ids = [c["card_id"] for _, _, c in cards]
    known |= set(ids)
    dupes = {i for i in ids if ids.count(i) > 1}

    all_errs = [f"duplicate card_id: {d}" for d in sorted(dupes)]
    for f, n, card in cards:
        for e in check_card(card, schema, known):
            all_errs.append(f"{f}:{n}: {e}")

    if all_errs:
        print(f"FAIL — {len(all_errs)} error(s) across {len(cards)} card(s):")
        for e in all_errs:
            print(f"  {e}")
        return 1
    print(f"OK — {len(cards)} card(s) valid, "
          f"{sum(1 for _, _, c in cards for _ in media_refs(c))} media refs checked")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
