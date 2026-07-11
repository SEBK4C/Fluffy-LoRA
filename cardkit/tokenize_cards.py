#!/usr/bin/env python3
"""tokenize_cards.py — the reference collate: card views -> gemma-4 tokens.

Proves every card in a manifest feeds processor.apply_chat_template after
cas:// resolution, and reports per-view token cost. This is the exact code
path the trainer's collate uses; if this runs green, the format works.

Usage: GEMMA4_SNAPSHOT=<snapshot dir> tokenize_cards.py cards.jsonl
"""
from __future__ import annotations

import json
import os
import sys

import cardlib


def resolve(content: list[dict]) -> list[dict]:
    out = []
    for item in content:
        item = dict(item)
        ref = item.get(item["type"])
        if isinstance(ref, str) and ref.startswith("cas://"):
            item[item["type"]] = cardlib.resolve_cas(ref)
        out.append(item)
    return out


def main() -> None:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(os.environ["GEMMA4_SNAPSHOT"])

    def toklen(content) -> int:
        out = processor.apply_chat_template(
            [{"role": "user", "content": resolve(content)}],
            tokenize=True, return_dict=True, add_generation_prompt=False)
        ids = out["input_ids"]
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        while ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

    total = 0
    for path in sys.argv[1:]:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            costs = {vn: toklen(v["content"])
                     for vn, v in card["views"].items()}
            costs |= {f"interleaved[{i}]": toklen(il["content"])
                      for i, il in enumerate(card.get("interleaved", []))}
            total += len(costs)
            print(f"{card['card_id']}: " + "  ".join(
                f"{k}={v}" for k, v in costs.items()))
    print(f"OK — {total} views tokenized")


if __name__ == "__main__":
    main()
