# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow"]
# ///
"""make_test_shard.py — rig-local smoke-test shard from cardkit output.

Builds ~290 exposures from the golden(15) + pilot(200) cards plus a v001
slice (text2text lane), packed with shards_v2. Purpose: exercise every
trainer lane (incl. audio + interleaved) in the A-series smokes. NOT
training data — negatives are padded to k=8 with random fills (miner
"random-fill-smoke") where the pilot miner only produced 4.

Usage (on PVE, where the card CAS lives):
  FLUFFY_CARDS_ROOT=/pool-ssd/fluffy-cards \
  python3 make_test_shard.py <v001-slice.jsonl> <out-dir>
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shards_v2 as sh

ROOT = os.environ.get("FLUFFY_CARDS_ROOT", "/pool-ssd/fluffy-cards")
K = 8
INSTR = {  # smoke placeholder strings; REAL strings fixed pre-launch (MERGE-RESEARCH §2D)
    "text2text": "Retrieve a text with the same meaning.",
    "text2image": "Retrieve the image matching the description.",
    "image2text": "Retrieve the description matching the image.",
    "text2audio": "Retrieve the audio matching the description.",
    "audio2text": "Retrieve the description matching the audio.",
    "interleaved2text": "Retrieve the description matching the document.",
}


def cas_bytes(ref: str) -> bytes:
    sha = ref[len("cas://"):]
    with open(os.path.join(ROOT, "cas", "sha256", sha[:2], sha), "rb") as f:
        return f.read()


def pack_content(content: list[dict], media: dict[str, bytes]) -> list[dict]:
    """cas:// refs -> member:// refs, collecting media bytes."""
    out = []
    for it in content:
        t = it["type"]
        if t == "text":
            out.append(it)
            continue
        data = cas_bytes(it[t])
        ext = "wav" if t == "audio" else "png"
        mname = sh.media_name(data, ext)
        media[mname] = data
        out.append({"type": t, t: f"member://{mname}"})
    return out


def main() -> None:
    v001_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(20260712)

    cards = []
    for p in (f"{ROOT}/golden/cards.jsonl", f"{ROOT}/pilot/cards.jsonl"):
        cards += [json.loads(ln) for ln in open(p)]
    by_id = {c["card_id"]: c for c in cards}
    print(f"cards: {len(cards)}")

    w = sh.ShardWriter(os.path.join(out_dir, "smoke-000000.tar"))
    n = 0

    def texts_of(card: dict) -> list[dict]:
        return card["views"]["text"]["content"]

    def negatives_for(card: dict, view: str, media: dict) -> list[dict]:
        """Card's mined negatives (text lane only in pilot) + random fill
        to k=8. Negative content = the negative card's <view> if present,
        else its text view."""
        negs, used = [], {card["card_id"]}
        for entry in (card.get("negatives", {}).get("text") or []):
            nc = by_id.get(entry["card_id"])
            if nc is None or nc["card_id"] in used:
                continue
            used.add(nc["card_id"])
            v = view if view in nc["views"] else "text"
            negs.append({"card": nc["card_id"], "view": v,
                         "miner": entry.get("miner", "?"),
                         "content": pack_content(nc["views"][v]["content"], media)})
            if len(negs) == K:
                return negs
        pool = [c for c in cards if c["card_id"] not in used]
        rng.shuffle(pool)
        for nc in pool:
            v = view if view in nc["views"] else "text"
            negs.append({"card": nc["card_id"], "view": v,
                         "miner": "random-fill-smoke",
                         "content": pack_content(nc["views"][v]["content"], media)})
            if len(negs) == K:
                break
        return negs

    def emit(lane: str, card: dict, a_view: str, p_view: str,
             a_content: list[dict], p_content: list[dict]) -> None:
        nonlocal n
        media: dict[str, bytes] = {}
        exp = {"lane": lane, "instruction": INSTR[lane],
               "anchor": {"card": card["card_id"], "view": a_view,
                          "content": pack_content(a_content, media)},
               "positive": {"card": card["card_id"], "view": p_view,
                            "content": pack_content(p_content, media)},
               "negatives": negatives_for(card, p_view, media)}
        w.add(f"smk-{n:06d}", exp, media)
        n += 1

    img_cards = [c for c in cards if "image" in c["views"]]
    aud_cards = [c for c in cards if "audio" in c["views"]]
    il_cards = [c for c in cards if c.get("interleaved")]
    rng.shuffle(img_cards); rng.shuffle(aud_cards); rng.shuffle(il_cards)

    for c in img_cards[:60]:
        emit("image2text", c, "image", "text",
             c["views"]["image"]["content"], texts_of(c))
    for c in img_cards[60:110] + img_cards[:10]:
        emit("text2image", c, "text", "image",
             texts_of(c), c["views"]["image"]["content"])
    for c in aud_cards[:40]:
        emit("audio2text", c, "audio", "text",
             c["views"]["audio"]["content"], texts_of(c))
    for c in aud_cards[40:80]:
        emit("text2audio", c, "text", "audio",
             texts_of(c), c["views"]["audio"]["content"])
    for c in il_cards[:30]:
        emit("interleaved2text", c, "interleaved", "text",
             c["interleaved"][0]["content"], texts_of(c))

    # text2text from the v001 slice: canonical <-> paraphrase, near-misses
    # as mined negatives (in-band per v1 convention), random fill to k=8.
    v001 = [json.loads(ln) for ln in open(v001_path)]
    t2t = 0
    for row in v001:
        paras = [p for p in (row.get("paraphrases") or []) if isinstance(p, str)]
        if not paras:
            continue
        media = {}
        negs = [{"card": row["card_id"], "view": "text-near-miss",
                 "miner": "v001-near-miss",
                 "content": [{"type": "text", "text": nm}]}
                for nm, s in zip(row.get("near_misses") or [],
                                 row.get("nm_sims") or [])
                if isinstance(nm, str) and 0.75 <= s <= 0.92][:K]
        pool = [r for r in v001 if r["card_id"] != row["card_id"]]
        while len(negs) < K:
            other = rng.choice(pool)
            negs.append({"card": other["card_id"], "view": "text",
                         "miner": "random-fill-smoke",
                         "content": [{"type": "text",
                                      "text": other["canonical_text"]}]})
        exp = {"lane": "text2text", "instruction": INSTR["text2text"],
               "anchor": {"card": row["card_id"], "view": "text",
                          "content": [{"type": "text",
                                       "text": row["canonical_text"]}]},
               "positive": {"card": row["card_id"], "view": "text-para",
                            "content": [{"type": "text",
                                         "text": rng.choice(paras)}]},
               "negatives": negs}
        w.add(f"smk-{n:06d}", exp, media)
        n += 1
        t2t += 1
        if t2t >= 60:
            break

    idx = w.close()
    print(f"wrote {n} exposures -> {w.tar_path}")
    print(f"index -> {idx}")
    print(json.dumps(json.load(open(idx))["counts"], indent=1))


if __name__ == "__main__":
    main()
