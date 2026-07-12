#!/usr/bin/env python3
"""synth_queries.py — quality bar §3.2: synthetic query diversification over
REAL images (MegaPairs pattern, adapted to available serving).

  synth_queries.py --subset MSCOCO_i2t --n-cards 2000 --per-card 2 \
      --llm http://127.0.0.1:8090 [--emb-out ...]

Recipe (documented deviation from MegaPairs: generation is CAPTION-
conditioned — the rig's gemma-4 QAT gguf serves TEXT via llama.cpp; no
vision path deployed there tonight — but the GATE is IMAGE-conditioned:
every generated query must clear the teacher band against the card's
IMAGE embedding, so the image stays the arbiter of acceptance):
  1. sample cards from the subset's cards-v2.jsonl (real photos, caption
     view = generation seed)
  2. gemma-4 generates N diverse queries per card (styles: keyword search /
     natural question / descriptive clause)
  3. queries -> items file; encode with Qwen3-VL-2B (encode_items.py — run
     it on a free rig GPU, or --device cpu for pilots)
  4. gate: cos(query_emb, card image_emb) >= subset calibration p25 of REAL
     positive sims (recorded per view; failed queries dropped, never ship)
  5. accepted queries -> alt views text-sq1/text-sq2 (source=synthetic,
     gen+gate recorded) + NEW text2image exposures with negatives RE-MINED
     per query (TopK-PercPos vs the subset's image item space) -> packed to
     shards/<subset>-synthq/ (same FORMAT CONTRACT).
Emits synth-views-<subset>.jsonl (audio-views join pattern) + REPORT.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import common
from common import log
import mmeb_spec

PROMPT = """You write search queries that would retrieve a given photo.
The photo's caption is: "{caption}"

Write {n} DIFFERENT search queries a user might type to find exactly this
photo. Vary the style: one short keyword query, one natural-language
question, one descriptive phrase. Each must be specific to THIS photo's
content, not generic. Output ONLY a JSON array of {n} strings."""


def llm_queries(url: str, caption: str, n: int) -> list[str]:
    body = json.dumps({
        "messages": [{"role": "user",
                      "content": PROMPT.format(caption=caption, n=n)}],
        "temperature": 0.8, "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    try:
        qs = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [" ".join(str(q).split())[:300] for q in qs
            if isinstance(q, str) and 8 <= len(q.strip()) <= 300][:n]


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--n-cards", type=int, default=2000)
    ap.add_argument("--per-card", type=int, default=2)
    ap.add_argument("--llm", default="http://127.0.0.1:8090")
    ap.add_argument("--encode-cmd", default="",
                    help="shell cmd template for encode_items (default: "
                    "local venv, cpu). Placeholders {items} {out}")
    args = ap.parse_args()
    sub = args.subset
    spec = mmeb_spec.SUBSETS[sub]
    assert spec["kind"] == "retrieval", "synth queries target retrieval subsets"
    root = common.SRC_ROOT["mmeb"]
    os.environ["FLUFFY_CARDS_ROOT"] = root
    stg = os.path.join(root, "staging", sub)
    import cardlib
    import minepack as mp
    import shards_v2

    t0 = time.time()
    # subset pairs + embeddings (image item space for gating + re-mining)
    _, _, pairs, emb, idx = mp.load_unit("mmeb", sub)
    by_id = {p["card_id"]: p for p in pairs}
    report_real = json.load(open(os.path.join(stg, "REPORT.json")))
    gate_floor = report_real["mining"]["calibration"]["positive_sim"]["p25"]
    log("synthq", f"{sub}: gate floor = real-positive p25 = {gate_floor:.3f}")

    rng = random.Random(common.SEED)
    sample = rng.sample(pairs, min(args.n_cards, len(pairs)))

    # ---- 1+2: generate -------------------------------------------------------
    gen, n_fail = [], 0
    for i, p in enumerate(sample):
        try:
            qs = llm_queries(args.llm, p["anchor_text"], args.per_card)
        except Exception as ex:  # noqa: BLE001
            n_fail += 1
            if n_fail > 50 and n_fail > 0.2 * (i + 1):
                raise SystemExit(f"LLM failing hard ({n_fail}/{i + 1}) — stop")
            continue
        for j, q in enumerate(qs):
            gen.append({"card_id": p["card_id"], "slot": j + 1, "query": q})
        if (i + 1) % 200 == 0:
            log("synthq", f"gen {i + 1}/{len(sample)} "
                f"({len(gen)} queries, {n_fail} llm fails)")
    log("synthq", f"generated {len(gen)} queries over {len(sample)} cards")

    # ---- 3: encode -----------------------------------------------------------
    items_path = os.path.join(stg, "synthq-items.jsonl")
    with open(items_path, "w") as f:
        for i, g in enumerate(gen):
            f.write(json.dumps({"id": f"sq{i:08d}", "kind": "text",
                                "text": g["query"], "image": None}) + "\n")
    emb_path = os.path.join(stg, "synthq-emb.npy")
    cmd = (args.encode_cmd or
           f"{sys.executable} {HERE}/encode_items.py --items {{items}} "
           f"--out {{out}} --model Qwen/Qwen3-VL-Embedding-2B "
           f"--media-root {root}/cas --device cpu")
    subprocess.run(cmd.format(items=items_path, out=emb_path), shell=True,
                   check=True, timeout=4 * 3600)
    qemb = np.load(emb_path).astype(np.float32)

    # ---- 4: image-conditioned teacher gate ----------------------------------
    def img_row(p):
        s = p["anchor"] if p["anchor"]["sha"] else p["positive"]
        return idx[common.item_id(s["kind"], s["text"], s["sha"])]

    kept, sims, kept_idx = [], [], []
    for i, (g, qe) in enumerate(zip(gen, qemb)):
        ie = emb[img_row(by_id[g["card_id"]])]
        s = float(qe @ ie)
        if s >= gate_floor:
            kept.append({**g, "sim_image": round(s, 4)})
            kept_idx.append(i)
            sims.append(s)
    rate = len(kept) / max(1, len(gen))
    log("synthq", f"gate: {len(kept)}/{len(gen)} pass ({rate:.0%}) "
        f"median sim {np.median(sims) if sims else 0:.3f}")
    if rate < 0.3:
        raise SystemExit("gate pass rate <30% — stop + post to T9 per brief")

    # ---- 5: views + re-mined negatives + pack --------------------------------
    QE = qemb[kept_idx]
    a_rows = np.array([img_row(by_id[k["card_id"]]) for k in kept])
    pos = np.einsum("ij,ij->i", QE, emb[a_rows])
    au = np.unique(np.array([img_row(p) for p in pairs]))
    rep = {}
    for p in pairs:
        rep.setdefault(int(img_row(p)), p["card_id"])
    uniq_pos = np.searchsorted(au, a_rows)
    nix, nsm, ncn = mp.ann(QE, emb[au], pos, uniq_pos, "synthq")

    views_path = os.path.join(stg, f"synth-views-{sub}.jsonl")
    exposures = []
    with open(views_path, "w") as f:
        for k, g in enumerate(kept):
            vname = f"text-sq{g['slot']}"
            view = {"content": [{"type": "text", "text": g["query"]}],
                    "source": "synthetic", "origin": "mmeb",
                    "native_id": by_id[g["card_id"]]["native_id"],
                    "gen": {"model": "gemma-4-12b-it-qat-q4_0",
                            "version": "synthq-v1"},
                    "gate": {"pass": True, "image_sim": g["sim_image"],
                             "floor": round(gate_floor, 4)}}
            f.write(json.dumps({"card_id": g["card_id"], "view": vname,
                                "obj": view}, ensure_ascii=False) + "\n")
            br = (f"topk-percpos-{common.PERCPOS}:"
                  f"ceil={round(float(common.PERCPOS * pos[k]), 4)}")
            negs = [{"card": rep[int(au[j])], "view": "image",
                     "sim": round(float(s), 4), "miner": "vl-ann-v1-synthq",
                     "band_rule": br}
                    for j, s in zip(nix[k][:ncn[k]], nsm[k][:ncn[k]])]
            exposures.append({"anchor": {"card": g["card_id"], "view": vname,
                                         "_text": g["query"]},
                              "positive": {"card": g["card_id"],
                                           "view": "image"},
                              "negatives": negs, "lane": "text2image",
                              "task": "retrieval-synthq", "source": "mmeb",
                              "subset": sub,
                              "instruction": common.INSTRUCTION})

    cards = {}
    for line in open(os.path.join(stg, "cards-v2.jsonl")):
        c = json.loads(line)
        cards[c["card_id"]] = c
    sdir = os.path.join(root, "shards", f"{sub}-synthq")
    os.makedirs(sdir, exist_ok=True)
    rng.shuffle(exposures)

    def entry(ref, media):
        if "_text" in ref:
            return {"card": ref["card"], "view": ref["view"],
                    "content": [{"type": "text", "text": ref["_text"]}]}
        card = cards[ref["card"]]
        content = []
        for item in card["views"][ref["view"]]["content"]:
            if item["type"] == "image":
                data = open(common.cas_path(root, item["image"][6:]),
                            "rb").read()
                mname = shards_v2.media_name(data, "jpg")
                media[mname] = data
                content.append({"type": "image", "image": f"member://{mname}"})
            else:
                content.append(item)
        return {"card": ref["card"], "view": ref["view"], "content": content}

    manifest = []
    for s0 in range(0, len(exposures), common.SHARD_SIZE):
        chunk = exposures[s0:s0 + common.SHARD_SIZE]
        name = f"synthq-{spec['tag']}-{s0 // common.SHARD_SIZE:06d}.tar"
        w = shards_v2.ShardWriter(os.path.join(sdir, name))
        for j, e in enumerate(chunk):
            media = {}
            exp = {"lane": e["lane"], "task": e["task"], "source": "mmeb",
                   "subset": sub, "instruction": e["instruction"],
                   "anchor": entry(e["anchor"], media),
                   "positive": entry(e["positive"], media),
                   "negatives": [dict(entry(n, media), miner=n["miner"],
                                      sim=n["sim"], band_rule=n["band_rule"])
                                 for n in e["negatives"]]}
            w.add(f"{s0 + j:08d}", exp, media)
        w.close()
        h = hashlib.sha256(open(os.path.join(sdir, name), "rb").read())
        manifest.append({"shard": name, "idx": name + ".idx.json",
                         "sha256": h.hexdigest(), "samples": len(chunk)})
    with open(os.path.join(sdir, "MANIFEST.jsonl"), "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(sdir, "SHA256SUMS"), "w") as f:
        for m in manifest:
            f.write(f"{m['sha256']}  {m['shard']}\n")

    hist = collections.Counter(len(e["negatives"]) for e in exposures)
    report = {"subset": sub, "stage": "synth-queries-v1",
              "sampled_cards": len(sample), "generated": len(gen),
              "gate_floor_p25": round(gate_floor, 4),
              "gate_pass": len(kept), "gate_rate": round(rate, 3),
              "exposures": len(exposures), "shards": len(manifest),
              "neg_hist": {str(k): v for k, v in sorted(hist.items())},
              "llm": "gemma-4-12b-it-qat-q4_0 (caption-seeded, image-gated)",
              "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(stg, "SYNTHQ-REPORT.json"), "w") as f:
        json.dump(report, f, indent=1)

    def upd(state):
        ss = common.subset_state(state, "mmeb", sub)
        ss["synthq"] = report
    common.update_state(upd)
    log("synthq", f"DONE {json.dumps(report)}")


if __name__ == "__main__":
    main()
