#!/usr/bin/env python3
"""build_queue.py — contamination/dedup guards + extract task fan-out.

  build_queue.py --source mmeb      # guards.json + 19 extract tasks
  build_queue.py --source colpali   # guards.json + 82 per-file tasks
  build_queue.py --source visrag    # 37 per-file tasks

Guards (image-eval-v1 stays FROZEN; warmup 50k never re-mined):
  eval_shas             sha256 of every eval-pin image (catches renamed COCO)
  eval_basenames        native_id basenames (warmup rule)
  eval_coco_ids         12-digit COCO ids of val2014 eval members (catches
                        COCO-2017-style renames, e.g. MSCOCO grounding)
  eval_qhash            dedup-hash of eval query texts (ColPali half)
  warmup_shas/captions  image-v001-warmup cards (dedup vs the 50k slice)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
from common import log
import mmeb_spec

EVAL_JSONL = os.environ.get(
    "EVAL_JSONL", "/pool-ssd/fluffy-cards/eval/image-eval-v1.jsonl")
EVAL_CAS_ROOT = os.environ.get("EVAL_CAS_ROOT", "/pool-ssd/fluffy-cards")
WARMUP_CARDS = os.environ.get(
    "WARMUP_CARDS", "/pool-ssd/fluffy/image-v001-warmup/cards-v2.jsonl")


def build_guards() -> dict:
    import cardlib
    g = {"eval_shas": [], "eval_basenames": [], "eval_coco_ids": [],
         "eval_qhash": [], "warmup_shas": [], "warmup_caption_hashes": []}
    with open(EVAL_JSONL) as f:
        for line in f:
            r = json.loads(line)
            g["eval_shas"].append(r["image"][6:])
            nid = r.get("native_id", "")
            g["eval_basenames"].append(os.path.basename(nid))
            m = re.search(r"(\d{12})", os.path.basename(nid))
            if m:
                g["eval_coco_ids"].append(m.group(1))
            g["eval_qhash"].append(cardlib.dedup_hash(r["text"]))
    with open(WARMUP_CARDS) as f:
        for line in f:
            c = json.loads(line)
            g["warmup_shas"].append(c["rights"]["source_sha256"])
            g["warmup_caption_hashes"].append(c["dedup"]["hash"])
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    choices=["mmeb", "colpali", "visrag"])
    args = ap.parse_args()
    source = args.source
    root = common.SRC_ROOT[source]
    os.makedirs(root, exist_ok=True)

    g = build_guards()
    gp = os.path.join(root, "guards.json")
    with open(gp + ".tmp", "w") as f:
        json.dump(g, f)
    os.rename(gp + ".tmp", gp)
    log("queue", f"{source}: guards.json — {len(g['eval_shas'])} eval shas, "
        f"{len(g['eval_coco_ids'])} coco ids, {len(g['eval_qhash'])} eval "
        f"queries, {len(g['warmup_shas'])} warmup shas")

    n = 0
    if source == "mmeb":
        for sub, spec in mmeb_spec.SUBSETS.items():
            payload = {"task": "extract", "source": "mmeb", "subset": sub}
            if spec.get("after"):
                payload["after"] = spec["after"]
            common.enqueue("mmeb", f"extract__mmeb__{sub}", payload)
            n += 1
    else:
        from extract_pages import SRC
        files = sorted(f for f in os.listdir(SRC[source])
                       if f.startswith("train-"))
        for i in range(len(files)):
            common.enqueue(source, f"extract__{source}__f{i:02d}",
                           {"task": "extract", "source": source, "file": i})
            n += 1

    def upd(state):
        src = state["sources"].setdefault(source, {"subsets": {}})
        src["queued_extract_tasks"] = n
    common.update_state(upd)
    log("queue", f"{source}: {n} extract tasks queued")


if __name__ == "__main__":
    main()
