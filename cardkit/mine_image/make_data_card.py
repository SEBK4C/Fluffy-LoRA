#!/usr/bin/env python3
"""make_data_card.py — per-source data card (OPS §3.9 datasheet discipline).

  make_data_card.py --source mmeb   -> OUT_ROOT/DATA-CARD-<source>.md

Aggregates staging/*/REPORT.json + extract stats into the datasheet that
ships with the HF upload: counts, lanes, gates, calibration, contamination
guards, dedup, rights (SIGNOFF-001), benchmark-shaped task coverage.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

SRC_DESC = {
    "mmeb": ("TIGER-Lab/MMEB-train (CORPUS-ACQ fetch @76dd0a4), 19 of 20 "
             "subsets (HatefulMemes excluded: binary labels are degenerate "
             "for contrastive pairs + content-audit risk)"),
    "colpali": ("vidore/colpali_train_set (CORPUS-ACQ fetch), train-*.parquet "
                "ONLY (test split wholly excluded — frozen image-eval-v1 "
                "draws from it); pages recompressed to max-dim 1280 JPEG q85 "
                "(page-recompress-v1; original sha kept in rights)"),
    "visrag": ("openbmb/VisRAG-Ret-Train-In-domain-data (CORPUS-ACQ fetch); "
               "pages recompressed as per ColPali"),
}

GUARDS_DESC = """\
- **Frozen evals untouched**: all 500 image-eval-v1 pin images excluded by
  sha256 (catches renamed COCO files in A-OKVQA/OK-VQA/VisDial/etc.), by
  native_id basename, and by 12-digit COCO id (catches COCO-2017-style
  renames in the MSCOCO grounding subset); ALL val2014 members excluded
  wholesale; ColPali TEST parquet never read + eval query dedup-hashes and
  original-image shas excluded.
- **Warmup dedup**: the 50k image-v001-warmup cards (MSCOCO_i2t +
  VisualNews_i2t) are excluded by image sha256 AND caption dedup-hash.
- **In-run dedup**: retrieval anchors deduped globally by normalized text
  hash (t2i subsets extract AFTER their i2t siblings and carry their hash
  sets); composite kinds dedup on (text, image-sha) keys; page sources
  dedup cross-file on (query-hash, page-sha) at mine time.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    source = args.source
    root = common.SRC_ROOT[source]

    reports = {}
    for rp in sorted(glob.glob(os.path.join(root, "staging", "*",
                                            "REPORT.json"))):
        r = json.load(open(rp))
        reports[r["subset"]] = r
    if not reports:
        raise SystemExit(f"{source}: no REPORT.json yet")

    state = json.load(open(common.STATE_PATH))
    src_state = state["sources"].get(source, {})

    tot_cards = sum(r["cards"] for r in reports.values())
    tot_exp = sum(r["exposures"] for r in reports.values())
    tot_bytes = sum(r["total_bytes"] for r in reports.values())
    tot_shards = sum(r["shards"] for r in reports.values())
    lanes: dict[str, int] = {}
    tasks: dict[str, int] = {}
    for r in reports.values():
        per_lane = r["exposures"] // len(r["lanes"])
        for ln in r["lanes"]:
            lanes[ln] = lanes.get(ln, 0) + per_lane
        tasks[r["task"]] = tasks.get(r["task"], 0) + r["exposures"]

    L = []
    L.append(f"# Fluffy-LoRA data card — image lane / {source}\n")
    L.append(f"Built {time.strftime('%Y-%m-%d %H:%MZ', time.gmtime())} by "
             f"MINE-IMAGE (CARD-SPEC v1.1, frozen gates).\n")
    L.append(f"**Source**: {SRC_DESC.get(source, source)}\n")
    L.append(f"**Totals**: {tot_cards:,} cards / {tot_exp:,} exposures / "
             f"{tot_shards} WDS shards / {tot_bytes / 1e9:.2f} GB\n")

    L.append("## Per-unit results\n")
    L.append("| unit | kind | cards | exposures | lanes | k=8 rate | "
             "pos-sim med | gate | shards | GB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for sub, r in sorted(reports.items()):
        hist = r["mining"]["negatives_histogram"]
        k8 = int(hist.get("8", 0))
        k8_rate = k8 / max(1, r["exposures"])
        med = r["mining"]["calibration"]["positive_sim"]["median"]
        L.append(f"| {sub} | {r['kind']} | {r['cards']:,} | "
                 f"{r['exposures']:,} | {','.join(r['lanes'])} | "
                 f"{k8_rate:.0%} | {med:.3f} | "
                 f"250-CLI PASS + bulk {r['gate']['bulk']} | "
                 f"{r['shards']} | {r['total_bytes'] / 1e9:.2f} |")
    L.append("")

    L.append("## Exposure volume by lane\n")
    for ln, n in sorted(lanes.items(), key=lambda x: -x[1]):
        L.append(f"- `{ln}`: ~{n:,}")
    L.append("\n## Benchmark-shaped task coverage (quality bar §3.8)\n")
    for t, n in sorted(tasks.items(), key=lambda x: -x[1]):
        L.append(f"- {t}: {n:,} exposures")
    L.append("")

    L.append("## Mining recipe (frozen)\n")
    L.append(f"- Teacher: {common.ENCODER} (fp16, 2048-dim, L2-normed), "
             "bulk encode on 2x RTX 4090")
    L.append(f"- Negatives: OWN-mined ANN top-k<={common.K_MAX} within unit, "
             f"TopK-PercPos band (ceiling = {common.PERCPOS} x query's "
             "positive sim, NV-Retriever rule), band_rule + sim recorded per "
             "negative (difficulty metadata for curriculum). Dataset-shipped "
             "negatives NEVER used.")
    L.append(f"- Instruction: stamped verbatim `{common.INSTRUCTION}` "
             "(frozen v2 stage-1 string); per-exposure `task` field enables "
             "instruction-template re-stamp once the frozen instruction set "
             "lands (quality bar §3.1).")
    L.append(f"- Shards: fluffy-exposure-shard-v1 (shards_v2 contract), "
             f"{common.SHARD_SIZE}/shard, .idx.json sidecars, deterministic "
             f"shuffle seed {common.SEED}, MANIFEST.jsonl + SHA256SUMS, "
             "writer re-hash + reader spot-check verified.\n")

    L.append("## Contamination + dedup guards\n")
    L.append(GUARDS_DESC)

    L.append("## Rights (SIGNOFF-001)\n")
    L.append("| tier | audit | training use | redistribution |")
    L.append("|---|---|---|---|")
    L.append("| source_audit_required | pending | YES | NO (HF repo stays "
             "PRIVATE until the rights audit clears) |")
    L.append("\nEvery card carries `rights.source_sha256`, per-view "
             "`native_id`, and origin provenance.\n")

    if src_state.get("staged_rig"):
        L.append("## Store-2 (rig) verification\n")
        L.append(f"```json\n{json.dumps(src_state['staged_rig'], indent=1)}"
                 "\n```\n")

    out = os.path.join(root, f"DATA-CARD-{source}.md")
    with open(out + ".tmp", "w") as f:
        f.write("\n".join(L))
    os.rename(out + ".tmp", out)
    print(f"wrote {out}: {tot_cards:,} cards, {tot_exp:,} exposures, "
          f"{tot_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
