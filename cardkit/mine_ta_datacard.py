#!/usr/bin/env python3
"""mine_ta_datacard.py — render DATACARD.md for a packed MINE-TA source
from its REPORT.json (+ pairs stats), per MINING-OPS §3.9 datasheet
discipline. Usage: mine_ta_datacard.py <report.json> [--extra k=v ...]
Writes DATACARD.md next to the report. The rights row (one line, pipe
table) is also appended to /pool-ssd/fluffy/state/mine-ta-rights-rows.md
for the HF README rights table (SIGNOFF-001)."""
from __future__ import annotations

import json
import os
import sys
import time

RIGHTS_ROWS = "/pool-ssd/fluffy/state/mine-ta-rights-rows.md"

TEMPLATE = """# Data card — {source}

- **Built by**: MINE-TEXTAUDIO (Fluffy-LoRA v2 relaunch prep), {packed_at}
- **Task type**: `{task_type}` (instruction currently frozen string
  `{instruction}`; per-exposure `task_type` allows restamp at freeze)
- **Cards / exposures**: {cards_or_pairs} cards -> {exposures} exposures
  ({lanes})
- **Shards**: {shards} WebDataset tars ({gb:.2f} GB), shards_v2 contract
  (.idx.json sidecars, MANIFEST.jsonl, SHA256SUMS)
- **Negatives**: {negatives}
- **Gates**: 250-sample cardkit CLI gate PASS + bulk validate 100%
  (schema, media rules, dedup-hash recompute, negative referential
  integrity{extra_gates})
- **Contamination guards**: {contamination}
- **Rights**: tier `{rights_tier}` — {license}. audit={audit};
  redistribution_ok={redistribution_ok}. HF repo stays PRIVATE until
  SIGNOFF-001 clears (MINING-OPS §2).
- **Provenance**: per-view origin + native_id; media by CAS sha256.
- **Difficulty metadata**: teacher sim per negative (+ pos_sim where the
  text teacher sees both sides) for curriculum / ANCE re-mining.
{extra_lines}
"""


def main() -> None:
    report_path = sys.argv[1]
    rep = json.load(open(report_path))
    extras = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    source = rep.get("source") or rep.get("subset")
    lanes = rep.get("lanes") or {"text2text": rep.get("exposures")}
    neg_desc = extras.pop("negatives", None) or (
        f"k={rep.get('k_max') or rep.get('k_text')} teacher-band "
        f"({rep.get('calibration', {}).get('band_rule', rep.get('band_rule'))}), "
        f"miner {rep.get('miner', 'qwen3emb8b-ann-v1')}; histogram "
        f"{rep.get('negatives_histogram')}")
    card = TEMPLATE.format(
        source=source,
        packed_at=rep.get("packed_at", time.strftime("%Y-%m-%d")),
        task_type=rep.get("task_type"),
        instruction=rep.get("instruction"),
        cards_or_pairs=rep.get("cards") or rep.get("pairs_kept"),
        exposures=rep.get("exposures"),
        lanes=", ".join(f"{k}={v}" for k, v in lanes.items()),
        shards=rep.get("shards"),
        gb=rep.get("bytes", 0) / 1e9,
        negatives=neg_desc,
        extra_gates=extras.pop("extra_gates", ""),
        contamination=extras.pop(
            "contamination",
            "eval splits wholesale-excluded at extraction (frozen evals "
            "use dev/test splits never mined)"),
        rights_tier=rep.get("rights_tier", "per-card (see cards-v2.jsonl)"),
        license=rep.get("license", "per-card"),
        audit=extras.pop("audit", "pending"),
        redistribution_ok=extras.pop("redistribution_ok", "false"),
        extra_lines="".join(f"- **{k}**: {v}\n" for k, v in extras.items()),
    )
    out = os.path.join(os.path.dirname(report_path), "DATACARD.md")
    with open(out, "w") as f:
        f.write(card)
    row = (f"| {source} | {rep.get('task_type')} | "
           f"{rep.get('exposures')} exposures | "
           f"{rep.get('rights_tier', 'per-card')} | "
           f"{rep.get('license', 'per-card')} | "
           f"{extras.get('audit', 'pending')} |\n")
    with open(RIGHTS_ROWS, "a") as f:
        f.write(row)
    print(f"wrote {out}\nrights row appended to {RIGHTS_ROWS}")


if __name__ == "__main__":
    main()
