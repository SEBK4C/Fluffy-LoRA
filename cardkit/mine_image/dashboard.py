#!/usr/bin/env python3
"""dashboard.py — one-screen MINE-IMAGE progress view (reads state + queue)."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

state = json.load(open(common.STATE_PATH))
for source, src in sorted(state.get("sources", {}).items()):
    subs = src.get("subsets", {})
    pairs = sum(ss.get("extracted", 0) for ss in subs.values())
    ct = sum(ss.get("chunks_total", 0) for ss in subs.values())
    cd = sum(ss.get("chunks_done", 0) for ss in subs.values())
    mined = sum(1 for ss in subs.values() if ss.get("mined"))
    cards = sum(ss.get("mined", {}).get("cards", 0) for ss in subs.values())
    exp = sum(ss.get("mined", {}).get("exposures", 0) for ss in subs.values())
    gb = sum(ss.get("packed", {}).get("bytes", 0) for ss in subs.values()) / 1e9
    q = len(glob.glob(os.path.join(common.task_dir(source), "*.json")))
    print(f"{source:8s} pairs={pairs:>8,} encode={cd}/{ct} "
          f"mined={mined}/{len(subs)} cards={cards:>8,} exp={exp:>9,} "
          f"packed={gb:6.1f}GB queue={q} "
          f"rig={'Y' if src.get('staged_rig') else '-'} "
          f"hf={'Y' if src.get('uploaded_hf') else '-'}")
claims = [c for c in os.listdir(common.CLAIMS)
          if any(c.startswith(p) for p in ("extract__", "encode__", "minepack__"))]
print(f"claims: {len(claims)} active")
for st, v in state.get("pending_stages", {}).items():
    print(f"  pending: {st} [{v['status']}]")
