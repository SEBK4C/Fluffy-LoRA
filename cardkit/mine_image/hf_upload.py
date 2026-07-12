#!/usr/bin/env python3
"""hf_upload.py — store-3 sync (OPS §2): shards + data card -> PRIVATE HF
dataset repo SEBK4C/Fluffy-LoRA-dataset under image/<source>/.

  hf_upload.py --source mmeb

SIGNOFF-001: repo is PRIVATE until the rights audit clears; this script
refuses to run against a public repo. The repo README carries a rights
table per source, maintained inside managed markers as sources land.
Per-subset upload_folder commits with retry (resumable at subset level:
already-uploaded identical files are skipped by the Hub's dedup).
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import log

REPO = "SEBK4C/Fluffy-LoRA-dataset"
MARK_A = "<!-- rights-table:begin -->"
MARK_B = "<!-- rights-table:end -->"
HEADER = ("| lane/source | cards | exposures | GB | rights tier | audit | "
          "release |\n|---|---|---|---|---|---|---|")


def rights_row(source: str, cards: int, exp: int, gb: float) -> str:
    return (f"| image/{source} | {cards:,} | {exp:,} | {gb:.1f} | "
            f"source_audit_required | pending | BLOCKED until audit |")


def update_readme(api, source: str, row: str) -> None:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(REPO, "README.md", repo_type="dataset")
        text = open(p).read()
    except Exception:  # noqa: BLE001
        text = ("# Fluffy-LoRA dataset (PRIVATE)\n\n"
                "Incremental off-site store of the Fluffy-LoRA mining week "
                "(MINING-OPS §2 store 3). PRIVATE until the rights audit "
                "clears (SIGNOFF-001).\n\n## Rights table\n\n"
                f"{MARK_A}\n{HEADER}\n{MARK_B}\n")
    if MARK_A not in text:
        text += f"\n## Rights table\n\n{MARK_A}\n{HEADER}\n{MARK_B}\n"
    head, _, rest = text.partition(MARK_A)
    block, _, tail = rest.partition(MARK_B)
    lines = [l for l in block.strip().splitlines()
             if l.strip() and f"| image/{source} " not in l]
    if not lines:
        lines = HEADER.splitlines()
    lines.append(row)
    text = head + MARK_A + "\n" + "\n".join(lines) + "\n" + MARK_B + tail
    api.upload_file(path_or_fileobj=io.BytesIO(text.encode()),
                    path_in_repo="README.md", repo_id=REPO,
                    repo_type="dataset",
                    commit_message=f"rights table: image/{source}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    source = args.source
    root = common.SRC_ROOT[source]
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(REPO)
    if not info.private:
        raise SystemExit(f"{REPO} is NOT private — SIGNOFF-001 forbids "
                         "uploading source_audit_required media. ABORT.")

    card = os.path.join(root, f"DATA-CARD-{source}.md")
    if not os.path.exists(card):
        raise SystemExit(f"{card} missing — run make_data_card.py first")

    t0 = time.time()
    subdirs = sorted(d for d in glob.glob(os.path.join(root, "shards", "*"))
                     if os.path.isdir(d))
    done_units = []
    for d in subdirs:
        unit = os.path.basename(d)
        dest = f"image/{source}/shards/{unit}"
        for attempt in range(1, 4):
            try:
                api.upload_folder(folder_path=d, path_in_repo=dest,
                                  repo_id=REPO, repo_type="dataset",
                                  commit_message=f"image/{source}: {unit}")
                break
            except Exception as ex:  # noqa: BLE001
                log("hf", f"{unit} upload attempt {attempt} failed: "
                    f"{type(ex).__name__}: {str(ex)[:200]}")
                if attempt == 3:
                    raise
                time.sleep(60 * attempt)
        done_units.append(unit)
        log("hf", f"uploaded {dest} ({len(done_units)}/{len(subdirs)})")

    api.upload_file(path_or_fileobj=card,
                    path_in_repo=f"image/{source}/DATA-CARD-{source}.md",
                    repo_id=REPO, repo_type="dataset",
                    commit_message=f"image/{source}: data card")

    import json
    reports = [json.load(open(p)) for p in glob.glob(
        os.path.join(root, "staging", "*", "REPORT.json"))]
    cards_n = sum(r["cards"] for r in reports)
    exp_n = sum(r["exposures"] for r in reports)
    gb = sum(r["total_bytes"] for r in reports) / 1e9
    update_readme(api, source, rights_row(source, cards_n, exp_n, gb))

    def upd(state):
        src = state["sources"].setdefault(source, {"subsets": {}})
        src["uploaded_hf"] = {
            "repo": REPO, "path": f"image/{source}", "units": done_units,
            "elapsed_s": round(time.time() - t0, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    common.update_state(upd)
    log("hf", f"{source}: upload COMPLETE in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
