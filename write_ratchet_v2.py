#!/usr/bin/env python3
"""Assemble state/ckpt-ratchet-v2.json from tonight's base baselines.

Per-lane ratchet state for the v2 window (T9 swap gate). eps rule follows
v1: eps = max(2*sigma, 0.002) per lane. best_checkpoint starts at "none"
(= base); the watch advances it kept-only, per lane, exactly as
ratchet_eval.py does for v1.
"""
import glob
import json
import sys
import time
from pathlib import Path

G0_PIN = "a6bc914be00afa15498580180e0633738d0c98a091938b611bf0764f56ef0761"
IMG_REPORT = sorted(glob.glob("logs/baseline-image-v2-*.json"))[-1]
rep = json.loads(Path(IMG_REPORT).read_text())

sig_img = max(rep["sigma"]["i2t_r1"], rep["sigma"]["t2i_r1"])
r0 = rep["runs"][0]

state = {
    "series": "v2",
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model": rep["model"],
    "snapshot": "0e2b1058541244490925fbacf8972041435691ac",
    "quant": rep["quant"],
    "eps_rule": "per lane: eps = max(2*sigma, 0.002); pointer advances only if lane r1 - best_r1 > eps (kept-only, ratchet_eval semantics)",
    "lanes": {
        "text": {
            "eval": "G0",
            "pin": G0_PIN,
            "baseline": {"r1": 0.008, "r5": 0.0213},
            "sigma": 0.0,
            "eps": 0.002,
            "n_pool": 1500, "n_queries": 3000,
            "protocol": "ratchet_eval.py UNCHANGED (language_model strip, tokenizer-only, no instruction) — continuity with v1 on-record baseline; rerun 2026-07-11T22:59:25Z reproduced r1/r5 exactly (secs 236.6); sigma 0.0 on record from 3 v1 re-evals",
        },
        "image": {
            "eval": "image-eval-v1",
            "pin": rep["pin"],
            "baseline": {
                "i2t": r0["i2t"],
                "t2i": r0["t2i"],
            },
            "sigma": sig_img,
            "sigma_detail": rep["sigma"],
            "eps": max(2 * sig_img, 0.002),
            "n_pairs": rep["n_pairs"],
            "protocol": "baseline_image_eval.py — FULL multimodal model (no strip), processor chat-template with role wrapper, cas:// resolved to local paths, lastpos (mask*arange).argmax pooling (train_v2 byte-match), text clip 2000 chars, query-side instruction prefix only",
            "instruction": rep["instruction"],
            "instruction_note": rep["instruction_note"],
            "runs": rep["runs"],
            "report": IMG_REPORT,
            "versions": rep["versions"],
            "secs_per_run": [r["secs"]["total"] for r in rep["runs"]],
            "img_batch": rep["img_batch"],
        },
        "audio": {
            "eval": "audio-eval-v1",
            "pin": "fe4b83d17e7d6af34c4c05aefc57b2144a7a349c611a1c5991c850fe9785dbce",
            "baseline": None,
            "note": "NOT a tonight-gate — lane enters day-2/3 (ORCH 22:47Z); baseline before audio lanes join FL_LANES",
        },
    },
    "best_checkpoint": "none",
    "kept": [],
    "rejected": [],
}

out = Path("state/ckpt-ratchet-v2.json")
out.write_text(json.dumps(state, indent=2) + "\n")
print(f"WROTE {out}")
print(json.dumps({"image_baseline": state["lanes"]["image"]["baseline"],
                  "image_eps": state["lanes"]["image"]["eps"],
                  "sigma": sig_img}))
