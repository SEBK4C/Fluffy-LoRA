# MINE-IMAGE brief — exhaust the real-photo pile (v2 relaunch prep)

Sebastian's order 2026-07-12 (~08:20Z): v2 stopped at step 589 for a DATA
BREADTH pivot. You own the IMAGE + INTERLEAVED lanes. Target: everything
gate-passing from the CAS image pile as CARD-SPEC v1.1 cards/exposures/
shards, staged to the rig's /pool-5tb/fluffy (now exists, seb-owned).
Timebox: post a data card within 48h; relaunch targets ~day 3. KISS binding.

READ FIRST: state/T9-STATUS.md (compute map + coordination), CARD-SPEC.md
(FROZEN v1.1), cardkit/build_image_lane.py (REUSE — it already did 50k pairs
correctly), TRAINING-CHECKLIST.md §E0/E, MERGE-RESEARCH.md §2 (interleave
order image→text→audio, TopK-PercPos). Rig connection = PRIVATE addendum in
/root/SYNTH-FORGE/FLUFFY-FORGE-BOOTSTRAP.md — never in commits.

## Sources (all in /pool-6b/corpus-acq — READ-ONLY pool)

1. **MMEB-train FULL** (~47.5G): the prior slice took 50k pairs from 2
   subsets — now take EVERYTHING that passes gates across all subsets
   (expect several hundred k pairs). Real photos stay dominant.
2. **ColPali train** (~52.7G): page-image↔query = the document lane.
3. **VisRAG** (~12.2G): doc-image↔query, second wave.

## Rules (unchanged, binding)

- Contamination guard EVERY source: exclude all members overlapping
  image-eval-v1 (extend the val2014 wholesale-exclusion pattern; eval pins
  in /pool-ssd/fluffy-cards/eval/). Evals stay frozen.
- Teacher: Qwen3-VL-2B bulk encode on the RIG 4090s (both free — this is
  your big win; ~100 img/s each measured), 8B for calibration + 1k-sample
  spot-verification per source. TopK-PercPos 0.95, k=8, band_rule recorded
  per negative. NO dataset-shipped negatives.
- Instruction string VERBATIM: "Retrieve the matching description."
- Provenance per row: CAS sha256, rights_tier (source_audit_required →
  audit:"pending"), native_id. Training-use yes, release gated.
- 250-sample cardkit gate before each source's bulk; >30% reject = stop+post.
- INTERLEAVED lane (new): c1-style composites, order image→text→audio,
  permutation negatives, ~10% of exposure volume; coordinate audio refs
  with MINE-TEXTAUDIO via T9-STATUS (use its gated audio views).
- Shards: WDS + idx sidecars + MANIFEST + SHA256SUMS → rsync to
  /pool-5tb/fluffy/shards/<lane>/ → sha -c on rig → HDD readback gate
  (harness at big-SSD fluffy/readback_gate.py; target 10x needed sps).
- Working sets on /pool-ssd/fluffy/ (PVE root fs is 86% — nothing there).

## Coordination

`[HH:MMZ] MINE-IMG:` lines in state/T9-STATUS.md at milestones (date -u);
commit scripts `ff:` (public repo: no media, no tailnet names). GPU claims
posted in T9 (CARDSPEC may want a 4090 for FLUX — negotiate there; you have
priority until your encode is done). DONE = data card in T9: pairs/exposures
per lane, gate pass-rates, shards+GB staged on 5TB, readback result.
