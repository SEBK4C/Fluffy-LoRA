# Morning runbook — CARDSPEC lane (for Sebastian, 2026-07-12 ~07:00Z)

Night result: **~121k training assets for ~$9.6 + ~3 kWh.** Full narrative
in `state/T9-STATUS.md`; spend in SYNTH-FORGE LEDGER. Details below are the
two actions that are yours, then the inventory.

## Your two actions

1. **FLUX gate** (~30 s): visit https://huggingface.co/black-forest-labs/FLUX.1-schnell
   logged in as SEBK4C and accept the terms (auto-gate, instant grant).
   Then say the word and I run:
   - the 1,856-prompt truncation re-run (~$0.50) — replaces the weak
     CLIP-77-truncated SDXL pairs;
   - (optional) a full FLUX tranche for generator-diverse noisy pairs
     (~$7 for another 70k at measured throughput).
2. **Swap review**: the 06:00Z decision is the orchestrator's lane —
   check T9-STATUS for the outcome. Post-swap, my queue is: 665-clip
   overlength recovery batch (re-synth at 1.15×), FLUX streaming from the
   3080 Ti per §E0.1, the multilingual (es/de/fr) audio wave if wanted
   (~17.5k more cards, Supertonic is 31-lang), and a same-text-different-
   voice alt-rendition pass (~10% of cards) for the §E voice-invariance
   positives the exposure sampler will want.

## Inventory (mining lane)

| Asset | Where | Count |
|---|---|---|
| Gated cards (MINER, image-lane, validator 50k/50k PASS) | /pool-ssd/fluffy/image-v001-warmup | 50,000 |
| Noisy image-text pairs (SDXL-Lightning 768px, provenance-stamped) | HF dataset SEBK4C/fluffy-noisy-tier (private), 36 tars | 70,241 |
| Gated audio views (Supertonic, frozen A3 gate + 30s cap) | /pool-ssd/fluffy-cards/bulk/ + CAS | ~15k by ~07:30Z |
| Merge-ready audio view objects | bulk/audio-views-v001.jsonl (rerun assemble_audio_views.py to refresh) | tracks the run |
| Frozen evals (real media, sha-pinned) | /pool-ssd/fluffy-cards/eval/ | image 500 + audio 352 |
| Weak-pair flags (CLIP-77) | dataset repo clip_trunc.jsonl | 1,856 |
| Overlength recovery tasks | bulk/recovery_overlength.jsonl | 665 |

## Night incidents (all resolved, all ledgered)

- Rig tailscale key expiry (21:32Z) — flagged, renewed, cleared.
- 10-worker audio scale-up regressed throughput — measured, reverted to 7.
- Cloud GPU-whisper path: 3 strikes (~$3) — abandoned for the proven local
  pipeline; fixed job script kept for post-mortem.
- **30s audio cap was unenforced** — 605 overlength clips caught by wav
  sanity-sampling, corrected in-manifest, pipeline now gates duration.
- SDXL CLIP-77 truncation — 3% of prompts, flagged, FLUX re-run covers it.
