# FLUFFY-BUILDER brief — trainer v2 + warmup data + THE SWAP by T-0 06:00Z

Authorized by Sebastian 2026-07-11 ~21:00Z: **"Make it happen in the next
nine hours."** T-0 = 2026-07-12 06:00Z. You are the builder. KISS is binding.
API/token spend is authorized (log every external spend in the LEDGER).

READ FIRST (all in this repo unless noted): TRAINING-CHECKLIST.md ·
MERGE-RESEARCH.md §2 (RATIFIED architecture — causal + last-token, native
4096-dim + MRL ladder, staged warmup, instruction prefix, TopK-PercPos) ·
CARD-SPEC.md v1.1 FROZEN + cardkit/ · DECISIONS-CARDSPEC.md · LEARNINGS-V1.md
(schedule math!) · /root/SYNTH-FORGE/FLUFFY-FORGE-BOOTSTRAP.md GROUND-TRUTH
ADDENDUM (rig connection details live there — PRIVATE, never commit them).

## Division of labor tonight

- **YOU**: trainer v2, miners, shards, smokes, baselines, the swap.
- **FLUFFY-ALIGN** (parallel tmux session): model downloads, staging tree,
  rsync, readback harness, rig hygiene. See ALIGN-BRIEF.md.
- **FLUFFY-EVAL** (existing session): finishing the benchmark on rig GPU0 —
  it frees GPU0 soon and notes it in `state/T9-STATUS.md`.
- Coordinate ONLY via `state/T9-STATUS.md` (append-only, timestamped lines,
  every agent) + repo commits (`ff:` prefix for you). Check it before every
  phase transition. Do not touch other agents' processes.

## Critical path (start ALL parallelizable items immediately)

1. **Trainer v2** (§B checklist + §2 MERGE-RESEARCH): full multimodal model
   (NO .language_model strip), towers frozen, LoRA targets unchanged, NF4,
   DDP world=2, alternating single-modality lane batches + interleaved
   minority lane (image→text→audio order), staged warmup config (stage 1 =
   text+image), instruction prefix at encode time, atomic saves + optimizer/
   scheduler/data-cursor state, rolling retention + >90% disk watermark, NaN
   tripwire, auto-restart wrapper. save_dir on the rig's big SSD mount.
   STEPS sized from MEASURED step-time (LEARNINGS-V1 §f — v1 died in warmup;
   warmup 2–3% of the REAL horizon, cosine annealing to the actual end).
2. **Smokes on rig GPU1 NOW** (GPU0 busy until eval frees it): A1 image
   path, A3 VRAM/batch-size, A4 grad flow, A6 20-step DDP alternating-lane
   smoke. Then the **kill -9 resume test at step ~50 — MANDATORY, no swap
   without it.**
3. **Text lane**: re-shard v001 pairs (224,474) into CARD-SPEC v1.1 exposure
   format (CPU, pool-ssd working set — PVE root fs is 86%, put NOTHING
   there).
4. **Image warmup slice**: when ALIGN posts Qwen3-VL-Embedding ready →
   re-derive band thresholds (~1k samples, TopK-PercPos 95% filter), then
   band-mine the first 20–50k MMEB pairs. FLUX noisy-tier gen takes rig
   GPU0 the moment EVAL frees it (target: whatever exists by T-2 ships;
   the 100–200k target streams from the 3080 Ti post-swap).
5. **Shards**: pack warmup shards (WebDataset per CARD-SPEC storage rules) →
   post in T9-STATUS → ALIGN rsyncs + runs the readback gate.
6. **Image-lane frozen eval + baselines** (gates the swap): smallest viable
   real-media image eval (real pages/photos on ≥1 side), byte-frozen; then
   per-lane BASE baselines → `state/ckpt-ratchet-v2.json`. Text lane
   baseline = existing G0 numbers, carry them in.
7. **THE SWAP at T-0.** Gates, ALL must be green: readback ✓ smokes ✓
   resume-test ✓ per-lane baselines ✓ retention/watchdog/auto-restart armed ✓
   auto-updates disabled (ALIGN) ✓. All green → execute under Sebastian's
   "make it happen" authorization, ledger it + echo the supervision-ack
   handshake, watch 30 min, write the Opus-watch v2 addendum +
   OPERATOR-HANDOVER-v2. **ANY gate red → NO swap** — post T9-STATUS, wait
   for Sebastian. A late clean launch beats an on-time broken one.

## Standing orders for the 14-day run (write them into the watch addendum)

- **Pre-registered kill criterion (Sebastian-approved): G0 R@1 < 0.05 at
  hour 36 → halt-and-alert**, don't burn the window on a flat line. v1's
  central lesson: loss-advance ≠ learning; per-lane eval movement is health.
- **Day-2/3 refresh-resumes are PRE-AUTHORIZED** (audio lane entry via
  Supertonic per CARD-SPEC v1.1 §gates, shard top-ups): each one ledgered,
  clean resume from saved state, honest lane labeling everywhere.
- Rig GPUs run NOTHING but the trainer post-swap. CORPUS-ACQ pool
  READ-ONLY. Public repo: no tailnet names/usernames/keys, no gated-rights
  manifests, no media.
