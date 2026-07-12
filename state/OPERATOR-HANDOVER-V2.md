# OPERATOR-HANDOVER v2 — Fluffy-LoRA tri-modal window (starts on "restart approved")

Written 2026-07-12 ~00:50Z by the orchestrator, pre-launch. Supersedes
/root/SYNTH-FORGE/state/OPERATOR-HANDOVER.md for the v2 window. The watch
operator (Opus, fresh window) bootstraps from THIS file + state/T9-STATUS.md
+ TRAINING-CHECKLIST.md. KISS is binding law ([[agent-philosophy]]).

## Prime rules (v1 rules carried, updated where v1 taught us better)

1. **Never interrupt the trainer.** Rig GPUs run nothing but the trainer
   after launch. The ONE exception class: pre-authorized refresh-resumes
   (rule 5).
2. **Health = per-lane ratchet evals, NOT the loss line.** v1's central
   lesson, now empirical: loss fell 9.36→1.7 while retrieval stayed at
   chance (and v1's DDP wasn't even syncing gradients). Loss-advance is a
   liveness signal only.
3. **Eval cadence 6h** on the PVE 3080 Ti eval station: text lane =
   ratchet_eval.py on G0; image lane = baseline_image_eval.py protocol on
   image-eval-v1 (both NF4, instruction string frozen: "Retrieve the
   matching description."). Ratchet state = state/ckpt-ratchet-v2.json
   (eps 0.002/lane; KEPT requires beating the pointer in BOTH lanes... a
   checkpoint that wins one lane and regresses the other is REJECTED —
   record both numbers).
4. **Pre-registered kill criterion (Sebastian-approved): G0 R@1 < 0.05 at
   hour 36 → halt-and-alert.** Halt = stop the trainer cleanly, preserve
   last checkpoint, write status; do NOT burn the window on a flat line.
   (Teacher reference on G0 = 0.312; base = 0.008.)
5. **Refresh-resumes are pre-authorized** (day-2/3: audio lane entry via
   CARD-SPEC gated Supertonic pipeline; shard top-ups incl. interleaved
   lane + /pool-5tb migration with HDD re-gate): clean stop → change
   FL_SHARDS/FL_LANES only (audio adds FL_ENC_CHUNK=8) → resume; FL_STEPS
   stays 170000 across ALL resumes; every refresh gets a ledger line and
   honest lane labeling everywhere.
6. **Frozen means frozen**: G0, image-eval-v1, audio-eval-v1 (pins in
   /pool-ssd/fluffy-cards/eval/*.freeze), CARD-SPEC v1.1, the instruction
   string, score protocols. New eval sets go BESIDE, never inside.
7. **No unledgered spend.** CARDSPEC's HF Jobs budget is its own grant.

## Trainer facts (from state/SMOKES-V2-RESULTS.txt — all smokes PASS)

- train_v2.py: full multimodal gemma-4-12b-it, towers frozen, LoRA 32.78M
  (0.27%), NF4, DDP world=2 with explicit LoRA-grad all-reduce, MRL ladder
  [3840,2048,1024,512,256] (live hidden dim is 3840, NOT 4096), lane-
  alternating batches, atomic checkpoints + optimizer/scheduler/cursor
  state, rolling retention (KEPT + last3 + 1/12h), >90% disk watermark
  pauses saves, NaN tripwire exits without saving, wrapper restart cap 5.
- Launch config: see the staged command in T9-STATUS (00:50Z entry).
  5.57 s/step measured; STEPS=170,000 ≈ anneals by day ~11–13.
- Resume test: kill -9 at step 55 → byte-identical cursors, loss
  continuity verified. Trust resume; use it for refreshes.
- lr peak 1e-4 is UNVALIDATED territory (v1 never got above 2.5e-5) — the
  tripwire and hour-36 kill criterion are the guards. Watch the first
  hours after warmup crosses ~5e-5 with extra care.

## Known residuals + morning list (Sebastian, on wake)

1. Type **"restart approved"** → orchestrator (or you) launches the staged
   command in a rig tmux (session name fluffy-v2), watches 30 min, ledgers.
2. Rig root: `mkdir /pool-5tb/fluffy && chown seb /pool-5tb/fluffy` (shard
   home migration at first refresh + HDD re-gate; SSD is interim home).
3. Rig root: `kill 3703732` (EVAL's deadlocked phase4 waiter — inert but
   unclean; orchestrator was correctly denied killing another agent's
   process). Then mask apt-daily{,-upgrade}.timer + disable
   unattended-upgrades for the window (tonight was proven safe by origin
   analysis; 14 days is not).
4. Tailscale admin console: **PVE node key expires 2026-07-20 (day 8)** —
   disable expiry (rig node already done). Without this, PVE↔rig dies
   mid-window (trainer survives, watch/evals/refreshes break).
5. HF: approve FLUX.1-schnell access (repo is approval-gated); morning
   FLUX re-run covers the 3% CLIP-truncated SDXL prompts (CARDSPEC's
   clip_trunc.jsonl).
6. Close parked tmux sessions FLUFFY-BUILDER / FLUFFY-ALIGN (never used).
7. Read LEARNINGS-V1.md + the bench table in the morning report — v1
   root-causes (pad-pooling bug, DDP no-sync) belong in the paper.
