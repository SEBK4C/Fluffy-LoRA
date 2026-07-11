# TRAINING-CHECKLIST — v1 safety + v2 deltas from current state

Verified against the live rig 2026-07-11 ~17:30Z (training alive: step 1400,
loss 1.66, both GPUs 100%). Checkboxes are the delta between what runs today
and what v2 needs. Owner: builder window, except section A (Opus watch).

> **DECISION 2026-07-11 ~18:00Z — v1 ABANDONED (Sebastian).** Keep latest
> weights as fluffy-text-v0, stop the trainer, benchmark the adapter on real
> text-embedding tasks for learnings. Owned by the FLUFFY-EVAL agent —
> see `EVAL-AGENT-BRIEF.md`. §A below is thereby SUPERSEDED (cleanup happens
> in wind-down, no prune policy needed). Rig GPUs free after the stop until
> the v2 swap.

## A. v1 run — checkpoint disk (act BEFORE ~day 12) ⚠️ [SUPERSEDED by v1 stop]

Measured reality: the trainer writes checkpoints to its home dir on the rig's
**root filesystem — 455G, 84% full, 74G free**. Cadence ~131MB every ~30min ≈
6.3GB/day → **root fs hits full at ~day 12 of 14**. A full root disk doesn't
just kill the run, it can wedge the whole machine.

- [ ] **Replace calendar "day-12 prune" with continuous rolling retention**,
      applied at every watch firing: keep ALL ratchet-KEPT + last 3 + one per
      12h; delete the rest. Steady state ≈ 5GB. Alert if root fs > 90%.
      **GATE: Sebastian types "prune policy approved" → watch amends
      OPERATOR-HANDOVER and starts pruning.**
- [ ] Do NOT move the checkpoint dir mid-run (that would touch the trainer).
      v2 fixes the location properly.
- [ ] Known quirk, no action: v1's schedule is sized for 200k steps but the
      window yields ~56k (~21.4s/step measured) — lr never leaves
      warmup/near-peak, cosine never anneals. v1 is a data-throughput run,
      not a schedule-complete run. v2 must size STEPS to the window.

## B. v2 trainer deltas (from current train.py)

- [ ] Full multimodal model — drop the `.language_model` strip; vision/audio
      towers frozen; LoRA targets unchanged (A1/A3/A4 smokes on the 3080 Ti)
- [ ] Alternating single-modality-lane batches, DDP-safe (A6: 20-step smoke)
- [ ] STEPS sized to the real window from measured step-time (re-measure with
      image batches — they're slower); cosine anneals to the actual end
- [ ] `save_dir` → the rig's big SSD mount (1.3T free), never the root fs
- [ ] Atomic saves (write tmp, rename); save optimizer + scheduler + data
      cursor so the run is resumable
- [ ] **Resume test before the swap: kill -9 at step ~50, resume, verify loss
      continuity** — the single cheapest insurance in long-run training
- [ ] Rolling retention built into the trainer (same policy as §A) + disk
      watermark guard: >90% → pause saves and alert, never crash
- [ ] NaN / loss-spike tripwire: halt-and-alert, don't save poisoned ckpts
- [ ] Auto-restart wrapper (systemd or tmux loop) with restart cap; every
      restart gets a ledger line

## C. Data — rig-local serving (network independence)

- [ ] ALL training shards on the rig's 5TB HDD pool (4.5T free, verified)
      BEFORE the swap; zero network mounts in the training loop
- [ ] sha256 manifest per tar shard; verify after rsync, before the gate
- [ ] READBACK GATE: 2-min dummy-DataLoader bench must sustain 10× the needed
      samples/s from the HDD
- [ ] Dataloader skips + logs corrupt samples — one bad shard must never kill
      day 9 of the run
- [ ] Verify base-model weights + tokenizer/processor are rig-local too

## D. Teacher upgrade — Qwen3-VL-Embedding-8B ✅ verified real

`Qwen/Qwen3-VL-Embedding-8B` exists (official, 1.3M downloads), text+image
multimodal embedding, sentence-transformers + GGUF community quants available.
The 2B sibling is a throughput option for bulk mining.

- [ ] Download 8B (and 2B); smoke the embedding path on the 3080 Ti
      (sentence-transformers first; GGUF/llama.cpp as fallback)
- [ ] **Re-derive similarity band thresholds on ~1k samples — the old text
      teacher's 0.75–0.92 band does NOT transfer to a different model's
      sim distribution**
- [ ] Keep v001's existing pairs as mined (old teacher); no retroactive
      re-banding of frozen assets
- [ ] Mine image↔text bands + ANN hard negatives with the VL teacher
- [ ] Audio: accept there is NO tri-modal embedding teacher — audio lanes use
      dataset ground truth + constructed negatives (§E)

## E0. BUILD THE TRI-MODAL CARD DATASET — ASAP, quality-gated ⚡

Priority order from Sebastian 2026-07-11: this is the first big build task.
Principle: take the REAL datasets and fill the missing modalities with
generative tools — but **nothing enters the training set without passing a
gate**. Generated ≠ trusted.

Fill matrix (real → generated):

| Missing modality | Generator | Quality gate (automatic, per asset) |
|---|---|---|
| Audio from text | Kokoro TTS, multi-voice (in-house) | ASR round-trip (whisper) → WER ≤ 10% vs source text |
| Text from image | VLM captioner (gemma-4 local) | teacher embed-sim caption↔image-source-text in band; dedup |
| Image from text | rendered typographic card (KISS default); image-gen API only where rendering is semantically wrong | independent VLM captions the image → embed-sim to source ≥ threshold (round-trip) |
| Text from audio | ASR (whisper) | reverse: TTS the transcript? No — gate on ASR confidence + teacher band vs any existing text |

Gates that apply to EVERY card, generated or real:
- [ ] Round-trip check per generated asset (table above), thresholds derived
      on a 200-sample pilot BEFORE bulk generation (the proven v002 pattern)
- [ ] 200-sample human spot-check gate: Sebastian eyeballs a stratified
      sample before bulk mining is unleashed
- [ ] Teacher-band dedup + near-miss mining per modality (§D/§E)
- [ ] Provenance per asset: real|generated, generator model+version, source
      CAS sha256, rights tier inherited from source (SIGNOFF-001)
- [ ] Generated assets NEVER appear on the eval side — frozen evals use real
      media on at least one side (§F)
- [ ] Ledger the pass/reject rates per gate — if a gate rejects >30%, stop
      and investigate the generator instead of grinding through

## E. Tri-modal cards (proposed scheme: text + image + audio per card)

Adaptation map for datasets we already have:

| Source | Text | Image | Audio |
|---|---|---|---|
| v001 cards (40,941) | native | rendered card (typographic) | TTS multi-voice |
| MMEB | native | native (real photos) | TTS of caption |
| ColPali / VisRAG | query | native (real pages) | TTS of query |
| LibriSpeech / MLS | transcript | rendered transcript | **native (real speech)** |
| FSD50K | labels | — | **native (real env sound)** |

- [ ] Kokoro multi-voice TTS pipeline (already in-house from the gemma4 work)
- [ ] Anti-shortcut rules: ≥~35% of audio exposure is REAL audio;
      same-voice-different-text = hard NEGATIVE (kills the TTS-voice
      shortcut); same-text-different-voice = positive (voice invariance);
      real photos stay dominant over rendered-text images in the image lane
- [ ] Hard negatives per modality: text = teacher band; image = VL-teacher
      ANN; audio = constructed pairs above
- [ ] Provenance columns on every row; media by CAS sha256 (rights gate
      SIGNOFF-001 unchanged — training yes, release gated)
- [ ] **Lane mix starting point: 65% text / 17.5% image / 17.5% audio** —
      adopted from BidirLM-Omni's published recipe (MAEB rank-3 on ~300K
      audio-text pairs proves the audio lane is winnable at our data scale).
      Supersedes the earlier invented 40-50/30-35/15-25 target as the START;
      re-derive after the 200-sample pilot stays in force.
- [ ] Wave-2 audio candidate: **Laion-Audio-300M** (env-sound↔text,
      complements FSD50K) — VERIFY name/availability/rights first and
      acquire through CORPUS-ACQ conventions (sample a slice, not 300M);
      not in the CAS today.

### Parked (real but NOT low-hanging — do not chase in this window)

- Bidirectional attention + MNTP conversion (BidirLM's recipe): their own
  ablation shows contrastive training is the dominant term (contrastive
  alone beats Bi+MNTP by 13+ MTEB points; bidir adds +0.8–9 on specific
  tasks). For our 12B it means retraining attention behavior, switching
  pooling, and breaking comparability with fluffy-text-v0/G0. Revisit
  after the window if v2 plateaus.

## F. Eval integrity (so we don't fool ourselves)

- [ ] Per-lane FROZEN eval sets with REAL media on at least one side —
      a synthetic-only audio eval would measure the TTS shortcut, not hearing
- [ ] Fresh per-lane baselines BEFORE the swap → `ckpt-ratchet-v2.json`
- [ ] G0 text eval stays byte-frozen, results comparable across v1/v2

## G. Rig hygiene for 14 unattended days

- [ ] Disable unattended-upgrades / kernel + driver auto-updates for the
      window (a mid-run driver bump is a classic run-killer)
- [ ] Audit cron/systemd timers + docker for anything that grabs GPU or
      writes the root fs
- [ ] Watch firing checklist gains: df on all three storage tiers
- [ ] THE SWAP stays hard-gated on Sebastian typing **"restart approved"**
