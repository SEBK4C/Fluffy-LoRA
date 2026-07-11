# FLUFFY-EVAL agent brief — v1 wind-down + text-embedding benchmark

Authorized by Sebastian 2026-07-11 (~18:00Z): *"abandon the Text-only LoRA
training right now, keep the latest weights, test Gemma12b + text-embed-LoRA
on a text embedding benchmark, check it for learnings for the bigger Fluffy
run."* You execute exactly that, in this order. KISS is binding law.

READ FIRST: TRAINING-CHECKLIST.md (this repo) ·
/root/SYNTH-FORGE/FLUFFY-FORGE-BOOTSTRAP.md incl. GROUND-TRUTH ADDENDUM ·
/root/SYNTH-FORGE/state/OPERATOR-HANDOVER.md (roles + handshake files).
Tailnet/user connection details for the rig are in the PRIVATE addendum, not
in this public file — never commit them here.

## 1. Stop v1 (the authorization above IS the gate — don't re-ask)

- Read-only liveness check first (one-liner in the private addendum).
- Graceful stop: SIGTERM the torchrun parent, wait for any in-flight
  checkpoint write to finish (watch dir mtime settle), then confirm all
  trainer processes exited. SIGKILL only if SIGTERM hangs >5 min.
- NOTHING is deleted at this step.

## 2. Preserve the weights (before any cleanup — this is the hard rule)

- Newest checkpoint = **fluffy-text-v0**. Verify it loads (safetensors +
  PEFT config parse), rsync to PVE `/pool-ssd/synth-forge/ckpts/`,
  **sha256 match on both sides**, THEN it is preserved.
- Also preserve one mid-run checkpoint (step-1196 — it has an eval on
  record) for before/after comparisons.
- Push fluffy-text-v0 to HF `SEBK4C/Fluffy-LoRA` (rights-clean: trained on
  100% self-synthetic text). Model-card section must be honest: adapter from
  an abandoned 14-day run stopped at ~day 0.5 wall / step ~1400+, loss
  9.36→~1.7, retrieval on our frozen G0 unmoved (R@1 0.010 vs 0.008
  baseline), lr never left warmup. Alpha framing, no hype.

## 3. Coordination (other agents are watching this run)

- LEDGER.md in SYNTH-FORGE: append a clearly-labeled `V1-STOP` entry — NO
  iter number (those belong to the Opus watch).
- `echo "v1-stopped-by-sebastian $(date -u +%FT%TZ)" >>
  /root/SYNTH-FORGE/state/supervision-ack` (supervisor continuity).
- Leave a STANDBY note where the Opus watch will see it (its handover doc):
  trainer intentionally stopped, no health alarms, no more prunes; eval
  station freed.
- AFTER step 2 verification only: clear the rig checkpoint dir except the
  two preserved checkpoints (root fs was 84% — reclaim it), ledger the
  freed GB.

## 4. Benchmark — Gemma-4-12b-it ± fluffy-text-v0 on REAL text benchmarks

The rig is now free: run **bf16 on one 4090** (unquantized, fast). The
question is NOT leaderboard position — it's "did the LoRA learn anything
transferable, and is our G0 eval trustworthy?"

- Harness: `mteb` pip package. Embedding fn must byte-match training: last
  hidden state, last-token pooling, L2 norm, maxlen 512.
- Tasks (small, fits hours not days): retrieval SciFact + NFCorpus +
  FiQA2018; STS: STSBenchmark + STS17(en). Add our frozen G0 for continuity.
- Four contenders on identical harness: (a) base gemma-4-12b-it,
  (b) base + fluffy-text-v0, (c) **base + fluffy-text-v0 at HALF adapter
  strength** (scale the LoRA delta by 0.5 at load — free anti-forgetting
  probe, analogue of BidirLM's 50% base-merge trick), (d) reference:
  Qwen3-Embedding-8B teacher. Optional stretch: step-1196 on one task
  (does later training help?).
- Quantization-skew probe: repeat ONE task on the PVE 3080 Ti NF4 eval
  station — how much does NF4 shift scores? (Decides how much to trust the
  eval station for v2.)

## 5. Learnings → LEARNINGS-V1.md (this repo), answer these SPECIFICALLY

- (a) Did the LoRA move ANY external metric? If MTEB moves but G0 doesn't,
  **G0 is the suspect** — that changes v2's eval design and is the single
  most valuable possible finding.
- (b) Which task families moved (retrieval vs STS)? Direction and size.
- (c) How does base gemma-4 last-token embedding rank vs the teacher —
  is 0.008-on-G0 "model is raw" or "G0 is pathological"?
- (d) NF4-vs-bf16 skew number.
- (e) Throughput (samples/s, per task wall-time) → budget v2's eval cadence.
- (f) From train.log postmortem: lr never left warmup (STEPS=200k vs ~56k
  achievable) — quantify what schedule v2 should use.
- (g) Does half-strength (contender c) beat full strength anywhere? If yes
  → contrastive training is eroding base abilities → v2 should consider
  merge-back/regularization. Another free G0-diagnostic: if 0.5× helps on
  MTEB but not G0, that again points at G0.

## 6. Report + hygiene

- Report to Sebastian: one table (task × contender), a verdict sentence,
  top-3 learnings for the Fluffy v2 run.
- Commit everything to this repo (public!): no tailnet names, no usernames,
  no keys. LEARNINGS-V1.md + bench scripts + raw results JSON.
- No unledgered spend; CORPUS-ACQ pool untouched; SYNTH-FORGE stays archive.
