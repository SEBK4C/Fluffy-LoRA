# LEARNINGS-V1 — what the abandoned text-only run actually taught us

v1 (text-only QLoRA contrastive embedder on gemma-4-12b-it) was stopped by
operator decision 2026-07-11T15:58Z at step 1500 of a 14-day window (~9h wall).
Weights preserved as **fluffy-text-v0** (step-1449) + step-1196. This file
answers the six questions in EVAL-AGENT-BRIEF §5 with measured numbers.
Benchmark harness: `bench_mteb.py` (mteb 2.18.0), raw JSON in `results/`.

## ⚠ THE HEADLINE FINDING: v1's last-token pooling read PAD positions

`gemma-4-12b-it`'s tokenizer defaults to **left padding**; train.py /
ratchet_eval.py pool at `attention_mask.sum(1)-1`, which is correct only for
**right** padding. Under left padding that index lands on a padding position
for every sequence that isn't the longest in its batch. Empirical probe
(bf16, CPU, batch of 3 mixed-length texts):

- cos(two UNRELATED texts, v1 pooling in padded batch) = **0.96** — versus
  cos = 0.75 for their true last-token embeddings. The pooled vectors collapse
  toward a shared direction, mostly erasing text identity.
- cos(v1-pooled vs true embedding, same text) = 0.89–0.93 — weak leakage
  only; the longest sequence per batch (and any text with more real tokens
  than the batch's pad length) gets the true embedding by luck.
- Correct pooling under left padding (`h[:, -1]`) reproduces the true
  embedding at cos ≥ 0.994.

**Consequences**: v1 trained InfoNCE largely on pad-position vectors (loss
could still fall via length shortcuts and the correctly-pooled minority),
and G0's "baseline ≈ random" measured the broken pooling, not the model.
This supersedes the run's earlier "loss-eval decoupling" reading: the loss
and the eval weren't measuring the same embedding function's quality — the
embedding function itself was mostly reading padding. Phase-3 bench
(`--pooling lastpos`) measures base gemma with fixed pooling to establish
what the model can actually do raw.

## (a) Did the LoRA move ANY external metric?

TBD — table lands when the bench completes.

## (b) Which task families moved (retrieval vs STS)?

TBD.

## (c) Base gemma-4 last-token embedding vs the teacher — is G0's 0.008 "model raw" or "G0 pathological"?

TBD.

## (d) NF4-vs-bf16 skew

TBD. (Probe ran on a rig 4090 with the identical NF4 config, not the PVE
3080 Ti — the eval station's teacher could not be paused in this session. The
NF4-vs-bf16 delta is the dominant term; the 3080Ti-vs-4090 hardware residual
is bounded by comparing rig-NF4 G0 numbers to the eval station's on-record
NF4 G0 numbers: baseline 0.008 / step-1196 0.010.)

## (e) Throughput → v2 eval-cadence budget

TBD (per-task wall-times + texts/s from results JSON).

## (f) train.log postmortem — the schedule never left warmup

Measured from `logs/train-v1-final.log` (local only, gitignored):

- **21.2 s/step** measured over the whole run (1500 steps in 8h51m,
  2×4090 @ 300 W, batch 16×2 ranks×4 accum = 128 pairs/step, NF4 base,
  grad checkpointing, eager attention).
- **STEPS=200,000 was ~3.6× the 14-day window**: at 21.2 s/step the window
  yields ~57k steps. Warmup (3% of STEPS = 6,000 steps) alone is ~35 h.
- **At stop (step 1500) lr was 2.47e-5 = 25% of the 1e-4 peak.** Loss fell
  9.36 → ~1.7–2.1 (noisy plateau ~2.0 from step ~300) with retrieval flat —
  the entire observed run happened inside the first quarter of warmup.
- **Data: 224,474 pairs → 1,754 steps/epoch at 128 pairs/step. The run died
  at 0.86 epochs** — it never even saw the corpus once. For scale: the full
  57k-step window ≈ 32 epochs over v001 text pairs.
- Had the run gone the full 14 days on this schedule, cosine would only have
  decayed to ~0.82 of peak at step 57k — **the schedule could never anneal**,
  by construction. v1 was a data-throughput run, not a schedule-complete run.

### What v2 must do (quantified)

1. **Size STEPS to the measured window**: STEPS = window_seconds /
   measured_step_seconds (re-measure with image batches — they are slower
   than text). For a 14-day text-only window that is ~57,000, not 200,000.
2. **Warmup 2–3% of the REAL horizon** (~1,200–1,700 steps ≈ 7–10 h at v1
   step time), cosine annealing to ~0 exactly at the real end.
3. **Peak lr is untested territory**: v1 never ran above 2.5e-5, so 1e-4 peak
   is unvalidated. The loss plateau at lr 1–2.5e-5 with flat retrieval says
   the bottleneck was probably not lr but the objective/data (see (a)–(c)),
   but v2 should still not assume 1e-4 is safe — it was never reached.
4. Checkpoint every 30 min ≈ 6.3 GB/day on the root fs was the other landmine
   (TRAINING-CHECKLIST §A) — v2 saves to the big SSD with rolling retention.
