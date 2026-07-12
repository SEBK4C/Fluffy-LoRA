# LEARNINGS-V1 — what the abandoned text-only run actually taught us

v1 (text-only QLoRA contrastive embedder on gemma-4-12b-it) was stopped by
operator decision 2026-07-11T15:58Z at step 1500 of a 14-day window (~9h wall).
Weights preserved as **fluffy-text-v0** (step-1449) + step-1196. This file
answers the six questions in EVAL-AGENT-BRIEF §5 with measured numbers.
Harness: `bench_mteb.py` (mteb 2.18.0), raw JSON in `results/`, all runs
2026-07-11/12 on 2×RTX 4090 (bf16, sharded) unless marked NF4.

## ⚠ THE HEADLINE FINDING: v1's last-token pooling read PAD positions

`gemma-4-12b-it`'s tokenizer defaults to **left padding**; train.py /
ratchet_eval.py pool at `attention_mask.sum(1)-1`, which is correct only for
**right** padding. Under left padding that index lands on a padding position
for every sequence that isn't the longest in its batch. Empirical probe
(bf16, batch of 3 mixed-length texts):

- cos(two UNRELATED texts, v1 pooling in padded batch) = **0.96** — versus
  cos = 0.75 for their true last-token embeddings. The pooled vectors collapse
  toward a shared direction, mostly erasing text identity.
- cos(v1-pooled vs true embedding, same text) = 0.89–0.93 — weak leakage
  only; the longest sequence per batch (and any text with more real tokens
  than the batch's pad length) gets the true embedding by luck.
- Correct pooling under left padding (`h[:, -1]`) reproduces the true
  embedding at cos ≥ 0.994.

**Consequences**: v1 trained InfoNCE largely on pad-position vectors (loss
could still fall via length shortcuts and the correctly-pooled minority), and
G0's "baseline ≈ random" measured the broken pooling, not the model. This
supersedes the run's earlier "loss-eval decoupling" reading: the embedding
function itself was mostly reading padding — during training AND every eval.

## The table (task × contender)

nDCG@10 for retrieval, Spearman for STS, R@1/R@5 for G0 (pool 1500, q 3000).
"fixed" = `--pooling lastpos` (correct); "mean" = masked mean pooling.

| Task | base | +v0 (lora) | base fixed | lora fixed | base mean | base NF4 | lora NF4 | teacher |
|---|---|---|---|---|---|---|---|---|
| SciFact | 0.000 | 0.004 | 0.002 | 0.002 | 0.001 | 0.001 | 0.001 | **0.788** |
| NFCorpus | 0.013 | 0.010 | 0.009 | — | — | — | — | **0.414** |
| FiQA2018 | 0.000 | 0.000 | — | — | — | — | — | **0.612** |
| STSBenchmark | 0.021 | 0.034 | 0.035 | — | 0.027 | — | — | **0.935** |
| STS17 (en-en) | 0.357 | 0.295 | 0.315 | — | 0.342 | — | — | **0.957** |
| G0 R@1 | 0.011 | 0.017 | 0.014 | **0.030** | 0.021 | 0.009 | 0.013 | **0.312** |
| G0 R@5 | 0.028 | 0.049 | 0.037 | **0.090** | 0.053 | 0.020 | 0.044 | **0.573** |

Stretch: step-1196 (bf16, v1 pooling) SciFact 0.0014 vs step-1449's 0.0043 —
later training changed nothing, as expected under broken pooling.
**Harness validity**: the teacher (Qwen3-Embedding-8B, its own card protocol)
reproduces its published MTEB scores on this exact pipeline — the gemma floor
is a real measurement, not an artifact.

## (a) Did the LoRA move ANY external metric?

**No.** Every external task is at the floor for base and +LoRA alike; deltas
(±0.004 nDCG, ±0.06 Spearman) are noise-level at these magnitudes. The one
real signal: under FIXED pooling the adapter doubles G0 (R@1 0.014→0.030,
R@5 0.037→0.090) — the run did learn something about the synthetic card
domain even through the corrupted training signal, but nothing transferable.
The brief's conditional ("if MTEB moves but G0 doesn't, G0 is the suspect")
resolved the other way: NOTHING moved, and the suspect was the embed fn.

## (b) Which task families moved (retrieval vs STS)?

Neither, meaningfully. Retrieval: floor everywhere (0.000–0.013). STS: the
nonzero STS17 scores (0.29–0.36 for every gemma variant incl. base) are what
anisotropy + length correlation give you for free; the adapter actually
*lowered* STS17 under v1 pooling (0.357→0.295). No family benefited.

## (c) Base gemma-4 vs the teacher — "model is raw" or "G0 pathological"?

**Model is raw — decisively.** With correct pooling and real benchmarks, raw
gemma-4 still scores ~0 (SciFact 0.002 vs teacher 0.788). Neither last-token
nor mean pooling rescues it: a decoder LM needs contrastive adaptation to be
an embedder at all — which is the Fluffy mission's premise, so this is
good news executed badly. **G0 is exonerated**: it tracks the external
benchmarks at every point, and the teacher reaches R@1 0.312 / R@5 0.573 on
it — G0 is HARD (near-duplicate canonicals cap the ceiling well below 1.0)
but it has real dynamic range (0.008 broken → 0.31 teacher) and stays as the
frozen text-lane Δ-instrument for v2.

## (d) NF4-vs-bf16 skew

Small and one-directional at floor levels: G0 R@1 drops 0.0113→0.0087 (base)
and 0.017→0.013 (lora) going bf16→NF4; SciFact shifts ≤0.003. Crucially,
**rig-NF4 reproduced the 3080 Ti eval station's numbers within 0.001**
(0.0087 vs the station's on-record 0.008 baseline) — the station is
trustworthy for v2. Probe ran on a 4090 (identical NF4 config) because the
station's teacher couldn't be paused in this session; the G0 cross-check
bounds the hardware residual at ~0.001. Practical rule for v2's ratchet:
treat cross-precision comparisons as having ±0.005 slack; same-precision
comparisons keep eps = max(2σ, 0.002).

## (e) Throughput → v2 eval-cadence budget

- gemma-12b bf16 sharded across 2×4090, batch 16, maxlen 512: **~7 texts/s**
  on benchmark corpora (full 5-task suite ≈ 2–2.75 h per contender; FiQA2018's
  57k docs alone ≈ 1.7–2.3 h); short G0 texts 32–43 texts/s → **G0 eval ≈
  2.5 min** on a free 4090, ~5 min on the station (NF4, teacher paused).
- Teacher 8B on one 4090: ~2× faster (full suite 66 min).
- Budget guidance: per-checkpoint G0-style lane evals are nearly free
  (minutes) — keep the 6 h cadence or tighten; full-MTEB checks are ~2.5 h —
  run at milestones only (e.g., day 3 refresh, mid-run, pre-release), and
  SciFact alone (~15 min) is a fine external canary between milestones.

## (f) train.log postmortem — the schedule never left warmup

Measured from the final train.log (local copy, gitignored — contains paths):

- **21.2 s/step** over the whole run (1500 steps in 8h51m, 2×4090 @ 300 W,
  128 pairs/step, NF4 base, grad ckpt, eager attention).
- **STEPS=200,000 was ~3.6× the window**: 14 days at 21.2 s/step ≈ 57k steps.
  Warmup (3% of STEPS = 6,000 steps) alone is ~35 h.
- At stop (step 1500) lr was 2.47e-5 = **25% of the 1e-4 peak**; loss fell
  9.36→~1.7–2.1 (noisy plateau ~2.0 from step ~300) with retrieval flat.
- **224,474 pairs → 1,754 steps/epoch: the run died at 0.86 epochs** — it
  never saw the corpus once. The full window ≈ 32 epochs.
- Had it run 14 days, cosine would only have decayed to ~0.82×peak — the
  schedule could never anneal, by construction.
- NOTE: because of the pooling bug, v1 says nothing about lr — the schedule
  findings are logistics lessons, not optimization lessons.

### What v2 must do (the checklist)

1. **Embed-fn smoke test BEFORE any training step** (the new hard gate):
   batch one short + one long text, require cos(batched, solo) ≥ 0.99 per
   text; assert the pooled position is a real token. Same test inside the
   eval harness. This single check would have saved the entire run.
2. Pool `h[:, -1]` under left padding (or force `padding_side="right"` and
   keep mask-sum pooling — pick one, test it, pin it).
3. Size STEPS to the measured window (re-measure step time with image
   batches); warmup 2–3% of the REAL horizon; cosine anneals to ~0 at the
   real end. ~57k steps for a 14-day text run at v1 speed.
4. Fresh start over fluffy-text-v0 is now empirically mandatory (adapter
   trained on pad vectors), not just provenance hygiene.
5. Keep G0 frozen as the text-lane Δ-instrument; add SciFact (~15 min) as an
   external canary at checkpoint-eval time; report Δ-over-baseline only.
6. Eval station (NF4/3080 Ti) approved for v2 lane evals; ±0.005 slack across
   precisions.

## Verdict (one sentence)

fluffy-text-v0 learned nothing transferable because v1 trained and evaluated
through a pooling bug that mostly read padding tokens; the benchmark proved
the harness sound, the eval station trustworthy, G0 hard-but-valid, and raw
gemma-4 unusable as an embedder without the contrastive adaptation v2 will
now do with a verified embedding function.
