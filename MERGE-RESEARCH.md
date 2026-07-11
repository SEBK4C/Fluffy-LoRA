# MERGE-RESEARCH — external-model research → Fluffy v2 architecture

Research session 2026-07-11 (Sebastian + research window). Every claim below
was web-verified this date; sources at the bottom. This document is the
**architecture handoff to the builder window** — decisions marked RATIFIED
are settled with Sebastian; PROPOSED items need discussion before build.

## 1. The landscape converged on our recipe

Three independent 2025–26 releases build omnimodal embedders the same way —
take a *native* omni generative backbone, keep the understanding half, train
it contrastively into one shared space:

| Model | Backbone | Method | Evidence |
|---|---|---|---|
| LCO-Embedding-Omni-7B | Qwen2.5-Omni Thinker | LoRA + contrastive, last-token pool | NeurIPS 2025; **MAEB #1** (Borda), MIEB SOTA |
| Omni-Embed-Nemotron-3B | Qwen2.5-Omni Thinker (Talker dropped) | contrastive | NVIDIA paper 2510.03458 |
| BidirLM-Omni-2.5B | 3× Qwen3-1.7B variants, weight-merged | bidir+MNTP + contrastive | arXiv 2604.02045; no verifiable eval tables |

**Fluffy v2 is this exact recipe on gemma-4-12b-it — untried at 12B scale.
We are filling the obvious next row of the table, not inventing a method.**

Most striking external result: LCO is **SOTA on MAEB without training on any
audio** — contrastive training on text/image transferred to audio through the
backbone's already-aligned encoders ("language-centric" transfer). Our
no-audio-teacher problem is smaller than feared: the shared space does much
of the lifting; real-audio data mostly needs to *calibrate*, not teach.

## 2. Architecture decisions — RATIFIED

**A. Attention stays CAUSAL; pooling stays LAST-TOKEN.**
The 2024 "flip to bidirectional" recipe (LLM2Vec, NV-Embed, NV-Retriever,
BidirLM) did NOT win out. 2026 evidence: Qwen3-Embedding-8B (top open text
embedder) is causal+last-token; LCO (omni SOTA) uses last-token pooling with
no mask-flip mentioned; Causal2Vec (2507.23386) argues mask-flipping
"undermines LLMs' ability to extract semantic information acquired during
pre-training"; NV-Retriever asserted bidir superiority with **no ablation
table**; BidirLM's own ablation shows contrastive training dominates
(+13 MTEB) over its Bi+MNTP stage. Bidir is CLOSED for v2 — revisit only as
a cheap A-series ablation if v2 retrieval stalls unexplainably.
(Bonus: v1/G0 embedding-fn comparability is preserved, though that was not
the deciding factor.)

**B. Output dim = native 4096, no projection head, + Matryoshka (MRL).**
gemma-4-12b hidden dim is 4096 (Sebastian-confirmed). No projection layer —
fewer moving parts, nothing extra to distill through. MRL gives smaller dims
on demand: same InfoNCE summed over nested prefix slices, uniform weights,
L2-normalize each prefix at use time.
- Ladder (fixed at training time): **4096 → 2048 → 1024 → 512 → 256**
- The 2048 rung = direct comparability with LCO / Nemotron / BidirLM
  (2048 is the field convention — thrice-confirmed across the table above)
- 256 rung = cheap ANN mining / bulk dedup
- Precedent: OpenAI text-embedding-3, Gemini Embedding, Qwen3-Embedding all
  ship MRL truncation.

**C. Card encoding: single-modality views are the BULK; interleaved views
are a first-class MINORITY lane (CORRECTED 2026-07-12).**
First writing of this section over-generalized Omni-Embed-Nemotron: their
separate-stream finding is scoped to **time-synchronized audio+video TMRoPE
interleaving** — we have no video lane, so it barely applies. The stronger
fact cuts the other way: **Gemma 4 was pretrained on interleaved modalities
under a fixed ordering convention (image BEFORE text, audio AFTER text),
and the 12B is Google's unified encoder-free variant ingesting raw
audio/image patches** (Gemma 4 tech report, 2607.02770). Interleaved
sequences are the backbone's native pretraining distribution; an embedding
recipe that never shows it interleaved input throws that away. Gemini
Embedding 2, BidirLM, and ATIR all embed interleaved.
- Bulk of v2 base exposures: canonical single-modality views (where teacher
  supervision and negative bands are well-defined).
- Interleaved exposures (incl. permutation negatives): minority lane in the
  v2 base mix, document-style, exact share re-derived at pilot.
- **Hard rule for interleaved views: follow the pretraining ordering
  convention — image → text → audio.**

**C2. PROPOSED (v2-recipe ablation, not card-spec): GRIT-style auxiliary
generative loss.** GRIT (arXiv 2402.09906, ICLR 2025; GritLM-7B) trains
generative next-token + contrastive embedding jointly at no loss to either;
LCO's Generation-Representation Scaling Law is the matching theory
(generative capability upper-bounds representation quality). Applied to
Fluffy: a small next-token term on the SAME interleaved cards alongside
InfoNCE would preserve the backbone's native interleaved competence instead
of letting contrastive-only training erode it. No multimodal GRIT exists in
the literature — same open-row situation as our main recipe. Cost: one
extra loss term, same data; needs an A-series smoke before it earns a place.

**D. Instruction prompts at encode time.**
Field standard (Qwen3-Embedding, LCO, BidirLM all do it). v2's embedding fn
gets a short task instruction prefix; train/eval must byte-match. Exact
strings to be fixed in the v2 spec before launch.

**E. Hard-negative mining gets the TopK-PercPos false-negative filter.**
NV-Retriever's actual contribution (their 51.4→60.5 NDCG@10 jump was mining,
not architecture): when mining hard negatives with the teacher, **drop any
candidate whose sim-to-query exceeds 95% of the positive's sim** — it is
probably an unlabeled positive that would poison InfoNCE. This anchors the
§D re-banding task: band ceilings derive from each query's own positive
score, not a global constant. Applies to every lane.

**F. Data mix: text-dominant, confirmed twice.**
BidirLM: 65/17.5/17.5 text/image/audio (already adopted as v2 start,
commit cd6d0cc). LCO independently: ~80% text / ~20% multimodal "to
calibrate the embedding space". Both agree: text carries the space,
multimodal calibrates it. Keep our anti-shortcut rules (real-audio floor,
same-voice-different-text negatives) regardless.

**G. Anti-forgetting: half-strength adapter as a free eval contender.**
BidirLM's 50% merge-back-with-base has a zero-cost LoRA analogue: evaluate
the adapter at 0.5 scale alongside full scale. (Already adopted, cd6d0cc.)

## 3. Teacher gate — audio-lane candidacy (lineup RATIFIED, gate PROPOSED)

No candidate earns anything by default. Gate: frozen G0 + 200-sample image
and audio retrieval probes, all candidates on the identical harness vs the
incumbent text/image teacher (Qwen3-VL-Embedding-8B). A candidate becomes a
teacher — audio lane only, where we have no teacher — iff it beats random
convincingly AND beats the incumbent where they overlap.

| Rank | Candidate | Standing |
|---|---|---|
| 1 | **LCO-Embedding-Omni-7B** | Apache-2.0, NeurIPS evals, MAEB #1, plain sentence-transformers, no trust_remote_code, GGUF quants exist |
| 2 | BidirLM-Omni-2.5B | Apache-2.0 but **zero verifiable evals**, weak cross-modal separation in own README demo, requires trust_remote_code |
| — | Omni-Embed-Nemotron-3B | **reference-only: "research and development only" license — never distill it into weights we release** |

Sandbox rule (binding, for any trust_remote_code model): disposable
container only — no tokens, no keys, no host access.

Note on LCO: attention mode is not explicitly stated in their README
(last-token pooling is confirmed from code). Verify causal-retained in their
paper during the gate before citing it as evidence anywhere formal.

## 4. Merge protocol (activation-difference shaping) — PROPOSED, not ratified

Sebastian's idea: run the same cards through an external model and ours, use
the embedding/activation differences to shape the Gemma4 LoRA.

Straight talk, now with external confirmation: a **true weight merge is
impossible** across different skeletons (2.5B/7B Qwen encoders vs 12B Gemma4
decoder — TIES/DARE/soups all require identical architecture). BidirLM
itself is the proof-by-demonstration: their merge worked *only because all
three donors were the same Qwen3-1.7B skeleton*.

The instinct maps to three real techniques, cheapest-first ladder:

0. **CKA/Procrustes probe** — embed ~1k paired cards in both models; measure
   shared structure (CKA; orthogonal Procrustes residual). Hours of work,
   no training. **Kill gate: low CKA → the spaces don't share enough
   structure; stop here, saved everything downstream.**
1. **RKD pilot (200 cards)** — add a relational-distillation term to
   InfoNCE: match the teacher's card×card similarity matrix. Dimension-free
   (sim matrices, not vectors) — the 4096-vs-2048 mismatch never matters.
   Was already in the unimplemented v1 full recipe.
2. **Feature distillation via small projector** — only if 0 and 1 both earn
   it; most machinery, weakest precedent at this scale.

Kill/keep criteria per rung: to be set in discussion before anything touches
the v2 recipe. Nothing here blocks the v2 build.

## 5. Open questions

- MRL loss weighting: uniform is the default; any evidence for weighted?
- Exact instruction-prompt strings per task family (fix before v2 launch).
- LCO paper deep-read: LoRA rank, data list, attention mode (during gate).
- Laion-Audio-300M rights tier (wave-2 acquisition, per cd6d0cc).

## Sources (all fetched 2026-07-11)

- BidirLM: https://arxiv.org/abs/2604.02045
- NV-Retriever: https://arxiv.org/abs/2407.15831
- Causal2Vec: https://arxiv.org/abs/2507.23386
- LCO-Embedding: https://github.com/LCO-Embedding/LCO-Embedding ·
  https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-7B
- Omni-Embed-Nemotron: https://arxiv.org/abs/2510.03458 ·
  https://huggingface.co/nvidia/omni-embed-nemotron-3b
- Nemotron Omni 3 docs (generative, bolted-encoder camp — low transfer):
  https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/omni3
- MAEB: https://arxiv.org/abs/2602.16008 · MMEB-V3: https://arxiv.org/abs/2604.23321
- MTEB 2026 state: https://www.codesota.com/benchmarks/mteb
