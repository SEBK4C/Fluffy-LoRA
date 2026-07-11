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
| Gemini Embedding 2 (closed) | full Gemini, all params | **bidir flip + mean pool**, 2-stage NCE (PFT text/image/code → FT all modalities), Gemini-as-data-engine, model soup, 3072-dim + projection + MRL | tech report 2605.27295; proprietary SOTA, interleaved omnimodal |

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

> **Confidence note (2026-07-12, GE-2 tech report)**: Gemini Embedding 2 —
> the proprietary omni SOTA — flips to bidirectional + mean pooling. Live
> counter-evidence, logged honestly. It does not flip the decision because
> the regimes differ: GE-2 full-retrains ALL parameters from Gemini in two
> stages at Google batch sizes — exactly where re-learning attention pays
> off. Our budget-class analogue (LCO: LoRA-only, frozen backbone) stayed
> causal/last-token and took MAEB #1. Status is therefore
> **closed-for-LoRA-scale**: if Fluffy ever graduates from LoRA to full
> fine-tune, this reopens FIRST.

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

**H. Staged modality warmup (adopted from GE-2, 2026-07-12).**
GE-2 trains in two stages: pre-finetuning on text/image/code ONLY (huge
noisy batches, in-batch negatives — its stated purpose: convert parameters
from generation to encoding, stably), then fine-tuning where audio/video
enter for the first time with curated hard-negative triplets. This is the
THIRD independent confirmation of build-the-space-on-text-first (BidirLM
65% text mix; LCO ~80/20). v2 adopts the scheduling analogue: a text+image-
heavy warmup phase before the full tri-modal lane mix — pure data
scheduling, no new machinery. Warmup length/mix = builder smoke decision.
GE-2's attached warning, now binding on the pilot: cross-modality balance
"was sensitive to hyper-parameters like sampling rates" — lane rates are a
measured knob, never an assumed one (reinforces §2F's re-derive-at-pilot).

**I. Model soup at eval (adopted from GE-2, 2026-07-12).**
GE-2 finishes by averaging checkpoints within and across fine-tuning runs
(weighted, e.g. 2:1 base:finetuned). LoRA analogue is near-zero cost: at
eval time, also score (i) the average of the last few checkpoints' adapter
weights and (ii) weighted blends with base (kin to §2G's half-strength
contender). Add both to the v2 eval contender list; adopt only what the
frozen evals reward.

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

## 4. Merge protocol (activation-difference shaping) — RATIFIED 2026-07-12

Sebastian's idea: run the same cards through an external model and ours, use
the embedding/activation differences to shape the Gemma4 LoRA.

Straight talk, now with external confirmation: a **true weight merge is
impossible** across different skeletons (2.5B/7B Qwen encoders vs 12B Gemma4
decoder — TIES/DARE/soups all require identical architecture). BidirLM
itself is the proof-by-demonstration: their merge worked *only because all
three donors were the same Qwen3-1.7B skeleton*.

**How modern is this process (checked 2026-07-12)?** Ingredients are 2019
classics (CKA: Kornblith; RKD: Park), but the ladder is current practice:
EMO (EMNLP 2025) does embedding-model distillation via intra-model
relational distillation + CKA; the Platonic Representation Hypothesis
(2024, + 2026 "language is the attractor of multimodal convergence"
follow-up) is the theory predicting shared structure exists — and directly
supports our language-centric design; vec2vec (NeurIPS 2025) demonstrated
different encoders' spaces are translatable with NO paired data (universal
geometry). Two modernizations adopted below: report mutual k-NN/CKNNA
(local structure) alongside CKA (global), and probe PER-LANE (Kabra 2026:
some cross-modal pairs show near-random similarity even between good
models — pooled scores can hide a dead lane).

The instinct maps to three real techniques, cheapest-first ladder:

0. **Structure probe** — embed ~1k paired cards in both models; report
   **CKA (global) + mutual k-NN/CKNNA (local, k=10)**, computed **per lane**
   (text/image/audio + cross-modal) — not pooled. Hours of work, no
   training; rides along with the §3 teacher-gate run (model already
   loaded). Calibration anchors, computed from the same embeddings:
   - FLOOR: shuffled-pairing score (random-structure baseline)
   - CEILING: Gemma4-base vs Qwen3-Embedding text teacher on the text lane
     (two models KNOWN to share structure)
   **KILL: candidate lands nearer FLOOR than CEILING on the lanes we'd
   distill (esp. audio). PROCEED: nearer ceiling on ≥1 target lane —
   distill only the lanes that pass.**
1. **RKD pilot (200 cards)** — add a relational-distillation term to
   InfoNCE: match the teacher's card×card similarity matrix. Dimension-free
   (sim matrices, not vectors) — the 4096-vs-2048 mismatch never matters.
   Teacher sims precomputed offline; per-step overhead ≈ 0. Was already in
   the unimplemented v1 full recipe.
   **KEEP: some frozen-lane metric improves > eval epsilon (0.002) with no
   lane regressing beyond epsilon, vs the identical run with the term off.
   KILL otherwise — one A/B, no second chances in this window.**
2. **Feature distillation via small projector** — only if 0 and 1 both earn
   it; most machinery, weakest precedent at this scale. NOT in this
   window's scope regardless (vec2vec-style translators likewise noted as
   evidence, not as machinery to build).

Nothing here blocks the v2 build; rung 0 is scheduled WITH the teacher gate.

## 5. Open questions

- MRL loss weighting: uniform is the default; any evidence for weighted?
- Exact instruction-prompt strings per task family (fix before v2 launch;
  IMG-H2's extraction-prompt A/B feeds this).
- LCO paper deep-read: LoRA rank, data list, attention mode (during gate).
- Laion-Audio-300M rights tier (wave-2 acquisition, per cd6d0cc).

## 6. External targets + image-lane hypothesis slate (RATIFIED targets,
## PROPOSED hypotheses — Sebastian, 2026-07-12)

**Primary external target: MAEB SOTA.** Rationale: current #1 (LCO) won
without training on audio — the lane is nearly empty; our audio plan (real
speech + real env sound + gated TTS fill + voice-invariance negatives) on
the largest backbone in the race is where the alpha concentrates.
**Secondary target: the document/visually-rich slice of MIEB** (ViDoRe-
style) — decoder-based embedders already beat CLIP-style there, Gemma4's
OCR-heavy pretraining + our ColPali/VisRAG/rendered-card lanes concentrate
exactly there. All-MIEB and MTEB-text are calibration references, not
targets. Banked regardless of charts: first open-weights single-backbone
tri-modal embedding adapter at 12B.

Image-lane hypotheses (cheapest first; IMG-H1/H2 are eval-only and should
run on the currently-free rig BEFORE v2 training exists):

- **IMG-H1 — inherited capacity probe (zero training).** LCO's scaling law
  + Gemma4-12b's generative image scores (above every ≤7B backbone on the
  MIEB podium) predict the UNTRAINED base, with instruction prompt +
  last-token pooling, already lands respectably on parts of MIEB. Test:
  base model on 3–4 MIEB-lite tasks vs LCO/GME published numbers. Measures
  how much of Google's training we inherit free; maps the real gaps.
- **IMG-H2 — modality-gap extraction prompt (E5-V, arXiv 2407.12580,
  verified).** "…Summary of the above image in one word:" collapses the
  text↔image modality gap BEFORE training; E5-V then trained on text pairs
  ONLY at ~5% cost and beat image-text-trained baselines. Synergy: if the
  prompt closes the gap, our 65%-text mix + staged warmup transfer to
  images the way LCO's transferred to audio. Test: A/B extraction prompts
  on the IMG-H1 probe; measure cross-modal sim distributions + retrieval.
  Feeds the §2D instruction-string decision.
- **IMG-H3 — document beachhead.** Formalized as the secondary target
  above. Test: the MIEB doc subset lives in the frozen image eval as its
  own tracked line.
- **IMG-H4 — classification gap is a data problem.** MLLM embedders lose
  to CLIP on zero-shot classification/clustering, plausibly for lack of
  label-style positives. Fix in existing card machinery: label-text views
  ("a photo of {class}") from datasets we hold. Test: 200-card pilot lane +
  one small zero-shot classification task in the eval.
- **IMG-H5 — permutation negatives buy compositionality.** CLIP-family
  fails compositional tasks (bag-of-words); hard compositional negatives
  are the published fix; our contrast taxonomy already contains them.
  Test: one MIEB compositionality task in the frozen eval — check the
  taxonomy actually cashes in.

**API mining guidance (image lane)** — spend API on text ABOUT real
images, never on bulk image generation or bulk captioning:
1. Synthetic retrieval queries over REAL images (2–3 diverse queries per
   MMEB/ColPali image) — the documented GE-2/Gecko synthetic-data win;
   real media stays on the image side (§F satisfied by construction);
   pairs gate through the teacher band like everything else.
2. Compositional hard negatives for IMG-H5: minimally-wrong captions
   (single attribute/relation swap) for real images — hard for small local
   models, pennies via API.
Anti-rules: no bulk API captioning (local gemma-4 is free and
distribution-matched); image GENERATION stays E0's narrow exception (real
photos dominant); every API asset gets generator+version provenance
(SIGNOFF-001) and ledgered spend.

## Sources (all fetched 2026-07-11)

- BidirLM: https://arxiv.org/abs/2604.02045
- NV-Retriever: https://arxiv.org/abs/2407.15831
- Causal2Vec: https://arxiv.org/abs/2507.23386
- LCO-Embedding: https://github.com/LCO-Embedding/LCO-Embedding ·
  https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-7B
- E5-V (extraction prompt kills modality gap; text-only training):
  https://arxiv.org/abs/2407.12580
- MIEB (task-type breakdown behind IMG-H3/H4/H5):
  https://arxiv.org/abs/2504.10471
- vec2vec / universal geometry: https://arxiv.org/abs/2505.12540
- EMO embedding distillation (EMNLP 2025):
  https://aclanthology.org/2025.emnlp-main.385.pdf
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987 ·
  language-as-attractor follow-up: https://arxiv.org/abs/2605.09352
- Gemini Embedding 2 tech report: https://arxiv.org/abs/2605.27295
- Gemini Embedding (v1, staged recipe + soup + Gemini-as-data-engine):
  https://arxiv.org/abs/2503.07891
- GRIT / GritLM: https://arxiv.org/abs/2402.09906
- Gemma 4 tech report (interleaved pretraining, ordering convention,
  12B unified encoder-free): https://arxiv.org/abs/2607.02770
- Omni-Embed-Nemotron: https://arxiv.org/abs/2510.03458 ·
  https://huggingface.co/nvidia/omni-embed-nemotron-3b
- Nemotron Omni 3 docs (generative, bolted-encoder camp — low transfer):
  https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/omni3
- MAEB: https://arxiv.org/abs/2602.16008 · MMEB-V3: https://arxiv.org/abs/2604.23321
- MTEB 2026 state: https://www.codesota.com/benchmarks/mteb
