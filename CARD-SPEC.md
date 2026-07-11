# CARD-SPEC v0.1 — tri-modal training cards for Fluffy-LoRA

Design goal: one card = one semantic anchor with renditions ("views") in all
three modalities, stored **in the exact message format gemma-4's processor
consumes** — so mining-time format and training-time format cannot drift.
Draft for Sebastian's review; becomes frozen spec at v1.0.

## Two-layer design: CARDS vs EXPOSURES

- **CARD** = curated asset: semantic anchor + per-modality views + mined
  negatives + provenance. Expensive to make, gate-checked, append-only.
- **EXPOSURE** = a training example sampled FROM cards: (anchor view,
  positive view, k hard negatives). Cheap, regenerable — resampling new
  exposure mixes never requires re-mining cards.

## Card schema (JSONL, one card per line; media by CAS ref, never inline)

```json
{
  "card_id": "flf-000123",
  "anchor_text": "canonical semantic statement of the card",
  "views": {
    "text":  {"content": [{"type": "text", "text": "..."}],
              "source": "real|synthetic", "origin": "v001|mmeb|colpali|librispeech|fsd50k"},
    "image": {"content": [{"type": "image", "image": "cas://<sha256>"}],
              "source": "real|rendered|genai",
              "gen": {"model": null, "version": null},
              "gate": {"roundtrip_sim": 0.83, "pass": true}},
    "audio": {"content": [{"type": "audio", "audio": "cas://<sha256>"}],
              "source": "real|tts",
              "gen": {"model": "kokoro", "voice": "af_heart"},
              "gate": {"asr_wer": 0.04, "pass": true}}
  },
  "interleaved": [
    {"recipe": "c1-permute",
     "content": [{"type": "image", "image": "cas://..."},
                 {"type": "text", "text": "..."},
                 {"type": "audio", "audio": "cas://..."}]}
  ],
  "negatives": {
    "text":  [{"card_id": "flf-000456", "sim": 0.81, "miner": "teacher-band-v2"}],
    "image": [{"card_id": "flf-000789", "miner": "vl-ann", "judge": 0.35}],
    "audio": [{"card_id": "flf-000123", "view": "audio-alt-voice",
               "miner": "same-voice-diff-text"}]
  },
  "rights": {"tier": "cc-by|source_audit_required|self-synthetic",
             "source_sha256": "...", "audit": "pending|clear"},
  "dedup": {"protocol": "v1", "hash": "..."}
}
```

Why `content` arrays: that is gemma-4's native chat-template item format
(`{"type": "text"|"image"|"audio", ...}`). The training collate is literally
`processor.apply_chat_template(card.views[m].content)` — zero translation
layer, and interleaved views come for free (they're just longer content
arrays). Same trick BidirLM-Omni and Gemini Embedding 2 use for interleaved
inputs; ATIR confirms the two-stage InfoNCE recipe on audio-text interleave.

## Exposure schema (what shards actually contain)

```json
{"anchor": {"card": "flf-000123", "view": "audio"},
 "positive": {"card": "flf-000123", "view": "text"},
 "negatives": [{"card": "flf-000456", "view": "text"}, "... k=8 total"],
 "lane": "audio2text", "instruction": "Retrieve the matching description."}
```

- **k=8 hard negatives per exposure** (ATIR's working point) + in-batch
  negatives as before.
- Optional `instruction` field: instruction-conditioned embedding is now
  standard (Gemini Embedding 2, Qwen3-VL-Emb) — cheap to carry, decide at
  v2 smoke whether to train with it.

## The contrast structure (negatives ARE the curriculum)

| Contrast type | Example | Purpose |
|---|---|---|
| Cross-modal positive | card's audio ↔ same card's text | the alignment signal |
| In-modal hard negative | text of near-miss card (teacher band 0.75–0.92 re-derived) | fine-grained text discrimination |
| Cross-modal hard negative | image of ANN-nearest other card | cross-modal discrimination |
| Anti-shortcut negative | SAME TTS voice, different text; same render font, different text | kills speaker/font shortcuts |
| Permutation negative | interleaved view with swapped media from another card | interleave understanding (c1 pattern) |

Known trap (published, not just our hunch): MMEB's bundled distractors are
too easy — we mine our OWN negatives for every lane and never trust
dataset-shipped ones. Negative hardness gets a judge score where ambiguous
(local gemma-4 as MLLM-judge, UniME-V2 pattern) so false negatives (actual
matches mislabeled negative) get filtered.

## Eval alignment (decided by what now exists)

Per-lane frozen evals adopt MTEB-ecosystem task formats: **MAEB** subset for
audio, **MIEB-lite** subset for image, alongside our G0 text eval — so
Fluffy's numbers are directly comparable to published leaderboards, which
the paper needs anyway. Real media on at least one side of every eval pair
(unchanged rule).

## Storage

- Cards: JSONL manifests on pool-ssd (append-only, versioned card-v2.jsonl).
- Media: CAS by sha256 (existing CORPUS-ACQ discipline; generated media get
  their own CAS namespace + generator provenance).
- Shards: WebDataset tars = exposures + referenced media co-packed, staged
  rig-local (checklist §C gates unchanged).
```
