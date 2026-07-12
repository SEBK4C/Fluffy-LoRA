# CARD-SPEC v1.1 — tri-modal training cards for Fluffy-LoRA

**FROZEN 2026-07-12** (Sebastian's decisions A3, B1, C–H yes, per
`DECISIONS-CARDSPEC.md`; H in its amended interleaved-friendly form).
**v1.1 additive amendment 2026-07-12 (Sebastian)** — see "v1.1 amendment"
below; gates and all v1.0 rules unchanged. Frozen means frozen: changes
require a new version and a new pin in `cardkit/FREEZE.sha256`.

## v1.1 amendment (additive only)

1. **TTS generator set** (supersedes decision B1): **Supertonic-3 PRIMARY**
   (`Supertone/supertonic-3`, 99M ONNX via `pip install supertonic`,
   voices F1–F5/M1–M5; MIT SDK + OpenRAIL-M weights, license reading in
   MERGE-RESEARCH §6) + **kokoro-82m SECONDARY** (generator diversity =
   free anti-shortcut axis). The A3 gate is unchanged and applies per-clip
   regardless of generator. Provenance via existing `gen.model`/`gen.voice`.
   Bulk audio unblocks when Supertonic's standard 200-sample pilot pass
   rate is published (expect ≥ Kokoro's 69%; a shifted sim distribution is
   reported, never re-tuned into the frozen thresholds).
2. **New alt rendition, document lane: `image-captioned`** — figure+caption
   composite (paper-figure realism). Normative layout rules
   (TRAINING-CHECKLIST §E0.1): caption strip **10–20% of canvas height**
   (recorded as `gen.layout.caption_frac`, validator-enforced), font size
   proportional to canvas width, enforced margins, **wrap-never-truncate**;
   the OCR round-trip ≥ 0.80 gate backstops any clipping that slips
   through. Doc-lane exposures only — NEVER the sole photo-lane image↔text
   positive (OCR shortcut). Reference renderer:
   `cardlib.render_figure_caption`.
3. **Two-tier clarification** (gate-bypass prevention): NOISY warmup pairs
   (FLUX.1-schnell loose image↔text, §E0.1) are **NOT cards** — they are
   exposure-style warmup shards outside the card store: no card IDs, never
   mined for negatives, never on the eval side, provenance still stamped.
   Card gates apply to the GATED tier only, unchanged.
4. **Sign-off recorded**: Sebastian approves bulk mining for the **text and
   image lanes now**; bulk **audio** unblocks when the Supertonic
   200-sample pilot passes (item 1).

## v1.1a amendment (additive only, 2026-07-12 — PENDING ratification)

**Origin-enum extension forced by the data-breadth pivot** (Sebastian's
12:03Z order postdates the freeze; the frozen enum anticipated audio
wave-2 sources but no broad-text sources). `views.*.origin` enum gains
two values: `kalm`, `allnli`. Nothing else changes: all v1.0/v1.1 cards
remain valid, all gates identical. Subset-level provenance rides in
`native_id` (`<subset>:<qid>`), matching MINE-IMAGE's convention of
top-level origins. New pin in `cardkit/FREEZE.sha256`. Posted to
T9-STATUS for Opus-manager + Sebastian ratification; a veto only requires
re-validating cards (provenance is recorded identically either way).

Design goal: one card = one semantic anchor with renditions ("views") in all
three modalities, stored **in the exact message format gemma-4's processor
consumes** — so mining-time format and training-time format cannot drift.

v0.2: format reality-tested against the actual gemma-4-12b-it processor
(transformers 5.13.1, snapshot `0e2b105`) — see "Measured reality" below.
Where the draft disagreed with the processor, the processor won.
v1.0: gate thresholds fixed from 200-card pilot measurements; interleaved
modality-order rule; TopK-PercPos negative-band metadata.

## Frozen gates & mining rules (v1.0 decisions)

| Rule | Frozen value | Basis |
|---|---|---|
| TTS gate | **WER ≤ 0.15 AND teacher round-trip sim ≥ 0.90** (whisper-small transcript vs source text) | A3: 67% yield at pilot; rejects semantic garble, admits minor mispronunciation |
| TTS generator | no-espeak Kokoro Q4 stays for v2 (B1) | yield acceptable under A3; espeak rebuild is a later upgrade path |
| Render gate | **OCR round-trip sim ≥ 0.80** | C: pilot n=50 min 0.814, median 0.977 |
| Negatives per exposure | **k = 8** + in-batch (D) | ATIR working point |
| Negative band | **TopK-PercPos**: per query, negative ceiling = 0.95 × that query's positive sim (NV-Retriever false-negative filter), recorded per negative as `band_rule` | H |
| `instruction` field | carried in exposures, OFF by default, A/B at v2 smoke (E; expected to land ON per MERGE-RESEARCH §2D) | |
| ColPali queries | instruction boilerplate stripped at mining time (G) | 11/41 pilot queries carried it |
| Interleaved exposures | first-class MINORITY lane in the v2 base mix, incl. permutation negatives; single-modality views remain the bulk (H, amended) | Gemma 4 interleaved pretraining |
| **Interleaved modality order** | **image → text → audio, validator-enforced** (all image items before all text items before all audio items) | Gemma 4 pretraining convention |

## Two-layer design: CARDS vs EXPOSURES

- **CARD** = curated asset: semantic anchor + per-modality views + mined
  negatives + provenance. Expensive to make, gate-checked, append-only.
- **EXPOSURE** = a training example sampled FROM cards: (anchor view,
  positive view, k hard negatives). Cheap, regenerable — resampling new
  exposure mixes never requires re-mining cards.

## Card schema (JSONL, one card per line; media by CAS ref, never inline)

Normative schema: `cardkit/card.schema.json` (JSON Schema 2020-12), enforced
by `cardkit/validate_card.py`. A real, gate-passed example card:
`cardkit/example_card.json`. Shape:

```json
{
  "card_id": "flf-000123",
  "anchor_text": "canonical semantic statement of the card",
  "views": {
    "text":  {"content": [{"type": "text", "text": "..."}],
              "source": "real|synthetic", "origin": "v001|mmeb|colpali|librispeech|fsd50k|...",
              "native_id": "id within the source"},
    "image": {"content": [{"type": "image", "image": "cas://<sha256>"}],
              "source": "real|rendered|genai",
              "gen": {"model": "pil-typographic-card", "version": "v1"},
              "gate": {"roundtrip_sim": 0.83, "ocr": "rapidocr", "pass": true}},
    "audio": {"content": [{"type": "audio", "audio": "cas://<sha256>"}],
              "source": "real|tts",
              "gen": {"model": "kokoro-82m-gguf", "voice": "af_heart"},
              "gate": {"asr_wer": 0.04, "roundtrip_sim": 0.97, "pass": true}},
    "audio-alt-voice": {"...": "alt renditions are extra keys: modality prefix + suffix"}
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
  "rights": {"tier": "source_audit_required", "license": "...",
             "audit": "pending|clear", "redistribution_ok": false},
  "dedup": {"protocol": "anchor-sha256-v1", "hash": "<sha256>"}
}
```

Schema decisions the reference build forced (v0.1 → v0.2):
- **`views` is a named map, not a fixed 3-key object.** Alt renditions
  (`audio-alt-voice` — the same-text-different-voice positive and the
  same-voice-different-text negative both need one) don't fit 3 fixed keys.
  Canonical keys `text`/`image`/`audio` are the default exposure views;
  suffixed keys (`^(text|image|audio)-[a-z0-9-]+$`) are alt renditions.
- **`rights.tier` uses CORPUS-ACQ's existing enum** (`commercial`,
  `commercial_after_attribution`, `source_audit_required`, `research_only`,
  `evaluation_only`, `quarantine`) instead of v0.1's invented three-value
  enum — one rights vocabulary across programs, and SIGNOFF-001 semantics
  carry over unchanged.
- **`source` covers the whole E0 fill matrix**: `real | synthetic |
  rendered | tts | genai | captioned | asr`. Every non-`real`/`synthetic`
  view MUST carry `gen` + a passing `gate` (validator-enforced; a failed
  gate never ships — the asset is dropped or regenerated, the card keeps
  its other views).
- **`native_id`** per view: provenance back to the source dataset row
  (utterance id, image filename, v001 card id).

Why `content` arrays: that is gemma-4's native chat-template item format
(`{"type": "text"|"image"|"audio", ...}`). The training collate is:

```python
processor.apply_chat_template(
    [{"role": "user", "content": resolve_cas(card.views[m].content)}],
    tokenize=True, return_dict=True)
```

Two facts the v0.1 draft got wrong, verified against the real processor:
the template REQUIRES the role wrapper (bare content arrays raise), and
`cas://` refs are not loadable — `resolve_cas` swaps each `cas://<sha256>`
for a local path (or PIL.Image / 16 kHz numpy array; all three are accepted).
That resolve is the ONLY translation layer; interleaved views still come for
free (they're just longer content arrays). Same trick BidirLM-Omni and
Gemini Embedding 2 use for interleaved inputs; ATIR confirms the two-stage
InfoNCE recipe on audio-text interleave.

## Measured reality (Phase 1, `cardkit/probe_chat_template.py`)

Probed on the 3080 Ti host, gemma-4-12b-it snapshot `0e2b105`, transformers
5.13.1. Full numbers in `cardkit/probe_report.json`.

| Item | Measured |
|---|---|
| Turn overhead | 6 tokens (`<bos><|turn>user\n` … `<turn|>\n`) |
| Image view | **256–266 soft tokens + 2 delimiters** (resolution-dependent; `max_soft_tokens: 280` is a ceiling, not a constant — budget 280, expect ~268) |
| Audio view | **exactly 25 soft tokens/sec** (40 ms/token, 16 kHz) + 2 delimiters |
| Interleaved | exact sum of parts — zero separator overhead |
| Payload forms | path, http(s) URL, base64, PIL.Image, raw numpy (audio presumed 16 kHz) |

**Hard rules the measurements force:**
- **Audio views ≤ 30 s.** The model's `audio_seq_length` is 750 tokens but the
  processor does NOT truncate (a 45 s clip emits 1125 audio tokens and would
  blow past the audio tower's budget at forward time). The validator enforces
  the cap; longer sources get clipped or split at mining time.
- **CAS audio is stored 16 kHz mono WAV.** Raw numpy arrays are presumed
  16 kHz by the processor; storing anything else invites silent resample bugs.

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
| In-modal hard negative | text of near-miss card (TopK-PercPos: ceiling = 0.95 × the query's positive sim, per H) | fine-grained text discrimination |
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
  their own CAS namespace + generator provenance). Reference layout
  (implemented): `$FLUFFY_CARDS_ROOT/cas/sha256/<2-prefix>/<sha>`, manifests
  under `golden/` and `pilot/`. CAS audio is 16 kHz mono WAV (hard rule
  above); images keep their source encoding.
- Shards: WebDataset tars = exposures + referenced media co-packed, staged
  rig-local (checklist §C gates unchanged).
```
