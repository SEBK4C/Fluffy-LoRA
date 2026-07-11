# CARD-SPEC v0.2 — tri-modal training cards for Fluffy-LoRA

Design goal: one card = one semantic anchor with renditions ("views") in all
three modalities, stored **in the exact message format gemma-4's processor
consumes** — so mining-time format and training-time format cannot drift.
Draft for Sebastian's review; becomes frozen spec at v1.0.

v0.2: format reality-tested against the actual gemma-4-12b-it processor
(transformers 5.13.1, snapshot `0e2b105`) — see "Measured reality" below.
Where the draft disagreed with the processor, the processor won.

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
