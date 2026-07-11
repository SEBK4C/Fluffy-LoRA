# Golden set build report — 2026-07-11

15 cards, 3 per §E source. Every generation gate run live (whisper-small
ASR WER + teacher round-trip sim for TTS; rapidocr + teacher embed-sim for
renders). Manifest + media live in `FLUFFY_CARDS_ROOT/golden/` (not
committed — gated-rights sources). Validator: **green, 45 media refs
checked**. Reference collate: **green, 43 views tokenized** through the real
gemma-4 processor.

## Gate outcomes (shipped views all pass; provisional thresholds)

Thresholds used: WER ≤ 0.10 (E0 target), render round-trip sim ≥ 0.60.

| Gate | Run | Passed | Measured range (passing) |
|---|---|---|---|
| TTS → whisper WER | 17 | 10 | WER 0.00–0.09; teacher ref↔transcript sim 0.87–1.00 |
| Render → OCR round-trip | 5 | 5 | sim 0.94–1.00 |

## Rejects (reject-and-advance, exactly what bulk generation will do)

7 TTS rejects out of 17 attempts (41%):

| Source | Rejects/attempts | WERs | Diagnosis |
|---|---|---|---|
| v001 | 3/6 | 0.11, 0.13, 0.14 | Kokoro (no-espeak Q4 GGUF) garbles technical vocabulary — "tomatoes"→"taas", "Quito–Cayambe"→"Quio K. Umb". Genuine audio defects, correctly rejected. |
| mmeb | 1/4 | 0.20 | Source caption contains typos ("form cincrete"); TTS of typo'd text can't round-trip. Gate catches upstream data noise. |
| colpali | 3/6 | 0.21, 0.22, 0.44 | Queries carry instruction boilerplate ("Your answer should be very brief.") and letter-list enumerations; both break TTS round-trip. Boilerplate should be stripped at mining time. |

The 41% aggregate is above the §E0 30% investigate-the-generator line —
investigated, root causes above. Consequences are decision points for the
spec freeze (better Kokoro build vs threshold vs source-text cleanup), not
grind-through.

## Token costs (reference collate, per view)

Text views 15–67 tok · image views 260–278 tok · audio views 29–474 tok
(25 tok/s) · interleaved (image+text+audio) 642 tok = exact sum of parts.

## Anti-shortcut structures exercised

- `flf-g001`: `audio-alt-voice` rendition (same text, bm_george vs af_heart)
  + `same-voice-diff-text` audio negative pointing at `flf-g003`.
- `flf-g002`: interleaved view, `c1-permute` recipe.
- All cards: teacher-kNN text negatives (k=2 within the golden set; sims
  recorded), cross-modal image negative where both sides have image views.
