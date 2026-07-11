# Phase 4 pilot — 200 cards under FROZEN CARD-SPEC v1.0 (2026-07-12)

Stratification: v001 80 · MMEB 30 · ColPali 30 · LibriSpeech 30 · FSD50K 30.
Result: **200/200 validator-green** (722 media refs, interleaved order rule
enforced), **552/552 views tokenize** through the real gemma-4 processor.
85 tri-modal cards carry an `ita-base` interleaved view (image → text →
audio per the v1.0 hard rule). Manifest + media: `FLUFFY_CARDS_ROOT/pilot/`
(not committed — gated-rights sources).

## Gate rates (frozen rules)

| Gate | Rule (frozen) | Attempts | Pass | Reject |
|---|---|---|---|---|
| TTS round-trip | WER ≤ 0.15 AND teacher sim ≥ 0.90 | 140 | 97 (69%) | **31%** |
| Render round-trip | OCR sim ≥ 0.80 | 50 | 50 (100%) | 0% |

Per-stratum TTS: v001 70% · MMEB 77% · ColPali 60%. Cards whose TTS fails
ship without an audio view (never with a failing one); real-audio strata
(LibriSpeech, FSD50K) have audio on 30/30 by construction.

## The 31% line, honestly

E0 says >30% rejection on a gate = stop and investigate the generator.
TTS rejection is 31% — at the line, not under it. It WAS investigated
(GOLDEN-REPORT.md + DECISIONS-CARDSPEC.md item B): root cause is the
no-espeak Kokoro Q4's grapheme-to-phoneme on technical vocabulary, and
Sebastian explicitly chose B1 (keep the generator, gate hard) at freeze.
ColPali is the weakest stratum (60%) — its queries are the most
technical-vocabulary-dense. If bulk mining needs higher audio yield, the
espeak-ng rebuild (decision B2) is the known upgrade path; threshold
loosening is not.

## Spot-check

`make_spotcheck.py` produces the stratified eyeball sample
(3 cards per stratum, media embedded, gate values shown):
`pilot/spotcheck.html`. **Sebastian's sign-off on that sample gates bulk
mining** — the builder starts from the frozen spec + this kit, not before.

---

# v1.1 amendment results (2026-07-12)

## Supertonic-3 pass-rate check (amendment item 1)

Same 140 stratified texts as the Kokoro run above, same frozen A3 gate
(WER ≤ 0.15 AND teacher sim ≥ 0.90), voices F1/F3/M1/M4 round-robin.
Full per-clip data: `pilot/supertonic_pilot.json`.

| Generator | overall | v001 | mmeb | colpali |
|---|---|---|---|---|
| kokoro-82m (no-espeak Q4, secondary) | 69% | 70% | 77% | 60% |
| **supertonic-3 (primary)** | **88%** | 95% | 83% | 73% |

Distributions: WER median 0.000 / p90 0.125; sim median 0.988 / p10 0.911.
The sim distribution is SHIFTED UP relative to Kokoro (per-origin medians
0.93–1.00 → 0.99) — reported per the amendment; thresholds untouched.
Supertonic largely fixes the technical-vocabulary G2P failures that
dominated Kokoro's rejects (v001 70% → 95%). ColPali remains the hardest
stratum (73%) — residual failures are enumeration-style queries, a text
property, not a generator one.

## Figure+caption composite smoke (amendment item 2)

`cardlib.render_figure_caption` + strip-cropped OCR gate
(`cardlib.composite_roundtrip`): 6/6 pass on golden MMEB photos + ColPali
pages (caption_frac 0.100–0.130, sims 0.89–1.00). Implementation note,
honestly: the OCR detector misses small caption strips on busy FULL
canvases (one MMEB photo OCR'd empty), so the gate crops to the caption
strip — whose boundary is known exactly from `gen.layout.fig_h` — making
the gate measure caption legibility, which is its job. Validator enforces
`caption_frac` ∈ [0.10, 0.20] (negative-tested: over/under/missing all
caught).

## Sign-off state (amendment item 4)

- **Text + image lanes: bulk mining APPROVED** (Sebastian, 2026-07-12).
- **Audio lane: UNBLOCKED** by the Supertonic result above
  (88% ≥ the 69% bar set in the amendment).
