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
