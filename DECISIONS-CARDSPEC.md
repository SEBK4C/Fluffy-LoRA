# CARD-SPEC freeze decisions — for Sebastian, 2026-07-11

Answer tersely ("A3, B1, yes, yes, yes, yes, yes" works). On your answers I
bump to v1.0, freeze, and pin the sha256. Evidence: `cardkit/GOLDEN-REPORT.md`
(15 golden cards, all gates live) and a provisional 200-card pilot
(`pilot_report.json` on pool-ssd; 200/200 validator-green, all 455 views
tokenize through the real gemma-4 processor).

## A. TTS gate rule (the big one)

E0's WER ≤ 10% rejects 39% of TTS clips at pilot scale — above the 30%
investigate line. Investigated: the no-espeak Kokoro Q4 garbles technical
vocabulary ("tomatoes"→"taas", "Quito–Cayambe"→"Quio K. Umb"). But teacher
round-trip sim shows most WER-0.10–0.15 clips are semantically intact minor
mispronunciations. Measured pass rates (n=140, stratified):

| Rule | overall | v001 | mmeb | colpali |
|---|---|---|---|---|
| A1: WER ≤ 0.10 (E0 as written) | 61% | 55% | 80% | 57% |
| A2: WER ≤ 0.15 | 80% | 79% | 83% | 80% |
| **A3: WER ≤ 0.15 AND sim ≥ 0.90 (rec)** | **67%** | 68% | 73% | 60% |
| A4: WER ≤ 0.10 OR sim ≥ 0.95 | 69% | 65% | 83% | 63% |
| A5: WER ≤ 0.20 AND sim ≥ 0.92 | 61% | 57% | 77% | 53% |

A3 admits WER-0.10–0.15 clips only when the transcript still embeds at
sim ≥ 0.90 to the source text (19 clips vs A1; the 5 worst-WER of them
inspected: WER 0.13–0.15 at sim 0.91–0.97, all minor pronunciation slips).
Both metrics are already computed per clip either way.

## B. TTS generator

The mispronunciations are a generator defect, not gate noise. Options:
- **B1 (rec): keep the no-espeak Kokoro + rule A3 for v2.** Yield ~67%,
  everything shipped is gate-clean; zero new work.
- B2: rebuild TTS.cpp/Kokoro with espeak-ng G2P first (≈a day, raises yield
  and audio quality; delays E0 bulk generation).

## C. Render gate threshold

Provisional 0.60 was far too loose: measured n=50 renders, min sim 0.814,
median 0.977. Propose **0.80**. yes/no?

## D. k = 8 hard negatives per exposure (ATIR working point), plus in-batch.
yes/no?

## E. Exposure `instruction` field: carry it in the exposure schema, OFF by
default, A/B at the v2 smoke (Gemini-Embedding-2/Qwen3-VL-Emb pattern).
yes/no?

## F. Ratify the reality-forced spec changes (already in v0.2, measured
against the real processor — these are facts more than choices):
role-wrapped collate + cas:// resolve step; audio views ≤ 30 s (processor
does NOT enforce the model's 750-token audio cap); CAS audio 16 kHz mono
WAV; `views` as named map (alt renditions); `rights.tier` = CORPUS-ACQ enum;
`source` enum covering the E0 fill matrix; per-view `native_id` provenance.
yes/no?

## G. ColPali mining rule: strip instruction boilerplate from queries
("Your answer should be very brief." etc. — 11/41 queries in the pilot
carried it; TTS of boilerplate fails round-trip and the boilerplate is task
noise, not semantics). yes/no?

---
Not blocking freeze, noted for the builder: MMEB captions carry typos
("form cincrete") — the TTS gate already rejects those clips; consider a
source-text quality filter if audio yield on mmeb matters.
