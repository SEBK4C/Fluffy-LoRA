# MINE-TA inventory — CAS text + audio holdings (posted before mining, per brief)

Surveyed 2026-07-12 ~12:30Z from /pool-6b/corpus-acq (READ-ONLY) via
/root/CORPUS-ACQ/registry/sources.yaml + kalm_subset_licenses.yaml +
state/*.verify.json. All rows below are ON DISK and sha-verified by
CORPUS-ACQ at fetch time.

## TEXT — kalm_sample (HIT-TMG/KaLM-embedding-pretrain-data, license-filtered draw)

121 GB parquet, **134.5M pairs**, uniform schema
`{query: str, pos: [str], neg: [] (EMPTY — no shipped negatives, good), relevance: float (KaLM teacher score)}`.
Only `commercial_sampling: true` subsets were drawn (research-only/UNKNOWN
constituents excluded at acquisition). rights_tier = source_audit_required
(training OK, release gated by SIGNOFF-001).

| subset | rows | GB | task shape | lang | constituent license |
|---|---|---|---|---|---|
| wikipedia | 37,500,000 | 13.5 | title↔passage | MULTI (en/fr/…) | CC BY-SA |
| falcon (RefinedWeb) | 22,951,012 | 60.3 | title↔web doc (noisy) | en | ODC-By 1.0 |
| s2orc | 20,685,705 | 9.7 | paper title↔abstract | en | ODC-By 1.0 |
| swim-ir-cross-lingual | 15,195,052 | 7.0 | non-EN query↔EN passage | MULTI | CC BY-SA 4.0 |
| stackoverflow | 11,759,634 | 9.7 | question↔answer | en | CC BY-SA 4.0 |
| paq | 9,004,577 | 4.2 | question↔wiki passage | en | CC BY-SA 3.0 |
| stackexchange | 6,759,266 | 7.8 | question↔answer | en/mixed | CC BY-SA 4.0 |
| dbpedia-entity | 4,629,644 | 1.0 | entity name↔abstract | en | CC BY-SA 3.0 |
| swim-ir-monolingual | 2,961,959 | 1.3 | non-EN query↔same-lang passage | MULTI | CC BY-SA 4.0 |
| codesearchnet | 1,204,732 | 0.5 | docstring↔code | code | MIT harness/OSS |
| big_patent | 460,383 | 5.7 | abstract↔patent body (LONG: 16-25k ch) | en | CC BY 4.0 |
| csl | 395,176 | 0.2 | zh title↔abstract | zh | Apache-2.0 |

Excluded at acquisition (NOT on disk): reddit, amazon-reviews,
multilingual_cc_news, skypile, wudao, nllb_*, zhihu, thucnews, newsroom,
webtext2019zh (research-only/UNKNOWN licenses).

**Multilingual text EXISTS in CAS**: swim-ir cross (15.2M) + mono (3.0M) +
wikipedia (multi) + csl (zh) — no acquisition needed for the multilingual
slice.

**Gaps (MS MARCO / NLI-class)**: CAS holds NO NLI/entailment pairs, no
human-query passage-ranking set (MS MARCO-class), no STS-style graded
pairs. Proposals in T9 (MS MARCO is research-only licensed — needs a call;
AllNLI = SNLI+MNLI is small + standard, CC BY-SA class).

## AUDIO (archives on disk, extraction needed — targets /pool-ssd only)

| source | GB (archives) | contents | rights |
|---|---|---|---|
| librispeech | 58 | train-clean-100 (~28.5k utts/100h) + train-clean-360 (~104k/360h) + train-other-500 (~149k/500h) + dev/test; 16kHz flac + transcripts | CC BY 4.0 (commercial_after_attribution) |
| mls_non_en | 90 | 7 langs opus (de≈476k, nl≈374k, fr≈258k, es≈220k, it≈60k, pt≈37k, pl≈25k utts) + transcripts | CC BY 4.0 |
| fsd50k | 23 | dev 40,966 clips + eval 10,231, multi-label ground truth; per-clip CC0/CC-BY/CC-BY-NC/Sampling+ | source_audit_required (per-clip) |
| common_voice | 200 | en/de/fr/es Scripted Speech 26.0, CC0 | wave-2 candidate, NOT in this window |

Contamination guards: audio-eval-v1 = 250 LibriSpeech **test-clean** + 102
FSD50K **eval** clips → test/dev splits wholesale-excluded from mining;
FSD50K mining uses **dev split only**; LibriSpeech mining uses train-* only.

TTS join: 15,080 gated views (audio-views-v001.jsonl) ride along →
real-audio ratio will be dominated by real (plan ≥200k real pairs).
