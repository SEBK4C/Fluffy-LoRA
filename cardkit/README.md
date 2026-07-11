# cardkit — CARD-SPEC reference kit

Working reference implementation of `../CARD-SPEC.md`. Everything here has
been run for real against the gemma-4-12b-it processor and live gate
services; nothing is aspirational.

| File | What it proves |
|---|---|
| `probe_chat_template.py` | Phase 1 reality check: measures actual token cost per modality against the real processor. Results in `probe_report.json`; conclusions folded into the spec's "Measured reality" section. |
| `card.schema.json` | JSON Schema (2020-12) for one card record. |
| `cardlib.py` | Shared helpers: CAS put/resolve, 16 kHz WAV normalization, Kokoro TTS client, typographic renderer, whisper-WER gate, OCR round-trip gate, teacher embeddings. |
| `validate_card.py` | The validator: schema + referential integrity (CAS refs resolve, negatives point at real cards, generated views carry passing gates, dedup hashes recompute, audio is 16 kHz mono ≤ 30 s). |
| `build_golden.py` | Builds the 15 golden cards (3 per §E source) with every generation gate run live, reject-and-advance on gate failure. |
| `tokenize_cards.py` | The reference collate: resolves `cas://` refs and feeds every view through `processor.apply_chat_template`. If this is green, the format works. |
| `example_card.json` | One real golden card (`flf-g001`, fully tri-modal, self-synthetic rights — the only rights tier committable here). |
| `supertonic_pilot.py` | v1.1: Supertonic-3 (PRIMARY TTS) pass-rate check over the same stratified pilot texts, frozen A3 gate unchanged. |
| `make_spotcheck.py` | Stratified eyeball sample as one self-contained HTML (media embedded); output never committed. |

## Environment

All host specifics come from env vars; defaults are localhost/local paths.

```
FLUFFY_CARDS_ROOT   card store root (default /pool-ssd/fluffy-cards)
                    layout: cas/sha256/<2-prefix>/<sha>, golden/, pilot/
GEMMA4_SNAPSHOT     local HF snapshot dir of google/gemma-4-12b-it
TEACHER_URL         llama.cpp embedding server (default http://127.0.0.1:9020)
TTS_URL             Kokoro tts-server, OpenAI /v1/audio/speech API
                    (default http://127.0.0.1:8096)
OCR_PY              python interpreter with rapidocr_onnxruntime (render gate)
SRC_*               source dataset overrides (see build_golden.py)
```

Python deps: `requirements.txt` (torch/torchvision CPU builds suffice —
only the processor is used, never model weights).

Golden card media and manifests live under `FLUFFY_CARDS_ROOT` and are NOT
committed: most source datasets are `source_audit_required` and never enter
the public repo (SIGNOFF-001 discipline).
