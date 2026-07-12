# INSTRUCTION SET v2 — DRAFT for freeze (MINE-TA, per MINING-OPS §3.1)

Status: **DRAFT** — Opus manager + Sebastian freeze. Until frozen, all
exposures ship the current frozen string verbatim
(`Retrieve the matching description.`) AND a `task_type` key, so
re-stamping instructions post-freeze is a pure task_type→string map at
pack time — no re-mining, no re-encoding.

Rules proposed for the freeze:
- Instruction applies to the ANCHOR side only (train_v2 anchor-side
  pattern, FL_INSTRUCT=1). Positives/negatives are never instructed.
- Byte-frozen strings; ids stable (it01…it15). New tasks = new ids, never
  edits. Frozen EVAL instructions stay as-is per eval (image re-baseline
  needed at relaunch per MINING-OPS §3.1 — Opus manager owns it).
- Every exposure carries `task_type` + `instruction` (validator can check
  the pair matches the frozen map).

| id | task_type | instruction (verbatim) | lanes/sources |
|---|---|---|---|
| it01 | general_match | `Retrieve the matching description.` | continuity default; v001 text; warmup image shards |
| it02 | qa_passage | `Given a question, retrieve the passage that answers it.` | paq, stackexchange, stackoverflow; MS MARCO if approved |
| it03 | title_doc | `Given a title, retrieve the document it belongs to.` | s2orc, big_patent, csl, wikipedia |
| it04 | web_query_doc | `Given a web search query, retrieve the most relevant document.` | falcon |
| it05 | entity_desc | `Given an entity name, retrieve its encyclopedic description.` | dbpedia-entity |
| it06 | code_search | `Given a description of what code does, retrieve the matching code.` | codesearchnet |
| it07 | crosslingual | `Given a query, retrieve the relevant passage in any language.` | swim-ir-cross-lingual |
| it08 | semantic_sim | `Retrieve a text with the same meaning as this one.` | NLI/STS-class when acquired; near-dup positives |
| it09 | nli_entail | `Given a premise, retrieve a hypothesis it entails.` | AllNLI when acquired |
| it10 | caption_image | `Retrieve the image described by this caption.` | MINE-IMG text2image |
| it11 | image_caption | `Retrieve the caption that describes this image.` | MINE-IMG image2text |
| it12 | page_query | `Given a query, retrieve the document page that answers it.` | ColPali/VisRAG (MINE-IMG) |
| it13 | speech_transcript | `Retrieve the exact transcript of this spoken audio.` (audio2text) / `Retrieve the spoken audio matching this transcript.` (text2audio) | LibriSpeech, MLS, TTS views |
| it14 | sound_label | `Retrieve the label that describes this sound.` (audio2text) / `Retrieve a sound matching this description.` (text2audio) | FSD50K |
| it15 | class_label | `Classify by retrieving the matching category label.` | classification-as-pairs (MAEB/MIEB coverage, §3.8) |

Notes for the freeze call:
1. it13/it14 are direction-split (one id, two strings keyed by lane) —
   alternative is 2 ids each; freezer's choice, stamping code handles both.
2. it01 stays byte-identical to the v2-stage-1 frozen string so already-
   baselined lanes remain comparable; image re-baseline at relaunch only.
3. MINE-IMG: please sanity-check it10-it12 against your lanes before the
   freeze; interleaved lane can ride it01 or get an it16 — your call.
