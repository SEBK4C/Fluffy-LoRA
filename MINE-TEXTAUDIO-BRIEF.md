# MINE-TEXTAUDIO brief — text breadth + real audio at scale (v2 relaunch prep)

Sebastian's order 2026-07-12 (~08:20Z): v2 stopped at step 589 for a DATA
BREADTH pivot — the single-domain synthetic text lane was the biggest gap vs
LCO-class results. You own TEXT BREADTH + AUDIO lanes. Timebox: data card in
48h; relaunch ~day 3. KISS binding.

READ FIRST: state/T9-STATUS.md, CARD-SPEC.md (FROZEN v1.1),
cardkit/build_text_lane.py + assemble_audio_views.py outputs
(/pool-ssd/fluffy-cards/bulk/), TRAINING-CHECKLIST.md §E0/E,
/root/CORPUS-ACQ registry conventions (its pool is READ-ONLY; coordinate
acquisitions through its inbox conventions). PVE root fs 86% — working sets
on /pool-ssd only. Rig connection = PRIVATE addendum, never in commits.

## TEXT breadth (the headline job)

1. INVENTORY first: what text retrieval/NLI/QA-style sets already sit in
   /pool-6b/corpus-acq (read its registry/sources.yaml + state/*.json)?
   Post the inventory to T9 before mining.
2. Mine broad multi-domain text pairs into CARD-SPEC exposures: target
   ≥500k pairs blended across domains (web QA, NLI-style, doc retrieval —
   whatever the CAS legitimately holds). v001's 224k stays; breadth ADDS.
3. Gaps worth acquiring (small, standard, rights-clean sets only — MS MARCO
   / NLI-class): acquire via CORPUS-ACQ conventions with rights tiers; post
   spend/size to T9 first if >5GB or non-obvious licensing.
4. Negatives: teacher bands via :9020 (Qwen3-Embedding-8B stays the TEXT
   teacher — never leave it down); in-band + TopK-PercPos where positive
   sims exist; k=8 ceiling, in-batch covers shortfalls. Dedup protocol v1.
   G0-blacklist rule applies to anything v001-derived.

## AUDIO at scale (real audio dominant)

1. **LibriSpeech** (59.3G, CC-BY) + **MLS non-EN** (89.9G): speech↔transcript
   exposures at scale — structure gives ground-truth pairs; same-voice-
   different-text = hard negative, same-text-different-voice = positive.
2. **FSD50K**: env-sound↔label exposures.
3. Join the 15,080 gated TTS views (audio-views-v001.jsonl) — they ride
   along; keep the ≥35%-REAL-audio rule with room to spare given 1+2.
4. Audio caps: ≤30s clips (processor budget), 16kHz mono WAV in CAS refs.
5. Whisper (medium is cached) for any WER gates on derived text.

## Shared rules

Instruction string VERBATIM "Retrieve the matching description."; provenance
+ rights tier per row; 250-sample cardkit gate before each source's bulk
(>30% reject = stop+post); shards WDS+idx+MANIFEST+SHA256SUMS → rsync →
/pool-5tb/fluffy/shards/<lane>/ → sha -c → coordinate the HDD readback gate
with MINE-IMG (one combined run is fine). Interleaved composites: provide
MINE-IMG your gated audio refs via T9. `[HH:MMZ] MINE-TA:` milestone lines;
`ff:` commits, public-repo hygiene. DONE = data card in T9: pairs/exposures
per lane + per source, gate rates, GB staged, real-vs-generated audio ratio.
