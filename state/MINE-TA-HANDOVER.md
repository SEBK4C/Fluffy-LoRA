# MINE-TA handover — resume protocol (MINING-OPS §1)

A fresh window needs ONLY: this file + MINE-TEXTAUDIO-BRIEF.md +
MINING-OPS.md + T9-STATUS tail + `/pool-ssd/fluffy/state/mine-textaudio.json`.
All compute runs detached (nohup); nothing lives in a chat session.

## FIRST CHECK on resume: is the autochain alive?
`pgrep -f mine_ta_autochain` + `tail /pool-ssd/fluffy/logs/mine-ta-autochain.log`.
The autochain (cardkit/mine_ta_autochain.sh) runs the WHOLE remaining tail
unattended: per-source mine_pack (text) -> datacard -> rig+sha -> HF, with
.staged/.FAILED markers under each source dir and T9 lines per event. If
it is running and no .FAILED markers exist, there is nothing to do but
wait. If dead: relaunch it (idempotent) + relaunch any dead embed workers
/ audio builds per the stage table below. .FAILED = gate/stage stopped on
purpose — investigate that source's log before anything else.

## Fixed facts
- Teacher: `http://127.0.0.1:9020` (llama-server, Qwen3-Emb-8B Q8, 3080Ti).
  NEVER leave it down; miners wait through downtime (mine_ta_lib.embed).
- Queue: `/pool-ssd/fluffy/queue/{text/kalm,audio}/`, claims in
  `queue/.claims/<chunk>__<host>` (owner file carries pid; dead-pid
  takeover same-host only).
- Working root: `/pool-ssd/fluffy/mine-ta/` (CAS, audio lanes, text emb).
- Logs: `/pool-ssd/fluffy/logs/mine-ta-*.log`.
- Instruction string in every exposure: `Retrieve the matching description.`
  (VERBATIM, stage-1 frozen). `task_type` stamped per exposure → restamp at
  instruction-set freeze = map by task_type at (re)pack, no re-mining.
- CARD-SPEC v1.1a additive origin amendment (kalm/allnli) PENDING
  ratification — see T9 13:26Z.

## Pipeline stages per source (all idempotent, resume = re-run command)

TEXT (kalm 12 subsets + allnli):
1. extract  — DONE for all 13 (queue chunks + EXTRACT-DONE markers).
2. embed    — `cardkit/mine_ta_text_embed_worker.py` (2 running; relaunch
   more of the same if dead; exits when queue drained + sentinel).
3. minepack — per subset once its chunks are embedded:
   `venv/bin/python cardkit/mine_ta_text_mine_pack.py --subset <s>`
4. stage    — `cardkit/mine_ta_stage.sh text kalm-<s>
               /pool-ssd/fluffy/mine-ta/text/<s>/shards`
5. datacard — `cardkit/mine_ta_datacard.py .../text/<s>/REPORT.json`
   (then hf-upload rides in stage script re-run, or upload DATACARD.md).

AUDIO (librispeech, mls, fsd50k, tts-v001):
1. extract — DONE for all (pairs.jsonl per source).
2-4. build — `venv/bin/python cardkit/mine_ta_audio_build.py --source <s>`
   (slice-resumable teacher embeds; gates; packs; REPORT.json = done).
5. stage   — `cardkit/mine_ta_stage.sh audio <s>
              /pool-ssd/fluffy/mine-ta/audio-lanes/<s>/shards`
6. datacard as above.

## End-of-mining tasks (after all sources packed)
- Cross-source dedup sweep (§3.4): `cardkit/mine_ta_crossdedup.py`
  --spec listing all emb npz/npy + MINE-IMG caption embeddings
  (coordinate via T9; run on a rig 4090 slot with --device cuda:0).
- HDD readback gate on /pool-5tb/fluffy/shards — ONE combined run with
  MINE-IMG (harness: rig big-SSD fluffy/readback_gate.py).
- Assembled dataset datasheet from per-source DATACARD.md files.
- HF README rights table: seed at /pool-ssd/fluffy/state/hf-readme-seed.md
  + rows accumulating in /pool-ssd/fluffy/state/mine-ta-rights-rows.md.

## Open decisions (T9)
- Instruction-set freeze (draft: state/INSTRUCTION-SET-DRAFT.md);
  reconcile exposure key name with MINE-IMG (`task_type` vs `task`).
- MS MARCO acquisition (research-only license) — Sebastian/manager call.
- v1.1a origin-enum ratification.

## Rights snapshot
LibriSpeech/MLS = CC BY 4.0 (audit clear); FSD50K per-clip (CC0 commercial
/ CC-BY attribution / BY-NC+Sampling research_only); kalm + allnli =
source_audit_required (SIGNOFF-001 pending); tts-v001 = self-synthetic
commercial. HF repo SEBK4C/Fluffy-LoRA-dataset stays PRIVATE.
