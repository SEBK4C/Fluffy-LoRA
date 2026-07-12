# MINING-OPS — restartability, three stores, and the quality bar

Binding on MINE-IMAGE, MINE-TEXTAUDIO, CARDSPEC, and the Opus manager.
Sebastian's requirements 2026-07-12: the mining process must be restartable,
Opus-managed for the week, and synced to HF as updates land. KISS binding.

## 1. Restartability contract (every miner, no exceptions)

- Each miner keeps a durable state file `/pool-ssd/fluffy/state/<agent>.json`:
  per source → {extracted, encoded, gated, packed, staged_rig, uploaded_hf}
  with counts + last completed shard id. Update it AFTER each stage completes
  (atomic write: tmp+rename).
- Every stage idempotent: re-running skips completed work by state + on-disk
  artifact checks (never by memory of a chat session).
- A fresh window resuming a dead miner needs ONLY: this file + the agent's
  brief + T9-STATUS. Test it: kill your own pipeline once mid-source and
  resume from state before declaring your first source DONE.
- Long jobs run nohup/tmux-safe with logs under /pool-ssd/fluffy/logs/ —
  never only inside a Claude session's foreground.

## 2. Three stores of the dataset (sync as updates are made)

| Store | Where | Role |
|---|---|---|
| 1 | PVE `/pool-ssd/fluffy/` | working master (cards, exposures, shards, state) |
| 2 | Rig `/pool-5tb/fluffy/shards/` | training-local copy (sha -c verified) |
| 3 | HF `SEBK4C/Fluffy-LoRA-dataset` (**PRIVATE**) | off-site, incremental |

- After each shard set passes its gates: rsync→rig+verify, THEN
  `hf upload` the shards + MANIFEST.jsonl + SHA256SUMS + the source's data
  card to the HF dataset repo under `<lane>/<source>/`. Post the upload to T9.
- **RIGHTS RULE (SIGNOFF-001)**: the HF repo stays PRIVATE until the rights
  audit clears — source_audit_required media may be backed up privately,
  never released. Self-synthetic + CC-BY content is unrestricted. The repo
  README carries a rights table per source, maintained as sources land.

## 3. The quality bar — what makes this LCO-grade, ranked

1. **Task/instruction diversity** (the big one): SOTA embedders train with
   MANY task instructions, not one. Build a FROZEN instruction set (~10-15
   templates: retrieval, question↔passage, caption↔image, page↔query,
   speech↔transcript, sound↔label, similarity, classification-style) and
   stamp per-exposure `instruction` by task type. Needs a 10-min image-lane
   re-baseline at relaunch (Opus manager owns it). Eval instructions stay
   as frozen per eval.
2. **Synthetic query diversification over REAL media** (MegaPairs pattern,
   checklist §6): local gemma-4 on the 3080 Ti generates 2-3 diverse queries
   per real image/page/doc → multiplies pairs per asset + query-style
   breadth; every generated query passes the teacher band gate.
3. **MLLM-judge filtering** (UniME-V2): score pair alignment with local
   gemma-4; drop the bottom decile per source; judge ambiguous negatives
   (false-negative kill). Sample-based (judge 10-20%, calibrate a
   teacher-sim proxy threshold for the rest — judge everything is too slow).
4. **Cross-source dedup**: embedding-space near-dup sweep (teacher, 0.95
   cosine) across ALL text and ALL captions after mining, not just within
   source. Media dedup by sha stays.
5. **Token-budget lane balancing**: images cost ~268 tokens, audio 25/s,
   text ~dozens — balance the mix by compute-exposure, not pair count, when
   setting FL_LANES at relaunch (Opus manager computes from shard stats).
6. **Difficulty metadata for curriculum**: keep teacher-sim per positive AND
   per negative on every exposure (already spec'd) — enables hard-negative
   curriculum + ANCE re-mining at refreshes with the live model
   (evidence-gated, wave 2).
7. **Multilingual slice**: MLS non-EN speech + any multilingual text the CAS
   holds (gemma-4 is multilingual; LCO's 119-language coverage came free
   from its backbone — ours can too, but only if some non-EN pairs exist).
8. **Benchmark-shaped coverage**: MAEB/MIEB task TYPES (retrieval,
   classification-as-pairs, clustering-ish, reranking) should each have
   training-task representation — coverage table in the final data card.
9. **Datasheet discipline**: per-source data card (counts, gates, rights,
   contamination guards) → assembled into one dataset datasheet. This is
   both paper material and what makes the release credible post-audit.

## 4. Relaunch gate (unchanged in spirit)

All lanes' data cards posted → HDD readback gate on /pool-5tb → token-budget
lane mix computed → instruction-set re-baseline (image lane) → **Sebastian's
word** → fresh start, FL_STEPS re-sized from measured pace on the real mix.

## 5. Scale-out contract — adding mining rigs (Sebastian, 2026-07-12)

Maximum parallelism now; more machines later. The design that gives both:

- **Work queue = directories, claims = atomic mkdir.** The task list lives at
  `/pool-ssd/fluffy/queue/<lane>/<source>/<chunk>.json` (each chunk ≈ one
  shard's worth of raw pairs). A worker claims a chunk by
  `mkdir /pool-ssd/fluffy/queue/.claims/<chunk>__<hostname>` — mkdir is
  atomic; success = yours, exists = someone else's. Finished chunks move
  their state into the miner state files (§1). Stale claims (>2h no
  progress) may be broken by the Opus manager only, ledgered.
- **Workers are location-agnostic**: a worker needs (1) tailnet access,
  (2) read access to its input chunk (rsync pull from PVE), (3) the encode
  model (Qwen3-VL-2B, 4.3GB — rsync once), (4) push-back of finished
  shards+sha to PVE. Everything over ssh/rsync — no shared-fs assumption.
- **Adding a rig** = install uv + rsync the worker kit
  (`cardkit/` + the miner's build script + model) + give it the claim dir
  path + start N workers for its GPU/CPU count. Target: a new rig
  contributes within ~30 min of getting a tailnet key. Miners MUST keep
  their build scripts runnable standalone (args: --chunk file, --out dir)
  to make this true — no orchestration logic inside the compute path.
- **Parallelism now**: PVE CPU = extraction/gates/packing workers (6-10);
  rig GPU0 + GPU1 = two independent encode workers; 3080Ti = text-teacher
  bands + judge scoring. Every worker writes progress into the state files
  so §1 restartability covers the whole fleet.
