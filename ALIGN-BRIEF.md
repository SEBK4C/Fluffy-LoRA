# FLUFFY-ALIGN brief — infrastructure alignment by T-0 06:00Z

Authorized by Sebastian 2026-07-11 ~21:00Z ("make it happen in nine hours").
You own INFRASTRUCTURE ONLY on both servers (PVE host + rig). You never
touch trainer code, miners, or other agents' processes. KISS is binding.

READ FIRST: TRAINING-CHECKLIST.md (§C/§D/§G) · BUILDER-BRIEF.md (your
counterpart) · /root/SYNTH-FORGE/FLUFFY-FORGE-BOOTSTRAP.md GROUND-TRUTH
ADDENDUM (rig connection — PRIVATE, never in commits). Coordinate ONLY via
`state/T9-STATUS.md` (append-only, timestamped) + `align:` commits.

Verified state 21:00Z (trust, then re-verify what you touch): v1 trainer
STOPPED, ckpts cleaned (step-1449+1196 preserved); EVAL agent benching on
rig GPU0 (frees it soon — watch T9-STATUS); rig GPU1 free for builder
smokes; PVE 3080 Ti runs the Qwen3 text teacher on :9020 (needed for gates
— never kill without restarting); PVE root fs 86% — working sets go to
pool-ssd; rig root fs ~84%.

## Task list, priority order

1. **Downloads to /pool-ssd/models (PVE)**: `Qwen/Qwen3-VL-Embedding-8B` +
   `Qwen/Qwen3-VL-Embedding-2B` (builder's banding is blocked on this —
   post to T9-STATUS the moment weights are complete + smoke-loaded);
   `black-forest-labs/FLUX.1-schnell`; verify Supertonic-3 assets exist
   (cardkit knows where); whisper for WER gates (small is cached; pull
   medium if bandwidth allows). Copy FLUX to the rig for GPU0 noisy-tier
   gen. Log sizes + sha in T9-STATUS.
2. **Teacher smoke**: Qwen3-VL-2B embedding path on the 3080 Ti
   (sentence-transformers). VRAM is tight next to the :9020 teacher
   (10.4G/12G used) — if it doesn't fit alongside, coordinate a teacher
   pause/restart window via T9-STATUS (eval_station.sh shows the pattern);
   never leave :9020 down.
3. **Rig staging**: create the staging tree on the 5TB pool (shards/,
   manifests/, eval/); measure real rsync throughput PVE→rig NOW with a
   ~2GB test file (direct vs relay — post the MB/s); build the readback
   harness (2-min dummy-DataLoader from the HDD, 4 prefetch workers,
   pass = 10× needed samples/s) ready to run when builder posts shards.
4. **Rig hygiene (§G)**: unattended-upgrades is RUNNING — disable it +
   kernel/driver auto-updates for the window; audit cron/systemd timers +
   docker for GPU or root-fs writers (report before disabling non-obvious
   things); confirm nothing else will touch the GPUs for 14 days.
5. **Rig HF cache check**: find where gemma-4-12b-it base weights live on
   the rig (v1 loaded them). If on the root fs, copy to the big SSD mount
   and post the correct HF_HOME for the builder's trainer env. Root fs must
   not carry the run.
6. **Final pre-swap df sweep**: post free GB on all five tiers (PVE root,
   pool-ssd, CAS pool; rig root, SSD, 5TB) into T9-STATUS + tick your
   checklist boxes with measured evidence (`align:` commit).

Public repo hygiene in every commit: pool names and sizes fine; NO tailnet
names, usernames, IPs, or keys. Anything ambiguous → put it in T9-STATUS as
a question instead of guessing.
