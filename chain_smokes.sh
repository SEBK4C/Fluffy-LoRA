#!/usr/bin/env bash
# chain_smokes.sh — one-shot swap-gate chain: A1 -> A4 -> A3 -> steptime ->
# kill-9 resume test -> A6 (DDP, both GPUs). Runs ON the rig.
# Env: FL_SHARDS, FL_OUT_BASE (+ optional FL_PY). GPU ids as args.
#   usage: ./chain_smokes.sh <solo-gpu> <ddp-gpus>   e.g. ./chain_smokes.sh 1 0,1
set -uo pipefail

SOLO="${1:?solo gpu id}"
DDP="${2:?ddp gpu ids}"
FROM="${3:-a1}"   # skip already-passed phases: a1|a4|a3|steptime|resume|a6
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${FL_SHARDS:?}" ; : "${FL_OUT_BASE:?}"
mkdir -p "$FL_OUT_BASE"
LOG="$FL_OUT_BASE/chain.log"
say() { echo "[$(date -u +%H:%M:%SZ)] CHAIN: $*" | tee -a "$LOG"; }
reached() { # phase order gate
  local order="a1 a4 a3 steptime resume a6" seen=0 p
  for p in $order; do
    [ "$p" = "$FROM" ] && seen=1
    [ "$p" = "$1" ] && { [ "$seen" = 1 ] && return 0 || return 1; }
  done; return 1
}

phase() { # phase <mode> <gpus>
  local mode="$1" gpus="$2" plog="$FL_OUT_BASE/$1.log"
  say "$mode START (gpu $gpus, log $plog)"
  if "$DIR/smokes_v2.sh" "$gpus" "$mode" >"$plog" 2>&1; then
    say "$mode PASS"
  else
    say "$mode FAIL rc=$? — tail:"; tail -8 "$plog" | tee -a "$LOG"; exit 1
  fi
}

say "chain start: shards=$FL_SHARDS out=$FL_OUT_BASE from=$FROM"
for m in a1 a4 a3 steptime; do
  if reached "$m"; then phase "$m" "$SOLO"; else say "$m SKIPPED (from=$FROM)"; fi
done

# --- kill -9 resume test (MANDATORY swap gate) ------------------------------
run_resume() {
RLA="$FL_OUT_BASE/resume-a.log" ; RLB="$FL_OUT_BASE/resume-b.log"
say "resume-a launching; will kill -9 at ~step 55 (ckpt every 25)"
"$DIR/smokes_v2.sh" "$SOLO" resume-a >"$RLA" 2>&1 &
WRAP=$!
for i in $(seq 1 900); do
  grep -q "step 5[5-9] " "$RLA" && break
  kill -0 "$WRAP" 2>/dev/null || { say "resume-a DIED before step 55 — tail:"; tail -8 "$RLA" | tee -a "$LOG"; exit 1; }
  sleep 2
done
grep -q "step 5[5-9] " "$RLA" || { say "resume-a never reached step 55"; exit 1; }
pkill -9 -f 'train_v2\.py' || true   # regex \. avoids pkill self-match
sleep 3; kill -9 "$WRAP" 2>/dev/null || true
say "KILLED -9 at: $(grep -oE 'step 5[0-9] .*' "$RLA" | tail -1)"
[ -d "$FL_OUT_BASE/resume/step-50" ] || { say "NO step-50 checkpoint — resume gate FAIL"; exit 1; }
# snapshot step-50 state NOW — rolling retention will prune it during resume-b
"${FL_PY:-$DIR/venv/bin/python}" - <<PY | tee -a "$LOG"
import torch
st = torch.load("$FL_OUT_BASE/resume/step-50/trainstate.pt", map_location="cpu", weights_only=False)
print(f"  step-50: cursors={st['cursors']} loss_ema={st['loss_ema']:.4f}")
PY
say "step-50 checkpoint present; relaunching (resume-b)"
if "$DIR/smokes_v2.sh" "$SOLO" resume-b >"$RLB" 2>&1; then
  say "resume-b completed"
else
  say "resume-b FAIL rc=$? — tail:"; tail -8 "$RLB" | tee -a "$LOG"; exit 1
fi
grep -m1 "RESUMED from step-50" "$RLB" >/dev/null || { say "resume-b did NOT resume from step-50"; tail -8 "$RLB" | tee -a "$LOG"; exit 1; }
say "RESUME EVIDENCE — run A steps 40-55:"
grep -E "step (4[0-9]|5[0-5]) " "$RLA" | tee -a "$LOG"
say "RESUME EVIDENCE — run B steps 50-65 (post-resume):"
grep -E "step (5[0-9]|6[0-5]) " "$RLB" | tee -a "$LOG"
say "cursor advance (latest surviving ckpt vs step-50 snapshot above):"
"${FL_PY:-$DIR/venv/bin/python}" - <<PY | tee -a "$LOG"
import torch
from pathlib import Path
ck = sorted(Path("$FL_OUT_BASE/resume").glob("step-*"),
            key=lambda d: int(d.name.split("-")[1]))[-1]
st = torch.load(ck / "trainstate.pt", map_location="cpu", weights_only=False)
print(f"  {ck.name}: cursors={st['cursors']} loss_ema={st['loss_ema']:.4f}")
PY
say "retention after resume-b (expect pruned set incl. 12h-bucket + last3):"
ls "$FL_OUT_BASE/resume" | tee -a "$LOG"
}
if reached resume; then run_resume; else say "resume SKIPPED (from=$FROM)"; fi

# --- A6: 20-step DDP world=2 (needs BOTH gpus free) -------------------------
if ! reached a6; then
  say "a6 SKIPPED (from=$FROM)"
elif nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  say "A6 SKIPPED — another process holds a GPU; run './smokes_v2.sh $DDP a6' when free"
else
  phase a6 "$DDP"
  say "A6 step-time lines:"
  grep -E "s/step|MEASURED" "$FL_OUT_BASE/a6.log" | tail -8 | tee -a "$LOG"
fi
say "CHAIN COMPLETE — all executed gates PASS"
