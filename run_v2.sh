#!/usr/bin/env bash
# run_v2.sh — auto-restart wrapper for train_v2.py (tmux-safe loop).
#
# Policy (BUILDER-BRIEF §1 / TRAINING-CHECKLIST §B):
#   exit 0  -> schedule complete, stop.
#   exit 3  -> TRIPWIRE (NaN/loss-spike): halt-and-alert, NEVER auto-restart.
#   other   -> crash: restart, cap FL_RESTART_CAP (default 5).
# Every start/stop gets a ledger line in $FL_OUT/run-ledger.log.
#
# Usage (rig):  FL_SHARDS=... FL_OUT=... FL_STEP_SECS=... ./run_v2.sh [world]
# All FL_* env is passed through to the trainer.
set -uo pipefail

WORLD="${1:-${FL_WORLD:-2}}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FL_PY:-$DIR/venv/bin/python}"
# FL_OUT must point at the rig's big SSD mount — passed at launch, the
# real path is never committed (public repo: no tailnet names in paths).
export FL_OUT="${FL_OUT:-/mnt/big-ssd/fluffy/checkpoints-v2}"
CAP="${FL_RESTART_CAP:-5}"
# 14-day fragmentation guard (long-lived allocator, variable seq lengths)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LEDGER="$FL_OUT/run-ledger.log"
LOG_DIR="${FL_LOG_DIR:-$FL_OUT/logs}"
mkdir -p "$FL_OUT" "$LOG_DIR"

ledger() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LEDGER"; }

n=0
ledger "WRAPPER START world=$WORLD cap=$CAP shards=${FL_SHARDS:-unset} steps=${FL_STEPS:-auto} step_secs=${FL_STEP_SECS:-unset} lanes=${FL_LANES:-default}"
while true; do
  LOG="$LOG_DIR/train-$(date -u +%Y%m%d-%H%M%S).log"
  ledger "LAUNCH attempt=$((n + 1)) log=$LOG"
  if [ "$WORLD" -gt 1 ]; then
    "$PY" -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
      "$DIR/train_v2.py" >>"$LOG" 2>&1
  else
    "$PY" "$DIR/train_v2.py" >>"$LOG" 2>&1
  fi
  code=$?
  if [ "$code" -eq 0 ]; then
    ledger "EXIT clean (schedule complete) — wrapper done"
    exit 0
  fi
  if [ "$code" -eq 3 ]; then
    ledger "ALERT TRIPWIRE exit=3 — NaN/loss-spike, NOT restarting (poison guard). Last log lines:"
    tail -5 "$LOG" | tee -a "$LEDGER"
    exit 3
  fi
  n=$((n + 1))
  ledger "CRASH exit=$code restart=$n/$CAP. Last log lines:"
  tail -5 "$LOG" | tee -a "$LEDGER"
  if [ "$n" -ge "$CAP" ]; then
    ledger "ALERT RESTART CAP REACHED ($CAP) — giving up, operator needed"
    exit 1
  fi
  sleep 30
done
