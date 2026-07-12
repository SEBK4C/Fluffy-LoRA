#!/usr/bin/env bash
# mine_img_autochain.sh — unattended TAIL of the MINE-IMAGE pipeline.
#
# The dir-claim queue (cpu_worker + encode_worker, MINING-OPS §5) already
# drives extract -> encode -> minepack per subset, and minepack itself runs
# the 250-sample cardkit gate + bulk validate + shard (fluffy-exposure-shard
# -v1) and writes staging/<subset>/REPORT.json. A >30% gate reject makes
# minepack write a durable <root>/.FAILED marker (see minepack.stop_source).
#
# THIS script drives the per-SOURCE tail the queue does NOT: once a source's
# whole queue has drained (every subset packed => REPORT.json each, no
# pending task, no active claim), it runs, per source:
#   make_data_card.py  -> DATA-CARD-<source>.md (datasheet, rights, guards)
#   stage_rig.py       -> rsync shards to rig /pool-5tb/fluffy/shards/image/
#                         <source>/ + rig-side sha256sum -c + HDD readback gate
#   hf_upload.py       -> PRIVATE SEBK4C/Fluffy-LoRA-dataset image/<source>/
#                         (shards + DATA-CARD + rights-table row; refuses if
#                         the repo is not private, SIGNOFF-001)
# Each sub-script updates /pool-ssd/fluffy/state/mine-image.json atomically
# (common.update_state, flock+rename). This loop only sequences them + posts
# T9 milestone lines.
#
# Restartable / idempotent (MINING-OPS §1) — per-source markers under <root>:
#   .DONE          staged+uploaded; skip forever
#   .FAILED        gate reject (minepack) OR tail error; skip + hold for a look
#   .FAILED.posted a T9 stop line was already emitted for this .FAILED
# A fresh window just re-launches this script; markers + the state file carry
# all progress. No chat-session memory is load-bearing.
#
# Launch (detached, survives the session):
#   nohup bash /root/FLUFFY-LORA/cardkit/mine_img_autochain.sh \
#     > /pool-ssd/fluffy/logs/mine-img-autochain.log 2>&1 &
set -u

CK=/root/FLUFFY-LORA/cardkit/mine_image
PY=/root/FLUFFY-LORA/venv/bin/python
FL=/pool-ssd/fluffy
T9=/root/FLUFFY-LORA/state/T9-STATUS.md
POLL=120
QUIET_NEEDED=2                     # consecutive drained polls before staging
SOURCES="mmeb colpali visrag"
declare -A ROOT=(
  [mmeb]="$FL/image-mmeb"
  [colpali]="$FL/image-colpali"
  [visrag]="$FL/image-visrag"
)

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; logger -t mine-img-autochain "$*" 2>/dev/null || true; }
t9()  { echo "[$(date -u +%H:%MZ)] MINE-IMG(auto): $*" >> "$T9"; }

# A source's queue is drained when it has no pending task files AND no active
# claim. Image task/claim names ALWAYS contain the source (mmeb|colpali|
# visrag); MINE-TA claims (audio-*, kalm-*) never do, so scanning the SHARED
# .claims dir by source substring is safe.
queue_empty() { # $1=source
  local s=$1
  compgen -G "$FL/queue/image/$s/*.json" >/dev/null && return 1
  compgen -G "$FL/queue/.claims/*${s}*"  >/dev/null && return 1
  return 0
}
n_reports() { ls "$1"/staging/*/REPORT.json 2>/dev/null | wc -l; }

log "autochain up — sources: $SOURCES; poll ${POLL}s; quiet-confirm ${QUIET_NEEDED}"
declare -A QUIET
while true; do
  alldone=1
  for s in $SOURCES; do
    root=${ROOT[$s]}

    [ -f "$root/.DONE" ] && continue

    if [ -f "$root/.FAILED" ]; then
      if [ ! -f "$root/.FAILED.posted" ]; then
        reason=$(head -1 "$root/.FAILED" 2>/dev/null | cut -c1-220)
        t9 "$s STOPPED (.FAILED): ${reason:-see $root/.FAILED} — source held, needs a manager/human look"
        log "$s: .FAILED observed — held"
        touch "$root/.FAILED.posted"
      fi
      continue
    fi

    alldone=0
    [ "$(n_reports "$root")" -eq 0 ] && continue   # nothing packed yet

    if queue_empty "$s"; then
      QUIET[$s]=$(( ${QUIET[$s]:-0} + 1 ))
    else
      QUIET[$s]=0
      continue
    fi
    [ "${QUIET[$s]:-0}" -ge "$QUIET_NEEDED" ] || continue

    reports=$(n_reports "$root")
    log "$s: queue drained ($reports subsets packed) — datacard -> stage -> upload"
    if $PY "$CK/make_data_card.py" --source "$s" >>"$FL/logs/mine-img-datacard-$s.log" 2>&1 \
       && $PY "$CK/stage_rig.py"   --source "$s" >>"$FL/logs/mine-img-stage-$s.log"    2>&1 \
       && $PY "$CK/hf_upload.py"   --source "$s" >>"$FL/logs/mine-img-hfupload-$s.log" 2>&1; then
      touch "$root/.DONE"
      t9 "$s STAGED+UPLOADED ($reports subsets): rig sha -c PASS + HDD readback PASS + HF image/$s/ (PRIVATE, DATA-CARD + rights row). counts in state .sources.$s"
      log "$s: DONE"
    else
      echo "$(date -u +%FT%TZ) $s tail FAILED (datacard/stage/upload) — see logs/mine-img-{datacard,stage,hfupload}-$s.log" >> "$root/.FAILED"
      t9 "$s TAIL FAILED (datacard/stage/upload) — logs/mine-img-{datacard,stage,hfupload}-$s.log; source held, needs a look"
      touch "$root/.FAILED.posted"
      log "$s: tail FAILED"
    fi
  done

  if [ "$alldone" = 1 ]; then
    log "ALL IMAGE SOURCES resolved (.DONE/.FAILED) — autochain exits"
    t9 "ALL IMAGE SOURCES resolved (.DONE/.FAILED) — image-lane tail complete. Joint remaining w/ MINE-TA (end-of-run): cross-source dedup sweep + combined HDD readback gate; instruction-set re-stamp at relaunch (Opus mgr owns)."
    exit 0
  fi
  sleep "$POLL"
done
