#!/usr/bin/env bash
# mine_ta_autochain.sh — unattended tail of the MINE-TA pipeline.
# Watches for per-source readiness and runs: (text) mine_pack -> datacard ->
# stage; (audio) datacard -> stage, as the detached builds finish.
# Gate failures leave a .FAILED marker + T9 line and never stage (>30%
# reject rule: stop + post). Idempotent: .staged markers skip done work.
# Run: nohup bash cardkit/mine_ta_autochain.sh > /pool-ssd/fluffy/logs/mine-ta-autochain.log 2>&1 &
set -u
CK=/root/FLUFFY-LORA/cardkit
PY=/root/FLUFFY-LORA/venv/bin/python
TA=/pool-ssd/fluffy/mine-ta
QD=/pool-ssd/fluffy/queue/text/kalm
T9=/root/FLUFFY-LORA/state/T9-STATUS.md
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; logger -t mine-ta-autochain "$*"; }
t9() { echo "[$(date -u +%H:%MZ)] MINE-TA(auto): $*" >> "$T9"; }

audio_sources="librispeech mls fsd50k tts-v001"
text_subsets="allnli big_patent codesearchnet csl dbpedia-entity falcon paq s2orc stackexchange stackoverflow swim-ir-cross-lingual swim-ir-monolingual wikipedia"

extras_for() {
  case "$1" in
    librispeech) echo "audit=clear real_audio=100% contamination=dev-*/test-* never extracted (frozen audio-eval-v1 uses test-clean)" ;;
    mls) echo "audit=clear real_audio=100% multilingual=7 non-EN languages ~13k pairs each" ;;
    *) echo "" ;;
  esac
}

all_chunks_embedded() { # subset
  local s=$1 c
  [ -f "$QD/EXTRACT-DONE-$s" ] || return 1
  for c in "$QD/kalm-$s-"*.json; do
    [ -e "$c" ] || return 1
    local cid; cid=$(basename "$c" .json)
    [ -f "$TA/text/emb/$cid.emb.npz" ] || return 1
  done
  return 0
}

while true; do
  alldone=1
  # ---- audio ----
  for s in $audio_sources; do
    d=$TA/audio-lanes/$s
    [ -f "$d/.staged" ] && continue
    [ -f "$d/.FAILED" ] && continue
    if [ -f "$d/REPORT.json" ]; then
      log "staging audio/$s"
      # shellcheck disable=SC2046
      if $PY "$CK/mine_ta_datacard.py" "$d/REPORT.json" $(extras_for "$s") \
         && bash "$CK/mine_ta_stage.sh" audio "$s" "$d/shards"; then
        touch "$d/.staged"
        t9 "audio/$s STAGED: rig sha -c PASS + HF uploaded (see $d/REPORT.json + DATACARD.md)"
        log "audio/$s staged OK"
      else
        touch "$d/.FAILED"
        t9 "audio/$s STAGE FAILED — needs a human/manager look ($d)"
      fi
    else
      alldone=0
    fi
  done
  # ---- text ----
  for s in $text_subsets; do
    d=$TA/text/$s
    [ -f "$d/.staged" ] && continue
    [ -f "$d/.FAILED" ] && continue
    if [ -f "$d/REPORT.json" ]; then
      log "staging text/kalm-$s"
      if $PY "$CK/mine_ta_datacard.py" "$d/REPORT.json" \
         && bash "$CK/mine_ta_stage.sh" text "kalm-$s" "$d/shards"; then
        touch "$d/.staged"
        t9 "text/kalm-$s STAGED: rig sha -c PASS + HF uploaded (see $d/REPORT.json)"
      else
        touch "$d/.FAILED"
        t9 "text/kalm-$s STAGE FAILED — needs a look ($d)"
      fi
    elif all_chunks_embedded "$s"; then
      log "mine_pack $s"
      if $PY "$CK/mine_ta_text_mine_pack.py" --subset "$s" \
           >> "/pool-ssd/fluffy/logs/mine-ta-minepack-$s.log" 2>&1; then
        log "mine_pack $s OK"
      else
        mkdir -p "$d"; touch "$d/.FAILED"
        t9 "text/kalm-$s MINE_PACK FAILED (gate? see logs/mine-ta-minepack-$s.log) — stopped per >30%-reject rule"
      fi
      alldone=0
    else
      alldone=0
    fi
  done
  [ "$alldone" = 1 ] && { log "ALL SOURCES STAGED — autochain exits"; t9 "ALL MINE-TA SOURCES STAGED — lanes complete; remaining: cross-source dedup sweep + combined HDD readback gate w/ MINE-IMG + instruction-set freeze restamp"; exit 0; }
  sleep 120
done
