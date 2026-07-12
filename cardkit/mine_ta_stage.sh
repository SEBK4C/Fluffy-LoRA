#!/usr/bin/env bash
# mine_ta_stage.sh — three-store sync for one gated source (MINING-OPS §2).
#   usage: mine_ta_stage.sh <lane> <source> <local_shards_dir>
#   e.g.:  mine_ta_stage.sh audio librispeech /pool-ssd/fluffy/mine-ta/audio-lanes/librispeech/shards
# Store 1 = PVE pool-ssd (already written by the packers)
# Store 2 = rig /pool-5tb/fluffy/shards/<lane>/<source>/ + sha256sum -c
# Store 3 = HF SEBK4C/Fluffy-LoRA-dataset (PRIVATE) under <lane>/<source>/
# Idempotent: rsync is incremental; hf upload skips identical files.
set -euo pipefail
LANE=${1:?lane}
SRC=${2:?source}
DIR=${3:?local shards dir}
RIG=seb@llm-serve.bunny-sunfish.ts.net
KEY=~/.ssh/corpus-acq-llmserve_ed25519
DEST=/pool-5tb/fluffy/shards/$LANE/$SRC
HF_REPO=SEBK4C/Fluffy-LoRA-dataset
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; logger -t mine-ta-stage "$*"; }

[ -f "$DIR/MANIFEST.jsonl" ] || { log "no MANIFEST in $DIR"; exit 1; }
[ -f "$DIR/SHA256SUMS" ] || { log "no SHA256SUMS in $DIR"; exit 1; }

log "rsync $SRC -> rig $DEST"
ssh -i "$KEY" "$RIG" "mkdir -p $DEST"
rsync -a --info=stats1 -e "ssh -i $KEY" "$DIR"/ "$RIG:$DEST/"

log "sha256sum -c on rig"
ssh -i "$KEY" "$RIG" "cd $DEST && sha256sum -c SHA256SUMS --quiet" \
  || { log "RIG SHA VERIFY FAILED for $SRC"; exit 1; }
log "rig sha -c PASS"

log "hf upload -> $HF_REPO/$LANE/$SRC (PRIVATE, SIGNOFF-001)"
/root/FLUFFY-LORA/venv/bin/hf upload "$HF_REPO" "$DIR" "$LANE/$SRC" \
  --repo-type dataset --commit-message "mine-ta: $LANE/$SRC shards" \
  >/dev/null
# data card + report ride along if present next to the shards dir
PARENT=$(dirname "$DIR")
for f in DATACARD.md REPORT.json; do
  if [ -f "$PARENT/$f" ]; then
    /root/FLUFFY-LORA/venv/bin/hf upload "$HF_REPO" "$PARENT/$f" \
      "$LANE/$SRC/$f" --repo-type dataset \
      --commit-message "mine-ta: $LANE/$SRC $f" >/dev/null
  fi
done
log "hf upload DONE for $LANE/$SRC"
