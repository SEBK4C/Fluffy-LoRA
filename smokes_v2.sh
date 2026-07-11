#!/usr/bin/env bash
# smokes_v2.sh — A-series swap-gate smokes for train_v2.py (BUILDER-BRIEF §2).
# Runs ON the rig. All host-specific paths come from env — nothing here may
# name hosts or users (public repo).
#
#   FL_SHARDS   exposure-shard dir (smoke shard)
#   FL_OUT_BASE base dir for per-smoke checkpoint dirs (big SSD mount)
#   FL_PY       python (default ./venv/bin/python next to this script)
#
# usage: ./smokes_v2.sh <gpu-ids> <a1|a4|a3|steptime|resume-a|resume-b|a6>
#   a1       image path through processor+model, 1 batch, forward only
#   a4       grad-flow audit: LoRA-only grads, towers frozen
#   a3       max per-device image batch (NF4+grad-ckpt, full step)
#   steptime 20-step world=1 loop on the mixed lane schedule (s/step)
#   resume-a phase A of the kill-9 resume test (run, ckpt every 25 steps)
#   resume-b phase B: resume after kill -9, verify continuity
#   a6       20-step DDP world=2 smoke (needs BOTH gpus, e.g. "0,1")
set -euo pipefail

GPUS="${1:?gpu ids, e.g. 1 or 0,1}"
MODE="${2:?mode}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FL_PY:-$DIR/venv/bin/python}"
: "${FL_SHARDS:?set FL_SHARDS}"
: "${FL_OUT_BASE:?set FL_OUT_BASE}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FL_SHARDS

# ENC_CHUNK=8: the smoke mix includes 30s-audio + interleaved lanes whose
# 758+ token sequences OOM eager attention at larger per-forward chunks.
common=(FL_BATCH=4 FL_ACCUM=1 FL_LOG_EVERY=1 FL_ENC_CHUNK=8)
# smoke shard carries ALL lanes incl. audio + interleaved — smokes prove the
# full path; the real stage-1 launch config is text+image only (FL_LANES).
ALL_LANES="text2text=0.35,image2text=0.15,text2image=0.15,audio2text=0.10,text2audio=0.10,interleaved2text=0.15"

case "$MODE" in
  a1)      env "${common[@]}" FL_SMOKE=a1 FL_OUT="$FL_OUT_BASE/a1" "$PY" "$DIR/train_v2.py" ;;
  a4)      env "${common[@]}" FL_SMOKE=a4 FL_OUT="$FL_OUT_BASE/a4" "$PY" "$DIR/train_v2.py" ;;
  a3)      env "${common[@]}" FL_SMOKE=a3 FL_OUT="$FL_OUT_BASE/a3" "$PY" "$DIR/train_v2.py" ;;
  steptime)
    rm -rf "$FL_OUT_BASE/steptime"
    env "${common[@]}" FL_LANES="$ALL_LANES" FL_MAX_STEPS=20 FL_STEPS=1000 \
        FL_OUT="$FL_OUT_BASE/steptime" "$PY" "$DIR/train_v2.py" ;;
  resume-a)
    rm -rf "$FL_OUT_BASE/resume"
    env "${common[@]}" FL_LANES="$ALL_LANES" FL_MAX_STEPS=100 FL_STEPS=1000 \
        FL_CKPT_STEPS=25 FL_OUT="$FL_OUT_BASE/resume" "$PY" "$DIR/train_v2.py" ;;
  resume-b)
    env "${common[@]}" FL_LANES="$ALL_LANES" FL_MAX_STEPS=100 FL_STEPS=1000 \
        FL_CKPT_STEPS=25 FL_OUT="$FL_OUT_BASE/resume" "$PY" "$DIR/train_v2.py" ;;
  a6)
    rm -rf "$FL_OUT_BASE/a6"
    env "${common[@]}" FL_LANES="$ALL_LANES" FL_MAX_STEPS=20 FL_STEPS=1000 \
        FL_OUT="$FL_OUT_BASE/a6" \
        "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 \
        "$DIR/train_v2.py" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
