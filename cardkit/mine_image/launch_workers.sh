#!/usr/bin/env bash
# launch_workers.sh — start the MINE-IMAGE fleet detached (OPS §1: nohup,
# logs under /pool-ssd/fluffy/logs/; the Claude session only monitors).
#   launch_workers.sh [N_CPU]        default 6 CPU workers + 2 GPU drivers
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(dirname "$(dirname "$HERE")")/venv/bin/python3"
LOGS=/pool-ssd/fluffy/logs
N_CPU="${1:-6}"
mkdir -p "$LOGS"
STAMP="$(date -u +%Y%m%d-%H%M%S)"

# OPS §1 resume hygiene: release OUR OWN dead workers' claims first
"$PY" "$HERE/clean_stale_claims.py"

# purge ORPHANED rig-side encode children (killing a PVE worker leaves its
# remote encode holding GPU VRAM — learned 2026-07-12). Whole-fleet launch
# only: never run this while other encode workers are alive.
if ! pgrep -f "encode_worke[r].py --gpu" > /dev/null; then
  # shellcheck disable=SC1091
  source /pool-ssd/fluffy/rig.env
  ssh -i "$RIG_KEY" -o BatchMode=yes -o ConnectTimeout=10 "$RIG_SSH" \
    'pkill -9 -f "encode_item[s].py" || true' || true
fi

for i in $(seq 1 "$N_CPU"); do
  OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    nohup "$PY" "$HERE/cpu_worker.py" \
    >> "$LOGS/cpu-worker-$i-$STAMP.log" 2>&1 &
  echo "cpu-worker-$i pid $!"
done

for gpu in 0 1; do
  nohup "$PY" "$HERE/encode_worker.py" --gpu "$gpu" \
    >> "$LOGS/encode-gpu$gpu-$STAMP.log" 2>&1 &
  echo "encode-gpu$gpu pid $!"
done
