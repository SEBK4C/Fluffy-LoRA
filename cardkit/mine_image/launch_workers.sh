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
