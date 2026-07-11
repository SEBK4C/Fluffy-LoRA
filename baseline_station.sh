#!/usr/bin/env bash
# Baseline station (v2 swap-gate): pause teacher -> G0 text baseline rerun
# + image-eval-v1 base baselines -> restart teacher. Pattern from
# eval_station.sh; the teacher restart is TRAPPED so :9020 comes back even
# if an eval fails mid-window.
set -u
cd /root/FLUFFY-LORA

restart_teacher() {
  nohup /root/llama.cpp-teacher/build/bin/llama-server \
    -m /pool-ssd/models/qwen3-embedding-8b/Qwen3-Embedding-8B-Q8_0.gguf \
    --embeddings --pooling last -ngl 999 -c 8192 -np 4 -ub 2048 \
    --port 9020 --host 0.0.0.0 >> /pool-ssd/synth-forge/logs/teacher9020.log 2>&1 &
  echo "teacher restart issued (pid $!)"
}
trap restart_teacher EXIT

for p in $(pgrep -f 'llama-server.*9020'); do kill "$p"; done
for i in $(seq 1 20); do
  pgrep -f 'llama-server.*9020' >/dev/null || break
  sleep 1
done
sleep 3   # let CUDA memory actually release

export FL_G0=/root/SYNTH-FORGE/eval/frozen/G0/eval-cards.jsonl
export HF_HOME=/pool-ssd/models/hf-cache

echo "=== G0 text baseline (protocol unchanged from ratchet_eval.py) ==="
uv run ratchet_eval.py --ckpt none --device cuda:0 --max-cards 1500
G0_RC=$?

echo "=== image-eval-v1 base baseline (repeats for sigma) ==="
uv run baseline_image_eval.py --repeats "${REPEATS:-3}" --img-batch "${IMG_BATCH:-2}"
IMG_RC=$?

echo "rc: g0=$G0_RC img=$IMG_RC"
exit $(( G0_RC != 0 || IMG_RC != 0 ))
