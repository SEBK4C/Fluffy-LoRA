#!/usr/bin/env bash
# Eval station: pause teacher -> ratchet_eval on 3080Ti -> restart teacher.
set -u
CKPT=${1:?usage: eval_station.sh <ckpt-path-or-none>}
cd /root/FLUFFY-LORA
for p in $(pgrep -f 'llama-server.*9020'); do kill $p; done
sleep 3
export FL_G0=/root/SYNTH-FORGE/eval/frozen/G0/eval-cards.jsonl
export HF_HOME=/pool-ssd/models/hf-cache
uv run ratchet_eval.py --ckpt "$CKPT" --device cuda:0 --max-cards 1500
EVAL_RC=$?
nohup /root/llama.cpp-teacher/build/bin/llama-server \
  -m /pool-ssd/models/qwen3-embedding-8b/Qwen3-Embedding-8B-Q8_0.gguf \
  --embeddings --pooling last -ngl 999 -c 8192 -np 4 -ub 2048 \
  --port 9020 --host 0.0.0.0 >> /pool-ssd/synth-forge/logs/teacher9020.log 2>&1 &
echo "teacher restarted; eval rc=$EVAL_RC"
