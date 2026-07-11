#!/usr/bin/env bash
# NF4-skew probe (EVAL-AGENT-BRIEF §4): repeat SciFact + G0 on the PVE 3080 Ti
# in NF4 to measure how much quantization shifts scores vs the rig's bf16 runs.
# Follows eval_station.sh: pause teacher -> eval -> restart teacher.
set -u
cd /root/FLUFFY-LORA
export FL_G0=/root/SYNTH-FORGE/eval/frozen/G0/eval-cards.jsonl
export HF_HOME=/pool-ssd/models/hf-cache

for p in $(pgrep -f 'llama-server.*9020'); do kill $p; done
sleep 3

for spec in "base" "lora --ckpt /pool-ssd/synth-forge/ckpts/step-1449"; do
  echo "=== START nf4 $spec $(date -u +%FT%TZ) ==="
  uv run bench_mteb.py --contender $spec --dtype nf4 --device cuda:0 --batch 16 \
      --tasks SciFact,G0 --g0 "$FL_G0" --out results
  echo "=== END nf4 $spec rc=$? $(date -u +%FT%TZ) ==="
done

nohup /root/llama.cpp-teacher/build/bin/llama-server \
  -m /pool-ssd/models/qwen3-embedding-8b/Qwen3-Embedding-8B-Q8_0.gguf \
  --embeddings --pooling last -ngl 999 -c 8192 -np 4 -ub 2048 \
  --port 9020 --host 0.0.0.0 >> /pool-ssd/synth-forge/logs/teacher9020.log 2>&1 &
sleep 5
curl -s -o /dev/null -w "teacher restarted http=%{http_code}\n" http://127.0.0.1:9020/health || true
echo "NF4 PROBE DONE $(date -u +%FT%TZ)"
