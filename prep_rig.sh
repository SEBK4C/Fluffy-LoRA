#!/usr/bin/env bash
# Stage the training environment on llm-serve (runs ON the rig, background-safe).
# 1. venv with torch cu128 + transformers/peft/bnb  2. gemma-4-12b-it weights
# 3. data staged from PVE at freeze time (rsync'd by stage_training.sh)
set -u
cd ~
mkdir -p ~/fluffy-lora/{data,checkpoints,logs}
if [ ! -d ~/fluffy-lora/venv ]; then
  python3 -m venv ~/fluffy-lora/venv
fi
source ~/fluffy-lora/venv/bin/activate
pip install -q -U pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
pip install -q "transformers>=4.57" peft bitsandbytes datasets numpy huggingface_hub hf_xet 2>&1 | tail -1
python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("google/gemma-4-12b-it",
                      allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
print("weights at:", p)
PY
python3 -c "import torch, transformers, peft, bitsandbytes; print('env ok', torch.__version__, torch.cuda.get_device_name(0))"
echo "PREP-DONE"
