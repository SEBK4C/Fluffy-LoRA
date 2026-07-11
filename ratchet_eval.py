# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.57", "peft", "bitsandbytes", "numpy"]
# ///
"""Checkpoint-ratchet evaluator (SYNTH-FORGE requirements §7, Molt semantics).

For a given adapter checkpoint: embed FROZEN-EVAL G0 canonicals (pool) and
queries (paraphrase[0] + intent per card), compute R@1/R@5 retrieval over the
pool, then apply the kept-only ratchet: best_checkpoint pointer advances only
if R@1 - best > eps (0.002 until sigma_ckpt measured; re-eval noise replaces
it later). State: state/ckpt-ratchet.json (kept-only accumulate).

Run on a free GPU (pause the teacher for the window) or CPU (slow, fine).
  python ratchet_eval.py --ckpt checkpoints/step-1234 [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("FL_BASE", "google/gemma-4-12b-it")
G0 = Path(os.environ.get("FL_G0", "data/g0-eval-cards.jsonl"))
STATE = Path("state/ckpt-ratchet.json")
EPS = 0.002


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-cards", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModel.from_pretrained(MODEL_ID, quantization_config=bnb,
                                     torch_dtype=torch.bfloat16,
                                     attn_implementation="eager")
    if hasattr(base, "language_model"):
        base = base.language_model
    model = base if args.ckpt in ("none", "base") else PeftModel.from_pretrained(base, args.ckpt)
    model.eval()
    dev = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    model.to(dev)

    cards = [json.loads(l) for l in G0.read_text().splitlines()][: args.max_cards]
    pool_texts = [c["canonical_text"] for c in cards]
    queries, qtarget = [], []
    for i, c in enumerate(cards):
        for q in ([p for p in (c.get("paraphrases") or []) if isinstance(p, str)][:1]
                  + ([c["intent_text"]] if isinstance(c.get("intent_text"), str) else [])):
            queries.append(q)
            qtarget.append(i)

    @torch.inference_mode()
    def embed(texts):
        out = []
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], padding=True, truncation=True,
                      max_length=args.maxlen, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            idx = enc["attention_mask"].sum(1) - 1
            out.append(F.normalize(h[torch.arange(h.shape[0]), idx].float(), dim=-1).cpu())
        return torch.cat(out)

    t0 = time.time()
    P = embed(pool_texts)
    Q = embed(queries)
    sims = Q @ P.T
    top5 = sims.topk(5, dim=1).indices.numpy()
    tgt = np.array(qtarget)
    r1 = float((top5[:, 0] == tgt).mean())
    r5 = float((top5 == tgt[:, None]).any(1).mean())
    res = {"ckpt": args.ckpt, "r1": round(r1, 4), "r5": round(r5, 4),
           "n_pool": len(pool_texts), "n_queries": len(queries),
           "secs": round(time.time() - t0, 1),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    STATE.parent.mkdir(exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else \
        {"best_r1": 0.0, "best_checkpoint": None, "kept": [], "rejected": []}
    if r1 - st["best_r1"] > EPS:
        st["best_r1"] = r1
        st["best_checkpoint"] = args.ckpt
        st["kept"].append(res)
        verdict = "KEPT (pointer advanced)"
    else:
        st["rejected"].append(res)
        verdict = "rejected (pointer holds)"
    STATE.write_text(json.dumps(st, indent=2))
    print(json.dumps(res))
    print(f"RATCHET: {verdict} | best_r1={st['best_r1']} best={st['best_checkpoint']}")


if __name__ == "__main__":
    main()
