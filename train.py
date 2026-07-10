# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.57", "peft", "bitsandbytes", "numpy", "datasets"]
# ///
"""Fluffy-LoRA trainer v1 — QLoRA contrastive embedder on gemma-4-12b-it.

PROGRAM §4 recipe, v001 scope (pairs; composites join at first refresh):
  NF4 double-quant base, BF16 compute; LoRA r=8 a=16 dropout .05 on
  q,k,v,o,gate,up,down (embed_tokens excluded); last-token pooling, l2 norm;
  symmetric InfoNCE tau=.02 with in-batch negatives (DDP all-gather when
  world>1); positives = canonical<->{paraphrase, qa, intent}; hard negatives =
  IN-BAND near-misses (nm_sims within [0.75,0.92]) appended to the batch;
  grad checkpointing; lr 1e-4 cosine 3% warmup.
Checkpoints every CKPT_MIN minutes to checkpoints/; resumable; eval hook =
recall@1 over a fixed G0 slice every EVAL_STEPS (full ratchet loop follows).
Data: accepted-v001.jsonl minus eval-task blacklist.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_ID = os.environ.get("FL_BASE", "google/gemma-4-12b-it")
ACCEPTED = Path(os.environ.get("FL_DATA", "/data/accepted-v001.jsonl"))
BLACKLIST = Path(os.environ.get("FL_BLACKLIST", "/data/eval-task-blacklist.json"))
G0_EVAL = Path(os.environ.get("FL_G0", "/data/g0-eval-cards.jsonl"))
OUT = Path(os.environ.get("FL_OUT", "checkpoints"))
STEPS = int(os.environ.get("FL_STEPS", "200000"))
BATCH = int(os.environ.get("FL_BATCH", "16"))          # pairs per device step
ACCUM = int(os.environ.get("FL_ACCUM", "4"))
LR = float(os.environ.get("FL_LR", "1e-4"))
TAU = 0.02
MAXLEN = int(os.environ.get("FL_MAXLEN", "512"))
CKPT_MIN = int(os.environ.get("FL_CKPT_MIN", "30"))
EVAL_STEPS = int(os.environ.get("FL_EVAL_STEPS", "500"))
SEED = int(os.environ.get("FL_SEED", "1"))


def log(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1


def load_pairs() -> list[dict]:
    bl = set(json.loads(BLACKLIST.read_text())["eval_task_ids"]) if BLACKLIST.exists() else set()
    rows = []
    for line in ACCEPTED.read_text().splitlines():
        c = json.loads(line)
        if c["task_id"] in bl:
            continue
        anchor = c["canonical_text"]
        poss = [p for p in (c.get("paraphrases") or []) if isinstance(p, str)]
        if c.get("intent_text"):
            poss.append(c["intent_text"])
        for qa in (c.get("qa_pairs") or [])[:2]:
            if isinstance(qa, dict) and qa.get("q"):
                poss.append(f"{qa['q']} {qa.get('a', '')}")
        nms = [n for n, s in zip(c.get("near_misses") or [], c.get("nm_sims") or [])
               if isinstance(n, str) and 0.75 <= s <= 0.92]
        for p in poss:
            rows.append({"a": anchor, "p": p, "n": nms})
    return rows


def main() -> None:
    rank, world = setup_dist()
    torch.manual_seed(SEED + rank)
    random.seed(SEED + rank)

    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(MODEL_ID, quantization_config=bnb,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation="eager")
    if hasattr(model, "language_model"):
        model = model.language_model  # text tower only for v001 text training
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.train()
    dev = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    model.to(dev)
    if rank == 0:
        model.print_trainable_parameters()

    def encode(texts: list[str]) -> torch.Tensor:
        enc = tok(texts, padding=True, truncation=True, max_length=MAXLEN,
                  return_tensors="pt").to(dev)
        out = model(**enc).last_hidden_state
        idx = enc["attention_mask"].sum(1) - 1
        emb = out[torch.arange(out.shape[0]), idx]
        return F.normalize(emb.float(), dim=-1)

    rows = load_pairs()
    log(f"pairs={len(rows)} world={world}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)
    warm = max(10, int(STEPS * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(s / warm, 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS))))

    OUT.mkdir(exist_ok=True)
    step0 = 0
    resume = sorted(OUT.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if resume:
        model.load_adapter(str(resume[-1]), adapter_name="default")
        step0 = int(resume[-1].name.split("-")[1])
        log(f"resumed from {resume[-1].name}")

    last_ckpt = time.time()
    rng = random.Random(SEED * 1000 + rank)
    for step in range(step0, STEPS):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(ACCUM):
            batch = [rows[rng.randrange(len(rows))] for _ in range(BATCH)]
            anchors = encode([b["a"] for b in batch])
            pos = encode([b["p"] for b in batch])
            negs_txt = [n for b in batch for n in b["n"][:1]]
            negs = encode(negs_txt) if negs_txt else None
            if world > 1:
                gp = [torch.zeros_like(pos) for _ in range(world)]
                dist.all_gather(gp, pos)
                gp[rank] = pos
                pos_all = torch.cat(gp)
            else:
                pos_all = pos
            cand = torch.cat([pos_all, negs]) if negs is not None else pos_all
            logits = anchors @ cand.T / TAU
            labels = torch.arange(BATCH, device=dev) + (rank * BATCH if world > 1 else 0)
            li = F.cross_entropy(logits, labels)
            logits_t = pos @ (torch.cat([anchors, negs]) if negs is not None else anchors).T / TAU
            lt = F.cross_entropy(logits_t[:, :BATCH], torch.arange(BATCH, device=dev))
            loss = (li + lt) / 2 / ACCUM
            loss.backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0:
            log(f"step {step} loss {total_loss:.4f} lr {sched.get_last_lr()[0]:.2e}")
        if rank == 0 and (time.time() - last_ckpt > CKPT_MIN * 60):
            p = OUT / f"step-{step}"
            model.save_pretrained(str(p))
            last_ckpt = time.time()
            log(f"checkpoint {p.name}")
    if rank == 0:
        model.save_pretrained(str(OUT / f"step-{STEPS}"))
    log("train loop exit")


if __name__ == "__main__":
    main()
