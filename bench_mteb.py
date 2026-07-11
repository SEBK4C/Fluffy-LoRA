# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.57", "peft", "bitsandbytes", "numpy", "mteb>=2.18"]
# ///
"""MTEB benchmark for the v1 wind-down (EVAL-AGENT-BRIEF §4).

Three contenders on one identical harness:
  base    — google/gemma-4-12b-it language tower, no adapter
  lora    — base + fluffy-text-v0 adapter (or any --ckpt)
  teacher — Qwen/Qwen3-Embedding-8B reference (its own card protocol:
            left-pad last-token pool, instructed queries)

The gemma embedding fn byte-matches training/ratchet_eval: right-pad tokenize
(truncation, max_length 512), eager attention, last_hidden_state, last-token
index = attention_mask.sum(1)-1, float() then L2 norm.

Tasks: MTEB retrieval (SciFact, NFCorpus, FiQA2018) + STS (STSBenchmark,
STS17 en-en) + our frozen G0 retrieval eval for continuity.

  python bench_mteb.py --contender base --tasks SciFact,G0 --out results/
  python bench_mteb.py --contender lora --ckpt checkpoints/step-1449 --dtype nf4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GEMMA_ID = "google/gemma-4-12b-it"
QWEN_ID = "Qwen/Qwen3-Embedding-8B"
MAXLEN = 512
MTEB_TASKS = ["SciFact", "NFCorpus", "FiQA2018", "STSBenchmark", "STS17"]

RETRIEVAL_INSTR = "Given a web search query, retrieve relevant passages that answer the query"
STS_INSTR = "Retrieve semantically similar text"


def make_meta(name: str, n_params: int, dim: int):
    from mteb.models import ModelMeta
    return ModelMeta(
        loader=None, name=name, revision="local", release_date="2026-07-11",
        languages=["eng-Latn"], n_parameters=n_params, memory_usage_mb=None,
        max_tokens=MAXLEN, embed_dim=dim, license="gemma", open_weights=True,
        public_training_code=None, public_training_data=None,
        framework=["PyTorch"], similarity_fn_name="cosine",
        use_instructions=False, training_datasets=None)


class CosineSimMixin:
    """similarity/similarity_pairwise required by mteb's EncoderProtocol."""

    @staticmethod
    def _t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)

    def similarity(self, a, b):
        a, b = F.normalize(self._t(a), dim=-1), F.normalize(self._t(b), dim=-1)
        return a @ b.T

    def similarity_pairwise(self, a, b):
        a, b = F.normalize(self._t(a), dim=-1), F.normalize(self._t(b), dim=-1)
        return (a * b).sum(-1)


class GemmaEncoder(CosineSimMixin):
    """Byte-matches train.py/ratchet_eval.py's embed fn."""

    def __init__(self, ckpt: str | None, dtype: str, device: str, batch: int,
                 pooling: str = "masksum"):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(GEMMA_ID)
        self.batch = batch
        # "masksum" = v1's pooling (attention_mask.sum-1), byte-matches training
        #   but reads a PAD position under gemma's left padding (the v1 bug).
        # "lastpos" = h[:, -1], the correct last real token under left padding.
        self.pooling = pooling
        kw = dict(torch_dtype=torch.bfloat16, attn_implementation="eager")
        if dtype == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
            kw["device_map"] = {"": device}
        else:
            kw["device_map"] = "auto"  # 12B bf16 ≈ 24 GB — shard over free GPUs
        base = AutoModel.from_pretrained(GEMMA_ID, **kw)
        if hasattr(base, "language_model"):
            base = base.language_model
        if ckpt:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, ckpt)
        else:
            self.model = base
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        name = f"fluffy/gemma-4-12b-it{'-lora' if ckpt else ''}-{dtype}"
        self.mteb_model_meta = make_meta(name, 12_000_000_000, self.model.config.hidden_size)

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(texts[i:i + self.batch], padding=True, truncation=True,
                           max_length=MAXLEN, return_tensors="pt").to(self.dev)
            h = self.model(**enc).last_hidden_state
            if self.pooling == "lastpos":
                emb = h[:, -1]
            elif self.pooling == "mean":
                m = enc["attention_mask"].to(h.device).unsqueeze(-1)
                emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                idx = (enc["attention_mask"].sum(1) - 1).to(h.device)
                emb = h[torch.arange(h.shape[0], device=h.device), idx]
            out.append(F.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(out)

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kw):
        out = [self.embed(list(b["text"])) for b in inputs]
        return torch.cat(out).numpy()


class QwenEncoder(CosineSimMixin):
    """Qwen3-Embedding-8B per its model card: left padding, last-token pool,
    instructed queries (retrieval: queries only; STS: both sides)."""

    def __init__(self, device: str, batch: int):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(QWEN_ID, padding_side="left")
        self.model = AutoModel.from_pretrained(
            QWEN_ID, torch_dtype=torch.bfloat16, device_map={"": device}).eval()
        self.batch = batch
        self.dev = next(self.model.parameters()).device
        self.mteb_model_meta = make_meta(
            "reference/qwen3-embedding-8b-bf16", 8_000_000_000,
            self.model.config.hidden_size)

    @staticmethod
    def _instr(instruction: str, q: str) -> str:
        return f"Instruct: {instruction}\nQuery:{q}"

    @torch.inference_mode()
    def embed(self, texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(texts[i:i + self.batch], padding=True, truncation=True,
                           max_length=MAXLEN, return_tensors="pt").to(self.dev)
            h = self.model(**enc).last_hidden_state
            emb = h[:, -1]  # left padding → last position is last real token
            out.append(F.normalize(emb.float(), dim=-1).cpu())
        return torch.cat(out)

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kw):
        ttype = getattr(task_metadata, "type", "") if task_metadata else ""
        is_sts = ttype == "STS"
        is_query = prompt_type is not None and str(prompt_type).endswith("query")
        out = []
        for b in inputs:
            texts = list(b["text"])
            if is_sts:
                texts = [self._instr(STS_INSTR, t) for t in texts]
            elif is_query:
                texts = [self._instr(RETRIEVAL_INSTR, t) for t in texts]
            out.append(self.embed(texts))
        return torch.cat(out).numpy()


def run_g0(enc, g0_path: str, instructed: bool) -> dict:
    """Frozen G0 retrieval, identical construction to ratchet_eval.py."""
    cards = [json.loads(l) for l in Path(g0_path).read_text().splitlines()][:1500]
    pool = [c["canonical_text"] for c in cards]
    queries, tgt = [], []
    for i, c in enumerate(cards):
        for q in ([p for p in (c.get("paraphrases") or []) if isinstance(p, str)][:1]
                  + ([c["intent_text"]] if isinstance(c.get("intent_text"), str) else [])):
            queries.append(enc._instr(RETRIEVAL_INSTR, q) if instructed else q)
            tgt.append(i)
    t0 = time.time()
    P = enc.embed(pool)
    Q = enc.embed(queries)
    top5 = (Q @ P.T).topk(5, dim=1).indices.numpy()
    tgt = np.array(tgt)
    return {"task": "G0", "r1": round(float((top5[:, 0] == tgt).mean()), 4),
            "r5": round(float((top5 == tgt[:, None]).any(1).mean()), 4),
            "n_pool": len(pool), "n_queries": len(queries),
            "secs": round(time.time() - t0, 1),
            "texts_per_s": round((len(pool) + len(queries)) / (time.time() - t0), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contender", required=True, choices=["base", "lora", "teacher"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "nf4"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pooling", default="masksum",
                    choices=["masksum", "lastpos", "mean"])
    ap.add_argument("--tasks", default=",".join(MTEB_TASKS + ["G0"]))
    ap.add_argument("--g0", default="data/g0-eval-cards.jsonl")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tag = (f"{args.contender}-{args.dtype}"
           + ("-1196" if args.ckpt and "1196" in args.ckpt else "")
           + {"lastpos": "-fixedpool", "mean": "-meanpool"}.get(args.pooling, ""))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",")

    if args.contender == "teacher":
        enc = QwenEncoder(args.device, args.batch)
    else:
        enc = GemmaEncoder(args.ckpt if args.contender == "lora" else None,
                           args.dtype, args.device, args.batch, args.pooling)

    report = {"contender": tag, "ckpt": args.ckpt, "batch": args.batch,
              "maxlen": MAXLEN, "pooling": args.pooling, "results": {}}

    mteb_names = [t for t in tasks if t != "G0"]
    if mteb_names:
        import mteb
        mt = mteb.get_tasks(tasks=mteb_names, languages=["eng"],
                            exclusive_language_filter=True)  # STS17 → en-en only
        t0 = time.time()
        res = mteb.evaluate(enc, mt, encode_kwargs={"batch_size": args.batch},
                            cache=None, raise_error=True)
        for tr in res.task_results:
            scores = tr.get_score()
            report["results"][tr.task_name] = {
                "main_score": scores,
                "evaluation_time": getattr(tr, "evaluation_time", None)}
        report["mteb_wall_s"] = round(time.time() - t0, 1)

    if "G0" in tasks:
        report["results"]["G0"] = run_g0(enc, args.g0, instructed=(args.contender == "teacher"))

    out = outdir / f"{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
