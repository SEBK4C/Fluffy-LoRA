# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers==5.13.1", "bitsandbytes", "numpy",
#                 "pillow", "torchvision", "accelerate", "safetensors"]
# ///
"""Image-lane BASE baseline for the v2 checkpoint ratchet (image-eval-v1).

Extends the ratchet_eval.py protocol to the frozen image eval: embeds the
500 image sides and 500 text sides of /pool-ssd/fluffy-cards/eval/
image-eval-v1.jsonl with base gemma-4-12b-it under the IDENTICAL NF4
double-quant config, computes R@1/R@5 in BOTH directions (i2t, t2i),
pool = all 500.

Byte-match rules (train_v2.py is the reference implementation):
  * FULL multimodal model — NO .language_model strip (v2 trains the full
    model; the text-lane G0 eval keeps ratchet_eval.py's stripped protocol
    for continuity with the on-record 0.008).
  * Collate: processor.apply_chat_template([{"role":"user","content":...}],
    tokenize=True, return_dict=True, padding=True) — role wrapper required,
    cas:// refs resolved to local paths (CARD-SPEC "Measured reality").
  * Instruction prefix on the QUERY (anchor) side only, prepended as a text
    item BEFORE the content — exactly train_v2.load_exposure's
    construction under FL_INSTRUCT=1 (default ON, MERGE-RESEARCH §2D).
    Pool (positive) sides are bare, as in training.
  * Pooling: last REAL token via (mask * arange).argmax(1) — train_v2's
    padding-side-robust indexing (LEARNINGS-V1 headline).
  * Text bodies clipped to 2000 chars (train_v2 FL_MAXCHARS default).

Instruction string: image-lane exposures did not exist at baseline time, so
the exact per-lane string is NOT yet frozen. This eval uses the CARD-SPEC
exposure-schema string on both query directions (noted in
state/ckpt-ratchet-v2.json); if real image exposures land a different
string, the watch must re-baseline to byte-match.

Run (teacher :9020 must be paused — see baseline_station.sh):
  uv run baseline_image_eval.py --repeats 3 --img-batch 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = os.environ.get("FL_BASE", "google/gemma-4-12b-it")
EVAL_JSONL = Path(os.environ.get(
    "FL_IMAGE_EVAL", "/pool-ssd/fluffy-cards/eval/image-eval-v1.jsonl"))
FREEZE = EVAL_JSONL.with_suffix(".freeze")
CAS_ROOT = Path(os.environ.get(
    "FLUFFY_CAS", "/pool-ssd/fluffy-cards/cas/sha256"))
INSTRUCTION = os.environ.get(
    "FL_EVAL_INSTRUCTION", "Retrieve the matching description.")
MAXCHARS = int(os.environ.get("FL_MAXCHARS", "2000"))
OUT_DIR = Path("logs")


def verify_pin() -> str:
    """Re-verify the frozen-eval sha256 pin before every use."""
    pinned = FREEZE.read_text().split()[0]
    actual = hashlib.sha256(EVAL_JSONL.read_bytes()).hexdigest()
    if actual != pinned:
        raise SystemExit(f"PIN MISMATCH {EVAL_JSONL}: {actual} != {pinned}")
    return pinned


def resolve_cas(ref: str) -> str:
    sha = ref.removeprefix("cas://")
    p = CAS_ROOT / sha[:2] / sha
    if not p.exists():
        raise SystemExit(f"CAS miss: {ref}")
    return str(p)


def load_pairs() -> list[dict]:
    rows = [json.loads(l) for l in EVAL_JSONL.read_text().splitlines()]
    for r in rows:
        r["image_path"] = resolve_cas(r["image"])
        r["text"] = r["text"][:MAXCHARS]
    return rows


def build_sides(rows: list[dict]):
    """Four content-array sets: instructed queries + bare pools per side."""
    instr = {"type": "text", "text": INSTRUCTION}
    img_q = [[dict(instr), {"type": "image", "image": r["image_path"]}] for r in rows]
    img_p = [[{"type": "image", "image": r["image_path"]}] for r in rows]
    txt_q = [[dict(instr), {"type": "text", "text": r["text"]}] for r in rows]
    txt_p = [[{"type": "text", "text": r["text"]}] for r in rows]
    return {"img_q": img_q, "img_p": img_p, "txt_q": txt_q, "txt_p": txt_p}


class Embedder:
    """train_v2 Encoder equivalent, inference-only."""

    def __init__(self, model, processor, dev):
        self.model, self.processor, self.dev = model, processor, dev
        self._fwd_keys = None

    def _filter(self, enc: dict) -> dict:
        if self._fwd_keys is None:
            import inspect
            self._fwd_keys = set(inspect.signature(self.model.forward).parameters)
        return {k: v for k, v in enc.items()
                if k in self._fwd_keys or "kwargs" in self._fwd_keys}

    @torch.inference_mode()
    def embed(self, contents: list[list[dict]], batch: int) -> torch.Tensor:
        out = []
        for lo in range(0, len(contents), batch):
            convs = [[{"role": "user", "content": c}] for c in contents[lo:lo + batch]]
            enc = self.processor.apply_chat_template(
                convs, tokenize=True, return_dict=True, return_tensors="pt",
                padding=True)
            enc = {k: (v.to(self.dev, torch.bfloat16) if v.is_floating_point()
                       else v.to(self.dev)) for k, v in enc.items()}
            outm = self.model(**self._filter(enc))
            h = getattr(outm, "last_hidden_state", None)
            if h is None:
                h = outm.hidden_states[-1]
            mask = enc["attention_mask"]
            idx = (mask * torch.arange(mask.shape[1], device=mask.device)).argmax(1)
            e = h[torch.arange(h.shape[0], device=h.device), idx]
            out.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out)


def recall(Q: torch.Tensor, P: torch.Tensor) -> tuple[float, float]:
    sims = Q @ P.T
    top5 = sims.topk(5, dim=1).indices.numpy()
    tgt = np.arange(Q.shape[0])
    return (float((top5[:, 0] == tgt).mean()),
            float((top5 == tgt[:, None]).any(1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--img-batch", type=int, default=2)
    ap.add_argument("--txt-batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU template/CAS check only, no model load")
    args = ap.parse_args()

    pin = verify_pin()
    rows = load_pairs()
    if args.limit:
        rows = rows[: args.limit]
    sides = build_sides(rows)
    print(f"pin OK {pin[:12]}… n={len(rows)} instruction={INSTRUCTION!r}", flush=True)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    if args.dry_run:
        for name, contents in sides.items():
            enc = processor.apply_chat_template(
                [[{"role": "user", "content": c}] for c in contents[:2]],
                tokenize=True, return_dict=True, return_tensors="pt", padding=True)
            print(f"dry-run {name}: keys={sorted(enc.keys())} "
                  f"ids={tuple(enc['input_ids'].shape)}", flush=True)
        print("DRY RUN OK")
        return

    from transformers import AutoModel, BitsAndBytesConfig
    import transformers
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": 0})
    assert hasattr(model, "language_model"), "expected full unified model"
    model.eval()
    dev = torch.device("cuda:0")
    emb = Embedder(model, processor, dev)
    load_secs = round(time.time() - t0, 1)
    print(f"model loaded in {load_secs}s (transformers {transformers.__version__}, "
          f"torch {torch.__version__})", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"baseline-image-v2-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    runs = []
    for rep in range(args.repeats):
        rt0 = time.time()
        E, secs = {}, {}
        for name, contents in sides.items():
            b = args.img_batch if name.startswith("img") else args.txt_batch
            st = time.time()
            E[name] = emb.embed(contents, b)
            secs[name] = round(time.time() - st, 1)
            print(f"  rep{rep} {name}: {secs[name]}s", flush=True)
        i2t_r1, i2t_r5 = recall(E["img_q"], E["txt_p"])
        t2i_r1, t2i_r5 = recall(E["txt_q"], E["img_p"])
        run = {"rep": rep,
               "i2t": {"r1": round(i2t_r1, 4), "r5": round(i2t_r5, 4)},
               "t2i": {"r1": round(t2i_r1, 4), "r5": round(t2i_r5, 4)},
               "secs": {**secs, "total": round(time.time() - rt0, 1)},
               "emb_checksum": {k: round(float(v.sum()), 6) for k, v in E.items()},
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        runs.append(run)
        print(json.dumps(run), flush=True)

    def sigma(key):
        vals = [r[key]["r1"] for r in runs]
        return float(np.std(vals))

    report = {
        "eval": "image-eval-v1", "pin": pin, "n_pairs": len(rows),
        "ckpt": "none", "model": MODEL_ID,
        "quant": "nf4-double-bf16 (ratchet_eval-identical)",
        "pooling": "lastpos (mask*arange).argmax — train_v2 byte-match",
        "instruction": INSTRUCTION,
        "instruction_note": ("query-side only, prepended text item "
                             "(train_v2 anchor construction, FL_INSTRUCT=1); "
                             "image-lane exposure string NOT yet frozen at "
                             "baseline time — CARD-SPEC schema string used"),
        "img_batch": args.img_batch, "txt_batch": args.txt_batch,
        "maxchars": MAXCHARS, "load_secs": load_secs,
        "versions": {"transformers": transformers.__version__,
                     "torch": torch.__version__},
        "runs": runs,
        "sigma": {"i2t_r1": round(sigma("i2t"), 6), "t2i_r1": round(sigma("t2i"), 6)},
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"WROTE {out_path}")
    print(json.dumps({"i2t": runs[0]["i2t"], "t2i": runs[0]["t2i"],
                      "sigma": report["sigma"]}))


if __name__ == "__main__":
    main()
