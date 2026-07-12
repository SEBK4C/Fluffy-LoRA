#!/usr/bin/env python3
"""encode_items.py — Qwen3-VL-Embedding-2B bulk encode of one item chunk.
Standalone compute path (OPS §5): runs on any host with the model + a CAS
mirror; no queue/orchestration logic in here.

  encode_items.py --items chunk.jsonl --out emb.npy \
      --model /path/Qwen3-VL-Embedding-2B --media-root /path/cas [--device cuda]

Items (one JSON/line): {"id", "kind": "text"|"image"|"imagetext",
                        "text": str|null, "image": "<sha256>"|null}
Output: fp16 L2-normed (n, 2048) .npy aligned to input order + <out>.done
        JSON {n, sha256} written LAST (atomic completion marker).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time


def cas(root: str, sha: str) -> str:
    return os.path.join(root, "sha256", sha[:2], sha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-text", type=int, default=256)
    ap.add_argument("--batch-image", type=int, default=64)
    ap.add_argument("--max-dim", type=int, default=1024,
                    help="cap image max dimension fed to the teacher "
                    "(native-res charts/pages OOM the dynamic-res vision "
                    "tower; photos are unaffected)")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items)]
    t0 = time.time()
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = SentenceTransformer(args.model, device=args.device,
                                trust_remote_code=True,
                                model_kwargs={"dtype": dtype})
    print(f"[{time.time()-t0:6.1f}s] model loaded; {len(items)} items", flush=True)

    emb = np.zeros((len(items), 2048), dtype=np.float16)
    groups: dict[str, list[int]] = {"text": [], "image": [], "imagetext": []}
    for i, it in enumerate(items):
        groups[it["kind"]].append(i)

    from PIL import Image

    def load_img(sha: str):
        im = Image.open(cas(args.media_root, sha)).convert("RGB")
        w, h = im.size
        if max(w, h) > args.max_dim:
            sc = args.max_dim / max(w, h)
            im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))),
                           Image.LANCZOS)
        return im

    def payload(it: dict):
        if it["kind"] == "text":
            return it["text"]
        if it["kind"] == "image":
            return {"image": load_img(it["image"])}
        return {"image": load_img(it["image"]), "text": it["text"]}

    def encode_batch(batch: list[int]) -> None:
        """CUDA-OOM-resilient (large doc pages): halve batch and recurse."""
        try:
            out = model.encode([payload(items[i]) for i in batch],
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            encode_batch(batch[:mid])
            encode_batch(batch[mid:])
            return
        emb[batch] = out.astype(np.float16)

    for kind, idxs in groups.items():
        if not idxs:
            continue
        bs = args.batch_text if kind == "text" else args.batch_image
        for s in range(0, len(idxs), bs):
            encode_batch(idxs[s:s + bs])
            done = min(s + bs, len(idxs))
            if done % (bs * 16) < bs or done == len(idxs):
                print(f"[{time.time()-t0:6.1f}s] {kind}: {done}/{len(idxs)} "
                      f"({done/(time.time()-t0):.1f}/s)", flush=True)

    tmp = args.out + ".tmp.npy"
    np.save(tmp, emb)
    os.rename(tmp, args.out)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    with open(args.out + ".done", "w") as f:
        json.dump({"n": len(items), "sha256": h, "max_dim": args.max_dim,
                   "elapsed_s": round(time.time() - t0, 1)}, f)
    print(f"[{time.time()-t0:6.1f}s] DONE {args.out} sha={h[:16]}", flush=True)


if __name__ == "__main__":
    main()
