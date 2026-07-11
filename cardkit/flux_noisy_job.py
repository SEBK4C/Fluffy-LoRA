# /// script
# requires-python = ">=3.11"
# dependencies = ["diffusers>=0.36", "torch", "transformers", "accelerate",
#                 "sentencepiece", "protobuf", "huggingface_hub", "pillow",
#                 "safetensors"]
# ///
"""flux_noisy_job.py — NOISY-tier FLUX.1-schnell generation on HF Jobs.

CARD-SPEC v1.1 item 3: noisy warmup pairs are NOT cards — loose image<->text
pairs, dedup + sanity only, provenance stamped, outside the card store.

Reads prompts.jsonl from the private dataset repo, generates images
(4 steps, guidance 0), packs WebDataset-style tars of (jpg + json) and
uploads each tar as it completes — a killed job loses at most one shard.

Env: OUT_REPO (dataset repo), START (int), COUNT (int), SIZE (px, default
512), BATCH (default 8), SHARD (imgs/tar, default 2000), RUN_TAG.
"""
import io
import json
import os
import tarfile
import time

import torch
from huggingface_hub import HfApi, hf_hub_download

OUT_REPO = os.environ["OUT_REPO"]
START = int(os.environ.get("START", 0))
COUNT = int(os.environ.get("COUNT", 64))
SIZE = int(os.environ.get("SIZE", 512))
BATCH = int(os.environ.get("BATCH", 8))
SHARD = int(os.environ.get("SHARD", 2000))
RUN_TAG = os.environ.get("RUN_TAG", "smoke")
GEN = os.environ.get("GEN", "flux")  # flux | sdxl-lightning

api = HfApi()
prompts_path = hf_hub_download(OUT_REPO, "prompts.jsonl", repo_type="dataset")
prompts = [json.loads(l) for l in open(prompts_path) if l.strip()]
work = prompts[START:START + COUNT]
print(f"{len(work)} prompts [{START}:{START + COUNT}] "
      f"gen={GEN} size={SIZE} batch={BATCH}")

if GEN == "flux":
    from diffusers import FluxPipeline

    MODEL = "black-forest-labs/FLUX.1-schnell"
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    STEPS, GUIDANCE = 4, 0.0
elif GEN == "sdxl-lightning":
    from diffusers import (EulerDiscreteScheduler, StableDiffusionXLPipeline,
                           UNet2DConditionModel)
    from safetensors.torch import load_file

    BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    MODEL = "ByteDance/SDXL-Lightning (4step) on sdxl-base-1.0"
    unet = UNet2DConditionModel.from_config(
        UNet2DConditionModel.load_config(BASE, subfolder="unet"))
    unet.load_state_dict(load_file(hf_hub_download(
        "ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, unet=unet.to(torch.bfloat16), torch_dtype=torch.bfloat16)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    STEPS, GUIDANCE = 4, 0.0
else:
    raise SystemExit(f"unknown GEN={GEN}")
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

t0 = time.time()
done = 0
shard_idx = START // SHARD
buf = io.BytesIO()
tar = tarfile.open(fileobj=buf, mode="w")


def flush_shard():
    global tar, buf, shard_idx
    tar.close()
    if buf.tell() > 0 and done > 0:
        name = f"shards/{RUN_TAG}-{shard_idx:05d}.tar"
        buf.seek(0)
        api.upload_file(path_or_fileobj=buf, path_in_repo=name,
                        repo_id=OUT_REPO, repo_type="dataset")
        print(f"uploaded {name} ({buf.getbuffer().nbytes // 1024} KiB)")
    shard_idx += 1
    buf = io.BytesIO()
    tar = tarfile.open(fileobj=buf, mode="w")


for i in range(0, len(work), BATCH):
    chunk = work[i:i + BATCH]
    gens = [torch.Generator("cuda").manual_seed(START + i + j)
            for j in range(len(chunk))]
    images = pipe([w["text"] for w in chunk], num_inference_steps=STEPS,
                  guidance_scale=GUIDANCE, height=SIZE, width=SIZE,
                  generator=gens).images
    for j, (w, img) in enumerate(zip(chunk, images)):
        jb = io.BytesIO()
        img.save(jb, format="JPEG", quality=90)
        meta = {"pid": w["pid"], "text": w["text"], "src": w["src"],
                "seed": START + i + j, "model": MODEL, "steps": STEPS,
                "size": SIZE, "tier": "noisy",
                "gen": {"model": GEN, "version": "hf-jobs"}}
        for ext, data in (("jpg", jb.getvalue()),
                          ("json", json.dumps(meta).encode())):
            ti = tarfile.TarInfo(f"{w['pid']}.{ext}")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
        done += 1
    if done % SHARD < BATCH:
        flush_shard()
    if done % 64 < BATCH:
        rate = done / (time.time() - t0)
        print(f"{done}/{len(work)}  {rate:.2f} img/s  "
              f"eta {(len(work) - done) / max(rate, 0.01) / 60:.1f} min",
              flush=True)

flush_shard()
elapsed = time.time() - t0
summary = {"run_tag": RUN_TAG, "start": START, "count": done, "size": SIZE,
           "batch": BATCH, "elapsed_s": round(elapsed, 1),
           "img_per_s": round(done / elapsed, 3)}
api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(summary, indent=1).encode()),
                path_in_repo=f"runs/{RUN_TAG}-{START}.json",
                repo_id=OUT_REPO, repo_type="dataset")
print("SUMMARY", json.dumps(summary))
