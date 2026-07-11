# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=5.13", "peft", "bitsandbytes",
#                 "numpy", "pillow", "torchvision", "safetensors"]
# ///
"""Fluffy-LoRA trainer v2 — tri-modal QLoRA contrastive embedder on
gemma-4-12b-it. Evolved from train.py (v1); architecture per
MERGE-RESEARCH.md §2 (RATIFIED), data per CARD-SPEC.md v1.1, schedule per
LEARNINGS-V1.md (v1 died inside warmup — STEPS is sized from MEASURED
step-time here, never assumed).

Deltas from v1, each load-bearing:
  * FULL multimodal model (NO .language_model strip); vision/audio towers
    frozen; LoRA r=8 a=16 on q/k/v/o/gate/up/down scoped to the LANGUAGE
    tower only (targets discovered at runtime — suffix-matching would also
    hit tower attention blocks).
  * Pooling is PADDING-SIDE-ROBUST: (mask * arange).argmax(1). v1 pooled
    PAD positions under gemma's default LEFT padding (LEARNINGS-V1
    headline); this indexing is correct for either side.
  * Data = CARD-SPEC v1.1 exposure shards (shards_v2 tars) with
    manifest-index shuffle + deterministic per-lane cursors -> exact
    resume. Corrupt sample => log + deterministic replacement, never crash.
  * Lane-alternating single-modality batches, DDP-safe (lane is a pure
    function of step => all ranks same lane); interleaved minority lane;
    STAGED warmup: stage 1 lanes/weights come from FL_LANES — audio lanes
    enter at a later refresh by changing FL_LANES only (no code change).
  * Objective: symmetric InfoNCE tau=0.02, last-token pool + L2, in-batch
    negatives + k=8 exposure hard negatives, instruction prefix on the
    anchor at encode time (byte-match at eval!), MRL ladder losses over
    prefix dims (native/2048/1024/512/256 — MERGE-RESEARCH §2B; NOTE: live
    text_config.hidden_size is 3840, not the 4096 in §2B, so "native"=3840).
  * NO DistributedDataParallel wrapper: v1 all-gathered embeddings but
    NEVER synced grads (ranks diverged silently). v2 does an explicit
    all_reduce(AVG) over LoRA grads before each optimizer step — correct
    under our multiple-forwards-per-backward pattern where DDP's reducer
    is not.
  * Robustness: atomic checkpoints (tmp+rename; adapter + optimizer +
    scheduler + data cursors + step), rolling retention (ratchet-KEPT +
    last 3 + one per 12h), >90% disk watermark pauses saves with an ALERT
    line, NaN/loss-spike tripwire exits 3 WITHOUT saving (wrapper alerts,
    never auto-restarts a tripwire).

Schedule: STEPS = FL_STEPS, or horizon/step_secs from FL_HORIZON_H (default
312 h = 13 d) x FL_STEP_SECS (measured in the A6 smoke). Warmup 2.5% of the
REAL horizon, cosine to ~0 at the REAL end (LEARNINGS-V1 §f: v1's 200k-step
schedule could never anneal by construction).
lr peak default 1e-4 as v1 intended — NEVER VALIDATED: v1 died at step 1500
having only reached 2.5e-5 (25% of warmup). Watch the tripwire early.

Smoke modes (FL_SMOKE): a1 = one image-lane batch forward; a3 = max
per-device image-batch search; a4 = grad-flow audit (LoRA-only, towers
frozen). FL_MAX_STEPS caps the loop for the A6 20-step DDP smoke and the
kill-9 resume test. In-train eval is intentionally absent: the external
ratchet/watch owns evals (v1 lesson: loss-advance != learning).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shards_v2

# --- config (env) -----------------------------------------------------------
MODEL_ID = os.environ.get("FL_BASE", "google/gemma-4-12b-it")
SHARD_DIR = os.environ.get("FL_SHARDS", "")           # dir containing *.tar
# Default save_dir is a neutral placeholder — the real rig path (big SSD
# mount) is passed via FL_OUT at launch and never committed (public repo).
OUT = Path(os.environ.get("FL_OUT", "/mnt/big-ssd/fluffy/checkpoints-v2"))
# Stage-1 lanes: text+image ONLY (MERGE-RESEARCH §2H staged warmup, GE-2
# pattern). Audio lanes enter at a refresh by changing FL_LANES — no code
# change. Interleaved = first-class minority lane (§2C).
LANES = os.environ.get(
    "FL_LANES",
    "text2text=0.60,image2text=0.15,text2image=0.15,interleaved2text=0.10")
BATCH = int(os.environ.get("FL_BATCH", "8"))          # exposures / device / micro-batch
ACCUM = int(os.environ.get("FL_ACCUM", "2"))
LR = float(os.environ.get("FL_LR", "1e-4"))           # UNVALIDATED peak (v1 never got past 2.5e-5)
TAU = float(os.environ.get("FL_TAU", "0.02"))
K_NEG = int(os.environ.get("FL_K_NEG", "8"))          # CARD-SPEC frozen k
NEG_GRAD = os.environ.get("FL_NEG_GRAD", "1") == "1"  # backprop through negatives
MRL = os.environ.get("FL_MRL", "native,2048,1024,512,256")
INSTRUCT = os.environ.get("FL_INSTRUCT", "1") == "1"  # expected ON (§2D)
MAXCHARS = int(os.environ.get("FL_MAXCHARS", "2000")) # text clip (chars, pre-tokenizer)
ENC_CHUNK = int(os.environ.get("FL_ENC_CHUNK", "32"))
STEPS_ENV = os.environ.get("FL_STEPS", "")
HORIZON_H = float(os.environ.get("FL_HORIZON_H", "312"))   # 13 days
STEP_SECS = os.environ.get("FL_STEP_SECS", "")             # measured (A6)
WARMUP_FRAC = float(os.environ.get("FL_WARMUP_FRAC", "0.025"))
CKPT_MIN = float(os.environ.get("FL_CKPT_MIN", "30"))
CKPT_STEPS = int(os.environ.get("FL_CKPT_STEPS", "0"))     # >0: also every N steps (tests)
RATCHET = os.environ.get("FL_RATCHET", "")                 # ckpt-ratchet-v2.json path
WATERMARK = float(os.environ.get("FL_DISK_WATERMARK", "0.90"))
SPIKE_MULT = float(os.environ.get("FL_SPIKE_MULT", "3.0"))
SPIKE_PATIENCE = int(os.environ.get("FL_SPIKE_PATIENCE", "10"))
MAX_STEPS = int(os.environ.get("FL_MAX_STEPS", "0"))       # 0 = full schedule (smokes cap this)
LOG_EVERY = int(os.environ.get("FL_LOG_EVERY", "5"))
SEED = int(os.environ.get("FL_SEED", "1"))
SMOKE = os.environ.get("FL_SMOKE", "")

EXIT_TRIPWIRE = 3


def log(*a, all_ranks: bool = False):
    r = int(os.environ.get("RANK", "0"))
    if r == 0 or all_ranks:
        print(f"[{time.strftime('%H:%M:%S')}] r{r}", *a, flush=True)


def setup_dist() -> tuple[int, int, torch.device]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        lr_ = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(lr_)
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), torch.device(f"cuda:{lr_}")
    return 0, 1, torch.device("cuda:0")


# --- model ------------------------------------------------------------------

def build_model(dev: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

    torch.manual_seed(SEED)  # IDENTICAL LoRA init on all ranks (v1 seeded per-rank
    #                          with no grad sync — adapters diverged)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": dev.index or 0})
    # FULL multimodal model — NO .language_model strip (v2 core delta).
    assert hasattr(model, "language_model"), "expected unified model with .language_model"

    # LoRA targets: the 7 projection suffixes, but ONLY inside the language
    # tower — plain suffix matching would also hit vision/audio tower blocks.
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
    targets = [n for n, _ in model.named_modules()
               if n.rsplit(".", 1)[-1] in suffixes and "language_model." in n]
    assert targets, "no LoRA targets found under language_model"
    n_tower = len([n for n, _ in model.named_modules()
                   if n.rsplit(".", 1)[-1] in suffixes]) - len(targets)
    log(f"LoRA targets: {len(targets)} lang-tower modules "
        f"({n_tower} tower modules with matching suffixes EXCLUDED)")

    for p in model.parameters():
        p.requires_grad_(False)  # towers + base frozen; PEFT re-enables LoRA only
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=targets)
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()  # needed: frozen embeddings + grad ckpt
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    hidden = model.config.text_config.hidden_size
    log(f"native hidden dim = {hidden} (MERGE-RESEARCH §2B said 4096 — live "
        f"config wins; MRL 'native' rung = {hidden})")
    return model, processor, hidden


def mrl_rungs(hidden: int) -> list[int]:
    rungs = []
    for tok in MRL.split(","):
        d = hidden if tok.strip() == "native" else int(tok)
        if d <= hidden and d not in rungs:
            rungs.append(d)
    return rungs


# --- data -------------------------------------------------------------------

def parse_lanes() -> tuple[list[str], list[float]]:
    lanes, weights = [], []
    for part in LANES.split(","):
        name, _, w = part.partition("=")
        w = float(w or 1.0)
        if w > 0:
            lanes.append(name.strip())
            weights.append(w)
    return lanes, weights


class LaneCursors:
    """Deterministic per-lane shuffled cursors over the manifest index.
    Identical on every rank (same seed, same shard set); rank r consumes
    slice [pos + r*B, pos + (r+1)*B), pos advances by B*world. State is
    exactly (epoch, pos) per lane — saved in every checkpoint."""

    def __init__(self, store: shards_v2.ExposureStore, lanes: list[str], seed: int):
        self.store, self.seed = store, seed
        missing = [ln for ln in lanes if ln not in store.lanes]
        if missing:
            raise SystemExit(f"FL_LANES {missing} absent from shards "
                             f"{list(store.lanes)} — stage config must match shard contents")
        self.state = {ln: {"epoch": 0, "pos": 0} for ln in lanes}
        self._perm: dict[str, list[int]] = {}

    def _order(self, lane: str) -> list[int]:
        if lane not in self._perm:
            st = self.state[lane]
            rng = random.Random(f"{self.seed}/{lane}/{st['epoch']}")
            idx = list(range(len(self.store.lanes[lane])))
            rng.shuffle(idx)
            self._perm[lane] = idx
        return self._perm[lane]

    def take(self, lane: str, rank: int, world: int, batch: int) -> list[tuple[int, str]]:
        st = self.state[lane]
        items = self.store.lanes[lane]
        if st["pos"] + batch * world > len(items):
            st["epoch"] += 1
            st["pos"] = 0
            self._perm.pop(lane, None)
        order = self._order(lane)
        lo = st["pos"] + rank * batch
        out = [items[order[i % len(items)]] for i in range(lo, lo + batch)]
        st["pos"] += batch * world
        return out

    def replacement(self, lane: str, step: int, rank: int, retry: int) -> tuple[int, str]:
        """Deterministic substitute for a corrupt sample; does NOT move the
        shared cursor, so ranks stay aligned."""
        items = self.store.lanes[lane]
        rng = random.Random(f"{self.seed}/bad/{lane}/{step}/{rank}/{retry}")
        return items[rng.randrange(len(items))]


def lane_for_step(lanes: list[str], weights: list[float], step: int) -> str:
    """Pure function of step => every rank picks the same lane (DDP-safe
    lane alternation). Weighted; interleaved rides as the minority lane."""
    rng = random.Random(SEED * 1_000_003 + step)
    return rng.choices(lanes, weights=weights, k=1)[0]


def load_exposure(store, cursors, lane, si, key, step, rank):
    """Materialize one exposure; corrupt sample => log + deterministic
    replacement (never crash — checklist §C: one bad shard must not kill
    day 9)."""
    for retry in range(8):
        try:
            exp, media = store.get(si, key)
            anchor = shards_v2.materialize(
                shards_v2.content_of(exp, exp["anchor"]), media, MAXCHARS)
            if INSTRUCT and exp.get("instruction"):
                # Instruction prefix on the ANCHOR side at encode time
                # (§2D; field standard = instruction first, even before
                # image items — eval must byte-match this construction).
                anchor = [{"type": "text", "text": exp["instruction"]}] + anchor
            pos = shards_v2.materialize(
                shards_v2.content_of(exp, exp["positive"]), media, MAXCHARS)
            negs = [shards_v2.materialize(shards_v2.content_of(exp, n), media, MAXCHARS)
                    for n in exp["negatives"][:K_NEG]]
            return anchor, pos, negs
        except Exception as e:  # noqa: BLE001
            log(f"CORRUPT sample shard={si} key={key}: {type(e).__name__}: {e} "
                f"— replacing (retry {retry})", all_ranks=True)
            si, key = cursors.replacement(lane, step, rank, retry)
    raise SystemExit(f"8 consecutive corrupt samples in lane {lane} — shard set is broken")


# --- encoding ---------------------------------------------------------------

class Encoder:
    def __init__(self, model, processor, dev):
        self.model, self.processor, self.dev = model, processor, dev
        self._fwd_keys = None

    def _filter(self, enc: dict) -> dict:
        if self._fwd_keys is None:
            import inspect
            base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
            self._fwd_keys = set(inspect.signature(base.forward).parameters)
        return {k: v for k, v in enc.items()
                if k in self._fwd_keys or "kwargs" in self._fwd_keys}

    def _forward(self, convs: list) -> torch.Tensor:
        enc = self.processor.apply_chat_template(
            convs, tokenize=True, return_dict=True, return_tensors="pt",
            padding=True)
        enc = {k: (v.to(self.dev, torch.bfloat16) if v.is_floating_point()
                   else v.to(self.dev)) for k, v in enc.items()}
        out = self.model(**self._filter(enc))
        h = getattr(out, "last_hidden_state", None)
        if h is None:
            h = out.hidden_states[-1]
        mask = enc["attention_mask"]
        # Last REAL token under EITHER padding side (gemma tokenizer default
        # is LEFT — v1 pooled pad positions; LEARNINGS-V1 headline finding).
        idx = (mask * torch.arange(mask.shape[1], device=mask.device)).argmax(1)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    def encode(self, contents: list[list[dict]], grad: bool = True) -> torch.Tensor:
        """Batch encode, GROUPED by modality signature — the processor never
        sees a mixed batch (some convs with pixel_values, some without)."""
        def sig(c):
            return (any(i["type"] == "image" for i in c),
                    any(i["type"] == "audio" for i in c))
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, c in enumerate(contents):
            groups[sig(c)].append(i)
        embs: list[torch.Tensor | None] = [None] * len(contents)
        ctx = torch.enable_grad if grad else torch.no_grad
        with ctx():
            for idxs in groups.values():
                for lo in range(0, len(idxs), ENC_CHUNK):
                    chunk = idxs[lo:lo + ENC_CHUNK]
                    convs = [[{"role": "user", "content": contents[i]}] for i in chunk]
                    e = self._forward(convs)
                    for j, i in enumerate(chunk):
                        embs[i] = e[j]
        return torch.stack(embs)  # [N, H] unnormalized; per-rung L2 in the loss


# --- loss -------------------------------------------------------------------

def gather_with_local_grad(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if world == 1:
        return x
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    xs[rank] = x  # local slice keeps its grad path
    return torch.cat(xs)


def mrl_infonce(a, p, n, rungs, rank, world) -> torch.Tensor:
    """Symmetric InfoNCE tau=TAU at every MRL rung (uniform weights, §2B).
    a,p: [B,H] local; n: [n_local,H] local or None.
    a->: candidates = all-rank positives + LOCAL hard negatives. Negatives
    are deliberately NOT all-gathered: per-exposure k varies in real shards
    (DATA text-v001 ships k=0..3; ORCH accepted the shortfall 22:47Z), and
    all_gather demands equal shapes across ranks — variable k would hang
    DDP. Rank-local hard negatives are standard practice and remove the
    hazard by construction.
    p->: candidates = all-rank anchors (hard negatives are positive-side
    views; keeping the reverse direction anchors-only avoids modality
    shortcuts in the candidate set)."""
    B = a.shape[0]
    labels = torch.arange(B, device=a.device) + rank * B
    a_all = gather_with_local_grad(a, rank, world)
    p_all = gather_with_local_grad(p, rank, world)
    total = a.new_zeros(())
    for d in rungs:
        af = F.normalize(a[:, :d].float(), dim=-1)
        pf_all = F.normalize(p_all[:, :d].float(), dim=-1)
        cand = pf_all if n is None else torch.cat(
            [pf_all, F.normalize(n[:, :d].float(), dim=-1)])
        li = F.cross_entropy(af @ cand.T / TAU, labels)
        pf = F.normalize(p[:, :d].float(), dim=-1)
        af_all = F.normalize(a_all[:, :d].float(), dim=-1)
        lt = F.cross_entropy(pf @ af_all.T / TAU, labels)
        total = total + (li + lt) / 2
    return total / len(rungs)


# --- checkpointing ----------------------------------------------------------

def disk_frac(path: Path) -> float:
    u = shutil.disk_usage(path)
    return u.used / u.total


def ratchet_kept() -> set[str]:
    if not RATCHET or not Path(RATCHET).exists():
        return set()
    return set(re.findall(r"step-\d+", Path(RATCHET).read_text()))


def save_checkpoint(model, opt, sched, cursors, step, loss_ema, rank) -> None:
    if rank != 0:
        return
    if disk_frac(OUT) > WATERMARK:
        log(f"ALERT: disk watermark {disk_frac(OUT):.0%} > {WATERMARK:.0%} on "
            f"{OUT} — PAUSING checkpoint saves (training continues)")
        return
    t0 = time.time()
    tmp = OUT / f".tmp-step-{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp / "adapter"))
    torch.save({"step": step,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "cursors": cursors.state,
                "loss_ema": loss_ema,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "config": {k: v for k, v in os.environ.items() if k.startswith("FL_")}},
               tmp / "trainstate.pt")
    (tmp / "meta.json").write_text(json.dumps(
        {"step": step, "saved_unix": time.time(), "loss_ema": loss_ema}))
    final = OUT / f"step-{step}"
    if final.exists():
        shutil.rmtree(final)
    os.rename(tmp, final)  # atomic publish
    apply_retention()
    log(f"checkpoint step-{step} saved in {time.time()-t0:.1f}s "
        f"(disk {disk_frac(OUT):.0%})")


def apply_retention() -> None:
    """Keep: ratchet-KEPT + last 3 + one per 12 h. Delete the rest."""
    ckpts = sorted([d for d in OUT.glob("step-*") if d.is_dir()],
                   key=lambda d: int(d.name.split("-")[1]))
    if len(ckpts) <= 3:
        return
    keep = set(ckpts[-3:]) | {d for d in ckpts if d.name in ratchet_kept()}
    buckets: dict[int, Path] = {}
    for d in ckpts:
        try:
            t = json.loads((d / "meta.json").read_text())["saved_unix"]
        except Exception:  # noqa: BLE001
            t = d.stat().st_mtime
        buckets.setdefault(int(t // 43200), d)  # oldest per 12h bucket
    keep |= set(buckets.values())
    for d in ckpts:
        if d not in keep:
            shutil.rmtree(d)
            log(f"retention: pruned {d.name}")


def find_resume() -> Path | None:
    ckpts = sorted([d for d in OUT.glob("step-*")
                    if d.is_dir() and (d / "trainstate.pt").exists()],
                   key=lambda d: int(d.name.split("-")[1]))
    return ckpts[-1] if ckpts else None


# --- smokes -----------------------------------------------------------------

def one_batch(store, cursors, enc, lane, B, rank, world, step=0):
    picks = cursors.take(lane, rank, world, B)
    anchors, poss, negss = [], [], []
    for si, key in picks:
        a, p, ns = load_exposure(store, cursors, lane, si, key, step, rank)
        anchors.append(a)
        poss.append(p)
        negss.extend(ns)
    return anchors, poss, negss


def smoke_a1(store, cursors, enc, rungs, rank, world):
    lane = "image2text"
    anchors, poss, negss = one_batch(store, cursors, enc, lane, 2, rank, world)
    log(f"A1: lane={lane} anchors={len(anchors)} pos={len(poss)} negs={len(negss)}")
    with torch.no_grad():
        ea = enc.encode(anchors, grad=False)
        ep = enc.encode(poss, grad=False)
    log(f"A1: anchor emb {tuple(ea.shape)} pos emb {tuple(ep.shape)} "
        f"norms a={ea.float().norm(dim=-1).mean():.1f} p={ep.float().norm(dim=-1).mean():.1f}")
    sims = (F.normalize(ea.float(), dim=-1) @ F.normalize(ep.float(), dim=-1).T)
    log(f"A1: cross-modal cos matrix (base, untrained):\n{sims.cpu().numpy().round(3)}")
    log(f"A1 PASS: image path through processor+full-multimodal forward OK "
        f"(peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB)")


def smoke_a4(model, store, cursors, enc, rungs, rank, world):
    anchors, poss, negss = one_batch(store, cursors, enc, "image2text", 2, rank, world)
    a = enc.encode(anchors)
    p = enc.encode(poss)
    n = enc.encode(negss, grad=NEG_GRAD)
    loss = mrl_infonce(a, p, n, rungs, rank, world)
    loss.backward()
    with_grad, lora_grad, tower_grad, tower_req = 0, 0, 0, 0
    for name, prm in model.named_parameters():
        towerish = any(t in name for t in
                       ("vision_tower", "audio_tower", "vision_model",
                        "audio_model", "multi_modal_projector", "embedder"))
        if "language_model" not in name and towerish:
            tower_req += int(prm.requires_grad)
            tower_grad += int(prm.grad is not None)
        if prm.grad is not None:
            with_grad += 1
            lora_grad += int("lora_" in name)
    assert with_grad == lora_grad, f"non-LoRA params got grads! ({with_grad} vs {lora_grad})"
    assert tower_req == 0 and tower_grad == 0, "tower params trainable or grad-carrying!"
    tr = [n_ for n_, p_ in model.named_parameters() if p_.requires_grad]
    assert all("lora_" in n_ and "language_model" in n_ for n_ in tr)
    log(f"A4 PASS: loss={loss.item():.4f}; {with_grad} params w/ grad, all LoRA "
        f"in language tower; {len(tr)} trainable; towers: 0 trainable, 0 grads")


def smoke_a3(model, opt, store, cursors, enc, rungs, rank, world):
    log("A3: max per-device image-lane batch (NF4 + grad-ckpt, k=8 negs, full step)")
    best = 0
    for B in (2, 4, 6, 8, 10, 12, 16):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            anchors, poss, negss = one_batch(store, cursors, enc, "image2text",
                                             B, rank, world)
            a = enc.encode(anchors)
            p = enc.encode(poss)
            n = enc.encode(negss, grad=NEG_GRAD)
            loss = mrl_infonce(a, p, n, rungs, rank, world)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [q for q in model.parameters() if q.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / 2**30
            log(f"A3: B={B:3d} OK loss={loss.item():.3f} peak={peak:.1f} GiB "
                f"step={time.time()-t0:.1f}s")
            best = B
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            log(f"A3: B={B} OOM")
            break
    log(f"A3 RESULT: max per-device image batch = {best} "
        f"(recommend FL_BATCH <= {max(best - 2, 2)} for 14-day headroom)")


# --- main -------------------------------------------------------------------

def main() -> None:
    rank, world, dev = setup_dist()
    OUT.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(str(p) for p in Path(SHARD_DIR).glob("*.tar"))
    if not shard_paths:
        raise SystemExit(f"no shards in FL_SHARDS={SHARD_DIR!r}")
    store = shards_v2.ExposureStore(shard_paths)
    log(f"shards={len(shard_paths)} lane counts={store.counts()}")

    lanes, weights = parse_lanes()
    log(f"active lanes (stage config): {dict(zip(lanes, [round(w,3) for w in weights]))}")

    model, processor, hidden = build_model(dev)
    rungs = mrl_rungs(hidden)
    log(f"MRL rungs: {rungs}")
    if rank == 0:
        model.print_trainable_parameters()
    enc = Encoder(model, processor, dev)
    cursors = LaneCursors(store, lanes, SEED)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)

    # --- schedule: sized to the MEASURED horizon (LEARNINGS-V1 §f) ---------
    if STEPS_ENV:
        steps_total = int(STEPS_ENV)
    elif STEP_SECS:
        steps_total = max(1000, int(HORIZON_H * 3600 / float(STEP_SECS)))
    elif SMOKE or MAX_STEPS:
        steps_total = max(MAX_STEPS, 1000)  # smoke runs: schedule shape irrelevant
    else:
        raise SystemExit("set FL_STEPS or FL_STEP_SECS (measured!) — v1 died "
                         "inside warmup because STEPS was assumed, not measured")
    warm = max(10, int(steps_total * WARMUP_FRAC))
    log(f"schedule: STEPS={steps_total} warmup={warm} ({WARMUP_FRAC:.1%}) "
        f"lr_peak={LR} (UNVALIDATED above 2.5e-5 — LEARNINGS-V1 §f.3) tau={TAU}")

    def lr_lambda(s: int) -> float:
        if s < warm:
            return s / warm
        prog = min(1.0, (s - warm) / max(1, steps_total - warm))
        return 0.5 * (1 + math.cos(math.pi * prog))  # -> ~0 at the REAL end

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # --- resume -------------------------------------------------------------
    step0, loss_ema = 0, None
    ck = find_resume()
    if ck is not None:
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        state = torch.load(ck / "trainstate.pt", map_location="cpu",
                           weights_only=False)
        sd = load_file(ck / "adapter" / "adapter_model.safetensors")
        set_peft_model_state_dict(model, sd)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        cursors.state.update(state["cursors"])
        step0 = state["step"]
        loss_ema = state["loss_ema"]
        log(f"RESUMED from {ck.name}: step={step0} loss_ema={loss_ema} "
            f"cursors={cursors.state}")

    torch.manual_seed(SEED * 7919 + rank * 13 + step0)  # dropout streams differ per rank

    # --- smokes --------------------------------------------------------------
    if SMOKE == "a1":
        smoke_a1(store, cursors, enc, rungs, rank, world)
        return
    if SMOKE == "a4":
        smoke_a4(model, store, cursors, enc, rungs, rank, world)
        return
    if SMOKE == "a3":
        smoke_a3(model, opt, store, cursors, enc, rungs, rank, world)
        return

    # --- train loop -----------------------------------------------------------
    end = min(steps_total, step0 + MAX_STEPS) if MAX_STEPS else steps_total
    last_ckpt_t = time.time()
    spike_run = 0
    step_times: list[float] = []
    log(f"training: step {step0} -> {end} batch={BATCH}x{ACCUM}accum x{world}ranks "
        f"= {BATCH*ACCUM*world} exposures/step, k={K_NEG} hard negs "
        f"(neg_grad={NEG_GRAD})")

    for step in range(step0, end):
        t0 = time.time()
        lane = lane_for_step(lanes, weights, step)  # same on every rank
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(ACCUM):
            anchors, poss, negss = one_batch(store, cursors, enc, lane,
                                             BATCH, rank, world, step)
            a = enc.encode(anchors)
            p = enc.encode(poss)
            n = enc.encode(negss, grad=NEG_GRAD) if negss else None
            loss = mrl_infonce(a, p, n, rungs, rank, world) / ACCUM
            loss.backward()
            step_loss += loss.item()
        if world > 1:  # explicit LoRA-grad sync (see module docstring)
            for prm in trainable:
                if prm.grad is not None:
                    dist.all_reduce(prm.grad, op=dist.ReduceOp.AVG)
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        dt = time.time() - t0
        step_times.append(dt)

        # --- tripwire (halt-and-alert, exit nonzero, NO save) ---------------
        if not math.isfinite(step_loss) or not torch.isfinite(gnorm):
            log(f"TRIPWIRE: non-finite loss={step_loss} gnorm={gnorm} at step "
                f"{step} lane={lane} — halting WITHOUT saving", all_ranks=True)
            sys.exit(EXIT_TRIPWIRE)
        loss_ema = step_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * step_loss
        if step > step0 + 20 and step_loss > SPIKE_MULT * loss_ema + 2.0:
            spike_run += 1
            if spike_run >= SPIKE_PATIENCE:
                log(f"TRIPWIRE: loss spike x{spike_run} (loss={step_loss:.2f} "
                    f"ema={loss_ema:.2f}) at step {step} — halting WITHOUT saving",
                    all_ranks=True)
                sys.exit(EXIT_TRIPWIRE)
        else:
            spike_run = 0

        if step % LOG_EVERY == 0:
            log(f"step {step} lane={lane:16s} loss {step_loss:.4f} "
                f"ema {loss_ema:.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"gnorm {float(gnorm):.2f} {dt:.1f}s/step")

        want_save = (time.time() - last_ckpt_t > CKPT_MIN * 60) or \
                    (CKPT_STEPS and (step + 1) % CKPT_STEPS == 0)
        if want_save:
            save_checkpoint(model, opt, sched, cursors, step + 1, loss_ema, rank)
            last_ckpt_t = time.time()

    save_checkpoint(model, opt, sched, cursors, end, loss_ema, rank)
    if step_times:
        med = sorted(step_times)[len(step_times) // 2]
        eff = BATCH * ACCUM * world
        log(f"MEASURED: median {med:.2f}s/step (n={len(step_times)}, "
            f"{eff} exposures/step) -> 13-day horizon = "
            f"{int(HORIZON_H*3600/med)} steps; set FL_STEP_SECS={med:.2f}")
    log("train loop exit (clean)")


if __name__ == "__main__":
    main()
