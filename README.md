# Fluffy-LoRA 🐶🐶🐶

> One body, three heads. Text, image, audio — one embedding space.

Named for **Fluffy**, Hagrid's three-headed dog (the family resemblance to Cerberus
is not a coincidence). Each head is an input modality; the body is a single shared
embedding space.

The trick: **no bolted-on encoders**. No CLIP, no CLAP, no Whisper stapled together
with projection layers. `google/gemma-4-12b-it` already ingests text, images, and
audio natively into one backbone — so we QLoRA the language tower, freeze the native
towers, and pool the last token into one L2-normed embedding. Three heads, one body.

## Status: raw alpha, building in public (2026-07-11)

- **v1 (text lane only)** is training right now on 2× RTX 4090. Loss is falling;
  retrieval has not moved yet. `state/ckpt-ratchet.json` is the honest scoreboard —
  `best_checkpoint` is still `"none"`, and it stays that way until a checkpoint
  beats the frozen eval by more than ε. No cherry-picking.
- **v2 (all three heads)** is being staged: real interleaved media
  (MMEB / ColPali / LibriSpeech / MLS) mixed with the frozen synthetic text corpus.

## Recipe

- **Base**: `google/gemma-4-12b-it`, NF4 double-quant QLoRA — r=8, α=16, dropout
  0.05, targets `q,k,v,o,gate,up,down` (~32.8M trainable, 0.27%)
- **Objective**: symmetric InfoNCE (τ=0.02), last-token pooling + L2 norm, in-batch
  negatives + in-band hard negatives (0.75 ≤ teacher sim ≤ 0.92)
- **Teacher** (text lanes): Qwen3-Embedding-8B for similarity bands + dedup;
  cross-modal lanes use dataset ground-truth pairs (image↔caption, speech↔transcript,
  page↔query)
- **Eval**: frozen retrieval sets, ratcheted — checkpoints are REJECTED by default

## Files

| File | What |
|---|---|
| `train.py` | v1 text-lane QLoRA trainer (DDP, uv inline deps) |
| `ratchet_eval.py` | kept-only checkpoint ratchet over the frozen eval |
| `eval_station.sh` | run evals on a 12 GB RTX 3080 Ti (NF4) |
| `prep_rig.sh` | stage the training env on the rig |
| `state/ckpt-ratchet.json` | live scoreboard — wins and losses, all of them |

Adapters land at [huggingface.co/SEBK4C/Fluffy-LoRA](https://huggingface.co/SEBK4C/Fluffy-LoRA)
when (if) the ratchet accepts one.
