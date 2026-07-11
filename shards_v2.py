# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow"]
# ///
"""shards_v2.py — CARD-SPEC v1.1 exposure shards: pack, index, read.

FORMAT CONTRACT (v2 trainer + all shard producers use this module):

  Shard = POSIX tar ("WebDataset tar" convention: members grouped by key).
  Each sample is SELF-CONTAINED:
    <key>.json            exposure (CARD-SPEC schema) with content arrays
                          MATERIALIZED: text inline; media items reference
                          tar members via "member://<mname>" where the tar
                          member is named "<key>.<mname>".
    <key>.<mname>         media bytes (mname = "<sha16>.png" | "<sha16>.wav").
                          CAS audio rule carries over: 16 kHz mono 16-bit WAV.

  Sidecar index = <shard>.idx.json (manifest-index shuffle needs random
  access, not streaming):
    {"format": "fluffy-exposure-shard-v1",
     "samples": {key: {"lane": str, "members": {name: [offset, size]}}},
     "lanes": {lane: [keys...]}, "counts": {lane: n}}

  Exposure JSON (CARD-SPEC v1.1 exposure schema + materialized content):
    {"lane": "image2text", "instruction": "Retrieve ...",
     "anchor":   {"card": "flf-...", "view": "image", "content": [...]},
     "positive": {"card": "flf-...", "view": "text",  "content": [...]},
     "negatives": [{"card": "...", "view": "...", "content": [...],
                    "miner": "..."} x k<=8]}

Interleaved contents obey the Gemma-4 ordering rule (image -> text -> audio),
validated at pack time. Reader is stdlib-only (+PIL/numpy at materialize).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import wave

_ORDER = {"image": 0, "text": 1, "audio": 2}
FORMAT = "fluffy-exposure-shard-v1"


def check_modality_order(content: list[dict]) -> None:
    """CARD-SPEC frozen rule: all image items before all text before audio."""
    ranks = [_ORDER[it["type"]] for it in content]
    if ranks != sorted(ranks):
        raise ValueError(f"modality order violation (need image->text->audio): "
                         f"{[it['type'] for it in content]}")


# --- packing ----------------------------------------------------------------

class ShardWriter:
    """Writes samples into a tar + builds the .idx.json sidecar."""

    def __init__(self, tar_path: str):
        self.tar_path = tar_path
        self.tmp = tar_path + ".tmp"
        self.tar = tarfile.open(self.tmp, "w")  # uncompressed: random access
        self.keys: list[str] = []
        self.lanes: dict[str, list[str]] = {}

    def add(self, key: str, exposure: dict, media: dict[str, bytes]) -> None:
        """media: {mname: bytes}; exposure content arrays must already use
        member://<mname> refs for non-text items."""
        for side in ("anchor", "positive"):
            check_modality_order(exposure[side]["content"])
        for n in exposure.get("negatives", []):
            check_modality_order(n["content"])
        blob = json.dumps(exposure, ensure_ascii=False).encode()
        self._member(f"{key}.json", blob)
        for mname, data in media.items():
            self._member(f"{key}.{mname}", data)
        self.keys.append(key)
        self.lanes.setdefault(exposure["lane"], []).append(key)

    def _member(self, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        self.tar.addfile(info, io.BytesIO(data))

    def close(self) -> str:
        self.tar.close()
        os.rename(self.tmp, self.tar_path)
        idx = build_index(self.tar_path)
        return idx


def build_index(tar_path: str) -> str:
    """Scan a shard tar, write <tar>.idx.json with member offsets."""
    samples: dict[str, dict] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            key, _, suffix = m.name.partition(".")
            s = samples.setdefault(key, {"lane": None, "members": {}})
            s["members"][suffix] = [m.offset_data, m.size]
        with open(tar_path, "rb") as f:
            for key, s in samples.items():
                off, size = s["members"]["json"]
                f.seek(off)
                s["lane"] = json.loads(f.read(size))["lane"]
    lanes: dict[str, list[str]] = {}
    for key, s in samples.items():
        lanes.setdefault(s["lane"], []).append(key)
    idx = {"format": FORMAT, "samples": samples, "lanes": lanes,
           "counts": {ln: len(ks) for ln, ks in lanes.items()}}
    idx_path = tar_path + ".idx.json"
    tmp = idx_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f)
    os.rename(tmp, idx_path)
    return idx_path


def media_name(data: bytes, ext: str) -> str:
    return f"{hashlib.sha256(data).hexdigest()[:16]}.{ext}"


# --- reading ----------------------------------------------------------------

class ExposureStore:
    """Random-access reader over a set of shards via their .idx.json.

    lane_index(lane) -> list of (shard_i, key); get(shard_i, key) ->
    (exposure dict, {mname: bytes}). Thread-safe reads via os.pread.
    """

    def __init__(self, shard_paths: list[str]):
        self.paths = sorted(shard_paths)
        self.idx: list[dict] = []
        self._fds: list[int | None] = [None] * len(self.paths)
        for p in self.paths:
            with open(p + ".idx.json") as f:
                idx = json.load(f)
            if idx.get("format") != FORMAT:
                raise ValueError(f"{p}: unknown shard format {idx.get('format')}")
            self.idx.append(idx)
        self.lanes: dict[str, list[tuple[int, str]]] = {}
        for i, idx in enumerate(self.idx):
            for lane, keys in idx["lanes"].items():
                self.lanes.setdefault(lane, []).extend((i, k) for k in sorted(keys))

    def counts(self) -> dict[str, int]:
        return {ln: len(v) for ln, v in self.lanes.items()}

    def _fd(self, shard_i: int) -> int:
        if self._fds[shard_i] is None:
            self._fds[shard_i] = os.open(self.paths[shard_i], os.O_RDONLY)
        return self._fds[shard_i]

    def get(self, shard_i: int, key: str) -> tuple[dict, dict[str, bytes]]:
        members = self.idx[shard_i]["samples"][key]["members"]
        fd = self._fd(shard_i)
        off, size = members["json"]
        exposure = json.loads(os.pread(fd, size, off))
        media = {}
        for mname, (off, size) in members.items():
            if mname != "json":
                media[mname] = os.pread(fd, size, off)
        return exposure, media


def content_of(exposure: dict, ref: dict) -> list[dict]:
    """Content array for an anchor/positive/negative entry. Supports both
    sample layouts in the wild tonight:
      - inline:   ref["content"] = [...]                (this module's packer)
      - resolved: exposure["resolved"]["<card>/<view>"] (DATA text-v001 tars)
    """
    if "content" in ref:
        return ref["content"]
    return exposure["resolved"][f"{ref['card']}/{ref['view']}"]


def wav_to_float32(data: bytes):
    """16 kHz mono 16-bit WAV bytes -> float32 numpy in [-1, 1] (CARD-SPEC:
    the processor presumes raw numpy audio is 16 kHz)."""
    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"CAS audio rule violated: sr={w.getframerate()} "
                             f"ch={w.getnchannels()} width={w.getsampwidth()}")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype("float32") / 32768.0


def materialize(content: list[dict], media: dict[str, bytes],
                max_text_chars: int = 4000) -> list[dict]:
    """member:// refs -> PIL.Image / 16 kHz float32 numpy, ready for
    processor.apply_chat_template (CARD-SPEC measured-reality payload forms).
    Text is clipped by CHARS here — never tokenizer-truncate multimodal
    sequences (cutting soft tokens desyncs pixel/audio features)."""
    from PIL import Image

    out = []
    for it in content:
        t = it["type"]
        if t == "text":
            out.append({"type": "text", "text": it["text"][:max_text_chars]})
        elif t == "image":
            ref = it["image"]
            data = media[ref[len("member://"):]]
            out.append({"type": "image",
                        "image": Image.open(io.BytesIO(data)).convert("RGB")})
        elif t == "audio":
            ref = it["audio"]
            data = media[ref[len("member://"):]]
            out.append({"type": "audio", "audio": wav_to_float32(data)})
        else:
            raise ValueError(f"unknown content type {t}")
    return out
