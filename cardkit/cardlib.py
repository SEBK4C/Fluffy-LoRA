"""cardlib.py — shared helpers for the CARD-SPEC reference kit.

Everything host-specific comes from env vars (all defaults are localhost /
local paths — nothing here may name external hosts):

  FLUFFY_CARDS_ROOT  card store root (default /pool-ssd/fluffy-cards)
  TEACHER_URL        llama.cpp embedding server (default http://127.0.0.1:9020)
  TTS_URL            Kokoro tts-server (default http://127.0.0.1:8096)
  OCR_PY             python interpreter with rapidocr_onnxruntime installed
                     (rendered-image round-trip gate; required for that gate)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import wave

import numpy as np
import requests

ROOT = os.environ.get("FLUFFY_CARDS_ROOT", "/pool-ssd/fluffy-cards")
TEACHER_URL = os.environ.get("TEACHER_URL", "http://127.0.0.1:9020")
TTS_URL = os.environ.get("TTS_URL", "http://127.0.0.1:8096")
SR = 16000  # CARD-SPEC hard rule: CAS audio is 16 kHz mono WAV
MAX_AUDIO_S = 30.0  # CARD-SPEC hard rule (audio_seq_length=750 @ 25 tok/s)


# --- CAS ------------------------------------------------------------------

def cas_path(sha: str) -> str:
    return os.path.join(ROOT, "cas", "sha256", sha[:2], sha)


def cas_put(data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    p = cas_path(sha)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.rename(tmp, p)
    return sha


def cas_ref(sha: str) -> str:
    return f"cas://{sha}"


def resolve_cas(ref: str) -> str:
    assert ref.startswith("cas://"), ref
    return cas_path(ref[6:])


# --- audio ----------------------------------------------------------------

def to_wav16k(data: np.ndarray, sr: int) -> bytes:
    """float or int array, any rate/channels -> 16 kHz mono 16-bit WAV bytes."""
    import soxr

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SR:
        data = soxr.resample(data, sr, SR)
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_info(path: str) -> dict:
    with wave.open(path, "rb") as w:
        return {"sr": w.getframerate(), "channels": w.getnchannels(),
                "duration_s": w.getnframes() / w.getframerate()}


# --- generators -----------------------------------------------------------

def tts(text: str, voice: str = "af_heart") -> bytes:
    """Kokoro TTS (SECONDARY generator, v1.1) -> 16 kHz mono WAV bytes."""
    import soundfile as sf

    r = requests.post(f"{TTS_URL}/v1/audio/speech",
                      json={"input": text, "voice": voice}, timeout=300)
    r.raise_for_status()
    data, sr = sf.read(io.BytesIO(r.content))
    return to_wav16k(np.asarray(data), sr)


_SUPERTONIC = None


def tts_supertonic(text: str, voice: str = "F1") -> bytes:
    """Supertonic-3 TTS (PRIMARY generator, v1.1) -> 16 kHz mono WAV bytes.

    Voices: F1-F5, M1-M5. ONNX, CPU-fast (~1.5 s/clip). MIT SDK,
    OpenRAIL-M weights (license reading: MERGE-RESEARCH §6).
    """
    global _SUPERTONIC
    if _SUPERTONIC is None:
        from supertonic import TTS as _SupertonicTTS
        _SUPERTONIC = _SupertonicTTS()  # supertonic-3 is the default model
    wav, _ = _SUPERTONIC.synthesize(
        text, voice_style=_SUPERTONIC.get_voice_style(voice))
    return to_wav16k(np.asarray(wav).squeeze(), _SUPERTONIC.sample_rate)


def render_text_card(text: str, width: int = 800) -> bytes:
    """Typographic card PNG (the KISS 'rendered' image generator)."""
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    margin, lh = 40, 40
    draw_probe = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    words, lines, line = text.split(), [], ""
    for w in words:
        cand = f"{line} {w}".strip()
        if draw_probe.textlength(cand, font=font) > width - 2 * margin:
            lines.append(line)
            line = w
        else:
            line = cand
    lines.append(line)
    img = Image.new("RGB", (width, 2 * margin + lh * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((margin, margin + lh * i), ln, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


CAPTION_FRAC_MIN, CAPTION_FRAC_MAX = 0.10, 0.20  # v1.1 §E0.1 layout bounds


def render_figure_caption(figure: bytes, caption: str,
                          width: int = 800) -> tuple[bytes, dict]:
    """Figure+caption composite (`image-captioned` doc-lane rendition).

    Normative layout rules (CARD-SPEC v1.1 / TRAINING-CHECKLIST §E0.1):
    caption strip 10-20% of canvas height, font size proportional to canvas
    width, enforced margins, wrap-never-truncate. If the wrapped caption
    would exceed 20%, the canvas widens (fewer lines, taller figure) until
    it fits; if under 10%, the strip pads with whitespace up to the floor.
    Returns (png bytes, layout dict for gen.layout).
    """
    from PIL import Image, ImageDraw, ImageFont

    fig = Image.open(io.BytesIO(figure)).convert("RGB")
    for _ in range(6):
        margin = max(12, width // 50)
        font_px = max(14, width // 32)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_px)
        except OSError:
            font = ImageFont.load_default()
        probe = ImageDraw.Draw(Image.new("RGB", (width, 8)))
        words, lines, line = caption.split(), [], ""
        for w in words:
            cand = f"{line} {w}".strip()
            if probe.textlength(cand, font=font) > width - 2 * margin and line:
                lines.append(line)
                line = w
            else:
                line = cand
        lines.append(line)
        line_h = int(font_px * 1.35)
        strip_h = 2 * margin + line_h * len(lines)
        fig_h = round(fig.height * width / fig.width)
        frac = strip_h / (fig_h + strip_h)
        if frac <= CAPTION_FRAC_MAX:
            break
        width = int(width * 1.3)  # widen -> fewer lines, taller figure
    if frac < CAPTION_FRAC_MIN:  # pad strip up to the floor
        strip_h = int(fig_h * CAPTION_FRAC_MIN / (1 - CAPTION_FRAC_MIN)) + 1
        frac = strip_h / (fig_h + strip_h)

    canvas = Image.new("RGB", (width, fig_h + strip_h), "white")
    canvas.paste(fig.resize((width, fig_h)), (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, ln in enumerate(lines):
        d.text((margin, fig_h + margin + line_h * i), ln,
               fill="black", font=font)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    layout = {"caption_frac": round(frac, 4), "font_px": font_px,
              "margin_px": margin, "canvas": [width, fig_h + strip_h],
              "fig_h": fig_h, "caption_lines": len(lines)}
    return buf.getvalue(), layout


def composite_roundtrip(image_path: str, layout: dict, caption: str) -> dict:
    """Composite gate: OCR the caption STRIP (its boundary is known from
    gen.layout), embed-sim vs caption. Cropping is what makes the gate
    about caption legibility rather than photo content — the OCR detector
    misses small strips on busy full canvases."""
    import tempfile

    from PIL import Image

    with Image.open(image_path) as im:
        strip = im.crop((0, layout["fig_h"], im.width, im.height))
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            strip.save(tmp.name)
            got = ocr_text(tmp.name)
    sim = cos(*teacher_embed([caption, got or " "]))
    return {"roundtrip_sim": round(sim, 4), "ocr": "rapidocr-strip",
            "ocr_text": got}


# --- gates ----------------------------------------------------------------

def teacher_embed(texts: list[str]) -> np.ndarray:
    r = requests.post(f"{TEACHER_URL}/v1/embeddings",
                      json={"input": texts}, timeout=600)
    r.raise_for_status()
    v = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


_WHISPER = None


def asr_wer(wav_path: str, ref_text: str) -> dict:
    """Whisper round-trip gate: transcribe and score WER vs source text."""
    global _WHISPER
    import jiwer
    from faster_whisper import WhisperModel

    if _WHISPER is None:
        _WHISPER = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = _WHISPER.transcribe(wav_path)
    hyp = " ".join(s.text for s in segs).strip()
    norm = jiwer.Compose([jiwer.ToLowerCase(),
                          jiwer.SubstituteRegexes({r"[-–—]": " "}),
                          jiwer.RemovePunctuation(),
                          jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                          jiwer.ReduceToListOfListOfWords()])
    wer = jiwer.wer(ref_text, hyp, reference_transform=norm,
                    hypothesis_transform=norm)
    return {"asr_wer": round(wer, 4), "asr_model": "faster-whisper-small-int8",
            "transcript": hyp}


def ocr_text(image_path: str) -> str:
    """OCR via the env-named interpreter that has rapidocr installed."""
    ocr_py = os.environ.get("OCR_PY")
    if not ocr_py:
        raise RuntimeError("set OCR_PY to a python with rapidocr_onnxruntime")
    code = (
        "import sys, json, numpy as np\n"
        "from rapidocr_onnxruntime import RapidOCR\n"
        "from PIL import Image\n"
        "res,_ = RapidOCR()(np.array(Image.open(sys.argv[1]).convert('RGB')))\n"
        "print(json.dumps(' '.join(r[1] for r in res) if res else ''))\n"
    )
    out = subprocess.run([ocr_py, "-c", code, image_path],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip())


def rendered_roundtrip(image_path: str, ref_text: str) -> dict:
    """Rendered-image gate: OCR the render, embed-sim vs source text."""
    got = ocr_text(image_path)
    sim = cos(*teacher_embed([ref_text, got or " "]))
    return {"roundtrip_sim": round(sim, 4), "ocr": "rapidocr", "ocr_text": got}


# --- ids / dedup ----------------------------------------------------------

DEDUP_PROTOCOL = "anchor-sha256-v1"


def dedup_hash(anchor_text: str) -> str:
    norm = re.sub(r"\s+", " ", anchor_text.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()
