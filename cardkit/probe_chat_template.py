#!/usr/bin/env python3
"""probe_chat_template.py — CARD-SPEC Phase 1 reality check.

Proves (or breaks) the spec's core bet: card views stored as gemma-4
chat-template content arrays feed straight into
processor.apply_chat_template. Measures actual token cost per modality.

Run:  GEMMA4_SNAPSHOT=<path-to-hf-snapshot> python cardkit/probe_chat_template.py <workdir>
Writes <workdir>/probe_report.json and prints a summary table.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from transformers import AutoProcessor

SNAPSHOT = os.environ.get("GEMMA4_SNAPSHOT")
if not SNAPSHOT:
    sys.exit("set GEMMA4_SNAPSHOT to the local gemma-4-12b-it snapshot dir")
WORKDIR = sys.argv[1] if len(sys.argv) > 1 else "probe-media"
os.makedirs(WORKDIR, exist_ok=True)

SR = 16000
SAMPLE_TEXT = (
    "A red fox pauses on a frost-covered stone wall at dawn, "
    "its breath visible in the cold air."
)


def render_card(path: str, size: tuple[int, int], text: str) -> str:
    """Minimal typographic card render (the E-map 'rendered' image path)."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) > size[0] // 12:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    for i, ln in enumerate(lines):
        d.text((16, 16 + 22 * i), ln, fill="black")
    img.save(path)
    return path


def synth_audio(path: str, seconds: float) -> str:
    """Placeholder waveform — token cost depends only on duration."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    wav = 0.1 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    sf.write(path, wav.astype(np.float32), SR)
    return path


def toklen(processor, content) -> tuple[int, list[int]]:
    out = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
    )
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    while ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids), [int(i) for i in ids]


def main() -> None:
    processor = AutoProcessor.from_pretrained(SNAPSHOT)
    tok = processor.tokenizer
    report: dict = {"snapshot": os.path.basename(SNAPSHOT.rstrip("/")),
                    "processor_class": type(processor).__name__, "cases": {}}

    # --- baselines -------------------------------------------------------
    n_empty, _ = toklen(processor, [{"type": "text", "text": ""}])
    n_text, ids_text = toklen(processor, [{"type": "text", "text": SAMPLE_TEXT}])
    body_tokens = n_text - n_empty
    report["cases"]["turn_overhead_tokens"] = n_empty
    report["cases"]["text_view"] = {"total": n_text, "body": body_tokens}

    # --- image: is 280/img fixed or resolution-dependent? -----------------
    img_cases = {}
    for label, size in [("320x240", (320, 240)), ("640x480", (640, 480)),
                        ("896x896", (896, 896)), ("1600x1200", (1600, 1200)),
                        ("wide_1600x400", (1600, 400))]:
        p = render_card(os.path.join(WORKDIR, f"card_{label}.png"), size, SAMPLE_TEXT)
        n, ids = toklen(processor, [{"type": "image", "image": p}])
        img_id = 258880
        img_cases[label] = {"total": n, "image_soft_tokens": ids.count(img_id),
                            "wrapper_tokens": n - n_empty - ids.count(img_id)}
    report["cases"]["image_views"] = img_cases

    # --- audio: tokens vs duration, cap behaviour -------------------------
    aud_cases = {}
    for secs in [1, 5, 10, 30, 45]:
        p = synth_audio(os.path.join(WORKDIR, f"a_{secs}s.wav"), secs)
        try:
            n, ids = toklen(processor, [{"type": "audio", "audio": p}])
            aud_id = 258881
            aud_cases[f"{secs}s"] = {"total": n, "audio_soft_tokens": ids.count(aud_id),
                                     "tokens_per_sec": round(ids.count(aud_id) / secs, 2)}
        except Exception as e:  # noqa: BLE001 — cap behaviour is exactly what we probe
            aud_cases[f"{secs}s"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    report["cases"]["audio_views"] = aud_cases

    # --- interleaved: sum of parts or extra separators? -------------------
    img_p = os.path.join(WORKDIR, "card_640x480.png")
    aud_p = os.path.join(WORKDIR, "a_5s.wav")
    n_il, ids_il = toklen(processor, [
        {"type": "image", "image": img_p},
        {"type": "text", "text": SAMPLE_TEXT},
        {"type": "audio", "audio": aud_p},
    ])
    report["cases"]["interleaved_view"] = {
        "total": n_il,
        "image_soft": ids_il.count(258880),
        "audio_soft": ids_il.count(258881),
        "text_body": body_tokens,
        "sum_of_single_view_totals_minus_shared_overhead":
            img_cases["640x480"]["total"] + aud_cases["5s"].get("total", 0)
            + n_text - 2 * n_empty,
    }

    # --- does a bare content array (no role wrapper) work? ----------------
    try:
        processor.apply_chat_template(
            [{"type": "text", "text": SAMPLE_TEXT}], tokenize=True)
        report["cases"]["bare_content_array"] = "accepted"
    except Exception as e:  # noqa: BLE001
        report["cases"]["bare_content_array"] = f"REJECTED: {type(e).__name__}: {e}"[:200]

    # --- does an unresolved cas:// ref work? -------------------------------
    try:
        toklen(processor, [{"type": "image", "image": "cas://deadbeef"}])
        report["cases"]["cas_ref_unresolved"] = "accepted (?!)"
    except Exception as e:  # noqa: BLE001
        report["cases"]["cas_ref_unresolved"] = f"REJECTED: {type(e).__name__}: {e}"[:200]

    # --- exact rendered string for the spec appendix ----------------------
    report["rendered_text_view"] = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": SAMPLE_TEXT}]}],
        tokenize=False, add_generation_prompt=False)
    report["decoded_interleaved_head"] = tok.decode(ids_il[:40])

    out = os.path.join(WORKDIR, "probe_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)[:4000])
    print(f"\nfull report: {out}")


if __name__ == "__main__":
    main()
