"""mmeb_spec.py — per-subset extraction spec for MMEB-train FULL.

Every MMEB subset maps to one task kind; a parser turns a parquet row into a
normalized pair record or None (drop, with a stat counter). Verified against
actual rows 2026-07-12 (recon of all 20 subsets).

Kinds -> card/view/lane shapes (CARD-SPEC v1.1; schema-checked):
  retrieval  views {image, text};              lanes image2text + text2image
  cls        views {image, text};              lane image2label
  vqa        views {image-question, text};     lane vqa2text
  grounding  views {image-query, image-crop};  lane imagetext2image
  composed   views {image-query, image};       lane imagetext2image
  i2i        views {image, image-target};      lane image2image
  dialog     views {text, image};              lane text2image
  webqa      views {text, image-info};         lane text2imagetext

MMEB's bundled neg_text/neg_image columns are IGNORED everywhere (documented
too easy — CARD-SPEC frozen rule: mine our own negatives).

EXCLUDED subsets:
  HatefulMemes — binary yes/no labels are degenerate for contrastive pairs
  (50% in-batch false-negative rate) + meme-hate content adds audit risk for
  zero embedding-signal gain. Documented in the data card.
"""
from __future__ import annotations

import re

IMG_TOKEN = "<|image_1|>\n"

Q_PREFIX = "Represent the given image with the following question: "


def _clean(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _strip_qry(qry: str) -> str:
    return _clean(qry.replace(IMG_TOKEN, ""))


def _after(s: str, sep: str) -> str | None:
    _, found, tail = s.partition(sep)
    return _clean(tail) if found else None


def _question(qry: str) -> str | None:
    return _after(_strip_qry(qry), Q_PREFIX)


# Each parser: row -> dict(anchor=(kind, member|None, text|None),
#                          positive=(kind, member|None, text|None),
#                          anchor_text=str, uniq=str-parts, cls_key=None|str)
# member paths keep the parquet's "images/..." prefix (stripped at zip read).

def p_i2t(r):  # MSCOCO_i2t, VisualNews_i2t: image -> caption
    cap = _clean(r["pos_text"])
    if not (20 <= len(cap) <= 600):
        return "drop_len"
    return {"anchor": ("image", r["qry_image_path"], None),
            "positive": ("text", None, cap),
            "anchor_text": cap, "uniq": ("text",)}


def p_t2i(r):  # MSCOCO_t2i / VisualNews_t2i: caption -> image
    q = _strip_qry(r["qry"])
    cap = (_after(q, "matches the given caption: ")
           or _after(q, "Retrieve an image of this news caption."))
    if cap is None:
        return "drop_parse"
    if not (20 <= len(cap) <= 600):
        return "drop_len"
    return {"anchor": ("text", None, cap),
            "positive": ("image", r["pos_image_path"], None),
            "anchor_text": cap, "uniq": ("text",)}


def p_n24(r):  # N24News: news image <-> real caption (labels discarded —
    # 24-way domain classification is near-degenerate; caption is the value)
    cap = _after(_strip_qry(r["qry"]), "for domain classification: ")
    if cap is None:
        return "drop_parse"
    if not (20 <= len(cap) <= 600):
        return "drop_len"
    return {"anchor": ("image", r["qry_image_path"], None),
            "positive": ("text", None, cap),
            "anchor_text": cap, "uniq": ("text",)}


def p_vqa(r):  # A-OKVQA OK-VQA DocVQA InfographicsVQA ChartQA Visual7W
    q = _question(r["qry"])
    a = _clean(r["pos_text"])
    if q is None:
        return "drop_parse"
    if not (10 <= len(q) <= 500) or not (1 <= len(a) <= 300):
        return "drop_len"
    return {"anchor": ("imagetext", r["qry_image_path"], q),
            "positive": ("text", None, a),
            "anchor_text": f"{q}\n{a}", "uniq": ("text", "anchor_image")}


def p_cls(r):  # ImageNet_1K, SUN397, VOC2007: image -> class text
    label = _clean(r["pos_text"])
    if not (2 <= len(label) <= 120):
        return "drop_len"
    return {"anchor": ("image", r["qry_image_path"], None),
            "positive": ("text", None, label),
            "anchor_text": label, "uniq": ("text", "anchor_image"),
            "cls_key": label}


def p_grounding(r):  # MSCOCO: full image + object phrase -> cropped image
    m = re.search(r'labeled as "([^"]+)"', r["qry"])
    if not m:
        return "drop_parse"
    label = _clean(m.group(1))
    if not (2 <= len(label) <= 120):
        return "drop_len"
    return {"anchor": ("imagetext", r["qry_image_path"], label),
            "positive": ("image", r["pos_image_path"], None),
            "anchor_text": label,
            "uniq": ("text", "anchor_image", "positive_image")}


def p_cirr(r):  # CIRR: reference image + modification text -> target image
    mod = _after(_strip_qry(r["qry"]), "with the described changes: ")
    if mod is None:
        return "drop_parse"
    if not (10 <= len(mod) <= 600):
        return "drop_len"
    return {"anchor": ("imagetext", r["qry_image_path"], mod),
            "positive": ("image", r["pos_image_path"], None),
            "anchor_text": mod, "uniq": ("text", "anchor_image")}


def p_nights(r):  # NIGHTS: image -> perceptually similar image
    return {"anchor": ("image", r["qry_image_path"], None),
            "positive": ("image", r["pos_image_path"], None),
            "anchor_text": f"similar-image {r['qry_image_path']}",
            "uniq": ("anchor_image", "positive_image")}


def p_visdial(r):  # VisDial: dialogue text -> image
    dlg = _after(_clean(r["qry"]), "used for image retrieval: ")
    if dlg is None:
        return "drop_parse"
    if not (20 <= len(dlg) <= 2500):
        return "drop_len"
    return {"anchor": ("text", None, dlg),
            "positive": ("image", r["pos_image_path"], None),
            "anchor_text": dlg, "uniq": ("text",)}


def p_webqa(r):  # WebQA: question -> wiki image + related caption
    q = _after(_clean(r["qry"]), "answers this question: ")
    cap = _after(_clean(r["pos_text"]), "related text information: ")
    if q is None or cap is None:
        return "drop_parse"
    if not (10 <= len(q) <= 500) or not (3 <= len(cap) <= 600):
        return "drop_len"
    return {"anchor": ("text", None, q),
            "positive": ("imagetext", r["pos_image_path"], cap),
            "anchor_text": q, "uniq": ("text",)}


# tag = card_id piece (schema ^flf-[a-z0-9][a-z0-9-]{2,30}$); cap = per-class
# cap for cls kinds; after = extract-task dependencies (t2i dedups vs i2t).
SUBSETS = {
    "MSCOCO_i2t":      dict(tag="mci2", kind="retrieval", parse=p_i2t),
    "VisualNews_i2t":  dict(tag="vni2", kind="retrieval", parse=p_i2t),
    "N24News":         dict(tag="n24", kind="retrieval", parse=p_n24),
    "MSCOCO_t2i":      dict(tag="mct2", kind="retrieval", parse=p_t2i,
                            after=["extract__mmeb__MSCOCO_i2t"],
                            carry_text=["MSCOCO_i2t"]),
    "VisualNews_t2i":  dict(tag="vnt2", kind="retrieval", parse=p_t2i,
                            after=["extract__mmeb__VisualNews_i2t"],
                            carry_text=["VisualNews_i2t"]),
    "VisDial":         dict(tag="vdial", kind="dialog", parse=p_visdial),
    "WebQA":           dict(tag="webqa", kind="webqa", parse=p_webqa),
    "CIRR":            dict(tag="cirr", kind="composed", parse=p_cirr),
    "NIGHTS":          dict(tag="nights", kind="i2i", parse=p_nights),
    "MSCOCO":          dict(tag="mgrnd", kind="grounding", parse=p_grounding),
    "ImageNet_1K":     dict(tag="in1k", kind="cls", parse=p_cls, cap=50),
    "SUN397":          dict(tag="sun", kind="cls", parse=p_cls, cap=60),
    "VOC2007":         dict(tag="voc", kind="cls", parse=p_cls, cap=100),
    "A-OKVQA":         dict(tag="aokvq", kind="vqa", parse=p_vqa),
    "OK-VQA":          dict(tag="okvq", kind="vqa", parse=p_vqa),
    "DocVQA":          dict(tag="dvqa", kind="vqa", parse=p_vqa),
    "InfographicsVQA": dict(tag="ivqa", kind="vqa", parse=p_vqa),
    "ChartQA":         dict(tag="cqa", kind="vqa", parse=p_vqa),
    "Visual7W":        dict(tag="v7w", kind="vqa", parse=p_vqa),
}

# kind -> (anchor_view, positive_view, lanes, negatives bucket of positive)
KIND = {
    "retrieval": dict(anchor_view="image", positive_view="text",
                      lanes=["image2text", "text2image"], task="retrieval"),
    "cls":       dict(anchor_view="image", positive_view="text",
                      lanes=["image2label"], task="classification"),
    "vqa":       dict(anchor_view="image-question", positive_view="text",
                      lanes=["vqa2text"], task="vqa"),
    "grounding": dict(anchor_view="image-query", positive_view="image-crop",
                      lanes=["imagetext2image"], task="grounding"),
    "composed":  dict(anchor_view="image-query", positive_view="image",
                      lanes=["imagetext2image"], task="composed-retrieval"),
    "i2i":       dict(anchor_view="image", positive_view="image-target",
                      lanes=["image2image"], task="similarity"),
    "dialog":    dict(anchor_view="text", positive_view="image",
                      lanes=["text2image"], task="dialog-retrieval"),
    "webqa":     dict(anchor_view="text", positive_view="image-info",
                      lanes=["text2imagetext"], task="multimodal-qa"),
    # document sources (ColPali / VisRAG): page image <-> query
    "docmatch":  dict(anchor_view="image", positive_view="text",
                      lanes=["page2query", "query2page"], task="doc-retrieval"),
}


def view_modality(view: str) -> str:
    return view.split("-")[0]
