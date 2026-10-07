"""Painted hints as vector strokes.

A stroke is a dict: {"rgb": [r, g, b] 0..255, "neutral": bool, "radius": r,
"points": [[x, y], ...]} with x, y and the radius as fractions of the frame
(the radius of the short side). Vector strokes stay small in the project
file, survive any resize, and rasterize to whatever size a pass works at.
A neutral stroke asks for no colour at all: a grey wall, a white shirt.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .hints import Hints


def stroke_ab(stroke: dict) -> tuple[float, float]:
    if stroke.get("neutral"):
        return 0.0, 0.0
    from .color import srgb_to_lab

    rgb = np.asarray(stroke.get("rgb", [128, 128, 128]), np.float32)[None, None] / 255.0
    lab = srgb_to_lab(rgb)[0, 0]
    return float(lab[1]), float(lab[2])


def rasterize(strokes: list[dict], w: int, h: int) -> Hints:
    """Strokes to a hint mask and a/b image at w x h. Later strokes paint
    over earlier ones."""
    import cv2

    ab = np.zeros((h, w, 2), np.float32)
    mask = np.zeros((h, w), np.float32)
    short = min(w, h)
    for st in strokes or []:
        pts = st.get("points") or []
        if not pts:
            continue
        r = max(1, int(round(float(st.get("radius", 0.01)) * short)))
        poly = np.array([[round(x * w), round(y * h)] for x, y in pts], np.int32)
        layer = np.zeros((h, w), np.uint8)
        if len(poly) == 1:
            cv2.circle(layer, tuple(int(v) for v in poly[0]), r, 255, -1)
        else:
            cv2.polylines(layer, [poly], False, 255, thickness=2 * r, lineType=cv2.LINE_8)
            for p in (poly[0], poly[-1]):
                cv2.circle(layer, tuple(int(v) for v in p), r, 255, -1)
        on = layer > 0
        a, b = stroke_ab(st)
        ab[on] = (a, b)
        mask[on] = 1.0
    return Hints(ab, mask)


def strokes_key(strokes) -> str:
    """Changes whenever the strokes do, rounded so float noise doesn't.
    Takes a list of strokes or a {frame: strokes} dict."""
    def norm(st):
        return {"rgb": [int(v) for v in st.get("rgb", [])], "n": bool(st.get("neutral")),
                "r": round(float(st.get("radius", 0)), 5),
                "p": [[round(x, 5), round(y, 5)] for x, y in st.get("points", [])]}
    if isinstance(strokes, dict):
        data = {str(k): [norm(s) for s in v] for k, v in sorted(strokes.items(), key=lambda kv: int(kv[0])) if v}
    else:
        data = [norm(s) for s in strokes or []]
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def hints_in(hints: dict | None, a: int, b: int) -> dict[int, list]:
    """The keyframes of a clip's hints that fall in frames [a, b)."""
    out = {}
    for k, v in (hints or {}).items():
        f = int(k)
        if a <= f < b and v:
            out[f] = v
    return out


def dabs(points, rgb, radius: float = 0.015) -> list[dict]:
    """Small round strokes, one per point: how a saved-colour match lands."""
    return [{"rgb": [int(c) for c in rgb], "neutral": False, "radius": radius, "points": [list(p)]}
            for p in points]
