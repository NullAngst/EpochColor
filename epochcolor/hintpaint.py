"""Painted hints as vector strokes.

A stroke is a dict: {"rgb": [r, g, b] 0..255, "neutral": bool, "radius": r,
"points": [[x, y], ...]} with x, y and the radius as fractions of the frame
(the radius of the short side). On video a stroke may also carry "reach":
how many frames either side its fix travels, 0 for its own frame only.
Without it a stroke reaches through its whole shot. "from" marks a stroke
carried over from another frame (paint by frame), so carrying again
replaces it instead of piling up copies. Vector strokes stay small in the project
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
        d = {"rgb": [int(v) for v in st.get("rgb", [])], "n": bool(st.get("neutral")),
             "r": round(float(st.get("radius", 0)), 5),
             "p": [[round(x, 5), round(y, 5)] for x, y in st.get("points", [])]}
        if reach_of(st) >= 0:
            d["reach"] = reach_of(st)  # only when set, so older projects keep their keys
        return d
    if isinstance(strokes, dict):
        data = {str(k): [norm(s) for s in v] for k, v in sorted(strokes.items(), key=lambda kv: int(kv[0])) if v}
    else:
        data = [norm(s) for s in strokes or []]
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def reach_of(stroke: dict) -> int:
    """Frames either side a stroke's fix travels; -1 is the whole shot."""
    r = stroke.get("reach")
    try:
        return -1 if r is None else max(-1, int(r))
    except (TypeError, ValueError):
        return -1


def group_by_reach(strokes: list) -> dict[int, list]:
    out: dict[int, list] = {}
    for st in strokes or []:
        out.setdefault(reach_of(st), []).append(st)
    return out


def move_strokes(strokes: list, flow: np.ndarray) -> list:
    """Strokes moved by optical flow F (from Flow(a, b): a(x) ~ b(x + F(x))),
    so they land on the same things in frame b. Each point moves by the
    median flow under the brush, which keeps a stroke whole when the flow is
    noisy at its edge."""
    h, w = flow.shape[:2]
    short = min(w, h)
    out = []
    for st in strokes:
        r = max(1, int(round(float(st.get("radius", 0.01)) * short)))
        pts = []
        for x, y in st.get("points") or []:
            px, py = int(round(x * (w - 1))), int(round(y * (h - 1)))
            win = flow[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1].reshape(-1, 2)
            dx, dy = (np.median(win, axis=0) if len(win) else (0.0, 0.0))
            pts.append([float(np.clip(x + dx / max(1, w - 1), 0, 1)), float(np.clip(y + dy / max(1, h - 1), 0, 1))])
        out.append({**st, "points": pts})
    return out


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
