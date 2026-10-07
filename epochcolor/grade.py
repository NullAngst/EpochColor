"""Colour grading: one engine for the preview, stills, photos and export.

A grade is a plain dict, so it saves in the project JSON and diffs cleanly.
It works on display-referred sRGB in 32-bit float and nothing is clipped
until the final quantize, so pushing gain up and pulling it back down
loses nothing. Order:

    white balance (linear light) -> offset, lift, gamma, gain -> contrast
    -> curves (master, then R, G, B) -> saturation, vibrance, hue curves
    -> secondary (HSL qualifier x shape mask) -> 3D LUT

Grades live per shot and every value can be keyframed: a track holds keys
at clip frames, and values in between are interpolated, linear or eased.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np

# --------------------------------------------------------------- defaults


def default_grade() -> dict:
    return {
        "lift": [0.0, 0.0, 0.0], "gamma": [0.0, 0.0, 0.0], "gain": [0.0, 0.0, 0.0],
        "offset": [0.0, 0.0, 0.0],
        "lift_m": 0.0, "gamma_m": 0.0, "gain_m": 0.0, "offset_m": 0.0,
        "temperature": 0.0, "tint": 0.0,
        "contrast": 1.0, "pivot": 0.435,
        "saturation": 1.0, "vibrance": 0.0,
        "curves": {"master": [[0, 0], [1, 1]], "r": [[0, 0], [1, 1]],
                   "g": [[0, 0], [1, 1]], "b": [[0, 0], [1, 1]]},
        "hue_sat": [],  # [[hue 0..1, multiplier]], periodic
        "hue_hue": [],  # [[hue 0..1, shift in turns]], periodic
        "qualifier": {"enabled": False, "hue": 0.0, "hue_width": 0.08, "sat_lo": 0.1, "sat_hi": 1.0,
                      "lum_lo": 0.0, "lum_hi": 1.0, "soft": 0.05,
                      "d_hue": 0.0, "d_sat": 1.0, "d_lum": 0.0},
        "mask": {"shape": "none", "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3, "angle": 0.0,
                 "feather": 0.1, "invert": False},
        "lut": None, "lut_mix": 1.0,
    }


def is_identity(g: dict | None) -> bool:
    return g is None or _key(g) == _IDENTITY_KEY


def _key(g: dict) -> str:
    return json.dumps(g, sort_keys=True, default=str)


_IDENTITY_KEY = _key(default_grade())


def merged(g: dict | None) -> dict:
    """A full grade with any missing keys filled from the defaults, so older
    projects and partial dicts both work."""
    base = default_grade()
    if not g:
        return base

    def fill(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict) and k != "curves":
                fill(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)
    fill(base, g)
    return base


# -------------------------------------------------------------- keyframes


def new_track(g: dict | None = None, frame: int = 0) -> dict:
    return {"keys": [{"frame": int(frame), "ease": "linear", "grade": merged(g)}]}


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def _lerp(a, b, t):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a if t < 0.5 else b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _lerp(a.get(k), b.get(k), t) if k in b else a[k] for k in a}
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_lerp(x, y, t) for x, y in zip(a, b)]
    return a if t < 0.5 else b  # strings, curves with different point counts


def evaluate(track: dict | None, frame: int) -> dict:
    """The grade at a clip frame."""
    if not track or not track.get("keys"):
        return default_grade()
    keys = sorted(track["keys"], key=lambda k: k["frame"])
    if len(keys) == 1 or frame <= keys[0]["frame"]:
        return merged(keys[0]["grade"])
    if frame >= keys[-1]["frame"]:
        return merged(keys[-1]["grade"])
    for a, b in zip(keys, keys[1:]):
        if a["frame"] <= frame <= b["frame"]:
            t = (frame - a["frame"]) / max(1, b["frame"] - a["frame"])
            if a.get("ease") == "ease":
                t = _smooth(t)
            return _lerp(merged(a["grade"]), merged(b["grade"]), t)
    return merged(keys[-1]["grade"])


def key_at(track: dict, frame: int) -> dict | None:
    return next((k for k in track.get("keys", []) if k["frame"] == frame), None)


def set_value(track: dict, frame: int, grade: dict, autokey: bool = True) -> dict:
    """Store an edited grade. With one key (or none) the grade is static and
    that key changes. With several, editing on a key changes it and editing
    between keys adds a key there."""
    keys = track.setdefault("keys", [])
    k = key_at(track, frame)
    if k is not None:
        k["grade"] = merged(grade)
    elif len(keys) <= 1 or not autokey:
        if keys:
            keys[0]["grade"] = merged(grade)
        else:
            keys.append({"frame": int(frame), "ease": "linear", "grade": merged(grade)})
    else:
        keys.append({"frame": int(frame), "ease": "linear", "grade": merged(grade)})
        keys.sort(key=lambda x: x["frame"])
    return track


# ----------------------------------------------------------------- curves


def _pchip(xs, ys, grid):
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(xs, ys, extrapolate=True)(grid)


@lru_cache(maxsize=256)
def _curve_lut(points_key: str, n: int = 1024) -> np.ndarray:
    pts = sorted(json.loads(points_key))
    xs = np.array([p[0] for p in pts], np.float64)
    ys = np.array([p[1] for p in pts], np.float64)
    grid = np.linspace(0, 1, n)
    if len(xs) < 2:
        return grid.astype(np.float32)
    # unique x, keep the last
    _, idx = np.unique(xs, return_index=True)
    xs, ys = xs[idx], ys[idx]
    if len(xs) == 2:
        out = np.interp(grid, xs, ys)
    else:
        out = _pchip(xs, ys, grid)
    # extend flat outside the first and last point, like every grading app
    out = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], out))
    return out.astype(np.float32)


@lru_cache(maxsize=256)
def _hue_lut(points_key: str, default: float, n: int = 360) -> np.ndarray:
    """Periodic curve over hue 0..1."""
    pts = sorted(json.loads(points_key))
    grid = np.linspace(0, 1, n, endpoint=False)
    if not pts:
        return np.full(n, default, np.float32)
    xs = np.array([p[0] % 1.0 for p in pts])
    ys = np.array([p[1] for p in pts])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    # wrap one period each side so the curve joins smoothly at red
    X = np.concatenate([xs - 1, xs, xs + 1])
    Y = np.concatenate([ys, ys, ys])
    if len(xs) == 1:
        return np.full(n, ys[0], np.float32)
    return _pchip(X, Y, grid).astype(np.float32)


def _apply_lut1(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    n = len(lut) - 1
    f = np.clip(x, 0.0, 1.0) * n
    i = np.minimum(f.astype(np.int32), n - 1)
    t = f - i
    y = lut[i] * (1 - t) + lut[i + 1] * t
    # beyond 0..1 the curve continues with the slope at its end
    lo, hi = x < 0, x > 1
    if lo.any():
        y = np.where(lo, lut[0] + (lut[1] - lut[0]) * n * x, y)
    if hi.any():
        y = np.where(hi, lut[-1] + (lut[-1] - lut[-2]) * n * (x - 1), y)
    return y.astype(np.float32)


def _is_identity_curve(pts) -> bool:
    return [list(map(float, p)) for p in sorted(pts)] == [[0.0, 0.0], [1.0, 1.0]]


# -------------------------------------------------------------------- LUTs


@lru_cache(maxsize=8)
def load_cube(path: str) -> tuple[np.ndarray, tuple, tuple]:
    """A .cube file: (table, domain_min, domain_max). 1D LUTs come back as
    3D tables too, so applying them is one code path."""
    size1 = size3 = None
    dmin, dmax = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    rows = []
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("TITLE"):
            continue
        if up.startswith("LUT_3D_SIZE"):
            size3 = int(line.split()[1])
        elif up.startswith("LUT_1D_SIZE"):
            size1 = int(line.split()[1])
        elif up.startswith("DOMAIN_MIN"):
            dmin = tuple(float(v) for v in line.split()[1:4])
        elif up.startswith("DOMAIN_MAX"):
            dmax = tuple(float(v) for v in line.split()[1:4])
        elif up[0].isalpha():
            continue  # LUT_3D_INPUT_RANGE and other extensions
        else:
            rows.append([float(v) for v in line.split()[:3]])
    data = np.array(rows, np.float32)
    if size3:
        if len(data) != size3 ** 3:
            raise ValueError(f"{path}: expected {size3 ** 3} rows, found {len(data)}")
        # .cube order: red changes fastest
        return data.reshape(size3, size3, size3, 3), dmin, dmax  # [b, g, r, ch]
    if size1:
        if len(data) != size1:
            raise ValueError(f"{path}: expected {size1} rows, found {len(data)}")
        n = 33
        grid = np.linspace(0, 1, n)
        src = np.linspace(0, 1, size1)
        chans = [np.interp(grid, src, data[:, c]) for c in range(3)]
        b, g, r = np.meshgrid(grid, grid, grid, indexing="ij")
        table = np.stack([np.interp(r, grid, chans[0]), np.interp(g, grid, chans[1]),
                          np.interp(b, grid, chans[2])], axis=-1).astype(np.float32)
        return table, dmin, dmax
    raise ValueError(f"{path}: no LUT_3D_SIZE or LUT_1D_SIZE")


def apply_cube(rgb: np.ndarray, table: np.ndarray, dmin, dmax) -> np.ndarray:
    """Trilinear lookup. rgb HxWx3 float."""
    n = table.shape[0]
    lo = np.asarray(dmin, np.float32)
    hi = np.asarray(dmax, np.float32)
    x = np.clip((rgb - lo) / (hi - lo), 0, 1) * (n - 1)
    i0 = np.minimum(x.astype(np.int32), n - 2)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    def T(b, g, r):
        return table[b, g, r]
    c00 = T(b0, g0, r0) * (1 - fr) + T(b0, g0, r0 + 1) * fr
    c01 = T(b0, g0 + 1, r0) * (1 - fr) + T(b0, g0 + 1, r0 + 1) * fr
    c10 = T(b0 + 1, g0, r0) * (1 - fr) + T(b0 + 1, g0, r0 + 1) * fr
    c11 = T(b0 + 1, g0 + 1, r0) * (1 - fr) + T(b0 + 1, g0 + 1, r0 + 1) * fr
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    return (c0 * (1 - fb) + c1 * fb).astype(np.float32)


def export_cube(g: dict, path: str | Path, size: int = 33, title: str = "EpochColor grade") -> list[str]:
    """Write the global part of a grade as a 3D LUT. Returns what had to be
    left out: a LUT maps colour to colour, so it can't hold a shape mask, and
    a qualifier only survives when it isn't limited to a shape."""
    g = merged(g)
    left_out = []
    if g["mask"]["shape"] != "none":
        left_out.append("the shape mask (and the secondary correction inside it)")
        g["qualifier"]["enabled"] = False
        g["mask"]["shape"] = "none"
    grid = np.linspace(0, 1, size, dtype=np.float32)
    b, gg, r = np.meshgrid(grid, grid, grid, indexing="ij")
    img = np.stack([r, gg, b], axis=-1).reshape(1, -1, 3)
    out = apply(img, g).reshape(-1, 3)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in np.clip(out, 0, 1)]
    Path(path).write_text("\n".join(lines) + "\n")
    return left_out


# ------------------------------------------------------------------ masks


def shape_mask(m: dict, h: int, w: int, offset=(0.0, 0.0)) -> np.ndarray | None:
    shape = m.get("shape", "none")
    if shape == "none":
        return None
    cx = (m["cx"] + offset[0]) * w
    cy = (m["cy"] + offset[1]) * h
    rw = max(1e-3, m["w"] * w / 2)
    rh = max(1e-3, m["h"] * h / 2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(m.get("angle", 0.0))
    dx, dy = xx - cx, yy - cy
    u = (dx * math.cos(a) + dy * math.sin(a)) / rw
    v = (-dx * math.sin(a) + dy * math.cos(a)) / rh
    if shape == "ellipse":
        d = np.sqrt(u * u + v * v)
    else:  # rectangle
        d = np.maximum(np.abs(u), np.abs(v))
    feather = max(1e-3, float(m.get("feather", 0.1)))
    out = np.clip((1.0 + feather - d) / (2 * feather), 0, 1).astype(np.float32)
    out = out * out * (3 - 2 * out)
    return 1.0 - out if m.get("invert") else out


def qualifier_matte(hsv: np.ndarray, luma: np.ndarray, q: dict) -> np.ndarray:
    soft = max(1e-3, float(q.get("soft", 0.05)))
    h = hsv[..., 0] / 360.0
    d = np.abs(((h - q["hue"]) + 0.5) % 1.0 - 0.5)
    mh = np.clip((q["hue_width"] / 2 + soft - d) / soft, 0, 1)
    s = hsv[..., 1]
    ms = np.clip((s - q["sat_lo"] + soft) / soft, 0, 1) * np.clip((q["sat_hi"] + soft - s) / soft, 0, 1)
    ml = np.clip((luma - q["lum_lo"] + soft) / soft, 0, 1) * np.clip((q["lum_hi"] + soft - luma) / soft, 0, 1)
    return (mh * ms * ml).astype(np.float32)


# ------------------------------------------------------------------- apply


WB_K = 0.8  # full slider = 2.2x gain on red against blue


def _wb_gains(temperature: float, tint: float) -> np.ndarray:
    """Temperature and tint as channel gains in linear light, -1..1 each.
    Warm lifts red and lowers blue; tint moves green against magenta."""
    t, n = float(temperature), float(tint)
    return np.array([math.exp(WB_K * t), math.exp(-WB_K * n), math.exp(-WB_K * t)], np.float32)


def wb_from_neutral(rgb: tuple[float, float, float]) -> tuple[float, float]:
    """Temperature and tint that turn this sRGB colour neutral."""
    from .color import srgb_to_linear

    lin = np.maximum(srgb_to_linear(np.asarray(rgb, np.float32)), 1e-4)
    t = math.log(lin[2] / lin[0]) / (2 * WB_K)
    g_target = math.sqrt(lin[0] * lin[2])
    n = math.log(lin[1] / g_target) / WB_K
    return float(np.clip(t, -1, 1)), float(np.clip(n, -1, 1))


def apply(rgb: np.ndarray, g: dict | None, mask_offset=(0.0, 0.0), matte_out: list | None = None) -> np.ndarray:
    """Grade an sRGB float image (HxWx3). Returns a new float32 array, not
    clipped. matte_out, when a list, receives the secondary matte."""
    if g is None or is_identity(g):
        if matte_out is not None:
            matte_out.append(None)
        return rgb.astype(np.float32, copy=True)
    import cv2

    from .color import linear_to_srgb, srgb_to_linear

    g = merged(g)
    x = rgb.astype(np.float32, copy=True)

    if g["temperature"] or g["tint"]:
        x = linear_to_srgb(srgb_to_linear(np.clip(x, 0, None)) * _wb_gains(g["temperature"], g["tint"]))

    off = np.asarray(g["offset"], np.float32) + g["offset_m"]
    lift = np.asarray(g["lift"], np.float32) + g["lift_m"]
    gain = 1.0 + np.asarray(g["gain"], np.float32) + g["gain_m"]
    gamma = np.asarray(g["gamma"], np.float32) + g["gamma_m"]
    if off.any():
        x = x + off
    if lift.any() or (gain != 1).any():
        x = gain * (x + lift * (1.0 - x))
    if gamma.any():
        x = np.sign(x) * np.power(np.abs(x), np.power(2.0, -gamma * 2.0).astype(np.float32))
    if g["contrast"] != 1.0:
        x = (x - g["pivot"]) * g["contrast"] + g["pivot"]

    cv = g["curves"]
    if not _is_identity_curve(cv["master"]):
        x = _apply_lut1(x, _curve_lut(json.dumps(cv["master"])))
    for c, name in enumerate("rgb"):
        if not _is_identity_curve(cv[name]):
            x[..., c] = _apply_lut1(x[..., c], _curve_lut(json.dumps(cv[name])))

    q = g["qualifier"]
    mshape = shape_mask(g["mask"], x.shape[0], x.shape[1], mask_offset)
    need_hsv = (g["saturation"] != 1.0 or g["vibrance"] or g["hue_sat"] or g["hue_hue"]
                or q["enabled"] or mshape is not None)
    matte = None
    if need_hsv:
        # cv2 HSV on float: H in degrees, S and V 0..1. Values outside 0..1
        # are folded into V by scaling, then restored, so nothing clips.
        peak = np.maximum(x.max(axis=-1, keepdims=True), 1.0)
        floor = np.minimum(x.min(axis=-1, keepdims=True), 0.0)
        xs = (x - floor) / (peak - floor)
        hsv = cv2.cvtColor(np.ascontiguousarray(xs), cv2.COLOR_RGB2HSV)
        hue = hsv[..., 0] / 360.0
        if g["hue_hue"]:
            lut = _hue_lut(json.dumps(g["hue_hue"]), 0.0)
            hsv[..., 0] = (hsv[..., 0] + 360.0 * lut[(hue * 360).astype(np.int32) % 360]) % 360.0
        s = hsv[..., 1]
        if g["saturation"] != 1.0:
            s = s * g["saturation"]
        if g["vibrance"]:
            s = s * (1.0 + g["vibrance"] * (1.0 - np.clip(s, 0, 1)))
        if g["hue_sat"]:
            lut = _hue_lut(json.dumps(g["hue_sat"]), 1.0)
            s = s * lut[(hue * 360).astype(np.int32) % 360]
        hsv[..., 1] = np.clip(s, 0, 1)

        if q["enabled"] or mshape is not None:
            luma = xs[..., 0] * 0.2126 + xs[..., 1] * 0.7152 + xs[..., 2] * 0.0722
            matte = qualifier_matte(hsv, luma, q) if q["enabled"] else np.ones(x.shape[:2], np.float32)
            if mshape is not None:
                matte = matte * mshape
            if q["d_hue"]:
                hsv[..., 0] = (hsv[..., 0] + 360.0 * q["d_hue"] * matte) % 360.0
            if q["d_sat"] != 1.0:
                hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + (q["d_sat"] - 1.0) * matte), 0, 1)
            if q["d_lum"]:
                hsv[..., 2] = hsv[..., 2] * (1.0 + q["d_lum"] * matte)
        xs = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        x = xs * (peak - floor) + floor

    if g["lut"] and g["lut_mix"] > 0:
        table, dmin, dmax = load_cube(str(g["lut"]))
        y = apply_cube(x, table, dmin, dmax)
        x = y if g["lut_mix"] >= 1 else x + (y - x) * g["lut_mix"]
    if matte_out is not None:
        matte_out.append(matte)
    return x.astype(np.float32)


# ------------------------------------------------------------- lookups


def grade_key(g: dict) -> str:
    return hashlib.sha1(_key(g).encode()).hexdigest()[:16]


class GradeBook:
    """Finds the grade for any clip frame: the track of the shot it is in,
    evaluated at that frame. grades: {clip_id: {str(shot_start): track}}."""

    def __init__(self, grades: dict, shots: dict[str, list]):
        self.grades = grades or {}
        self.shots = shots or {}
        self._cache: OrderedDict = OrderedDict()

    def shot_start(self, clip_id: str, frame: int) -> int:
        start = 0
        for a, b in self.shots.get(clip_id, []):
            if a <= frame < b:
                return a
            if a <= frame:
                start = a
        return start

    def track(self, clip_id: str, frame: int) -> dict | None:
        return self.grades.get(clip_id, {}).get(str(self.shot_start(clip_id, frame)))

    def at(self, clip_id: str, frame: int) -> dict | None:
        t = self.track(clip_id, frame)
        if not t:
            return None
        g = evaluate(t, frame)
        if is_identity(g):
            return None
        return g

    def mask_offset(self, clip_id: str, frame: int) -> tuple[float, float]:
        t = self.track(clip_id, frame)
        off = (t or {}).get("track", {}).get(str(frame))
        return (float(off[0]), float(off[1])) if off else (0.0, 0.0)
