"""Film restoration: the things old film brings that colorization models
handle badly.

- Negative inversion for black and white negatives, scanned or copied
  with a camera. Density based: the film base becomes black, each pixel's
  density above it becomes brightness, with black and white points and a
  contrast exponent on top. Done through a lookup table on the 16-bit grey,
  so it costs one array index per frame.
- Deflicker: old film pulses in brightness from frame to frame. Each
  frame's 5th and 95th percentile brightness is pulled toward a smoothed
  version of itself over about half a second, per shot, so real fades and
  lighting changes survive and the pulsing goes.
- Dust: a speck of dirt is on one frame only. Where a frame differs
  sharply from both its motion-compensated neighbours, in the same
  direction, while the neighbours agree with each other, it's dirt, and
  gets their average instead. Scratches running through many frames are
  not caught this way.
- Regrain: synthetic grain after grading, for footage cleaned hard or
  matched to a look. Mono by default, with an optional faint colour grain.

Deflicker and dust can run on the model's copy only (the default for a
new project: the model stops reading flicker and dirt as colour) or on the
output too, which changes the picture's brightness and is opt-in. The
original luma is otherwise left alone, as everywhere in EpochColor.
"""

from __future__ import annotations

import numpy as np

from .color import srgb_to_linear

# ------------------------------------------------------------------ negative


def _gray(img: np.ndarray) -> np.ndarray:
    """Gamma-encoded grey 0..1 from grey or RGB."""
    img = np.asarray(img, np.float32)
    if img.ndim == 3:
        from .color import linear_to_srgb

        lin = srgb_to_linear(img)
        return linear_to_srgb(lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722)
    return img


def measure_negative(samples: list[np.ndarray], border: float = 0.04) -> dict:
    """Film base, density range and contrast for a negative, from one image
    or a handful of frames (gamma-encoded grey 0..1 or RGB).

    The film base is the clearest part of the film: the brightest of the
    edge strips, where the rebate between frames usually shows. A scan
    cropped tight to the picture has no rebate, and then the brightest
    part of the picture stands in for it.
    """
    edges, alls = [], []
    for s in samples:
        g = _gray(s)
        h, w = g.shape
        b = max(2, int(border * min(h, w)))
        edges.append(np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()]))
        alls.append(g[:: max(1, h // 400), :: max(1, w // 400)].ravel())
    edge = np.concatenate(edges)
    every = np.concatenate(alls)
    base = float(np.percentile(edge, 99.0))
    if base < float(np.percentile(every, 95.0)):  # no rebate in the edges
        base = float(np.percentile(every, 99.8))
    base_lin = float(max(1e-4, srgb_to_linear(np.float32(base))))
    D = np.log10(base_lin / np.maximum(srgb_to_linear(every), 1e-5))
    D = np.maximum(D, 0)
    return {"base": round(base, 5), "dmin": round(float(np.percentile(D, 0.5)), 4),
            "dmax": round(float(max(np.percentile(D, 99.5), np.percentile(D, 0.5) + 0.05)), 4), "gamma": 1.0}


def negative_lut(p: dict) -> np.ndarray:
    """(65536,) float32: 16-bit negative grey -> gamma-encoded positive 0..1."""
    x = np.arange(65536, dtype=np.float32) / 65535.0
    base_lin = max(1e-4, float(srgb_to_linear(np.float32(p.get("base", 1.0)))))
    D = np.log10(base_lin / np.maximum(srgb_to_linear(x), 1e-5))
    dmin, dmax = float(p.get("dmin", 0.0)), float(p.get("dmax", 2.0))
    n = np.clip((D - dmin) / max(1e-3, dmax - dmin), 0, 1)
    g = float(p.get("gamma", 1.0))
    return (n ** g if g != 1.0 else n).astype(np.float32)


_lut_cache: dict[tuple, np.ndarray] = {}


def lut_for(p: dict | None) -> np.ndarray | None:
    if not p:
        return None
    key = tuple(sorted((k, float(v)) for k, v in p.items() if isinstance(v, (int, float))))
    if key not in _lut_cache:
        if len(_lut_cache) > 16:
            _lut_cache.clear()
        _lut_cache[key] = negative_lut(p)
    return _lut_cache[key]


def invert(gray: np.ndarray, p: dict | None) -> np.ndarray:
    """Apply a negative's inversion to decoded grey: uint16 or uint8 in,
    float32 0..1 out. Float input (0..1) goes through the 16-bit table."""
    lut = lut_for(p)
    if lut is None:
        g = np.asarray(gray)
        if g.dtype == np.uint16:
            return g.astype(np.float32) / 65535.0
        if g.dtype == np.uint8:
            return g.astype(np.float32) / 255.0
        return g.astype(np.float32)
    g = np.asarray(gray)
    if g.dtype == np.uint8:
        return lut[np.arange(256) * 257][g]
    if g.dtype != np.uint16:
        g = np.clip(np.rint(_gray(g) * 65535.0), 0, 65535).astype(np.uint16)
    return lut[g]


def invert_rgb(img: np.ndarray, p: dict | None) -> np.ndarray:
    """A photo: float RGB or grey 0..1 in, positive grey as RGB out."""
    pos = invert(_gray(img), p)
    return np.repeat(pos[..., None], 3, axis=2) if np.asarray(img).ndim == 3 else pos


# ----------------------------------------------------------------- deflicker


def frame_stats(L: np.ndarray) -> tuple[float, float]:
    """5th and 95th percentile L* of a (small) frame."""
    lo, hi = np.percentile(L, (5.0, 95.0))
    return float(lo), float(hi)


def deflicker_targets(stats, shots, fps: float, seconds: float = 0.5) -> np.ndarray:
    """(n, 2) smoothed percentiles each frame is pulled toward, per shot, so
    no smoothing reaches across a cut."""
    from scipy.ndimage import gaussian_filter1d

    st = np.asarray(stats, np.float32).reshape(-1, 2)
    out = st.copy()
    sigma = max(1.0, fps * seconds / 2.0)
    for a, b in shots:
        b = min(b, len(st))
        if b - a >= 3:
            out[a:b] = gaussian_filter1d(st[a:b], sigma, axis=0, mode="nearest")
    return out


def deflicker(L: np.ndarray, stats, target, strength: float = 1.0) -> np.ndarray:
    """Pull a frame's brightness range onto its target range."""
    if strength <= 0 or stats is None or target is None:
        return L
    lo, hi = float(stats[0]), float(stats[1])
    tlo, thi = float(target[0]), float(target[1])
    k = (thi - tlo) / max(1.0, hi - lo)
    k = float(np.clip(k, 0.5, 2.0))  # a frame that's mostly black stays sane
    fixed = (L - lo) * k + tlo
    return np.clip(L + strength * (fixed - L), 0.0, 100.0).astype(np.float32)


# ---------------------------------------------------------------------- dust


def dust_mask(cur: np.ndarray, wp: np.ndarray, wn: np.ndarray, thr: float,
              max_area: float = 0.002) -> np.ndarray:
    """Pixels that are dirt: far from both warped neighbours, in the same
    direction, where the neighbours agree. Blobs bigger than max_area of
    the frame are left alone; that's motion the flow missed, not dirt."""
    import cv2

    d1 = cur - wp
    d2 = cur - wn
    m = (d1 * d2 > 0) & (np.minimum(np.abs(d1), np.abs(d2)) > thr) & (np.abs(wp - wn) < 0.5 * thr)
    m = m.astype(np.uint8)
    if not m.any():
        return m.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    limit = max(64, int(max_area * m.size))  # small frames still allow an 8x8 speck
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] <= limit
    m = keep[lab].astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def remove_dust(cur: np.ndarray, prev: np.ndarray | None, nxt: np.ndarray | None, flow,
                thr: float = 8.0, flows=None) -> tuple[np.ndarray, np.ndarray]:
    """(cleaned L, mask). Needs both neighbours; at a shot's ends it does nothing.
    flows: (F to prev, F to next) at cur's size, when already computed."""
    from .video.temporal import warp

    if prev is None or nxt is None:
        return cur, np.zeros(cur.shape, bool)
    Fp, Fn = flows if flows is not None else (flow(cur, prev), flow(cur, nxt))
    wp, wn = warp(prev, Fp), warp(nxt, Fn)
    m = dust_mask(cur, wp, wn, thr)
    if not m.any():
        return cur, m
    out = cur.copy()
    out[m] = 0.5 * (wp[m] + wn[m])
    return out, m


def scaled_flow(flow, small_a: np.ndarray, small_b: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Flow computed on small frames, scaled up to full size."""
    import cv2

    F = flow(small_a, small_b)
    h, w = shape
    sh, sw = F.shape[:2]
    up = cv2.resize(F, (w, h), interpolation=cv2.INTER_LINEAR)
    up[..., 0] *= w / sw
    up[..., 1] *= h / sh
    return up


# ------------------------------------------------------------------- regrain


def regrain(rgb: np.ndarray, amount: float, size: float = 1.0, chroma: float = 0.0, seed: int = 0) -> np.ndarray:
    """Add grain to finished sRGB float 0..1. amount: grain strength in
    percent of full scale at mid grey; size: grain size in pixels at 1080
    lines, scaled to the frame; chroma: 0..1, how much independent colour
    grain rides on top. Same seed, same grain, so renders repeat exactly."""
    if amount <= 0:
        return rgb
    import cv2

    h, w = rgb.shape[:2]
    rng = np.random.default_rng(seed)
    sigma = max(0.0, float(size) * h / 1080.0)

    def field():
        n = rng.standard_normal((h, w)).astype(np.float32)
        if sigma > 0.35:
            n = cv2.GaussianBlur(n, (0, 0), sigma)
            n /= max(1e-6, float(n.std()))
        return n

    y = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    k = (0.25 + 3.0 * np.clip(y, 0, 1) * (1 - np.clip(y, 0, 1)))[..., None]  # strongest in the mids
    a = float(amount) / 100.0
    out = rgb + a * k * field()[..., None]
    if chroma > 0:
        c = np.stack([field() for _ in range(3)], -1)
        out = out + a * float(chroma) * 0.5 * k * c
    return np.clip(out, 0, 1).astype(np.float32)
