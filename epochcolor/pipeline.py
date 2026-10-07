"""The photo pipeline, milestone 1.

    source sRGB -> L (full size, grain intact)
                -> L at working size -> denoise -> model -> chroma
                                                -> hint correction
                -> chroma scaled up to full size, snapped to edges of L
    recombine: L (grain slider) + chroma -> sRGB

The model never sees the full-size luma and never touches it. The output
luma is the source luma unless you pull the grain slider down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .color import lab_to_srgb, srgb_to_l
from .denoise import denoise_l, estimate_noise
from .hints import Hints, empty_hints
from .models import ColorModel
from .propagate import propagate


@dataclass
class PhotoSettings:
    working_size: int = 512  # short side for chroma work
    grain: float = 100.0  # percent of original luma in the output
    denoise: float | None = None  # L* units, None = auto
    spread: float | None = None  # None = per-model default
    guided: bool = True  # snap upscaled chroma to edges of the full-size luma
    saturation: float = 1.0  # plain chroma multiplier, real grading is milestone 7


@dataclass
class Report:
    size: tuple[int, int] = (0, 0)
    working: tuple[int, int] = (0, 0)
    noise_sigma: float = 0.0
    hint_pixels: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _resize(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    interp = cv2.INTER_AREA if w < arr.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(arr, (w, h), interpolation=interp)


def guided_upsample(ab_small: np.ndarray, guide: np.ndarray, eps: float) -> np.ndarray:
    """Scale chroma up to the guide's size and pull its edges onto the
    guide's edges (He et al. guided filter). eps is in L* units squared;
    luma variation below it, grain for instance, does not shape the colour."""
    import cv2

    h, w = guide.shape
    sh = ab_small.shape[0]
    up = cv2.resize(ab_small, (w, h), interpolation=cv2.INTER_CUBIC)
    scale = h / sh
    if scale <= 1.01:
        return up.astype(np.float32)
    r = int(max(2, round(1.5 * scale)))
    k = (2 * r + 1, 2 * r + 1)
    I = guide.astype(np.float32)

    def box(x):
        return cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)

    mean_I = box(I)
    var_I = box(I * I) - mean_I * mean_I
    out = np.empty_like(up, dtype=np.float32)
    for c in range(2):
        p = up[..., c].astype(np.float32)
        mean_p = box(p)
        cov = box(I * p) - mean_I * mean_p
        a = cov / (var_I + eps)
        b = mean_p - a * mean_I
        out[..., c] = box(a) * I + box(b)
    return out


def colorize_photo(
    img: np.ndarray,
    model: ColorModel,
    settings: PhotoSettings | None = None,
    hints: Hints | None = None,
) -> tuple[np.ndarray, Report]:
    s = settings or PhotoSettings()
    rep = Report()
    t0 = time.perf_counter()

    def mark(name):
        nonlocal t0
        t = time.perf_counter()
        rep.timings[name] = round(t - t0, 3)
        t0 = t

    L = srgb_to_l(img)
    h, w = L.shape
    rep.size = (w, h)
    if img.ndim == 3:
        tint = np.abs(img - img.mean(axis=2, keepdims=True)).max()
        if tint > 0.08:
            rep.warnings.append(
                "source has visible colour in it; it was reduced to luminance first"
            )

    k = min(1.0, s.working_size / min(h, w))
    ww, wh = max(8, round(w * k)), max(8, round(h * k))
    rep.working = (ww, wh)
    Lw = _resize(L, ww, wh)
    rep.noise_sigma = round(estimate_noise(Lw), 3)
    Lw_dn = denoise_l(Lw, s.denoise)
    mark("prepare")

    hw = empty_hints(wh, ww)
    if hints is not None and hints.count:
        hw = hints.resized(ww, wh).binarized(0.3)
        rep.hint_pixels = hints.count
        if hints.mask.mean() > 0.5:
            rep.warnings.append(
                "over half the hint image counts as painted; if it is a tinted copy "
                "of the photo, paint on a transparent layer instead"
            )

    ab = model.predict(Lw_dn, hw if model.info.takes_hints else None)
    mark("model")

    if hw.count:
        spread = s.spread
        if spread is None:
            spread = 3.0 if model.info.mode == "propagation" else 0.15
        ab = propagate(Lw_dn, ab, hw, spread=spread)
        mark("hints")
    elif model.info.mode == "propagation":
        rep.warnings.append("hints model with no hints gives a grey image")

    if s.saturation != 1.0:
        ab = ab * float(s.saturation)

    if s.guided:
        # grain measured at full size sets how much luma texture the
        # colour ignores
        eps = float(np.clip((2.0 * estimate_noise(L)) ** 2, 1.0, 100.0))
        ab_full = guided_upsample(ab, L, eps)
    else:
        ab_full = _resize(ab, w, h)
    mark("upscale")

    g = float(np.clip(s.grain, 0.0, 100.0)) / 100.0
    if g < 1.0:
        L_dn = denoise_l(L, s.denoise)
        L_out = g * L + (1.0 - g) * L_dn
    else:
        L_out = L
    mark("grain")

    rgb = lab_to_srgb(L_out, ab_full)
    mark("recombine")
    return rgb, rep
