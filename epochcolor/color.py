"""Color space math. Everything is float32 numpy, sRGB primaries, D65 white.

L is CIE L* in 0..100, a and b are CIE a*/b* (roughly -110..110).
"""

from __future__ import annotations

import numpy as np

# D65 reference white
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883

_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)

_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(np.asarray(v, dtype=np.float32), 0.0, None)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055).astype(np.float32)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)


def _finv(t: np.ndarray) -> np.ndarray:
    t3 = t * t * t
    return np.where(t3 > _EPS, t3, (116.0 * t - 16.0) / _KAPPA)


def y_to_l(y: np.ndarray) -> np.ndarray:
    """Relative luminance (linear, 0..1) to L*."""
    return (116.0 * _f(np.asarray(y, dtype=np.float32)) - 16.0).astype(np.float32)


def l_to_y(L: np.ndarray) -> np.ndarray:
    return _finv((np.asarray(L, dtype=np.float32) + 16.0) / 116.0).astype(np.float32)


def srgb_to_l(img: np.ndarray) -> np.ndarray:
    """sRGB image (HxW gray or HxWx3, float 0..1) to L* (HxW).

    A colour scan of a black and white print is reduced to luminance in
    linear light, so a slight tint in the scan does not leak into L.
    """
    img = np.asarray(img, dtype=np.float32)
    lin = srgb_to_linear(img)
    if lin.ndim == 3:
        y = lin @ _RGB2XYZ[1].astype(np.float32)
    else:
        y = lin
    return y_to_l(y)


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(img).astype(np.float64)
    xyz = lin @ _RGB2XYZ.T
    fx = _f(xyz[..., 0] / _XN)
    fy = _f(xyz[..., 1] / _YN)
    fz = _f(xyz[..., 2] / _ZN)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_linear(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    L = L.astype(np.float64)
    fy = (L + 16.0) / 116.0
    fx = fy + ab[..., 0].astype(np.float64) / 500.0
    fz = fy - ab[..., 1].astype(np.float64) / 200.0
    xyz = np.stack([_finv(fx) * _XN, _finv(fy) * _YN, _finv(fz) * _ZN], axis=-1)
    return (xyz @ _XYZ2RGB.T).astype(np.float32)


def lab_to_srgb(L: np.ndarray, ab: np.ndarray, gamut_iters: int = 12) -> np.ndarray:
    """L (HxW) plus ab (HxWx2) to sRGB 0..1, keeping L intact.

    Out-of-gamut pixels get their chroma pulled toward neutral until they fit,
    instead of each RGB channel being clipped. Clipping channels changes the
    brightness; shrinking chroma at constant L does not, since the neutral
    axis is always in gamut.
    """
    ab = ab.astype(np.float32).copy()
    lin = lab_to_linear(L, ab)
    tol = 1e-5
    bad = np.any((lin < -tol) | (lin > 1.0 + tol), axis=-1)
    if bad.any():
        idx = np.nonzero(bad)
        Lb = L[idx]
        abb = ab[idx]
        lo = np.zeros(len(Lb), dtype=np.float32)  # scale known to fit
        hi = np.ones(len(Lb), dtype=np.float32)  # scale known not to fit
        for _ in range(gamut_iters):
            mid = (lo + hi) * 0.5
            rgb = lab_to_linear(Lb, abb * mid[:, None])
            ok = np.all((rgb >= -tol) & (rgb <= 1.0 + tol), axis=-1)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        ab[idx] = abb * lo[:, None]
        lin = lab_to_linear(L, ab)
    return linear_to_srgb(np.clip(lin, 0.0, 1.0))
