"""Spatial denoising for photos.

The denoised copy feeds the model and the edge detection, never the output
luma unless the grain slider is pulled below 100%. Video gets temporal
denoising in milestone 2; a photo has no neighbouring frames, so it is
spatial only.
"""

from __future__ import annotations

import numpy as np


def estimate_noise(L: np.ndarray) -> float:
    """Noise sigma in L* units, Immerkaer's fast estimator.

    Works on a central crop so a 100 MP scan does not cost much.
    """
    import cv2

    h, w = L.shape
    cy, cx = h // 2, w // 2
    r = 1024
    crop = L[max(0, cy - r): cy + r, max(0, cx - r): cx + r].astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(crop, -1, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    hh, ww = conv.shape
    if hh <= 0 or ww <= 0:
        return 0.0
    return float(np.sqrt(np.pi / 2.0) * np.abs(conv).sum() / (6.0 * hh * ww))


def denoise_l(L: np.ndarray, strength: float | None = None, passes: int = 2) -> np.ndarray:
    """Edge-preserving denoise on L* (0..100), float32 throughout.

    strength is the range sigma in L* units; None picks it from the estimated
    noise. 0 returns the input unchanged. Bilateral filtering in float keeps
    full precision, which matters since this copy can be mixed into the
    output luma by the grain slider.
    """
    import cv2

    if strength is None:
        strength = 2.5 * estimate_noise(L)
    out = L.astype(np.float32, copy=True)
    if strength <= 0.05:
        return out
    for _ in range(passes):
        out = cv2.bilateralFilter(out, d=7, sigmaColor=float(strength), sigmaSpace=2.5)
    return out
