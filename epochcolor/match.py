"""Saved colours: find the same object elsewhere and offer its colour.

A saved colour is a stroke you painted plus a name ("Anna's coat, navy").
Matching uses the colorization model's own mid-level features, which
already encode what kind of thing a region is, since that's what predicting
its colour takes. The exemplar is the average feature under the stroke;
in another shot or photo, places whose features point the same way are
candidates. It's matching at inference time: nothing is trained.

This finds the same coat in the next shot well enough to be useful, and it
also finds other things that merely look alike. That's why the default is to
mark a match and wait for a click (project setting), not to paint it.
"""

from __future__ import annotations

import numpy as np

from .hintpaint import rasterize


def feature_mask(strokes: list, fh: int, fw: int) -> np.ndarray:
    import cv2

    h = rasterize(strokes, fw * 8, fh * 8)
    return cv2.resize(h.mask, (fw, fh), interpolation=cv2.INTER_AREA)


def exemplar(model, L: np.ndarray, strokes: list) -> np.ndarray | None:
    """Average unit feature under the strokes, unit length. None when the
    model has no features to match with."""
    if not hasattr(model, "features"):
        return None
    F = model.features(L)  # torch (C, h, w)
    C, fh, fw = F.shape
    m = feature_mask(strokes, fh, fw)
    if m.sum() < 1e-3:
        return None
    import torch

    mt = torch.from_numpy(m).to(F.device)
    e = (F * mt).sum(dim=(1, 2)) / mt.sum()
    e = e / (e.norm() + 1e-6)
    return e.cpu().numpy()


def similarity(model, L: np.ndarray, exemplars: list[np.ndarray]) -> list[np.ndarray]:
    import torch

    F = model.features(L)
    out = []
    for e in exemplars:
        et = torch.from_numpy(np.asarray(e, np.float32)).to(F.device)
        out.append(torch.einsum("c,chw->hw", et, F).cpu().numpy())
    return out


def peaks(S: np.ndarray, threshold: float, max_points: int = 3) -> tuple[float, list[list[float]]]:
    """Best score and up to a few well separated points above the
    threshold, as fractions of the frame."""
    best = float(S.max())
    if best < threshold:
        return best, []
    work = S.copy()
    h, w = S.shape
    pts = []
    rad = max(1, int(round(0.08 * min(h, w))))
    floor = max(threshold, best - 0.06)
    for _ in range(max_points):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        if work[y, x] < floor:
            break
        pts.append([(x + 0.5) / w, (y + 0.5) / h])
        work[max(0, y - rad):y + rad + 1, max(0, x - rad):x + rad + 1] = -1
    return best, pts
