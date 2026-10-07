"""Painted hints on video: fix one frame, carry the fix through the shot.

1. At each painted frame (a keyframe), the model runs again with the hints
   (models that read hints get them as input), then the strokes are spread
   to the object's edges by the edge-aware solver. Where that differs from
   the plain model output is the fix: a target colour and a weight.
2. The fix travels forward and backward from its keyframe along optical
   flow, frame by frame. Where the motion doesn't line up (occlusion,
   something new entering) its weight drops, and an edge-aware fill lets it
   reach parts of the object that come into view.
3. With several keyframes in a shot, each frame takes the nearest fixes,
   weighted by how far away they are and how sure the carry is.
4. The result, target blended over the model's chroma by weight, is what the
   stabilizer then smooths, same as an unpainted shot.

The fix is an absolute colour, not an offset, so a coat painted navy stays
navy even where the model drifts to a different wrong colour later on.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from ..hintpaint import rasterize
from ..models import ColorModel
from .temporal import Flow, confidence, warp


def _box(x, r):
    import cv2

    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided(p: np.ndarray, I: np.ndarray, r: int = 4, eps: float = 4.0) -> np.ndarray:
    """He et al. guided filter: smooth p, keeping the edges of guide I."""
    mI = _box(I, r)
    vI = _box(I * I, r) - mI * mI
    if p.ndim == 3:
        return np.stack([guided(p[..., c], I, r, eps) for c in range(p.shape[2])], -1)
    mp = _box(p, r)
    a = (_box(I * p, r) - mI * mp) / (vI + eps)
    b = mp - a * mI
    return _box(a, r) * I + _box(b, r)


def keyframe_fix(model: ColorModel, source: Path, fps: Fraction, frame: int, strokes: list,
                 working: tuple[int, int], chroma: tuple[int, int], ab_model_k: np.ndarray | None = None,
                 denoise=None, spread: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Target a/b and weight (both at chroma size) for one painted frame."""
    import cv2

    from ..color import srgb_to_l
    from ..denoise import denoise_l
    from ..media import FrameReader
    from ..propagate import propagate

    ww, wh = working
    cw, ch = chroma
    r = FrameReader(source, fps, "gray16le", cache=2)
    try:
        g16 = r.get(frame)
    finally:
        r.close()
    if g16 is None:
        raise RuntimeError(f"{Path(source).name}: frame {frame} could not be decoded for its hints")
    L = srgb_to_l(g16.astype(np.float32) / 65535.0)
    Lw = cv2.resize(L, (ww, wh), interpolation=cv2.INTER_AREA)
    Lw = denoise_l(Lw, denoise)
    hints = rasterize(strokes, ww, wh)
    # what the strokes change is measured against the model on this same
    # input without them, so differences between this single-frame run and
    # the clip's temporally denoised run don't count as part of the fix
    plain = model.predict(Lw, None)
    ab = model.predict(Lw, hints) if model.info.takes_hints else plain
    ab = propagate(Lw, ab, hints, spread=spread)
    T = cv2.resize(ab, (cw, ch), interpolation=cv2.INTER_AREA)
    P = cv2.resize(plain, (cw, ch), interpolation=cv2.INTER_AREA)
    m = cv2.resize(hints.mask, (cw, ch), interpolation=cv2.INTER_AREA)
    diff = np.linalg.norm(T - P, axis=-1)
    W = np.clip(diff / 6.0, 0.0, 1.0)
    W = np.maximum(W, np.clip(m * 2, 0, 1))
    W = cv2.GaussianBlur(W, (0, 0), 1.0)
    return T.astype(np.float32), np.clip(W, 0, 1).astype(np.float32)


def carry(Ldn, ab_raw, a: int, b: int, keys: dict[int, tuple[np.ndarray, np.ndarray]],
          flow: Flow, tol: float, out) -> None:
    """Fill out[0 : b-a] with the hinted chroma for frames [a, b).

    Ldn, ab_raw: whole-clip arrays (chroma size). keys: clip frame ->
    (target, weight). out: array-like of the shot's length.
    """
    n = b - a
    ch, cw = Ldn[a].shape
    accT = np.zeros((n, ch, cw, 2), np.float32) if n * ch * cw * 12 < 6e8 else None
    if accT is None:  # very long shot: accumulate on disk
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        accT = np.lib.format.open_memmap(str(tmp / "t.npy"), "w+", np.float32, (n, ch, cw, 2))
        accA = np.lib.format.open_memmap(str(tmp / "a.npy"), "w+", np.float32, (n, ch, cw))
        accW = np.lib.format.open_memmap(str(tmp / "w.npy"), "w+", np.float32, (n, ch, cw))
    else:
        accA = np.zeros((n, ch, cw), np.float32)
        accW = np.zeros((n, ch, cw), np.float32)

    def L(t):
        return np.asarray(Ldn[t], np.float32)

    for k, (T0, W0) in keys.items():
        for step in (1, -1):
            T, W = T0, W0
            t = k
            while True:
                i = t - a
                near = 1.0 / (1.0 + abs(t - k) / 24.0)  # a fix one second away counts half
                wa = W * near
                accT[i] += T * wa[..., None]
                accA[i] += wa
                accW[i] = np.maximum(accW[i], W)
                t2 = t + step
                if not (a <= t2 < b) or (step == -1 and t2 < a):
                    break
                F = flow(L(t2), L(t))
                Tw = warp(T, F)
                Ww = warp(W, F)
                c = confidence(L(t2), warp(L(t), F), tol)
                Ww = Ww * np.clip((c - 0.3) / 0.5, 0, 1)
                # reach parts of the object that come into view
                gW = np.clip(guided(Ww, L(t2)), 0, 1)
                gT = guided(Tw * Ww[..., None], L(t2))
                T = np.where(gW[..., None] > 1e-3, gT / np.maximum(gW[..., None], 1e-3), Tw)
                W = np.clip(np.maximum(Ww, 0.9 * gW), 0, 1)
                t = t2
                if W.max() < 0.02:
                    break  # nothing left to carry
    for i in range(n):
        A = np.maximum(accA[i], 1e-6)[..., None]
        T = accT[i] / A
        Wm = accW[i][..., None]
        raw = np.asarray(ab_raw[a + i], np.float32)
        out[i] = Wm * T + (1 - Wm) * raw
