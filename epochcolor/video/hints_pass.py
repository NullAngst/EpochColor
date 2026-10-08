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


PAINT_VERSION = 2  # bump when what a stroke does changes, so cached fixes get redone


def frame_luma(source: Path, fps: Fraction, frame: int, size: tuple[int, int], negative: dict | None = None,
               denoise=None) -> np.ndarray:
    """One frame's L* at size (w, h), lightly denoised, for painting."""
    import cv2

    from ..color import srgb_to_l
    from ..denoise import denoise_l
    from ..film import invert
    from ..media import FrameReader

    r = FrameReader(source, fps, "gray16le", cache=2)
    try:
        g16 = r.get(frame)
    finally:
        r.close()
    if g16 is None:
        raise RuntimeError(f"{Path(source).name}: frame {frame} could not be decoded for its hints")
    L = srgb_to_l(invert(g16, negative))
    return denoise_l(cv2.resize(L, size, interpolation=cv2.INTER_AREA), denoise)


def keyframe_fix(model: ColorModel | None, source: Path, fps: Fraction, frame: int, strokes: list,
                 working: tuple[int, int], chroma: tuple[int, int], ab_model_k: np.ndarray | None = None,
                 denoise=None, spread: float = 0.35, negative: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Target a/b and weight (both at chroma size) for one painted frame.

    The strokes fill their own objects out to the objects' edges
    (propagate.paint_fill) and nothing else; the weight says where. The
    model isn't involved, so painting works the same whatever model is
    picked, and the model's colour everywhere you didn't paint is left
    exactly as it was."""
    from ..propagate import paint_fill

    cw, ch = chroma
    L = frame_luma(source, fps, frame, (cw, ch), negative, denoise)
    return paint_fill(L, rasterize(strokes, cw, ch), spread=spread)


def carry(Ldn, ab_raw, a: int, b: int, keys, flow: Flow, tol: float, out, protect=None) -> None:
    """Fill out[0 : b-a] with the hinted chroma for frames [a, b).

    Ldn, ab_raw: whole-clip arrays (chroma size). keys: either a dict of
    clip frame -> (target, weight), carried through the whole shot, or a
    list of (frame, target, weight, reach) where reach is how many frames
    either side the fix may travel: -1 for the whole shot, 0 for its own
    frame only. out: array-like of the shot's length. protect, when given
    ((b-a, h, w) floats), gets the weight of the frame-limited fixes, so
    the stabilizer that runs next can be kept from averaging them away.
    """
    if isinstance(keys, dict):
        keys = [(k, T, W, -1) for k, (T, W) in keys.items()]
    n = b - a
    ch, cw = Ldn[a].shape
    accT = np.zeros((n, ch, cw, 2), np.float32) if n * ch * cw * 12 < 6e8 else None
    tmp = None
    if accT is None:  # very long shot: accumulate on disk, on the cache disk, not /tmp
        import tempfile

        from .pipeline import cache_root

        (cache_root() / "tmp").mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=cache_root() / "tmp"))
        accT = np.lib.format.open_memmap(str(tmp / "t.npy"), "w+", np.float32, (n, ch, cw, 2))
        accA = np.lib.format.open_memmap(str(tmp / "a.npy"), "w+", np.float32, (n, ch, cw))
        accW = np.lib.format.open_memmap(str(tmp / "w.npy"), "w+", np.float32, (n, ch, cw))
    else:
        accA = np.zeros((n, ch, cw), np.float32)
        accW = np.zeros((n, ch, cw), np.float32)

    def L(t):
        return np.asarray(Ldn[t], np.float32)

    for k, T0, W0, reach in keys:
        for step in ((1,) if reach == 0 else (1, -1)):
            T, W = T0, W0
            t = k
            while True:
                i = t - a
                near = 1.0 / (1.0 + abs(t - k) / 24.0)  # a fix one second away counts half
                wa = W * near
                if protect is not None and reach >= 0:
                    protect[i] = np.maximum(protect[i], W)
                accT[i] += T * wa[..., None]
                accA[i] += wa
                accW[i] = np.maximum(accW[i], W)
                t2 = t + step
                if not (a <= t2 < b) or (step == -1 and t2 < a):
                    break
                if reach >= 0 and abs(t2 - k) > reach:
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
    if tmp is not None:
        import shutil

        del accT, accA, accW
        shutil.rmtree(tmp, ignore_errors=True)
