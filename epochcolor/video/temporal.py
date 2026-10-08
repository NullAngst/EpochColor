"""Shot detection, optical flow and the two temporal filters.

Everything here runs at working size on L* (0..100).
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ shots


def thumb(L: np.ndarray, width: int = 96) -> np.ndarray:
    import cv2

    h, w = L.shape
    return cv2.resize(L, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def cut_score(prev: np.ndarray, cur: np.ndarray) -> float:
    """Difference between a frame's thumbnail and the one before, after
    removing each one's mean brightness. Old film pulses in exposure;
    subtracting the mean keeps a flicker from reading as a cut."""
    a = prev - prev.mean()
    b = cur - cur.mean()
    return float(np.float32(np.abs(a - b).mean()))


def cut_scores(thumbs: list[np.ndarray]) -> np.ndarray:
    s = np.zeros(len(thumbs), np.float32)
    for i in range(1, len(thumbs)):
        s[i] = cut_score(thumbs[i - 1], thumbs[i])
    return s


def detect_shots(scores: np.ndarray, threshold: float = 6.0, ratio: float = 3.0,
                 min_len: int = 6) -> list[tuple[int, int]]:
    """Return shots as [start, end) frame ranges.

    A cut needs an absolute jump above threshold (in L* units) and a jump
    well above the local motion level, so a fast pan does not get chopped
    up while a cut between two similar shots still registers.
    """
    n = len(scores)
    cuts = [0]
    for i in range(1, n):
        lo, hi = max(1, i - 8), min(n, i + 9)
        around = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        local = float(np.median(around)) if around.size else 0.0
        if scores[i] > threshold and scores[i] > ratio * max(local, 0.5):
            if i - cuts[-1] >= min_len:
                cuts.append(i)
    cuts.append(n)
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


# ------------------------------------------------------------------- flow


class Flow:
    """OpenCV DIS optical flow on the CPU.

    flow(a, b) returns F such that a(x) ~ b(x + F(x)), which is what warp()
    needs to pull frame b onto frame a. A GPU path (RAFT) comes later behind
    the same call.
    """

    def __init__(self, preset: str = "medium"):
        import cv2

        p = {"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
             "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
             "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST}[preset]
        self.dis = cv2.DISOpticalFlow_create(p)
        self._memo: list = []  # the last few (a, b, F): dust removal and the denoise ask for the same pairs

    @staticmethod
    def _u8(L: np.ndarray) -> np.ndarray:
        return np.clip(L * 2.55, 0, 255).astype(np.uint8)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        for ma, mb, F in self._memo:
            if ma is a and mb is b:  # the very same arrays, held here so they can't be recycled
                return F
        F = self.dis.calc(self._u8(a), self._u8(b), None)
        self._memo = (self._memo + [(a, b, F)])[-4:]
        return F


def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    import cv2

    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def confidence(L: np.ndarray, L_warped: np.ndarray, tol: float) -> np.ndarray:
    """1 where the warp lines up, falling toward 0 where it does not
    (occlusion, new objects, flow failure).

    The warped frame is matched to the current one in brightness and
    contrast first, so old film's exposure flicker does not read as a
    mismatch and cut the colour memory short.
    """
    import cv2

    mc, sc = float(L.mean()), float(L.std()) + 1e-3
    mw, sw = float(L_warped.mean()), float(L_warped.std()) + 1e-3
    matched = (L_warped - mw) * (sc / sw) + mc
    d = cv2.GaussianBlur(np.abs(L - matched), (0, 0), 1.0)
    return np.exp(-(d * d) / (2.0 * tol * tol)).astype(np.float32)


# ------------------------------------------------------------ the filters


def temporal_denoise(prev: np.ndarray | None, cur: np.ndarray, nxt: np.ndarray | None,
                     flow: Flow, sigma: float) -> np.ndarray:
    """Average a frame with its motion-compensated neighbours. Grain changes
    every frame and the picture mostly does not, so this removes grain with
    far less smearing than a spatial filter of the same strength."""
    acc = cur.astype(np.float32).copy()
    wsum = np.ones_like(acc)
    tol = max(1.0, 2.5 * sigma)
    for nb in (prev, nxt):
        if nb is None:
            continue
        wp = warp(nb, flow(cur, nb))
        c = confidence(cur, wp, tol)
        acc += c * wp
        wsum += c
    return acc / wsum


def _recursive(L, ab, flow: Flow, order, n_max: float, tol: float, out, count) -> None:
    """One direction of the chroma stabilizer.

    A running average carried along the motion: out[t] mixes the model's
    chroma for frame t with the previous result warped onto frame t. The mix
    is a true average of up to n_max frames, and the frame count drops back
    toward 1 wherever the warp does not line up (occlusion, new objects,
    flow failure), so stale colour is never dragged onto something new.
    count[t] keeps the effective number of frames behind out[t].
    """
    prev_i = prev_out = prev_n = None
    for i in order:
        cur = ab[i].astype(np.float32)
        if prev_i is None:
            res, n = cur, np.ones(cur.shape[:2], np.float32)
        else:
            F = flow(L[i], L[prev_i])
            c = confidence(L[i], warp(L[prev_i], F), tol)
            n = np.minimum(n_max, 1.0 + c * warp(prev_n, F))
            a = (1.0 / n)[..., None]
            res = a * cur + (1.0 - a) * warp(prev_out, F)
        out[i] = res
        count[i] = n
        prev_i, prev_out, prev_n = i, res, n


def stabilize_chroma(L, ab, out, strength: float = 0.9, tol: float = 3.0,
                     flow: Flow | None = None, scratch=None) -> None:
    """Forward and backward running averages along the flow, merged by how
    many frames stand behind each. Near the start of a shot the backward
    pass has the longer memory and wins, near the end the forward one does,
    so nothing lags and no frame gets less smoothing than its neighbours.

    strength sets the memory: 0.9 averages up to 10 frames each way. L, ab
    and out are indexable sequences for one shot (arrays or memmaps).
    scratch holds the forward pass; it defaults to an in-memory array.
    """
    n = len(ab)
    if n == 0:
        return
    if strength <= 0:
        for i in range(n):
            out[i] = ab[i]
        return
    flow = flow or Flow()
    n_max = 1.0 / float(np.clip(1.0 - strength, 0.01, 1.0))
    fwd = scratch if scratch is not None else np.empty((n,) + ab[0].shape, np.float32)
    nf = np.empty((n,) + ab[0].shape[:2], np.float32)
    _recursive(L, ab, flow, range(n), n_max, tol, fwd, nf)

    prev_i = prev_out = prev_n = None
    for i in range(n - 1, -1, -1):  # backward pass, merged as it goes
        cur = ab[i].astype(np.float32)
        if prev_i is None:
            b, nb = cur, np.ones(cur.shape[:2], np.float32)
        else:
            F = flow(L[i], L[prev_i])
            c = confidence(L[i], warp(L[prev_i], F), tol)
            nb = np.minimum(n_max, 1.0 + c * warp(prev_n, F))
            a = (1.0 / nb)[..., None]
            b = a * cur + (1.0 - a) * warp(prev_out, F)
        f = np.asarray(fwd[i], np.float32)
        nfi = nf[i][..., None]
        nbi = nb[..., None]
        # both passes contain this frame's own chroma once, so count it once
        out[i] = (nfi * f + nbi * b - cur) / (nfi + nbi - 1.0)
        prev_i, prev_out, prev_n = i, b, nb
