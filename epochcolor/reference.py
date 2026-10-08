"""Reference images: a colour photo guides a shot's or a photo's colours.

Give it a colour photo of the same place, the same kind of scene, or the
same era (a period postcard of the street, an Autochrome, a colour still
from the same set) and the colours it finds there steer the colorization.

How: the colorization model's own mid-level features describe what each
part of a picture is (sky, brick, skin, foliage), since predicting colour
takes knowing that. Each part of the black and white frame is matched
against every part of the reference by those features, plus how close
their brightness is, and takes a soft average of the colours of its best
matches. Where nothing in the reference looks like it, the match is weak
and the model's own colour stays. Nothing gets trained; it's matching at
inference time, the same machinery as saved colours.

The result is a fix like a painted keyframe's: a target a/b and a weight,
carried through the shot along the motion by the hint pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

LUMA_WEIGHT = 0.3  # how much a brightness mismatch counts against a feature match
TEMPERATURE = 0.05  # lower takes only the very best matches' colour
FLOOR, SPAN = 0.35, 0.35  # feature similarity below FLOOR counts as no match, FLOOR+SPAN as full


def reference_id(ref: dict | None) -> str | None:
    """Changes when the reference file, its strength, or (for painting used
    as a reference) the painted strokes do."""
    if not ref:
        return None
    if ref.get("kind") == "painted":
        from .hintpaint import strokes_key

        srcs = [[x.get("path"), x.get("frame"), strokes_key(x.get("strokes") or []), x.get("negative")]
                for x in ref.get("sources") or []]
        if not srcs:
            return None
        raw = json.dumps([srcs, round(float(ref.get("strength", 1.0)), 3)], sort_keys=True, default=str)
        return "p" + hashlib.sha1(raw.encode()).hexdigest()[:11]
    if not ref.get("path"):
        return None
    p = Path(ref["path"])
    try:
        st = p.stat()
        stamp = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    raw = f"{p.resolve()}|{stamp}|{round(float(ref.get('strength', 1.0)), 3)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_reference(path: str | Path, short: int = 512):
    """(L, ab) of a colour photo at about `short` px on the short side."""
    import cv2

    from .color import srgb_to_lab
    from .imageio import load_image

    img, _ = load_image(path)
    if img.ndim == 2:
        raise ValueError(f"{Path(path).name} is black and white; a reference has to be in colour")
    h, w = img.shape[:2]
    k = min(1.0, short / min(h, w))
    img = cv2.resize(img, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA)
    lab = srgb_to_lab(img)
    return lab[..., 0].astype(np.float32), lab[..., 1:].astype(np.float32)


def _source_luma(src: dict, short: int) -> np.ndarray:
    """L* of a painted frame or photo, at about `short` px on the short side."""
    import cv2

    from .color import srgb_to_l
    from .film import invert, invert_rgb

    if src.get("frame") is None:
        from .imageio import load_image

        img, _ = load_image(src["path"])
        L = srgb_to_l(invert_rgb(img, src.get("negative")) if src.get("negative") else img)
    else:
        from fractions import Fraction

        from .media import FrameReader

        r = FrameReader(src["path"], Fraction(src["fps_num"], src["fps_den"]), "gray16le", cache=2)
        try:
            g = r.get(int(src["frame"]))
        finally:
            r.close()
        if g is None:
            raise RuntimeError(f"{Path(src['path']).name}: frame {src['frame']} could not be decoded")
        L = srgb_to_l(invert(g, src.get("negative")))
    h, w = L.shape
    k = min(1.0, short / min(h, w))
    return cv2.resize(L, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA)


def painted_reference(model, ref: dict, short: int, cache: dict | None = None):
    """Features, brightness and colour of every painted place in the painted
    frames a reference points at: (R (C, N) on the model's device, L (N,), ab (N, 2)).
    Only places the strokes actually fill count, so unpainted parts of
    those frames can't hand their grey to anything."""
    import cv2
    import torch

    from .hintpaint import rasterize, strokes_key
    from .propagate import paint_fill

    Rs, Ls, abs_ = [], [], []
    for src in ref.get("sources") or []:
        key = ("painted", src.get("path"), src.get("frame"), strokes_key(src.get("strokes") or []), id(model))
        if cache is not None and key in cache:
            part = cache[key]
        else:
            L = _source_luma(src, short)
            F = model.features(L)
            C, fh, fw = F.shape
            ch_, cw_ = max(8, L.shape[0] // 2), max(8, L.shape[1] // 2)
            Lc = cv2.resize(L, (cw_, ch_), interpolation=cv2.INTER_AREA)
            T, W = paint_fill(Lc, rasterize(src.get("strokes") or [], cw_, ch_))
            Ts = cv2.resize(T * W[..., None], (fw, fh), interpolation=cv2.INTER_AREA)
            Ws = cv2.resize(W, (fw, fh), interpolation=cv2.INTER_AREA)
            ok = (Ws > 0.6).ravel()
            Ts = (Ts / np.maximum(Ws, 1e-3)[..., None]).reshape(-1, 2)
            Lsm = cv2.resize(L, (fw, fh), interpolation=cv2.INTER_AREA).ravel()
            part = (F.reshape(C, -1)[:, torch.from_numpy(ok).to(F.device)], Lsm[ok], Ts[ok])
            if cache is not None:
                cache[key] = part
        if part[0].shape[1]:
            Rs.append(part[0])
            Ls.append(part[1])
            abs_.append(part[2])
    if not Rs:
        return None
    return torch.cat(Rs, 1), np.concatenate(Ls).astype(np.float32), np.concatenate(abs_).astype(np.float32)


def reference_fix(model, L: np.ndarray, ref: dict, out_size: tuple[int, int] | None = None,
                  cache: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Target a/b and weight for the frame whose L* is given (working size),
    at out_size (w, h), default L's size. ref: {"path", "strength"} for a
    colour photo, or {"kind": "painted", "sources": [...], "strength"} for
    painted frames used as the reference (painted_reference).
    cache: optional dict to keep the reference's features between frames."""
    import cv2
    import torch

    from .video.hints_pass import guided

    if not hasattr(model, "features"):
        raise RuntimeError(f"{model.info.name} can't use a reference image; pick a network model")
    floor = FLOOR
    if ref.get("kind") == "painted":
        got = painted_reference(model, ref, max(L.shape), cache)
        if got is None:
            w, h = out_size or (L.shape[1], L.shape[0])
            return np.zeros((h, w, 2), np.float32), np.zeros((h, w), np.float32)
        Fr, Lr_s, abr_s = got
        floor = FLOOR + 0.1  # painted places are few; ask a closer match before they spread
    else:
        key = (str(ref["path"]), id(model))
        if cache is not None and key in cache:
            Fr, Lr_s, abr_s = cache[key]
        else:
            Lr, abr = load_reference(ref["path"], max(L.shape))
            Fr = model.features(Lr)  # (C, hr, wr), unit length per position
            C, hr, wr = Fr.shape
            Lr_s = cv2.resize(Lr, (wr, hr), interpolation=cv2.INTER_AREA)
            abr_s = cv2.resize(abr, (wr, hr), interpolation=cv2.INTER_AREA)
            if cache is not None:
                cache[key] = (Fr, Lr_s, abr_s)
    Fk = model.features(L)
    C, hk, wk = Fk.shape
    Lk_s = cv2.resize(L, (wk, hk), interpolation=cv2.INTER_AREA)
    dev = Fk.device
    with torch.inference_mode():
        K = Fk.reshape(C, -1).float()
        R = Fr.to(dev).reshape(Fr.shape[0], -1).float()
        sim = K.T @ R  # (Nk, Nr)
        dl = torch.from_numpy(Lk_s.reshape(-1, 1)).to(dev) - torch.from_numpy(Lr_s.reshape(1, -1)).to(dev)
        score = sim - LUMA_WEIGHT * dl.abs() / 100.0
        A = torch.softmax(score / TEMPERATURE, dim=1)
        ab = A @ torch.from_numpy(abr_s.reshape(-1, 2)).to(dev)
        best = sim.max(dim=1).values
    T = ab.cpu().numpy().reshape(hk, wk, 2).astype(np.float32)
    W = np.clip((best.cpu().numpy().reshape(hk, wk) - floor) / SPAN, 0, 1).astype(np.float32)
    W *= float(ref.get("strength", 1.0))
    w, h = out_size or (L.shape[1], L.shape[0])
    I = cv2.resize(L, (w, h), interpolation=cv2.INTER_AREA)
    T = cv2.resize(T, (w, h), interpolation=cv2.INTER_LINEAR)
    W = cv2.resize(W, (w, h), interpolation=cv2.INTER_LINEAR)
    r = max(2, int(round(min(w, h) / 64)))
    T = guided(T, I, r)  # colour edges follow the frame's own edges
    W = np.clip(guided(W, I, r), 0, 1)
    return T.astype(np.float32), W.astype(np.float32)


def keyframes(a: int, b: int, every: int = 48) -> list[int]:
    """Frames of a shot the reference gets matched on: the middle, and one
    every couple of seconds in a long shot, so the fix doesn't have to
    travel far along the motion."""
    n = b - a
    if n <= every:
        return [a + n // 2]
    k = int(np.ceil(n / every))
    step = n / k
    return [a + int(step * (i + 0.5)) for i in range(k)]
