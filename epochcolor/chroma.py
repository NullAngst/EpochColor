"""Adjustments to predicted chroma before it meets the original luma:
colour cast removal and saturation.

Cast removal is for the all-over tint some models lay on a whole scene,
most often an orange or sepia wash. The idea is the photographer's grey
world, aimed only where it's safe: the least colourful part of the
picture is where the model had the least reason to put colour, so if even
that part leans orange, the lean is the model's cast and not the scene.
The cast is the median a/b of the 40% least colourful pixels, and removing
it shifts all the chroma back by that much (times the strength).

A scene that's honestly warm (a sunset, a desert, a candlelit room) keeps
neutral-ish pixels somewhere and so reads as little cast. A scene with no
neutral at all reads as cast and gets cooled; turn the strength down for
those shots or set a per-shot white balance in the grade.

For video the cast is measured per shot, from frames spread across it, so
it doesn't flicker frame to frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LOW_FRACTION = 0.4


def estimate_cast(ab: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
    """(2,) a/b offset. ab: (..., 2) chroma, any leading shape, any dtype."""
    x = np.asarray(ab).reshape(-1, 2)
    if len(x) > max_samples:
        x = x[:: int(np.ceil(len(x) / max_samples))]
    x = x.astype(np.float32)
    if not len(x):
        return np.zeros(2, np.float32)
    c = np.hypot(x[:, 0], x[:, 1])
    cut = np.quantile(c, LOW_FRACTION)
    low = x[c <= cut]
    if len(low) < 16:
        low = x
    return np.median(low, axis=0).astype(np.float32)


def adjust(ab: np.ndarray, saturation: float = 1.0, cast: float = 0.0,
           bias: np.ndarray | None = None) -> np.ndarray:
    """Cast removal, then saturation. Returns float32; ab itself is left alone."""
    out = np.asarray(ab, np.float32)
    if cast and bias is not None:
        out = out - float(cast) * np.asarray(bias, np.float32)
    if saturation != 1.0:
        out = out * float(saturation)
    return out


def adjust_rgb(rgb: np.ndarray, saturation: float = 1.0, cast: float = 0.0) -> np.ndarray:
    """The same on a finished sRGB picture (a colorized photo): luma kept."""
    if saturation == 1.0 and not cast:
        return rgb
    from .color import lab_to_srgb, srgb_to_lab

    lab = srgb_to_lab(np.asarray(rgb, np.float32))
    bias = estimate_cast(lab[..., 1:]) if cast else None
    return lab_to_srgb(lab[..., 0], adjust(lab[..., 1:], saturation, cast, bias))


class ShotCasts:
    """Per-shot cast of an analysed clip, measured lazily from its ab.npy
    and remembered. Shots come from the analysis's meta.json."""

    def __init__(self, analysis_dir: str | Path, ab=None, samples: int = 12):
        self.dir = Path(analysis_dir)
        self.ab = ab
        self.samples = samples
        self.shots: list[tuple[int, int]] = []
        self.cache: dict[tuple[int, int], np.ndarray] = {}
        try:
            meta = json.loads((self.dir / "meta.json").read_text())
            stab = meta.get("stab") or {}
            self.shots = sorted(tuple(int(v) for v in k.split("-")) for k in stab)
        except (OSError, ValueError):
            pass

    def _ab(self):
        if self.ab is None:
            self.ab = np.load(self.dir / "ab.npy", mmap_mode="r")
        return self.ab

    def shot_of(self, frame: int) -> tuple[int, int]:
        for a, b in self.shots:
            if a <= frame < b:
                return a, b
        n = len(self._ab())
        return 0, n  # no shot list: the whole clip

    def at(self, frame: int) -> np.ndarray:
        sh = self.shot_of(frame)
        if sh not in self.cache:
            a, b = sh
            ab = self._ab()
            b = min(b, len(ab))
            idx = np.unique(np.linspace(a, max(a, b - 1), self.samples).round().astype(int))
            self.cache[sh] = estimate_cast(np.stack([np.asarray(ab[i]) for i in idx]))
        return self.cache[sh]
