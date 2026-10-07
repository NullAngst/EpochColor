"""Paint hints from an image file.

Two ways to make one:

1. A layer with transparency (RGBA PNG/TIFF/WebP). Every pixel with alpha
   above half is a hint, its colour is the hint colour. Paint neutral grey
   with alpha to force an area to stay uncoloured.
2. A plain RGB copy of the photo with colour painted straight on it. Any
   pixel with visible chroma counts as a hint, everything grey is ignored.

The hint image is scaled to the photo if the sizes differ, so painting on a
smaller export works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .color import srgb_to_lab


@dataclass
class Hints:
    ab: np.ndarray  # HxWx2 float32
    mask: np.ndarray  # HxW float32, 0 or 1 (fractional after resizing)

    @property
    def count(self) -> int:
        return int((self.mask > 0.5).sum())

    def resized(self, w: int, h: int) -> "Hints":
        """Area-resize keeping hint colour unbiased by the empty pixels."""
        import cv2

        if self.mask.shape == (h, w):
            return self
        m = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_AREA)
        abm = cv2.resize(self.ab * self.mask[..., None], (w, h), interpolation=cv2.INTER_AREA)
        ab = np.where(m[..., None] > 1e-6, abm / np.maximum(m[..., None], 1e-6), 0.0)
        return Hints(ab.astype(np.float32), m.astype(np.float32))

    def binarized(self, thresh: float = 0.3) -> "Hints":
        m = (self.mask >= thresh).astype(np.float32)
        return Hints(self.ab * m[..., None], m)


def empty_hints(h: int, w: int) -> Hints:
    return Hints(np.zeros((h, w, 2), np.float32), np.zeros((h, w), np.float32))


def load_hints(path: str | Path, h: int, w: int, chroma_thresh: float = 6.0) -> Hints:
    """Read a hint image and return hints at h x w."""
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"could not read hint image {path}")
    if arr.ndim == 2:
        raise RuntimeError("hint image is greyscale, so it has no colour to give")
    maxv = 65535.0 if arr.dtype == np.uint16 else 255.0
    arr = arr.astype(np.float32) / maxv
    rgb = arr[..., 2::-1]  # BGR(A) to RGB
    lab = srgb_to_lab(np.ascontiguousarray(rgb))
    if arr.shape[2] == 4:
        mask = (arr[..., 3] > 0.5).astype(np.float32)
    else:
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        mask = (chroma > chroma_thresh).astype(np.float32)
    hints = Hints(lab[..., 1:].astype(np.float32) * mask[..., None], mask)
    return hints.resized(w, h).binarized(0.5) if hints.mask.shape != (h, w) else hints
