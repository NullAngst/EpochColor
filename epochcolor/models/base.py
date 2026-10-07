"""The model adapter interface.

Every colorization model sits behind one adapter: luma in, a/b chroma out,
with optional hints. Swapping models costs an adapter and nothing else.

Milestone 1 handles one frame at a time. Batches of frames and reference
images come with the video work in milestone 2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..hints import Hints


@dataclass(frozen=True)
class ModelInfo:
    name: str
    mode: str  # "automatic", "hint", "exemplar", "propagation"
    description: str
    takes_hints: bool
    needs_weights: bool
    needs_torch: bool
    license: str = ""


class ColorModel(ABC):
    info: ModelInfo

    def load(self, device: str) -> None:
        """Load weights onto a device. Adapters that need nothing can skip it."""

    @abstractmethod
    def predict(self, L: np.ndarray, hints: Hints | None = None) -> np.ndarray:
        """L is HxW float32 L* (0..100), already denoised. Returns HxWx2
        float32 a/b at the same size as L. Hints, when given, are at that
        size too. An adapter that cannot use hints ignores them; the pipeline
        enforces them afterwards either way."""
