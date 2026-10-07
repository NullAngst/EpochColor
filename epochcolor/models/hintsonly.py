"""Colour from painted hints alone, no neural network.

Starts from neutral grey and spreads every stroke out to the edges around
it. Needs no weights and no PyTorch, which makes it the quickest way to check
the grain split and the hint handling. It also covers the case where you want
full control and are willing to paint every region.
"""

from __future__ import annotations

import numpy as np

from .base import ColorModel, ModelInfo


class HintsOnly(ColorModel):
    info = ModelInfo(
        name="hints",
        mode="propagation",
        description="No network. Spreads painted hints to edges, everything unpainted stays grey.",
        takes_hints=True,
        needs_weights=False,
        needs_torch=False,
        license="GPL-3.0",
    )

    def predict(self, L, hints=None):
        # The colour itself comes from the pipeline's hint pass, which this
        # adapter asks to spread much further than a correction would.
        return np.zeros(L.shape + (2,), np.float32)
