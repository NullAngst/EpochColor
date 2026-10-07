"""Model registry."""

from __future__ import annotations

from .base import ColorModel, ModelInfo
from .hintsonly import HintsOnly
from .zhang import ECCV16, SIGGRAPH17

REGISTRY: dict[str, type[ColorModel]] = {
    "siggraph17": SIGGRAPH17,
    "eccv16": ECCV16,
    "hints": HintsOnly,
}

DEFAULT_MODEL = "siggraph17"


def create(name: str, weights: str | None = None) -> ColorModel:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}, pick one of {', '.join(REGISTRY)}") from None
    if cls.info.needs_weights:
        return cls(weights=weights)
    return cls()


__all__ = ["ColorModel", "ModelInfo", "REGISTRY", "DEFAULT_MODEL", "create"]
