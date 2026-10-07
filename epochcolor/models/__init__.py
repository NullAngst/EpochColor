"""Model registry: adapters by architecture, models by id.

A model id names an installed model (see manager.py) or the built-in
"hints" model. Every installed model points at an adapter by its
architecture, so a new set of weights for a known architecture needs no
new code.
"""

from __future__ import annotations

from .base import ColorModel, ModelInfo
from .hintsonly import HintsOnly

DEFAULT_MODEL = "siggraph17"


def adapters() -> dict[str, type]:
    from .ddcolor import DDColorAdapter
    from .zhang import ECCV16, SIGGRAPH17

    return {"eccv16": ECCV16, "siggraph17": SIGGRAPH17, "ddcolor": DDColorAdapter}


def info_for(e: dict) -> ModelInfo:
    cls = adapters()[e["architecture"]]
    return ModelInfo(
        name=e["id"], mode=e.get("mode", cls.info.mode), description=e.get("description", ""),
        takes_hints=cls.info.takes_hints, needs_weights=True, needs_torch=True,
        license=e.get("license", ""),
    )


def create(name: str, weights: str | None = None) -> ColorModel:
    """name: an installed model id, a catalog id (with weights=path), or "hints"."""
    if name == "hints":
        return HintsOnly()
    from .manager import ModelError, entry, installed

    man = installed().get(name)
    if man is None:
        man = entry(name)
        if man is None:
            known = ", ".join(sorted(set(installed()) | {"hints"}))
            raise ValueError(f"unknown model {name!r}; installed: {known}")
        if weights is None:
            raise ModelError(f"{name} is not downloaded. Download it in Colour > Model manager, "
                             f"or run: epochcolor fetch {name}")
    path = weights or man["_path"]
    cls = adapters()[man["architecture"]]
    m = cls(weights=str(path), info=info_for(man), params=man.get("params", {}))
    # what the analysis cache is keyed on: the installed weights' hash, or an
    # explicitly given file's path
    m.cache_id = str(weights) if weights else (man.get("sha256") or "")
    return m


def cache_id_for(name: str) -> str:
    """The cache identity of an installed model, the same one create() sets."""
    if name == "hints":
        return ""
    from .manager import installed

    man = installed().get(name)
    return (man or {}).get("sha256") or ""


def choices() -> list[tuple[str, str, bool]]:
    """(id, label, installed) for every model a project can pick."""
    from .manager import catalog, installed

    have = installed()
    out = []
    for e in catalog():
        out.append((e["id"], e["name"], e["id"] in have))
    for mid, man in have.items():
        if not any(c[0] == mid for c in out):
            out.append((mid, man.get("name", mid), True))
    return out


__all__ = ["ColorModel", "ModelInfo", "DEFAULT_MODEL", "create", "choices", "adapters", "cache_id_for"]
