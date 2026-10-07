"""Weight files: where they live, how they get fetched, how they load.

This is the bare minimum the CLI needs. The in-app model manager with a
remote catalog comes in milestone 8.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightEntry:
    model: str
    url: str
    filename: str
    sha256_prefix: str  # torch hub style: the filename carries the hash prefix
    size_mb: int
    license: str


CATALOG: dict[str, WeightEntry] = {
    "eccv16": WeightEntry(
        model="eccv16",
        url="https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
        filename="colorization_release_v2-9b330a0b.pth",
        sha256_prefix="9b330a0b",
        size_mb=130,
        license="BSD-2-Clause",
    ),
    "siggraph17": WeightEntry(
        model="siggraph17",
        url="https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth",
        filename="siggraph17-df00044c.pth",
        sha256_prefix="df00044c",
        size_mb=140,
        license="BSD-2-Clause",
    ),
}


def models_dir() -> Path:
    env = os.environ.get("EPOCHCOLOR_MODELS")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "EpochColor" / "models"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "epochcolor" / "models"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_weights(model: str, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"weights file not found: {p}")
        return p
    entry = CATALOG.get(model)
    if entry is None:
        raise FileNotFoundError(f"no catalog entry for {model}, pass --weights")
    p = models_dir() / entry.filename
    if not p.is_file():
        raise FileNotFoundError(
            f"{model} weights are not downloaded. Run: epochcolor fetch {model}"
        )
    return p


def fetch(model: str, progress=True) -> Path:
    """Download weights with resume, then check the hash before keeping them."""
    entry = CATALOG[model]
    dest_dir = models_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / entry.filename
    if dest.is_file():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(entry.url, headers={"User-Agent": "EpochColor"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored the range, start over
        total = resp.headers.get("Content-Length")
        total = int(total) + have if total else None
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    print(f"\r{model}: {done * 100 // total}% of {total >> 20} MB", end="", file=sys.stderr)
    if progress:
        print(file=sys.stderr)
    digest = sha256_file(part)
    if not digest.startswith(entry.sha256_prefix):
        part.unlink()
        raise RuntimeError(f"hash mismatch for {entry.filename}, download deleted")
    part.rename(dest)
    return dest


def load_state_dict(path: Path):
    """Load weights without running pickled code.

    safetensors when the file is one, otherwise torch.load with
    weights_only=True, which refuses anything but tensors and plain types.
    """
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd
