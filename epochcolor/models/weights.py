"""Where weight files live and how they load. The catalog, downloads and
installs are in manager.py."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def config_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "EpochColor"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "epochcolor"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def read_settings() -> dict:
    try:
        return json.loads(settings_path().read_text())
    except (OSError, ValueError):
        return {}


def write_setting(key: str, value) -> None:
    s = read_settings()
    if value is None:
        s.pop(key, None)
    else:
        s[key] = value
    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text(json.dumps(s, indent=1))


def default_models_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "EpochColor" / "models"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "epochcolor" / "models"


def models_dir() -> Path:
    """EPOCHCOLOR_MODELS wins, then the location chosen in the model
    manager (shared by the editor and the command line), then the default."""
    env = os.environ.get("EPOCHCOLOR_MODELS")
    if env:
        return Path(env).expanduser()
    chosen = read_settings().get("models_dir")
    if chosen:
        return Path(chosen).expanduser()
    return default_models_dir()


def sha256_file(path: Path, progress=None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress("verify", done, total)
    return h.hexdigest()


def load_state_dict(path: Path):
    """Load weights without running pickled code.

    safetensors when the file is one, otherwise torch.load with
    weights_only=True, which refuses anything but tensors and plain types.
    A plain PyTorch pickle can run arbitrary code when loaded, and a model
    from a stranger shouldn't get that chance.
    """
    import torch

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if isinstance(sd, dict) and sd and all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}  # saved from DataParallel
    return sd
