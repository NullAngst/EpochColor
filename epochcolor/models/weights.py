"""Where weight files live and how they load. The catalog, downloads and
installs are in manager.py."""

from __future__ import annotations

import hashlib
import json
import pickle
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


class Inert:
    """Stands in for any pickled object that isn't a tensor or a plain value.
    Keeps whatever arguments and state it was given, and does nothing."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args, obj.kwargs = args, kwargs
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state

    def __call__(self, *args, **kwargs):  # a stand-in for a function: calling it gives another stand-in
        return Inert(*args, **kwargs)

    def __repr__(self):
        return f"<inert {type(self).__name__}>"


_BUILTINS = {"slice", "set", "frozenset", "complex", "int", "float", "bool", "str", "bytes",
             "bytearray", "list", "tuple", "dict", "range"}
_PLAIN = {"builtins": _BUILTINS, "__builtin__": _BUILTINS,  # the second is how older pickles spell it
          "collections": {"OrderedDict"}}


def _allowed_torch(module: str, name: str):
    """PyTorch's own pieces a weights file legitimately needs: tensor and
    parameter rebuilders, storages, dtypes, sizes and devices. None if not."""
    import importlib

    import torch

    if not (module == "torch" or module.startswith("torch.")):
        return None
    try:
        obj = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None
    if module == "torch._utils" and name.startswith("_rebuild_"):
        return obj
    if module in ("torch.serialization",) and name == "_get_layout":
        return obj
    if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
        return obj
    if isinstance(obj, type) and (issubclass(obj, (torch.Size, torch.device))
                                  or name.endswith("Storage") or name in ("Tensor", "TypedStorage",
                                                                          "UntypedStorage")):
        return obj
    if obj is torch.device or obj is torch.Size:
        return obj
    return None


def _inert_pickle():
    """A pickle module for torch.load whose unpickler only hands out plain
    types and PyTorch's own rebuilders; every other global becomes Inert."""
    import types

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in _PLAIN.get(module, ()):
                return super().find_class(module, name)
            obj = _allowed_torch(module, name)
            if obj is not None:
                return obj
            return type(f"{module}.{name}", (Inert,), {})

    mod = types.ModuleType("epochcolor_inert_pickle")
    mod.Unpickler = Unpickler
    mod.load = lambda f, **kw: Unpickler(f, **kw).load()
    mod.__name__ = "epochcolor_inert_pickle"
    return mod


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
    try:
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Older checkpoints (DeOldify's, from 2019) carry training leftovers
        # next to the weights: optimizer settings, functools.partial, slice.
        # PyTorch's safe loader refuses the whole file over them. This loader
        # is just as strict about what runs, and keeps those leftovers as
        # inert stand-ins instead; only the weights get used.
        sd = torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=_inert_pickle())
    for key in ("params_ema", "params", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    if isinstance(sd, dict) and sd and all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}  # saved from DataParallel
    return sd
