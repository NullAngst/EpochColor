"""Export presets as JSON, the same idea as HandBrake presets.

Built-in presets ship with the app. Saved ones live in
~/.config/epochcolor/presets/<name>.json and override a built-in of the same
name. A path to a .json file works anywhere a preset name does.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .plan import ExportError, ExportSettings

BUILTIN: dict[str, dict] = {
    "archive-h265": dict(codec="h265", bits=10, container=".mkv", rf=16, speed="slow",
                         tune_grain=True),
    "share-h264": dict(codec="h264", bits=8, container=".mp4", rf=20, speed="medium",
                       audio_codec="aac"),
    "small-av1": dict(codec="av1", bits=10, container=".mkv", rf=30, speed="6", audio_codec="opus"),
    "edit-prores": dict(codec="prores", bits=10, chroma="422", container=".mov", rf=None,
                        audio_codec="aac"),
    "master-ffv1": dict(codec="ffv1", bits=10, chroma="444", container=".mkv", rf=None,
                        audio_codec="flac"),
}

DESCRIPTIONS = {
    "archive-h265": "H.265 10-bit RF 16, slow, grain tuned. For keeping.",
    "share-h264": "H.264 8-bit RF 20 in MP4 with AAC. Plays on everything.",
    "small-av1": "SVT-AV1 10-bit RF 30 with Opus. Smallest files.",
    "edit-prores": "ProRes 422 HQ in MOV. For editing elsewhere.",
    "master-ffv1": "FFV1 10-bit 4:4:4, lossless. True archive master.",
}


def presets_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "EpochColor" / "presets"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "epochcolor" / "presets"


def _valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name))


def load_preset(name_or_path: str) -> tuple[ExportSettings, list[str]]:
    p = Path(name_or_path).expanduser()
    if p.suffix == ".json" and p.is_file():
        data = json.loads(p.read_text())
    else:
        saved = presets_dir() / f"{name_or_path}.json"
        if _valid_name(name_or_path) and saved.is_file():
            data = json.loads(saved.read_text())
        elif name_or_path in BUILTIN:
            data = dict(BUILTIN[name_or_path])
        else:
            raise ExportError(f"no preset called {name_or_path!r}; see `epochcolor presets`")
    if not isinstance(data, dict):
        raise ExportError(f"preset {name_or_path} is not a JSON object")
    data.pop("name", None)
    data.pop("description", None)
    s, unknown = ExportSettings.from_dict(data)
    warnings = [f"preset key {k!r} ignored" for k in unknown]
    return s, warnings


def save_preset(name: str, s: ExportSettings, description: str = "") -> Path:
    if not _valid_name(name):
        raise ExportError("preset names use letters, digits, dot, dash and underscore")
    d = presets_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    data = {"name": name, "description": description, **s.to_dict()}
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def list_presets() -> list[tuple[str, str, str]]:
    """(name, source, description)"""
    rows = {n: (n, "built in", DESCRIPTIONS.get(n, "")) for n in BUILTIN}
    d = presets_dir()
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            try:
                desc = json.loads(f.read_text()).get("description", "")
            except (OSError, ValueError):
                desc = "(unreadable)"
            rows[f.stem] = (f.stem, "saved", desc)
    return list(rows.values())
