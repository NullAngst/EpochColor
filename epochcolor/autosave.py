"""Autosave and recovery.

Every couple of minutes, a project with unsaved changes is written to a
side file in the state folder (never over your project file). Saving
deletes it, closing normally deletes it. If EpochColor dies or the machine
does, the side file is still there, and opening that project (or starting
EpochColor, for a project that was never saved) offers to bring the
changes back.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path


def autosave_dir() -> Path:
    from .resources import log_dir

    return log_dir().parent / "autosave"


def path_for(project_path: str | Path | None) -> Path:
    if project_path is None:
        return autosave_dir() / "untitled.epochcolor"
    key = hashlib.sha1(str(Path(project_path).expanduser().resolve()).encode()).hexdigest()[:16]
    return autosave_dir() / f"{key}.epochcolor"


def write(project) -> Path:
    from .project import FORMAT_VERSION

    out = path_for(project.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"epochcolor_project": FORMAT_VERSION, **asdict(project.d),
            "_autosave": {"for": str(project.path) if project.path else None, "time": time.time()}}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(out)
    return out


def find(project_path: str | Path | None) -> tuple[Path, float] | None:
    """(side file, when it was written) if it holds changes newer than the project file."""
    p = path_for(project_path)
    if not p.exists():
        return None
    try:
        when = float(json.loads(p.read_text()).get("_autosave", {}).get("time", p.stat().st_mtime))
    except (OSError, ValueError):
        return None
    if project_path is not None:
        try:
            if Path(project_path).stat().st_mtime >= when:
                return None  # saved since
        except OSError:
            pass
    return p, when


def recover(side: Path, project_path: str | Path | None):
    """The project as autosaved, pointed at its real file and marked unsaved."""
    from .project import Project

    p = Project.load(side)
    p.path = Path(project_path) if project_path else None
    p.dirty = True
    return p


def clear(project_path: str | Path | None) -> None:
    try:
        path_for(project_path).unlink()
    except OSError:
        pass
