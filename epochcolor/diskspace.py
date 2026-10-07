"""Free-space checks and readable "disk full" errors.

The usual surprise: /tmp. Many distros (openSUSE, Fedora, Arch) mount /tmp
as tmpfs, which lives in RAM and is capped at about half of it, so a big
unpack there fails with "no space left on device" while the home disk has
hundreds of gigabytes free. EpochColor keeps its own temp files next to
where the result goes for that reason, and these helpers say which
filesystem actually ran out.
"""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

GB = 1024 ** 3


def _existing(path: str | Path) -> Path:
    p = Path(path).expanduser().absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def free_bytes(path: str | Path) -> int:
    return shutil.disk_usage(_existing(path)).free


def mount_of(path: str | Path) -> tuple[str, str]:
    """(mount point, filesystem type) holding path, from /proc/mounts.
    ("", "") where that can't be read (Windows, odd sandboxes)."""
    p = str(_existing(path).resolve())
    best = ("", "")
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return best
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mnt = parts[1].replace("\\040", " ")
        if (p == mnt or p.startswith(mnt.rstrip("/") + "/") or mnt == "/") and len(mnt) >= len(best[0]):
            best = (mnt, parts[2])
    return best


def describe(path: str | Path) -> str:
    """One line: where path lives and how much room is left there."""
    mnt, fs = mount_of(path)
    free = free_bytes(path) / GB
    where = f"{mnt} ({fs})" if mnt else str(_existing(path))
    note = ", which is in RAM" if fs in ("tmpfs", "ramfs") else ""
    return f"{path} is on {where}{note}, {free:.1f} GB free"


class DiskFull(OSError):
    """ENOSPC with a message that already says where and how much."""

    def __init__(self, message: str, path: str | Path):
        super().__init__(errno.ENOSPC, message, str(path))
        self.message = message

    def __str__(self) -> str:
        return self.message


def is_full(e: BaseException) -> bool:
    return isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT)


def explain(e: OSError, path: str | Path | None = None) -> str:
    """Replace a bare ENOSPC with which folder filled up and how to move it."""
    if isinstance(e, DiskFull):
        return e.message
    target = path or e.filename or os.getcwd()
    try:
        line = describe(target)
    except OSError:
        line = str(target)
    return (f"No space left on device while writing {target}.\n{line}.\n"
            "If that is not the disk you expected, set the folder in the model manager "
            "(Storage folder...) or with EPOCHCOLOR_MODELS / EPOCHCOLOR_CACHE / TMPDIR.")


def need(path: str | Path, nbytes: int, what: str) -> None:
    """Raise early, before a long download, if path can't hold nbytes."""
    free = free_bytes(path)
    if free < nbytes:
        raise DiskFull(f"{what} needs about {nbytes / GB:.1f} GB but only {free / GB:.1f} GB is free. "
                       f"{describe(path)}.", path)
