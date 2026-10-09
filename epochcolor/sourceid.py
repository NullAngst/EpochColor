"""What a source file is, by its content.

Caches used to be named after a source's path and modification time. Both
change for reasons that have nothing to do with the picture: a drive that
mounts at /run/media/you/Disk1 instead of /run/media/you/Disk, a copy to
another disk, a sync or backup tool touching the file, a filesystem that
reports the time with less precision after a remount. Any of those threw
the whole colorize away.

source_id names a file by its size plus a hash of five 1 MB samples spread
through it (the whole file when it's small). That reads 5 MB however big
the file is, so it's quick even for a 200 GB scan, and it's remembered per
process. Two different videos with the same size and the same bytes at
all five places don't happen in practice; an edit to the picture changes
the size or the samples.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20
_memo: dict[tuple, str] = {}


def source_id(path: str | Path) -> str:
    p = Path(path).resolve()
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    got = _memo.get(k)
    if got is not None:
        return got
    n = st.st_size
    h = hashlib.sha256(f"epochcolor-source:{n}".encode())
    with open(p, "rb") as f:
        if n <= 5 * CHUNK:
            h.update(f.read())
        else:
            for off in (0, n // 4, n // 2, 3 * n // 4, n - CHUNK):
                f.seek(off)
                h.update(f.read(CHUNK))
    v = h.hexdigest()[:24]
    _memo[k] = v
    return v


def adopt(legacy: Path, new: Path) -> bool:
    """Move a cache entry from its old path-and-time name to its content
    name, if the old one exists and the new one doesn't. True if it moved."""
    if new.exists() or not legacy.exists():
        return False
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(new)
        return True
    except OSError:  # another process got there first, or a different disk
        return False
