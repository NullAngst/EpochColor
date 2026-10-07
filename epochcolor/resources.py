"""Keeping heavy work from taking the whole machine down.

Three parts:

- The worker runs at a lower CPU priority (nice 10) and leaves a core free,
  so the desktop stays usable while a model pass or a render runs.
- A memory watchdog stops the work before the system starts swapping
  itself to a standstill. Linux only lets the OOM killer step in once swap
  is exhausted, which on a big swap file can mean ten minutes of a frozen
  desktop; stopping early with a clear message is far better.
- The worker writes everything it prints (Python and the C libraries under
  it, ROCm included) to a log file, and Python's fault handler dumps a stack
  trace there on a crash, so a dead worker leaves something to read.

Linux reads /proc; elsewhere the watchdog is a no-op.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

GB = 1024 ** 3


# ----------------------------------------------------------------- settings


def _setting(key: str, default):
    from .models.weights import read_settings

    try:
        v = read_settings().get(key, default)
    except Exception:
        return default
    return default if v is None else v


def cpu_threads() -> int:
    """Threads for the heavy work. Default: all cores but one."""
    n = os.cpu_count() or 2
    want = _setting("threads", 0)
    try:
        want = int(want)
    except (TypeError, ValueError):
        want = 0
    return max(1, min(n, want) if want > 0 else n - 1)


def niceness() -> int:
    try:
        return max(0, min(19, int(_setting("nice", 10))))
    except (TypeError, ValueError):
        return 10


# ------------------------------------------------------------------ memory


def meminfo() -> tuple[int, int] | None:
    """(total, available) bytes, or None where /proc/meminfo isn't there."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    vals = {}
    for line in text.splitlines():
        k, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            vals[k] = int(parts[0]) * 1024
    if "MemTotal" not in vals:
        return None
    return vals["MemTotal"], vals.get("MemAvailable", vals.get("MemFree", 0))


def children(pid: int) -> list[int]:
    out = []
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except OSError:
        return out
    for t in tasks:
        try:
            out += [int(x) for x in (t / "children").read_text().split()]
        except (OSError, ValueError):
            pass
    return out


def tree_rss(pid: int) -> int:
    """Resident memory of a process and everything it started (ffmpeg)."""
    total, seen, todo = 0, set(), [pid]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError):
            continue
        todo += children(p)
    return total


def memory_limit(total: int) -> int:
    """Most the worker may hold. Settings key memory_limit_gb, else 75% of RAM."""
    try:
        gb = float(_setting("memory_limit_gb", 0) or 0)
    except (TypeError, ValueError):
        gb = 0
    return int(gb * GB) if gb > 0 else int(total * 0.75)


def low_water(total: int) -> int:
    """Free memory below which work stops: 6% of RAM, at most 2 GB."""
    return int(min(2 * GB, total * 0.06))


def over_limit(pid: int) -> str | None:
    """Why pid should be stopped now, or None. Cheap enough to call twice a second."""
    m = meminfo()
    if m is None:
        return None
    total, avail = m
    rss = tree_rss(pid)
    lim = memory_limit(total)
    if rss > lim:
        return (f"it was using {rss / GB:.1f} GB, over its limit of {lim / GB:.1f} GB "
                f"(of {total / GB:.0f} GB RAM)")
    if avail < low_water(total) and rss > 0.5 * GB:
        return (f"the system was down to {avail / GB:.1f} GB of free memory "
                f"(the worker had {rss / GB:.1f} GB of it)")
    return None


class Watchdog(threading.Thread):
    """In-process watchdog for the command line: on a trip it prints why and
    ends the process (and its ffmpeg children) instead of letting it swap."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True, name="memory-watchdog")
        self.interval = interval

    def run(self) -> None:
        me = os.getpid()
        while True:
            time.sleep(self.interval)
            why = over_limit(me)
            if why:
                print(f"\nepochcolor: stopped to keep the system usable: {why}. "
                      "Lower --working-size, or raise memory_limit_gb in settings.json.",
                      file=sys.stderr, flush=True)
                for c in children(me):
                    try:
                        os.kill(c, 9)
                    except OSError:
                        pass
                os._exit(3)


# ----------------------------------------------------------------- limits


def apply_limits(log=None) -> None:
    """Lower priority and cap threads for this process. Call before numpy,
    OpenCV or torch start their thread pools (the env vars), and again after
    importing them (the setters) does no harm."""
    n = cpu_threads()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ.setdefault(var, str(n))
    if hasattr(os, "nice"):
        try:
            cur = os.nice(0)
            if cur < niceness():
                os.nice(niceness() - cur)
        except OSError:
            pass
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:
        pass
    if log:
        log(f"limits: {n} threads, nice {os.nice(0) if hasattr(os, 'nice') else '-'}")


def torch_threads() -> None:
    try:
        import torch

        torch.set_num_threads(cpu_threads())
    except Exception:
        pass


# -------------------------------------------------------------------- logs


def log_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "EpochColor"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "epochcolor"
    return base / "logs"


def open_log(name: str, max_bytes: int = 5 * 1024 * 1024):
    """Append-mode log file, rotated to name.1 once it passes max_bytes."""
    d = log_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    try:
        if p.stat().st_size > max_bytes:
            p.replace(p.with_name(name + ".1"))
    except OSError:
        pass
    return open(p, "a", buffering=1, errors="replace")


def capture_output(name: str) -> Path:
    """Send this process's stdout and stderr, at the file descriptor level,
    to a log file, and turn on the fault handler there. C libraries that
    print straight to stderr (ROCm's libdrm does) land in the log too."""
    import faulthandler

    f = open_log(name)
    try:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
    except OSError:
        pass
    sys.stdout = sys.stderr = f
    faulthandler.enable(file=f, all_threads=True)
    return Path(f.name)


def tail(path: Path, lines: int = 25) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            data = f.read().decode(errors="replace")
    except OSError:
        return ""
    return "\n".join(data.splitlines()[-lines:])


def describe_exit(code: int | None) -> str:
    import signal

    if code is None:
        return "it stopped without an exit code"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        what = {"SIGSEGV": "it crashed inside native code (a segmentation fault, most often the GPU "
                           "driver or the GPU build of PyTorch)",
                "SIGKILL": "it was killed, most often by the kernel's out-of-memory killer",
                "SIGABRT": "it aborted inside native code",
                "SIGBUS": "it crashed on a memory access (a full disk under a memory-mapped file "
                          "does this)"}.get(name, f"it was stopped by {name}")
        return f"{what} ({name})"
    return f"it exited with code {code}"
