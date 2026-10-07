"""First-run PyTorch install.

Releases don't bundle PyTorch: a CUDA build alone runs to several gigabytes,
and AMD users should never download CUDA. Instead the right wheel for the
GPU in this machine goes into its own folder on first run:

    ~/.local/share/epochcolor/torch-cpython-312   (Linux)
    %LOCALAPPDATA%\\EpochColor\\torch-cpython-312  (Windows)

(one folder per Python version, since the wheels are built per version)
and that folder is added to the import path at startup. Removing the folder
uninstalls it. A PyTorch you installed yourself (a venv, a distro package)
always wins, since the folder goes at the end of the path.

Which build: AMD with /dev/kfd gets ROCm, NVIDIA gets the newest CUDA build
its driver can run, Intel gets XPU, everything else CPU. The PyTorch index
is checked live, newest first, so this does not go stale as new builds
come out.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE = "https://download.pytorch.org/whl"
ROCM = ["rocm7.4", "rocm7.3", "rocm7.2", "rocm7.1", "rocm7.0", "rocm6.4", "rocm6.3", "rocm6.2"]
# CUDA build -> lowest Linux driver it runs on
CUDA = [("cu130", 580), ("cu129", 575), ("cu128", 570), ("cu126", 560), ("cu124", 550),
        ("cu121", 530), ("cu118", 520)]
KFD = Path("/dev/kfd")  # the ROCm compute device
SIZES = {"cpu": "about 200 MB", "xpu": "about 2 GB", "cuda": "about 3 GB", "rocm": "4 to 6 GB"}
# Rough room needed while installing: the wheel plus its unpacked copy,
# which sit side by side until pip finishes. Unpacked ROCm and CUDA builds
# are several times the download.
NEED_GB = {"cpu": 2, "xpu": 8, "cuda": 12, "rocm": 20}


def _base() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "EpochColor"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "epochcolor"


def torch_dir() -> Path:
    """Where this Python's PyTorch goes. One folder per Python version,
    since the wheels are built per version: the AppImage runs 3.12, a
    source checkout runs whatever your distro ships."""
    return _base() / f"torch-{sys.implementation.cache_tag}"


def _legacy() -> Path | None:
    """The unversioned folder older releases used, if it suits this Python."""
    d = _base() / "torch"
    tag = sys.implementation.cache_tag  # cpython-312
    short = f"cp{sys.version_info.major}{sys.version_info.minor}"  # Windows names
    if (d / "torch").is_dir() and (any((d / "torch").glob(f"_C.{tag}*")) or any((d / "torch").glob(f"_C.{short}*"))):
        return d
    return None


def active_dir() -> Path | None:
    d = torch_dir()
    if (d / "torch").is_dir():
        return d
    return _legacy()


def activate() -> bool:
    """Put the installed PyTorch on the import path. Cheap, safe to call twice."""
    d = active_dir()
    if d is not None and str(d) not in sys.path:
        sys.path.append(str(d))
        return True
    return False


def installed_variant() -> str | None:
    d = active_dir()
    try:
        return (d / "variant.txt").read_text().strip() or None if d else None
    except OSError:
        return None


def have_torch() -> bool:
    import importlib.util

    activate()
    return importlib.util.find_spec("torch") is not None


# ------------------------------------------------------------- detection


def gpu_vendors() -> set[str]:
    """Vendors of the display adapters, from sysfs. Linux only."""
    found = set()
    for vendor in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            v = vendor.read_text().strip().lower()
        except OSError:
            continue
        found.add({"0x1002": "amd", "0x10de": "nvidia", "0x8086": "intel"}.get(v, v))
    return found


def nvidia_driver() -> int | None:
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
    except OSError:
        return None
    m = re.search(r"Kernel Module(?: for [^ ]+)?\s+(\d+)\.", text)
    return int(m.group(1)) if m else None


def detect() -> tuple[str, str]:
    """(family, reason): family is rocm, cuda, xpu or cpu."""
    v = gpu_vendors()
    if "nvidia" in v:
        drv = nvidia_driver()
        if drv:
            return "cuda", f"NVIDIA GPU, driver {drv}"
        return "cpu", "NVIDIA GPU found but no NVIDIA driver loaded (nouveau can't run CUDA)"
    if "amd" in v:
        if KFD.exists():
            return "rocm", "AMD GPU with /dev/kfd"
        return "cpu", "AMD GPU but no /dev/kfd, so ROCm can't reach it"
    if "intel" in v:
        return "xpu", "Intel GPU"
    return "cpu", "no supported GPU found"


def _index_has(name: str) -> bool:
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    url = f"{BASE}/{name}/torch/"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "EpochColor"}),
                                    timeout=30) as r:
            page = r.read().decode(errors="replace")
    except Exception:
        return False
    arch = "x86_64" if sys.platform != "win32" else "win_amd64"
    return bool(re.search(rf"{py}-{py}[^\"<]*{arch}", page))


def pick_index(family: str) -> str:
    """The newest PyTorch index for this family that has a wheel for this
    Python, and for CUDA, that the installed driver can run."""
    if family == "rocm":
        names = ROCM
    elif family == "cuda":
        drv = nvidia_driver() or 0
        names = [n for n, need in CUDA if drv >= need]
        if not names:
            raise RuntimeError(f"NVIDIA driver {drv} is older than any CUDA build PyTorch ships; "
                               f"update the driver or use --variant cpu")
    elif family in ("xpu", "cpu"):
        names = [family]
    else:
        raise ValueError(f"unknown variant {family!r}")
    for n in names:
        if _index_has(n):
            return n
    raise RuntimeError(f"no {family} build of PyTorch for Python "
                       f"{sys.version_info.major}.{sys.version_info.minor} found on {BASE}")


# --------------------------------------------------------------- install


def install(family: str = "auto", log=print) -> Path:
    from . import diskspace

    if family == "auto":
        family, why = detect()
        log(f"detected: {why}, so {family}")
    index = pick_index(family)
    url = f"{BASE}/{index}"
    log(f"installing torch from {url} ({SIZES.get(family, '')})")
    final = torch_dir()
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name("torch.new")
    # pip unpacks into the system temp folder before moving files into
    # --target. /tmp is often tmpfs (RAM, half its size at most), far too
    # small for a GPU build, so pip gets a temp folder on the same disk as
    # the install. Same disk also makes the final move a rename.
    work = final.with_name("pip-tmp")
    for d in (tmp, work):
        shutil.rmtree(d, ignore_errors=True)
    diskspace.need(final.parent, NEED_GB.get(family, 12) * diskspace.GB, f"PyTorch ({family})")
    work.mkdir()
    env = {**os.environ, "TMPDIR": str(work), "TEMP": str(work), "TMP": str(work)}
    # --no-cache-dir: otherwise pip keeps a second copy of the wheel in
    # ~/.cache/pip, gigabytes for something installed once
    cmd = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
           "--no-cache-dir", "--target", str(tmp), "--index-url", url, "torch"]
    log(f"working folder: {diskspace.describe(work)}")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        full = False
        for line in proc.stdout:
            line = line.rstrip()
            full = full or "No space left on device" in line
            log(line)
        if proc.wait() != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            if full:
                raise RuntimeError(f"pip ran out of disk space; nothing was changed. "
                                   f"{diskspace.describe(final.parent)}")
            raise RuntimeError("pip failed; nothing was changed")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    # numpy comes with the app; a second copy in here would shadow nothing
    # (this folder is last on the path) but wastes space
    for extra in tmp.glob("numpy*"):
        shutil.rmtree(extra, ignore_errors=True)
    (tmp / "variant.txt").write_text(index + "\n")
    old = final.with_name("torch.old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    leg = _legacy()
    if leg is not None:  # replaced by the versioned folder
        shutil.rmtree(leg, ignore_errors=True)
    log(f"done: {final}")
    return final


def remove() -> bool:
    gone = False
    for d in (torch_dir(), _legacy()):
        if d is not None and d.exists():
            shutil.rmtree(d)
            gone = True
    return gone