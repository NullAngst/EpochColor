#!/usr/bin/env python3
"""Run EpochColor straight from a source checkout: python3 epochcolor.py

No arguments (or files) opens the editor; a subcommand runs the command
line, same as the AppImage. On the first run it offers to make a virtual
environment in .venv next to this file and install what it needs there,
so nothing touches your system Python. After that it switches into .venv
by itself every time.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
COMMANDS = {"photo", "video", "render", "fetch", "models", "device", "cache", "encoders", "presets",
            "gui", "setup-torch"}
NEEDED = ["numpy", "scipy", "cv2", "PIL", "tifffile", "av", "safetensors"]


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def missing(gui: bool) -> list[str]:
    import importlib.util

    names = NEEDED + (["PySide6"] if gui else [])
    return [n for n in names if importlib.util.find_spec(n) is None]


def make_venv() -> None:
    import venv

    if not venv_python().exists():
        print(f"making {VENV}")
        venv.create(VENV, with_pip=True)
    py = str(venv_python())
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([py, "-m", "pip", "install", "-e", f"{HERE}[gui,raw,heic]"])


def main() -> int:
    args = sys.argv[1:]
    if not args or (args[0] not in COMMANDS and not args[0].startswith("-")):
        args = ["gui"] + args
    gui = args[0] == "gui"

    # already set up: hop into the venv
    if VENV.exists() and not in_venv() and venv_python().exists():
        py = str(venv_python())
        if sys.platform == "win32":
            return subprocess.call([py, __file__] + sys.argv[1:])
        os.execv(py, [py, __file__] + sys.argv[1:])

    sys.path.insert(0, str(HERE))
    gone = missing(gui)
    if gone:
        if not sys.stdin.isatty():
            print(f"missing Python packages: {', '.join(gone)}\n"
                  f"install them with: {sys.executable} -m pip install -e \"{HERE}[gui,raw,heic]\"",
                  file=sys.stderr)
            return 1
        print(f"missing Python packages: {', '.join(gone)}")
        ask = (f"finish installing them into {VENV}? [Y/n] " if in_venv() else
               f"make a virtual environment in {VENV} and install them there? [Y/n] ")
        ans = input(ask).strip().lower()
        if ans not in ("", "y", "yes"):
            return 1
        try:
            make_venv()
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"setting up {VENV} failed: {e}", file=sys.stderr)
            return 1
        py = str(venv_python())
        if sys.platform == "win32":
            return subprocess.call([py, __file__] + sys.argv[1:])
        os.execv(py, [py, __file__] + sys.argv[1:])

    import shutil

    if shutil.which("ffmpeg") is None:
        print("note: no ffmpeg on PATH; photos work, video import and export need it", file=sys.stderr)

    from epochcolor.cli import main as cli

    return cli(args)


if __name__ == "__main__":  # the guard matters: worker processes re-import this file
    sys.exit(main())
