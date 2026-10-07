"""Find out what this ffmpeg can actually encode.

An encoder showing up in `ffmpeg -encoders` only means it was compiled in.
Whether the driver behind it works is a separate question, so every
hardware encoder gets a two-frame test encode at the bit depth asked for.
Results are cached per ffmpeg binary, since a test takes a second or two.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .codecs import ENCODERS, EncoderSpec


def _run(cmd: list[str], timeout: float = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except OSError as e:
        return 127, str(e)


def find_vaapi_device() -> str | None:
    dri = Path("/dev/dri")
    if not dri.exists():
        return None
    nodes = sorted(dri.glob("renderD*"))
    return str(nodes[0]) if nodes else None


class Detector:
    def __init__(self, ffmpeg: str, cache_file: Path | None = None,
                 vaapi_device: str | None = None):
        self.ffmpeg = ffmpeg
        self.vaapi_device = vaapi_device or find_vaapi_device()
        st = os.stat(ffmpeg)
        self.ident = f"{os.path.realpath(ffmpeg)}:{st.st_size}:{st.st_mtime_ns}"
        self.cache_file = cache_file
        self.cache: dict = {}
        if cache_file and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                if data.get("ffmpeg") == self.ident:
                    self.cache = data.get("tests", {})
            except (OSError, ValueError):
                pass
        self._encoders: set[str] | None = None
        self._pix: dict[str, list[str]] = {}

    # ---- what's compiled in

    def encoders(self) -> set[str]:
        if self._encoders is None:
            rc, out = _run([self.ffmpeg, "-hide_banner", "-encoders"])
            names = set()
            for line in out.splitlines():
                m = re.match(r"\s*[VAS][\.A-Z]{5}\s+(\S+)", line)
                if m:
                    names.add(m.group(1))
            self._encoders = names
        return self._encoders

    def pix_fmts(self, encoder: str) -> list[str]:
        if encoder not in self._pix:
            rc, out = _run([self.ffmpeg, "-hide_banner", "-h", f"encoder={encoder}"])
            m = re.search(r"Supported pixel formats:\s*(.+)", out)
            self._pix[encoder] = m.group(1).split() if m else []
        return self._pix[encoder]

    # ---- does it work

    def test_command(self, spec: EncoderSpec, pix_fmt: str) -> list[str]:
        cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        vf = f"format={pix_fmt}"
        if spec.hw == "vaapi":
            cmd += ["-vaapi_device", self.vaapi_device or "/dev/dri/renderD128"]
            vf += ",hwupload"
        cmd += ["-f", "lavfi", "-i", "testsrc2=size=256x144:rate=24:duration=0.1",
                "-vf", vf, "-c:v", spec.name, "-frames:v", "2", "-f", "null", "-"]
        return cmd

    def check(self, spec: EncoderSpec, bits: int, chroma: str) -> str | None:
        """None if the encoder works for this format, else why not."""
        if spec.name not in self.encoders():
            return f"{spec.name} is not in this ffmpeg build"
        pix = spec.pix_fmt(bits, chroma)
        fmts = self.pix_fmts(spec.name)
        if fmts and pix not in fmts and not (spec.hw and spec.hw in ("vaapi", "qsv")):
            return f"{spec.name} in this ffmpeg build does not take {pix}"
        if spec.hw is None:
            return None
        if spec.hw == "vaapi" and not self.vaapi_device:
            return "no /dev/dri render node for VAAPI"
        key = f"{spec.name}:{pix}"
        if key not in self.cache:
            rc, out = _run(self.test_command(spec, pix), timeout=30)
            err = out.strip().splitlines()[-1] if out.strip() else f"exit {rc}"
            self.cache[key] = None if rc == 0 else err[:200]
            self._save()
        res = self.cache[key]
        return None if res is None else f"{spec.name} failed a test encode: {res}"

    def _save(self) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps({"ffmpeg": self.ident, "tests": self.cache}))
        except OSError:
            pass

    def clear(self) -> None:
        self.cache = {}
        self._save()

    def table(self) -> list[tuple[str, str, str, str]]:
        """(encoder, codec, 8-bit result, 10-bit result) for every known encoder."""
        rows = []
        for spec in ENCODERS.values():
            res = []
            for bits in (8, 10):
                if bits not in spec.depths:
                    res.append("n/a")
                    continue
                chroma = spec.chromas[0]
                why = self.check(spec, bits, chroma)
                if why is None:
                    res.append("works")
                elif "not in this ffmpeg build" in why:
                    res.append("not built in")
                elif "no /dev/dri" in why:
                    res.append("no device")
                else:
                    res.append("FAILS")
            rows.append((spec.name, spec.codec, res[0], res[1]))
        return rows
