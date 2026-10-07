"""Video in and out.

Decoding goes through PyAV, frame by frame, as 16-bit grey. Taking only the
luma plane sidesteps the YUV matrix question: a black and white source has
neutral chroma, so R=G=B=Y' whatever matrix it was tagged with.

Encoding pipes 16-bit RGB into an ffmpeg subprocess, which converts to YUV
with BT.709 and writes the tags explicitly. Audio is copied from the source.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


class InputError(RuntimeError):
    """The clip breaks the input assumption (VFR, unreadable, no video)."""


CONVERT_HINT = (
    "Convert it first to a constant frame rate, deinterlaced, at the project "
    "resolution. HandBrake does this well: Video tab, Constant Framerate."
)


@dataclass
class ClipInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frames_estimate: int
    duration: float
    start_time: float
    codec: str
    audio_streams: list[str] = field(default_factory=list)
    interlaced: bool = False

    @property
    def fps_float(self) -> float:
        return float(self.fps)


def probe(path: str | Path, scan_frames: int = 240) -> ClipInfo:
    import av

    path = Path(path)
    try:
        c = av.open(str(path))
    except Exception as e:
        raise InputError(f"cannot open {path.name}: {e}") from None
    with c:
        if not c.streams.video:
            raise InputError(f"{path.name} has no video stream")
        vs = c.streams.video[0]
        rate = vs.average_rate or vs.guessed_rate or vs.base_rate
        if not rate:
            raise InputError(f"{path.name}: no frame rate in the stream. {CONVERT_HINT}")
        tb = vs.time_base
        audio = [f"{a.index}: {a.codec_context.name} {a.channels}ch {a.rate} Hz"
                 for a in c.streams.audio]

        # Look at real timestamps instead of trusting the header.
        pts, interlaced = [], False
        vs.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(vs)):
            if fr.pts is not None:
                pts.append(fr.pts)
            interlaced |= bool(getattr(fr, "interlaced_frame", False))
            if i + 1 >= scan_frames:
                break
        if not pts:
            raise InputError(f"{path.name}: no decodable video frames")
        if len(pts) >= 3 and tb:
            d = np.diff(np.sort(np.asarray(pts, np.int64))).astype(np.float64) * float(tb)
            expect = 1.0 / float(rate)
            off = np.abs(d - expect) > 0.25 * expect
            if off.mean() > 0.02:
                raise InputError(
                    f"{path.name} has a variable frame rate "
                    f"({off.sum()} of the first {len(d)} frame gaps differ). {CONVERT_HINT}"
                )
        duration = float(c.duration / 1e6) if c.duration else (
            float(vs.duration * tb) if vs.duration and tb else 0.0)
        start = float(vs.start_time * tb) if vs.start_time is not None and tb else 0.0
        frames = vs.frames or int(round(duration * float(rate)))
        return ClipInfo(
            path=path, width=vs.codec_context.width, height=vs.codec_context.height,
            fps=Fraction(rate), frames_estimate=frames, duration=duration,
            start_time=start, codec=vs.codec_context.name, audio_streams=audio,
            interlaced=interlaced,
        )


def iter_gray(path: str | Path, size: tuple[int, int] | None = None,
              limit: int | None = None) -> Iterator[np.ndarray]:
    """Yield frames as float32 grey, gamma-encoded 0..1, HxW.

    size=(w, h) scales in the decoder with area averaging, which is faster
    than decoding full size and resizing in Python.
    """
    import av

    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        n = 0
        for fr in c.decode(vs):
            if size:
                fr = fr.reformat(width=size[0], height=size[1], format="gray16le",
                                 interpolation="AREA")
                arr = fr.to_ndarray()
            else:
                arr = fr.to_ndarray(format="gray16le")
            yield arr.astype(np.float32) / 65535.0
            n += 1
            if limit and n >= limit:
                return


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found in PATH")
    return exe


@dataclass
class EncodeSettings:
    rf: float = 18.0
    preset: str = "medium"
    tune_grain: bool = False
    audio: bool = True


class Encoder:
    """x265 10-bit at a constant RF. The full codec matrix is milestone 3."""

    def __init__(self, out: str | Path, info: ClipInfo, s: EncodeSettings,
                 frames: int | None = None):
        out = Path(out)
        w, h = info.width, info.height
        cmd = [
            ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostats", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
            "-framerate", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "-",
        ]
        map_audio = s.audio and info.audio_streams
        if map_audio:
            # shift audio so the first video frame lands at zero, same as ours
            cmd += ["-itsoffset", f"{-info.start_time:.6f}", "-i", str(info.path),
                    "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy"]
            if frames:
                cmd += ["-t", f"{frames / info.fps_float:.6f}"]
        x265 = "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:log-level=error"
        if s.tune_grain:
            x265 += ":tune=grain"
        cmd += [
            "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,"
                   "format=yuv420p10le",
            "-c:v", "libx265", "-crf", f"{s.rf:g}", "-preset", s.preset, "-x265-params", x265,
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-color_range", "tv",
        ]
        if out.suffix.lower() == ".mp4":
            cmd += ["-tag:v", "hvc1", "-movflags", "+faststart"]
        cmd.append(str(out))
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.count = 0

    def write(self, rgb16: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(rgb16, dtype="<u2").tobytes())
        except BrokenPipeError:
            raise RuntimeError(f"ffmpeg stopped: {self._err()}") from None
        self.count += 1

    def _err(self) -> str:
        try:
            return self.proc.stderr.read().decode(errors="replace").strip()
        except Exception:
            return ""

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {self._err()}")

    def abort(self) -> None:
        try:
            self.proc.kill()
        finally:
            self.proc.wait()
