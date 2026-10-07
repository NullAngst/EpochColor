"""Video in and out.

Decoding goes through PyAV, frame by frame, as 16-bit grey. Taking only the
luma plane sidesteps the YUV matrix question: a black and white source has
neutral chroma, so R=G=B=Y' whatever matrix it was tagged with.

Encoding lives in epochcolor.export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

from ..export.plan import AudioStream


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
    audio: list[AudioStream] = field(default_factory=list)
    interlaced: bool = False

    @property
    def audio_streams(self) -> list[str]:
        return [f"{a.index + 1}: {a.codec} {a.channels}ch" + (f" {a.title}" if a.title else "")
                for a in self.audio]

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
        audio = []
        for i, a in enumerate(c.streams.audio):
            md = dict(a.metadata or {})
            title = md.get("title") or md.get("language") or ""
            ch = getattr(a.codec_context, "channels", 0) or getattr(a, "channels", 0) or 2
            try:
                layout = a.codec_context.layout.name
            except AttributeError:
                layout = ""
            audio.append(AudioStream(i, a.codec_context.name, int(ch), title, layout))

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
            start_time=start, codec=vs.codec_context.name, audio=audio,
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
