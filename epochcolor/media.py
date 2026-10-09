"""Proxies, thumbnails, waveforms and frame-accurate reading.

Importing a clip decodes it once. That one pass writes a small H.264 proxy
for scrubbing, keeps a thumbnail strip for the timeline, and scores every
frame for shot detection. Audio gets a peak envelope per stream for the
waveform display. All of it lives in the cache and can be deleted.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .project import AudioInfo, Clip, new_id

PROXY_SHORT = 360
THUMB_W = 96
WAVE_RATE = 100  # envelope samples per second

Progress = Callable[[str, int, int], None]


def media_dir(path: str | Path) -> Path:
    """Where a clip's proxy, thumbnails and waveforms live, named after the
    source's content (sourceid.py). An entry under the old path-and-time
    name gets renamed the first time it's looked up."""
    from .sourceid import adopt, source_id
    from .video.pipeline import cache_root

    p = Path(path).resolve()
    d = cache_root() / "media" / hashlib.sha256(f"{source_id(p)}:m2".encode()).hexdigest()[:20]
    if not d.exists():
        st = p.stat()
        legacy = hashlib.sha256(f"{p}:{st.st_size}:{st.st_mtime_ns}:m1".encode()).hexdigest()[:20]
        adopt(cache_root() / "media" / legacy, d)
    return d


def _even(x: float) -> int:
    return max(16, int(round(x / 2)) * 2)


def proxy_size(w: int, h: int) -> tuple[int, int]:
    k = min(1.0, PROXY_SHORT / min(w, h))
    return _even(w * k), _even(h * k)


@dataclass
class MediaInfo:
    dir: Path
    frames: int
    proxy: Path
    proxy_size: tuple[int, int]
    thumbs: Path
    thumb_stride: int
    scores: list[float]
    waves: list[Path]

    @classmethod
    def load(cls, d: Path) -> "MediaInfo | None":
        m = d / "media.json"
        if not m.exists():
            return None
        j = json.loads(m.read_text())
        info = cls(d, j["frames"], d / "proxy.mp4", tuple(j["proxy_size"]), d / "thumbs.npy",
                   j["thumb_stride"], j["scores"], [d / w for w in j["waves"]])
        if not info.proxy.exists() or not info.thumbs.exists():
            return None
        return info


def import_clip(path: str | Path, progress: Progress | None = None,
                shot_threshold: float = 6.0) -> tuple[Clip, MediaInfo]:
    """Probe, build the proxy, detect shots. Reuses the cache when present."""
    from .video.io import probe
    from .video.temporal import detect_shots

    info = probe(path)
    d = media_dir(path)
    media = MediaInfo.load(d) or build_media(info, d, progress)
    shots = detect_shots(np.asarray(media.scores, np.float32), threshold=shot_threshold)
    clip = Clip(
        id=new_id(), path=str(Path(path).resolve()), width=info.width, height=info.height,
        fps_num=info.fps.numerator, fps_den=info.fps.denominator, frames=media.frames,
        start_time=info.start_time, codec=info.codec,
        audio=[AudioInfo(a.codec, a.channels, getattr(a, "layout", ""), a.title) for a in info.audio],
        shots=[[a, b] for a, b in shots],
    )
    return clip, media


def build_media(info, d: Path, progress: Progress | None = None) -> MediaInfo:
    import av

    from .color import srgb_to_l
    from .export.writer import ffmpeg_path
    from .video.temporal import cut_scores, thumb

    d.mkdir(parents=True, exist_ok=True)
    pw, ph = proxy_size(info.width, info.height)
    proxy = d / "proxy.mp4"
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{pw}x{ph}",
           "-framerate", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-tune", "fastdecode", "-crf", "20",
           "-g", "12", "-bf", "0", "-pix_fmt", "yuv420p", str(proxy)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    est = max(1, info.frames_estimate)
    stride = max(1, est // 6000)
    thumbs, small = [], []
    n = 0
    try:
        with av.open(str(info.path)) as c:
            vs = c.streams.video[0]
            vs.thread_type = "AUTO"
            for fr in c.decode(vs):
                g = fr.reformat(width=pw, height=ph, format="gray", interpolation="AREA").to_ndarray()
                proc.stdin.write(np.ascontiguousarray(g).tobytes())
                L = srgb_to_l(g.astype(np.float32) / 255.0)
                small.append(thumb(L))
                if n % stride == 0:
                    import cv2

                    tw = THUMB_W
                    th = max(1, round(ph * tw / pw))
                    thumbs.append(cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA))
                n += 1
                if progress and n % 10 == 0:
                    progress("proxy", n, max(n, est))
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0 or n == 0:
        raise RuntimeError(f"proxy build failed for {info.path.name}")
    scores = cut_scores(small)
    np.save(d / "thumbs.npy", np.stack(thumbs))
    waves = []
    for k in range(len(info.audio)):
        if progress:
            progress(f"waveform {k + 1}", k, len(info.audio))
        wp = d / f"wave{k}.npy"
        try:
            np.save(wp, audio_envelope(info.path, k))
            waves.append(wp.name)
        except RuntimeError:
            waves.append("")
    meta = {"frames": n, "proxy_size": [pw, ph], "thumb_stride": stride,
            "scores": [round(float(s), 3) for s in scores], "waves": waves}
    (d / "media.json").write_text(json.dumps(meta))
    if progress:
        progress("proxy", n, n)
    return MediaInfo.load(d)


def audio_envelope(path: Path, stream: int, rate: int = WAVE_RATE) -> np.ndarray:
    """Peak level per 1/rate second, mono, 0..1."""
    from .export.writer import ffmpeg_path

    sr = 8000
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
           "-map", f"0:a:{stream}", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-300:])
    x = np.abs(np.frombuffer(p.stdout, np.float32))
    hop = sr // rate
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, np.float16)
    return np.clip(x[: n * hop].reshape(n, hop).max(axis=1), 0, 1).astype(np.float16)


# ------------------------------------------------------------- reading


class FrameReader:
    """Frame-accurate random access into a video file, by frame number.

    Sequential reads decode straight on; a jump seeks to the keyframe before
    the target and decodes forward. Proxies use a short GOP so jumps stay
    cheap. Not thread-safe: one reader per thread.
    """

    def __init__(self, path: str | Path, fps: Fraction, fmt: str = "gray",
                 size: tuple[int, int] | None = None, cache: int = 48):
        import av

        self.c = av.open(str(path))
        self.vs = self.c.streams.video[0]
        self.vs.thread_type = "AUTO"
        self.tb = self.vs.time_base
        self.fps = Fraction(fps)
        self.t0 = float(self.vs.start_time * self.tb) if self.vs.start_time is not None else 0.0
        self.fmt, self.size = fmt, size
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_n = cache
        self._it = None
        self._next = None  # frame number the iterator will give next

    def close(self) -> None:
        self.c.close()

    def _index(self, fr) -> int:
        t = float(fr.pts * self.tb) if fr.pts is not None else 0.0
        return int(round((t - self.t0) * float(self.fps)))

    def _convert(self, fr) -> np.ndarray:
        if self.size:
            fr = fr.reformat(width=self.size[0], height=self.size[1], format=self.fmt,
                             interpolation="AREA")
            return fr.to_ndarray()
        return fr.to_ndarray(format=self.fmt)

    def _seek(self, idx: int) -> None:
        t = self.t0 + max(0, idx) / float(self.fps)
        self.c.seek(int(t / self.tb), stream=self.vs, backward=True, any_frame=False)
        self._it = self.c.decode(self.vs)
        self._next = None

    def get(self, idx: int) -> np.ndarray | None:
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if self._it is None or self._next is None or not (self._next <= idx <= self._next + 24):
            self._seek(idx)
        back = 0
        while True:
            try:
                fr = next(self._it)
            except StopIteration:
                return None
            n = self._index(fr)
            if self._next is None and n > idx:
                back += 1  # the seek landed past the target
                if back > 4:
                    return None
                self._seek(idx - 12 * back)
                continue
            self._next = n + 1
            if n >= idx - 2:  # keep the frames just before too, for reverse play
                arr = self._convert(fr)
                self.cache[n] = arr
                if len(self.cache) > self.cache_n:
                    self.cache.popitem(last=False)
            if n >= idx:
                return self.cache.get(idx, self._convert(fr) if n == idx else None)


def iter_range(path: str | Path, fps: Fraction, start: int, end: int,
               fmt: str = "gray16le") -> Iterator[np.ndarray]:
    """Frames [start, end) of a file, decoded in order after one seek."""
    r = FrameReader(path, fps, fmt, cache=2)
    try:
        for i in range(start, end):
            f = r.get(i)
            if f is None:
                raise RuntimeError(f"{Path(path).name}: frame {i} could not be decoded")
            yield f
    finally:
        r.close()
