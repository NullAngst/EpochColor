"""The video pipeline, milestone 2.

Four passes, each streaming frames so memory stays flat however long the
clip is. Working-size data lives in float16 memmaps in the cache folder.

1. Decode at working size -> L, shot detection scores
2. Temporal denoise -> the model's copy
3. Model, shot by shot -> raw chroma (cached, so re-renders skip it)
4. Chroma stabilizer per shot, then the full-size render: original luma,
   chroma scaled up along its edges, piped to the encoder

Propagation and the stabilizer never cross a shot boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..color import lab_to_srgb, srgb_to_l
from ..denoise import denoise_l, estimate_noise
from ..imageio import quantize
from ..models import ColorModel
from ..pipeline import guided_upsample
from ..export.plan import ExportPlan
from ..export.writer import VideoWriter
from .io import ClipInfo, iter_gray
from .temporal import Flow, cut_scores, detect_shots, stabilize_chroma, temporal_denoise, thumb


@dataclass
class VideoSettings:
    working_size: int = 512
    grain: float = 100.0
    denoise: float | None = None
    stabilize: float = 0.9  # 0 = off, 0.9 = average up to 10 frames each way
    shot_threshold: float = 6.0
    guided: bool = True
    saturation: float = 1.0
    frames: int | None = None  # process only the first N frames
    flow_preset: str = "medium"


@dataclass
class VideoReport:
    frames: int = 0
    working: tuple[int, int] = (0, 0)
    shots: list[tuple[int, int]] = field(default_factory=list)
    noise_sigma: float = 0.0
    cached_shots: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def cache_root() -> Path:
    env = os.environ.get("EPOCHCOLOR_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "EpochColor" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "epochcolor"


def cache_key(info: ClipInfo, s: VideoSettings, model: ColorModel) -> str:
    st = info.path.stat()
    ident = {
        "src": str(info.path.resolve()), "size": st.st_size, "mtime": st.st_mtime_ns,
        "ws": s.working_size, "dn": s.denoise, "frames": s.frames,
        "model": model.info.name, "weights": str(getattr(model, "weights", "") or ""),
        "v": 1,
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:20]


class _Progress:
    def __init__(self, label: str, total: int, quiet: bool):
        self.label, self.total, self.quiet = label, max(1, total), quiet
        self.t0 = time.perf_counter()
        self.last = 0.0

    def __call__(self, done: int) -> None:
        if self.quiet:
            return
        now = time.perf_counter()
        if now - self.last < 0.5 and done < self.total:
            return
        self.last = now
        rate = done / max(now - self.t0, 1e-6)
        print(f"\r  {self.label}: {done}/{self.total} ({rate:.1f} fps)   ", end="", file=sys.stderr)

    def done(self) -> float:
        el = time.perf_counter() - self.t0
        if not self.quiet:
            print(f"\r  {self.label}: {self.total}/{self.total} in {el:.1f}s" + " " * 12,
                  file=sys.stderr)
        return round(el, 2)


def _memmap(path: Path, shape, mode="w+"):
    return np.lib.format.open_memmap(str(path), mode=mode, dtype=np.float16, shape=shape)


def colorize_video(info: ClipInfo, model: ColorModel, out: str | Path, plan: ExportPlan,
                   s: VideoSettings | None = None, use_cache: bool = True,
                   quiet: bool = False) -> VideoReport:
    s = s or VideoSettings()
    rep = VideoReport()
    W, H = info.width, info.height
    k = min(1.0, s.working_size / min(W, H))
    # even sizes keep the decoder's scaler happy
    ww, wh = max(16, round(W * k / 2) * 2), max(16, round(H * k / 2) * 2)
    rep.working = (ww, wh)
    if info.interlaced:
        rep.warnings.append("frames are flagged interlaced; deinterlace first if you see combing")

    cdir = cache_root() / cache_key(info, s, model)
    cdir.mkdir(parents=True, exist_ok=True)
    meta_path = cdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if (use_cache and meta_path.exists()) else {}
    flow = Flow(s.flow_preset)

    # ---------------------------------------------- 1. decode at working size
    if meta.get("decoded") and (cdir / "L.npy").exists():
        n = meta["frames"]
        L = np.load(cdir / "L.npy", mmap_mode="r")
        scores = np.asarray(meta["scores"], np.float32)
    else:
        raw = cdir / "L.raw"
        thumbs = []
        prog = _Progress("decode", s.frames or info.frames_estimate, quiet)
        n = 0
        with open(raw, "wb") as f:
            for g in iter_gray(info.path, (ww, wh), s.frames):
                Lf = srgb_to_l(g)
                thumbs.append(thumb(Lf))
                f.write(Lf.astype(np.float16).tobytes())
                n += 1
                prog(n)
        prog.total = n
        rep.timings["decode"] = prog.done()
        if n == 0:
            raise RuntimeError("no frames decoded")
        L = _memmap(cdir / "L.npy", (n, wh, ww))
        L[:] = np.fromfile(raw, np.float16).reshape(n, wh, ww)
        L.flush()
        raw.unlink()
        scores = cut_scores(thumbs)
        meta = {"frames": n, "decoded": True, "scores": scores.tolist(), "model_done": []}
        meta_path.write_text(json.dumps(meta))
        L = np.load(cdir / "L.npy", mmap_mode="r")
    rep.frames = n
    shots = detect_shots(scores, threshold=s.shot_threshold)
    rep.shots = shots
    sigma = estimate_noise(np.asarray(L[min(n - 1, n // 2)], np.float32))
    rep.noise_sigma = round(sigma, 3)
    if not quiet:
        print(f"  {n} frames, {len(shots)} shot(s), chroma at {ww}x{wh}, "
              f"grain sigma {sigma:.2f} L*", file=sys.stderr)

    # ------------------------------------------- 2+3. denoise and model, per shot
    ab_raw_path = cdir / "ab_raw.npy"
    ldn_path = cdir / "Ldn.npy"
    done = set(tuple(x) for x in meta.get("model_done", []))
    if ab_raw_path.exists() and ldn_path.exists():
        ab_raw = np.load(ab_raw_path, mmap_mode="r+")
        Ldn = np.load(ldn_path, mmap_mode="r+")
    else:
        done = set()
        ab_raw = _memmap(ab_raw_path, (n, wh, ww, 2))
        Ldn = _memmap(ldn_path, (n, wh, ww))
    todo = [sh for sh in shots if tuple(sh) not in done]
    rep.cached_shots = len(shots) - len(todo)
    total = sum(b - a for a, b in todo)
    prog = _Progress("model", total, quiet)
    count = 0
    for a, b in todo:
        for t in range(a, b):
            cur = np.asarray(L[t], np.float32)
            prev = np.asarray(L[t - 1], np.float32) if t > a else None
            nxt = np.asarray(L[t + 1], np.float32) if t + 1 < b else None
            if s.denoise == 0:
                dn = cur
            else:
                dn = temporal_denoise(prev, cur, nxt, flow, sigma)
                # light spatial pass for whatever the neighbours did not cover
                dn = denoise_l(dn, None if s.denoise is None else s.denoise * 0.5)
            Ldn[t] = dn
            ab_raw[t] = model.predict(dn, None)
            count += 1
            prog(count)
        ab_raw.flush()
        Ldn.flush()
        done.add((a, b))
        meta["model_done"] = sorted(list(x) for x in done)
        meta_path.write_text(json.dumps(meta))  # resume point after each shot
    rep.timings["model"] = prog.done() if todo else 0.0

    # ------------------------------------------------- 4a. stabilize chroma
    ab = _memmap(cdir / "ab.npy", (n, wh, ww, 2))
    prog = _Progress("stabilize", n, quiet)
    count = 0
    for a, b in shots:
        scratch = np.empty((b - a, wh, ww, 2), np.float32) if (b - a) * wh * ww * 8 < 2e9 \
            else _memmap(cdir / "fwd.npy", (b - a, wh, ww, 2))
        stabilize_chroma(_Slice(Ldn, a, b), _Slice(ab_raw, a, b), _Slice(ab, a, b),
                         strength=s.stabilize, tol=max(2.0, 3.0 * sigma), flow=flow,
                         scratch=scratch)
        count += b - a
        prog(count)
    ab.flush()
    rep.timings["stabilize"] = prog.done()

    # ------------------------------------------------------ 4b. render
    eps = None
    g = float(np.clip(s.grain, 0.0, 100.0)) / 100.0
    encoder = VideoWriter(plan, out, W, H, info.fps, info.path, info.start_time,
                          cdir / "export", frames_limit=n if s.frames else None)
    prog = _Progress("render", n, quiet)
    t = 0
    try:
        for gray in iter_gray(info.path, None, n):
            Lfull = srgb_to_l(gray)
            if eps is None:
                eps = float(np.clip((2.0 * estimate_noise(Lfull)) ** 2, 1.0, 100.0))
            abw = np.asarray(ab[t], np.float32)
            if s.saturation != 1.0:
                abw = abw * float(s.saturation)
            if s.guided:
                abf = guided_upsample(abw, Lfull, eps)
            else:
                import cv2

                abf = cv2.resize(abw, (W, H), interpolation=cv2.INTER_CUBIC)
            Lout = Lfull if g >= 1.0 else g * Lfull + (1 - g) * denoise_l(Lfull, s.denoise)
            encoder.write(quantize(lab_to_srgb(Lout, abf), 16))
            t += 1
            prog(t)
        rep.timings["render"] = prog.done()
        if plan.two_pass:
            tp = time.perf_counter()
            encoder.close(on_pass=lambda k: quiet or print(f"  encode pass {k} of 2", file=sys.stderr))
            rep.timings["two-pass"] = round(time.perf_counter() - tp, 2)
        else:
            encoder.close()
    except BaseException:
        encoder.abort()
        raise
    if t != n:
        raise RuntimeError(f"render decoded {t} frames, analysis saw {n}")
    return rep


class _Slice:
    """A view of frames [a, b) that reads and writes through to a memmap."""

    def __init__(self, arr, a: int, b: int):
        self.arr, self.a, self.b = arr, a, b

    def __len__(self):
        return self.b - self.a

    def __getitem__(self, i):
        return np.asarray(self.arr[self.a + i], np.float32)

    def __setitem__(self, i, v):
        self.arr[self.a + i] = v


def clear_cache() -> int:
    root = cache_root()
    if not root.exists():
        return 0
    n = sum(1 for _ in root.iterdir())
    shutil.rmtree(root)
    return n
