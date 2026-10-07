"""The video pipeline.

Analysis, once per clip, streaming so memory stays flat however long the
clip is. Working-size data lives in float16 memmaps in the cache folder.

1. Decode at working size -> L, shot detection scores
2. Temporal denoise -> the model's copy
3. Model, shot by shot -> raw chroma (resumable, re-renders skip it)
4. Chroma stabilizer per shot -> ab.npy, the chroma every render reads

Render, as often as you like: the timeline's pieces in order, each frame
decoded at full size, its original luma recombined with the cached chroma
scaled up along its edges, piped to the encoder.

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
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

from ..color import lab_to_srgb, srgb_to_l
from ..denoise import denoise_l, estimate_noise
from ..export.plan import ExportPlan
from ..export.writer import AudioTimeline, VideoWriter
from ..imageio import quantize
from ..models import ColorModel
from ..pipeline import guided_upsample
from .io import ClipInfo, iter_gray
from .temporal import Flow, cut_scores, detect_shots, stabilize_chroma, temporal_denoise, thumb

ProgressFn = Callable[[str, int, int], None]


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
    chroma_size: int = 256  # short side of the stored chroma

    @classmethod
    def from_project(cls, d: dict) -> "VideoSettings":
        keys = {"working_size", "grain", "denoise", "stabilize", "shot_threshold", "saturation",
                "chroma_size"}
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class Analysis:
    dir: Path
    frames: int = 0
    working: tuple[int, int] = (0, 0)
    shots: list[tuple[int, int]] = field(default_factory=list)
    noise_sigma: float = 0.0
    cached_shots: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ab_path(self) -> Path:
        return self.dir / "ab.npy"


VideoReport = Analysis  # the CLI's report is the analysis plus render timings


def cache_root() -> Path:
    env = os.environ.get("EPOCHCOLOR_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "EpochColor" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "epochcolor"


def cache_key(path: Path, s: VideoSettings, model_name: str, weights: str | None) -> str:
    st = path.stat()
    ident = {
        "src": str(path.resolve()), "size": st.st_size, "mtime": st.st_mtime_ns,
        "ws": s.working_size, "dn": s.denoise, "frames": s.frames,
        "model": model_name, "weights": str(weights or ""), "cs": s.chroma_size, "v": 2,
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:20]


def analysis_dir(path: Path, s: VideoSettings, model_name: str, weights: str | None = None) -> Path:
    return cache_root() / "clips" / cache_key(Path(path), s, model_name, weights)


def finished_analysis(path: Path, s: VideoSettings, model_name: str,
                      weights: str | None = None) -> Path | None:
    """The cache folder if this clip is fully analysed with these settings."""
    d = analysis_dir(path, s, model_name, weights)
    m = d / "meta.json"
    if not m.exists() or not (d / "ab.npy").exists():
        return None
    try:
        meta = json.loads(m.read_text())
    except ValueError:
        return None
    return d if meta.get("stab") is not None else None


class _Progress:
    def __init__(self, label: str, total: int, quiet: bool, cb: ProgressFn | None = None):
        self.label, self.total, self.quiet, self.cb = label, max(1, total), quiet, cb
        self.t0 = time.perf_counter()
        self.last = 0.0

    def __call__(self, done: int) -> None:
        now = time.perf_counter()
        if now - self.last < 0.25 and done < self.total:
            return
        self.last = now
        if self.cb:
            self.cb(self.label, done, self.total)
        if self.quiet:
            return
        rate = done / max(now - self.t0, 1e-6)
        print(f"\r  {self.label}: {done}/{self.total} ({rate:.1f} fps)   ", end="", file=sys.stderr)

    def done(self) -> float:
        el = time.perf_counter() - self.t0
        if self.cb:
            self.cb(self.label, self.total, self.total)
        if not self.quiet:
            print(f"\r  {self.label}: {self.total}/{self.total} in {el:.1f}s" + " " * 12,
                  file=sys.stderr)
        return round(el, 2)


def _memmap(path: Path, shape, dtype=np.float16, mode="w+"):
    return np.lib.format.open_memmap(str(path), mode=mode, dtype=dtype, shape=shape)


def working_size(w: int, h: int, short: int) -> tuple[int, int]:
    k = min(1.0, short / min(w, h))
    # even sizes keep the decoder's scaler happy
    return max(16, round(w * k / 2) * 2), max(16, round(h * k / 2) * 2)


# ------------------------------------------------------------- analysis


def analyze(info: ClipInfo, model: ColorModel, s: VideoSettings | None = None,
            shots: list[tuple[int, int]] | None = None, use_cache: bool = True,
            quiet: bool = False, progress: ProgressFn | None = None) -> Analysis:
    """Run passes 1 to 4 on a clip. shots, when given, replace detection
    (the GUI passes the shots the user sees and may have edited).

    What stays on disk is kept small, since a feature has ~130,000 frames:
    chroma at chroma_size (256 px short side by default, which is the
    models' own output size and already softer than 4:2:0 video chroma)
    stored as int8, plus the denoised luma at the same size for the
    stabilizer's motion. The working-size luma only lives until the model
    pass is done.
    """
    s = s or VideoSettings()
    W, H = info.width, info.height
    ww, wh = working_size(W, H, s.working_size)
    cw, ch = working_size(W, H, min(s.working_size, s.chroma_size))
    cdir = analysis_dir(info.path, s, model.info.name, getattr(model, "weights", None))
    cdir.mkdir(parents=True, exist_ok=True)
    rep = Analysis(cdir, working=(ww, wh))
    if info.interlaced:
        rep.warnings.append("frames are flagged interlaced; deinterlace first if you see combing")
    meta_path = cdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if (use_cache and meta_path.exists()) else {}
    flow = Flow(s.flow_preset)
    L_path = cdir / "L.npy"

    def decode() -> int:
        raw = cdir / "L.raw"
        thumbs = []
        prog = _Progress("decode", s.frames or info.frames_estimate, quiet, progress)
        n = 0
        with open(raw, "wb") as f:
            for g in iter_gray(info.path, (ww, wh), s.frames):
                Lf = srgb_to_l(g)
                thumbs.append(thumb(Lf))
                f.write(Lf.astype(np.float16).tobytes())
                n += 1
                prog(n)
        prog.total = max(1, n)
        rep.timings["decode"] = prog.done()
        if n == 0:
            raise RuntimeError("no frames decoded")
        Lm = _memmap(L_path, (n, wh, ww))
        Lm[:] = np.fromfile(raw, np.float16).reshape(n, wh, ww)
        Lm.flush()
        del Lm
        raw.unlink()
        if "scores" not in meta:
            meta.update({"frames": n, "scores": cut_scores(thumbs).tolist(), "model_done": []})
        meta["decoded"] = True
        meta_path.write_text(json.dumps(meta))
        return n

    # ---------------------------------------------- 1. decode at working size
    if not meta.get("scores"):
        decode()
    n = meta["frames"]
    scores = np.asarray(meta["scores"], np.float32)
    rep.frames = n
    if shots:
        shots = [(max(0, int(a)), min(n, int(b))) for a, b in shots if min(n, b) > a]
    else:
        shots = detect_shots(scores, threshold=s.shot_threshold)
    rep.shots = shots

    # ------------------------------------------- 2+3. denoise and model, per shot
    ab_raw_path = cdir / "ab_raw.npy"
    ldn_path = cdir / "Ldn.npy"
    done = set(tuple(x) for x in meta.get("model_done", []))
    if ab_raw_path.exists() and ldn_path.exists() and meta.get("model_done") is not None:
        ab_raw = np.load(ab_raw_path, mmap_mode="r+")
        Ldn = np.load(ldn_path, mmap_mode="r+")
    else:
        done = set()
        ab_raw = _memmap(ab_raw_path, (n, ch, cw, 2), np.int8)
        Ldn = _memmap(ldn_path, (n, ch, cw))
    todo = [sh for sh in shots if tuple(sh) not in done]
    rep.cached_shots = len(shots) - len(todo)
    if todo and not L_path.exists():
        decode()  # the working-size luma was dropped after an earlier pass
    sigma = float(meta.get("sigma", 0.0))
    if todo:
        L = np.load(L_path, mmap_mode="r")
        sigma = estimate_noise(np.asarray(L[min(n - 1, n // 2)], np.float32))
        meta["sigma"] = sigma
        meta["stab"] = None  # chroma is about to change
        meta_path.write_text(json.dumps(meta))
    rep.noise_sigma = round(sigma, 3)
    if not quiet:
        print(f"  {n} frames, {len(shots)} shot(s), model input {ww}x{wh}, chroma {cw}x{ch}, "
              f"grain sigma {sigma:.2f} L*", file=sys.stderr)
    import cv2

    total = sum(b - a for a, b in todo)
    prog = _Progress("model", total, quiet, progress)
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
            ab_w = model.predict(dn, None)
            Ldn[t] = cv2.resize(dn, (cw, ch), interpolation=cv2.INTER_AREA)
            ab_raw[t] = _to_i8(cv2.resize(ab_w, (cw, ch), interpolation=cv2.INTER_AREA))
            count += 1
            prog(count)
        ab_raw.flush()
        Ldn.flush()
        done.add((a, b))
        meta["model_done"] = sorted(list(x) for x in done)
        meta_path.write_text(json.dumps(meta))  # resume point after each shot
    rep.timings["model"] = prog.done() if todo else 0.0
    if todo:
        del L
    if all(tuple(sh) in done for sh in shots) and L_path.exists():
        L_path.unlink()  # biggest file; only needed again if shots change

    # ------------------------------------------------- 4. stabilize chroma
    stab_id = {"strength": s.stabilize, "shots": [list(x) for x in shots]}
    ab_path = cdir / "ab.npy"
    if meta.get("stab") == stab_id and ab_path.exists():
        return rep
    ab = _memmap(ab_path, (n, ch, cw, 2), np.int8)
    prog = _Progress("stabilize", n, quiet, progress)
    count = 0
    for a, b in shots:
        if (b - a) * ch * cw * 2 * 4 < 5e8:
            scratch = np.empty((b - a, ch, cw, 2), np.float32)
            out = np.empty((b - a, ch, cw, 2), np.float32)
        else:  # a very long shot: keep the working arrays on disk, not in RAM
            scratch = _memmap(cdir / "fwd.npy", (b - a, ch, cw, 2))
            out = _memmap(cdir / "out.npy", (b - a, ch, cw, 2))
        stabilize_chroma(_Slice(Ldn, a, b), _Slice(ab_raw, a, b), out,
                         strength=s.stabilize, tol=max(2.0, 3.0 * sigma), flow=flow,
                         scratch=scratch)
        ab[a:b] = _to_i8(out)
        count += b - a
        prog(count)
    ab.flush()
    del ab, scratch, out
    for tmp in ("fwd.npy", "out.npy"):
        (cdir / tmp).unlink(missing_ok=True)
    rep.timings["stabilize"] = prog.done()
    meta["stab"] = stab_id
    meta_path.write_text(json.dumps(meta))
    return rep


def _to_i8(ab: np.ndarray) -> np.ndarray:
    """a/b to int8. One unit of a/b is about at the threshold of seeing a
    difference, and the chroma gets interpolated on the way up anyway."""
    return np.clip(np.rint(ab), -127, 127).astype(np.int8)


# --------------------------------------------------------------- render


@dataclass
class Piece:
    """Frames [src_in, src_out) of one analysed clip."""
    path: Path
    fps: Fraction
    width: int
    height: int
    analysis: Path
    src_in: int
    src_out: int

    @property
    def length(self) -> int:
        return self.src_out - self.src_in


class FrameRenderer:
    """Full-size colour for one frame: original luma plus cached chroma."""

    def __init__(self, s: VideoSettings):
        self.s = s
        self.eps: float | None = None
        self.g = float(np.clip(s.grain, 0.0, 100.0)) / 100.0

    def __call__(self, gray: np.ndarray, ab_small: np.ndarray) -> np.ndarray:
        """gray: float 0..1 HxW. Returns sRGB float HxWx3."""
        Lfull = srgb_to_l(gray)
        if self.eps is None:
            self.eps = float(np.clip((2.0 * estimate_noise(Lfull)) ** 2, 1.0, 100.0))
        ab = np.asarray(ab_small, np.float32)
        if self.s.saturation != 1.0:
            ab = ab * float(self.s.saturation)
        if self.s.guided:
            abf = guided_upsample(ab, Lfull, self.eps)
        else:
            import cv2

            abf = cv2.resize(ab, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
        Lout = Lfull if self.g >= 1.0 else self.g * Lfull + (1 - self.g) * denoise_l(Lfull, self.s.denoise)
        return lab_to_srgb(Lout, abf)


def render(pieces: list[Piece], plan: ExportPlan, out: str | Path, s: VideoSettings,
           audio_source: tuple[Path, float] | None = None,
           audio_timeline: AudioTimeline | None = None, frames_limit: int | None = None,
           quiet: bool = False, progress: ProgressFn | None = None) -> dict[str, float]:
    """Render pieces in order into one file. audio_source passes a single
    untouched clip's audio through; audio_timeline cuts and joins tracks."""
    from ..media import iter_range

    if not pieces:
        raise RuntimeError("nothing to render")
    W, H, fps = pieces[0].width, pieces[0].height, pieces[0].fps
    total = sum(p.length for p in pieces)
    workdir = cache_root() / "export" / hashlib.sha256(str(out).encode()).hexdigest()[:12]
    src, start = audio_source if audio_source else (None, 0.0)
    writer = VideoWriter(plan, out, W, H, fps, src, start, workdir,
                         frames_limit=frames_limit, audio_timeline=audio_timeline)
    rend = FrameRenderer(s)
    timings = {}
    prog = _Progress("render", total, quiet, progress)
    done = 0
    try:
        for p in pieces:
            ab = np.load(p.analysis / "ab.npy", mmap_mode="r")
            for i, g16 in enumerate(iter_range(p.path, p.fps, p.src_in, p.src_out)):
                gray = g16.astype(np.float32) / 65535.0
                writer.write(quantize(rend(gray, ab[p.src_in + i]), 16))
                done += 1
                prog(done)
            del ab
        timings["render"] = prog.done()
        if plan.two_pass:
            tp = time.perf_counter()

            def on_pass(k):
                if progress:
                    progress(f"encode pass {k} of 2", 0, 1)
                if not quiet:
                    print(f"  encode pass {k} of 2", file=sys.stderr)

            writer.close(on_pass=on_pass)
            timings["two-pass"] = round(time.perf_counter() - tp, 2)
        else:
            writer.close()
    except BaseException:
        writer.abort()
        raise
    return timings


def colorize_video(info: ClipInfo, model: ColorModel, out: str | Path, plan: ExportPlan,
                   s: VideoSettings | None = None, use_cache: bool = True,
                   quiet: bool = False) -> Analysis:
    """Analyse one clip and render all of it: what the CLI does."""
    s = s or VideoSettings()
    rep = analyze(info, model, s, use_cache=use_cache, quiet=quiet)
    piece = Piece(info.path, info.fps, info.width, info.height, rep.dir, 0, rep.frames)
    rep.timings.update(render([piece], plan, out, s, audio_source=(info.path, info.start_time),
                              frames_limit=rep.frames if s.frames else None, quiet=quiet))
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
