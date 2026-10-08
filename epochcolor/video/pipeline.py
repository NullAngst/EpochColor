"""The video pipeline.

Analysis, once per clip, streaming so memory and disk stay flat however
long the clip is: nothing at working size is ever written out.

1. Decode at working size -> shot detection scores (and nothing else kept)
2+3. Decode again, shot by shot, temporal denoise -> model -> raw chroma
   at chroma size (resumable, re-renders skip it)
4. Chroma stabilizer per shot -> ab.npy, the chroma every render reads

Everything lives in the working folder (cache_root), which the user can
point at another disk.

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
from .temporal import Flow, detect_shots, stabilize_chroma, temporal_denoise, thumb

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
    cast: float = 0.0  # 0..1, how much of the measured colour cast to take out (chroma.py)
    frames: int | None = None  # process only the first N frames
    flow_preset: str = "medium"
    chroma_size: int = 256  # short side of the stored chroma

    @classmethod
    def from_project(cls, d: dict) -> "VideoSettings":
        keys = {"working_size", "grain", "denoise", "stabilize", "shot_threshold", "saturation",
                "chroma_size", "cast"}
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


CACHE_PARTS = ("clips", "photos", "media", "export", "tmp", "match")  # all cache_root ever holds


def cache_root() -> Path:
    """The working folder: EPOCHCOLOR_CACHE, else the "cache_dir" setting,
    else ~/.cache/epochcolor."""
    env = os.environ.get("EPOCHCOLOR_CACHE")
    if env:
        return Path(env).expanduser()
    from ..models.weights import read_settings

    chosen = read_settings().get("cache_dir")
    if chosen:
        return Path(chosen).expanduser()
    return default_cache_root()


def set_cache_root(path: str | Path | None) -> Path:
    """Use another working folder (None: back to the default). Nothing is
    moved: the cache is rebuilt as needed, and moving it between disks can
    take longer than that."""
    from ..models.weights import write_setting

    if path:
        p = Path(path).expanduser().absolute()
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".epochcolor-write-test"
        probe.write_text("ok")
        probe.unlink()
        write_setting("cache_dir", str(p))
        return p
    write_setting("cache_dir", None)
    return default_cache_root()


def cache_size(root: Path | None = None) -> int:
    root = root or cache_root()
    total = 0
    for part in CACHE_PARTS:
        d = root / part
        if d.exists():
            total += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return total


def sweep_stale(root: Path | None = None) -> int:
    """Delete working-size luma left by versions before 0.5.2 (L.raw, L.npy):
    the biggest files by far, and never read any more."""
    root = root or cache_root()
    freed = 0
    for pat in ("clips/*/L.raw", "clips/*/L.npy"):
        for f in root.glob(pat):
            try:
                freed += f.stat().st_size
                f.unlink()
            except OSError:
                pass
    return freed


def default_cache_root() -> Path:
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
    """weights: the model's cache id; None looks up the installed model's."""
    if weights is None:
        from ..models import cache_id_for

        weights = cache_id_for(model_name)
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
    m = np.lib.format.open_memmap(str(path), mode=mode, dtype=dtype, shape=shape)
    if mode == "w+":
        # reserve the space now: writing into a sparse memmap on a full disk
        # kills the process with SIGBUS, reserving fails with a plain ENOSPC
        try:
            with open(path, "r+b") as f:
                os.posix_fallocate(f.fileno(), 0, os.path.getsize(path))
        except AttributeError:
            pass  # not on this platform
        except OSError as e:
            del m
            path.unlink(missing_ok=True)
            from ..diskspace import explain, is_full

            if is_full(e):
                from ..diskspace import DiskFull

                raise DiskFull(explain(e, path), path) from None
            raise
    return m


def estimate_cache_bytes(frames: int, chroma: tuple[int, int]) -> int:
    """What a clip's analysis keeps on disk: raw chroma, stabilized chroma
    (int8 a/b each) and the small denoised luma (float16)."""
    cw, ch = chroma
    return int(frames * cw * ch * (2 + 2 + 2) * 1.05)


def working_size(w: int, h: int, short: int) -> tuple[int, int]:
    k = min(1.0, short / min(w, h))
    # even sizes keep the decoder's scaler happy
    return max(16, round(w * k / 2) * 2), max(16, round(h * k / 2) * 2)


# ------------------------------------------------------------- analysis


def wanted_stab(shots, strength: float, hints: dict | None) -> dict[str, str]:
    """What each shot's stabilized chroma has to have been made from: the
    stabilizer strength and the hints painted in that shot. A shot whose id
    changes is redone; the rest stay as they are."""
    from ..hintpaint import hints_in, strokes_key

    out = {}
    for a, b in shots:
        h = hints_in(hints, a, b)
        out[f"{a}-{b}"] = f"{strength}:{strokes_key(h) if h else '-'}"
    return out


def analyze(info: ClipInfo, model: ColorModel, s: VideoSettings | None = None,
            shots: list[tuple[int, int]] | None = None, use_cache: bool = True,
            quiet: bool = False, progress: ProgressFn | None = None,
            hints: dict | None = None) -> Analysis:
    """Run passes 1 to 4 on a clip. shots, when given, replace detection
    (the GUI passes the shots the user sees and may have edited). hints:
    painted strokes, {clip frame: [stroke, ...]}.

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
    cdir = analysis_dir(info.path, s, model.info.name, getattr(model, "cache_id", None))
    cdir.mkdir(parents=True, exist_ok=True)
    rep = Analysis(cdir, working=(ww, wh))
    if info.interlaced:
        rep.warnings.append("frames are flagged interlaced; deinterlace first if you see combing")
    meta_path = cdir / "meta.json"
    meta = json.loads(meta_path.read_text()) if (use_cache and meta_path.exists()) else {}
    flow = Flow(s.flow_preset)
    for stale in ("L.raw", "L.npy"):  # from versions that kept working-size luma
        (cdir / stale).unlink(missing_ok=True)

    def decode() -> int:
        """Pass 1: only the cut scores and the noise sample are kept."""
        from .temporal import cut_score

        prog = _Progress("decode", s.frames or info.frames_estimate, quiet, progress)
        mid = max(0, (s.frames or info.frames_estimate) // 2)
        scores, prev_t, sample = [], None, None
        n = 0
        for g in iter_gray(info.path, (ww, wh), s.frames):
            Lf = srgb_to_l(g)
            t = thumb(Lf)
            scores.append(0.0 if prev_t is None else cut_score(prev_t, t))
            prev_t = t
            if n <= mid:
                sample = Lf  # ends up as the middle frame, or the last if the clip is shorter
            n += 1
            prog(n)
        prog.total = max(1, n)
        rep.timings["decode"] = prog.done()
        if n == 0:
            raise RuntimeError("no frames decoded")
        meta.update({"frames": n, "scores": scores, "model_done": [],
                     "sigma": float(estimate_noise(sample)), "decoded": True})
        meta_path.write_text(json.dumps(meta))
        return n

    # ---------------------------------------------- 1. decode at working size
    if not meta.get("scores") or "sigma" not in meta:
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
        from ..diskspace import need

        need(cdir, estimate_cache_bytes(n, (cw, ch)), f"Colorizing {Path(info.path).name} ({n} frames)")
        done = set()
        ab_raw = _memmap(ab_raw_path, (n, ch, cw, 2), np.int8)
        Ldn = _memmap(ldn_path, (n, ch, cw))
    todo = sorted(sh for sh in shots if tuple(sh) not in done)
    rep.cached_shots = len(shots) - len(todo)
    sigma = float(meta.get("sigma", 0.0))
    if todo:
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
    frames = _LumaStream(info, (ww, wh)) if todo else None
    try:
        for a, b in todo:
            cur = frames.get(a)
            prev = None
            for t in range(a, b):
                nxt = frames.get(t + 1) if t + 1 < b else None
                if s.denoise == 0:
                    dn = cur
                else:
                    dn = temporal_denoise(prev, cur, nxt, flow, sigma)
                    # light spatial pass for whatever the neighbours did not cover
                    dn = denoise_l(dn, None if s.denoise is None else s.denoise * 0.5)
                ab_w = model.predict(dn, None)
                Ldn[t] = cv2.resize(dn, (cw, ch), interpolation=cv2.INTER_AREA)
                ab_raw[t] = _to_i8(cv2.resize(ab_w, (cw, ch), interpolation=cv2.INTER_AREA))
                prev, cur = cur, nxt
                count += 1
                prog(count)
            ab_raw.flush()
            Ldn.flush()
            done.add((a, b))
            meta["model_done"] = sorted(list(x) for x in done)
            meta_path.write_text(json.dumps(meta))  # resume point after each shot
    finally:
        if frames is not None:
            frames.close()
    rep.timings["model"] = prog.done() if todo else 0.0

    # ------------------------------------- 4. hints, then stabilize chroma
    want = wanted_stab(shots, s.stabilize, hints)
    have = meta.get("stab") if isinstance(meta.get("stab"), dict) else {}
    ab_path = cdir / "ab.npy"
    redo = [(a, b) for a, b in shots if have.get(f"{a}-{b}") != want[f"{a}-{b}"] or not ab_path.exists()]
    if not redo:
        return rep
    if ab_path.exists():
        ab = np.load(ab_path, mmap_mode="r+")
        if ab.shape != (n, ch, cw, 2):
            del ab
            ab = _memmap(ab_path, (n, ch, cw, 2), np.int8)
            redo = list(shots)
    else:
        ab = _memmap(ab_path, (n, ch, cw, 2), np.int8)
        redo = list(shots)
    from ..hintpaint import group_by_reach, hints_in, strokes_key
    from .hints_pass import carry, keyframe_fix

    tol = max(2.0, 3.0 * sigma)
    total = sum(b - a for a, b in redo)
    prog = _Progress("stabilize", total, quiet, progress)
    count = 0
    scratch = out = None
    for a, b in redo:
        src = _Slice(ab_raw, a, b)
        shot_hints = hints_in(hints, a, b)
        if shot_hints:
            key = strokes_key(shot_hints)
            hpath = cdir / f"hint_{a}_{b}_{key}.npy"
            ppath = cdir / f"protect_{a}_{b}_{key}.npy"  # where frame-limited fixes are, see carry()
            for old in list(cdir.glob(f"hint_{a}_{b}_*.npy")) + list(cdir.glob(f"protect_{a}_{b}_*.npy")):
                if old not in (hpath, ppath):
                    old.unlink()
            if not hpath.exists():
                if progress:
                    progress("hints", count, max(1, total))
                keys = []
                used = set()
                for f, strokes in sorted(shot_hints.items()):
                    for reach, group in group_by_reach(strokes).items():
                        # each painted frame's fix is cached on its own, so painting
                        # frame after frame doesn't redo the frames already done
                        kf = cdir / f"kf_{f}_{reach}_{strokes_key(group)}.npz"
                        used.add(kf.name)
                        if kf.exists():
                            with np.load(kf) as z:
                                T, W = z["T"], z["W"]
                        else:
                            T, W = keyframe_fix(model, info.path, info.fps, f, group, (ww, wh), (cw, ch),
                                                np.asarray(ab_raw[f], np.float32), s.denoise)
                            np.savez(kf, T=T.astype(np.float16), W=W.astype(np.float16))
                        keys.append((f, np.asarray(T, np.float32), np.asarray(W, np.float32), reach))
                for old_kf in cdir.glob("kf_*.npz"):
                    try:
                        fr = int(old_kf.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    if a <= fr < b and old_kf.name not in used:
                        old_kf.unlink()
                hinted = _memmap(hpath.with_suffix(".tmp.npy"), (b - a, ch, cw, 2), np.int8)
                buf = np.empty((b - a, ch, cw, 2), np.float32) if (b - a) * ch * cw * 8 < 5e8 else \
                    _memmap(cdir / "hbuf.npy", (b - a, ch, cw, 2))
                prot = _memmap(ppath.with_suffix(".tmp.npy"), (b - a, ch, cw), np.uint8) \
                    if any(k[3] >= 0 for k in keys) else None
                pbuf = np.zeros((b - a, ch, cw), np.float32) if prot is not None else None
                carry(Ldn, ab_raw, a, b, keys, flow, tol, buf, protect=pbuf)
                hinted[:] = _to_i8(np.asarray(buf))
                hinted.flush()
                del hinted, buf
                (cdir / "hbuf.npy").unlink(missing_ok=True)
                if prot is not None:
                    prot[:] = np.clip(pbuf * 255 + 0.5, 0, 255).astype(np.uint8)
                    prot.flush()
                    del prot, pbuf
                    ppath.with_suffix(".tmp.npy").rename(ppath)
                hpath.with_suffix(".tmp.npy").rename(hpath)
            src = _Slice(np.load(hpath, mmap_mode="r"), 0, b - a)
            protect = np.load(ppath, mmap_mode="r") if ppath.exists() else None
        else:
            protect = None
            for old in list(cdir.glob(f"hint_{a}_{b}_*.npy")) + list(cdir.glob(f"protect_{a}_{b}_*.npy")):
                old.unlink()
        if (b - a) * ch * cw * 2 * 4 < 5e8:
            scratch = np.empty((b - a, ch, cw, 2), np.float32)
            out = np.empty((b - a, ch, cw, 2), np.float32)
        else:  # a very long shot: keep the working arrays on disk, not in RAM
            scratch = _memmap(cdir / "fwd.npy", (b - a, ch, cw, 2))
            out = _memmap(cdir / "out.npy", (b - a, ch, cw, 2))
        stabilize_chroma(_Slice(Ldn, a, b), src, out, strength=s.stabilize, tol=tol, flow=flow,
                         scratch=scratch)
        if protect is not None:  # frame-by-frame strokes keep their own frame's colour
            for i in range(b - a):
                pw = np.asarray(protect[i], np.float32)[..., None] / 255.0
                if pw.max() > 0:
                    out[i] = pw * np.asarray(src[i], np.float32) + (1 - pw) * np.asarray(out[i], np.float32)
        ab[a:b] = _to_i8(out)
        count += b - a
        prog(count)
        have[f"{a}-{b}"] = want[f"{a}-{b}"]
        meta["stab"] = {k: v for k, v in have.items() if k in want}
        ab.flush()
        meta_path.write_text(json.dumps(meta))  # resume point after each shot
    del ab, scratch, out
    for tmp in ("fwd.npy", "out.npy"):
        (cdir / tmp).unlink(missing_ok=True)
    rep.timings["stabilize"] = prog.done()
    return rep


class _LumaStream:
    """Frames of a clip at working size as L*, read in order. Jumps (to the
    next shot still to do) seek; everything else decodes straight on. Only
    the frames in hand are in memory."""

    def __init__(self, info: ClipInfo, size: tuple[int, int]):
        from ..media import FrameReader

        self.info, self.size = info, size
        self.r = FrameReader(info.path, info.fps, "gray16le", size=size, cache=3)
        self.fallback = None  # counting decoder, for files whose timestamps don't seek

    def get(self, t: int) -> np.ndarray:
        if self.fallback is None:
            g = self.r.get(t)
            if g is not None:
                return srgb_to_l(g.astype(np.float32) / 65535.0)
            self.fallback = [iter_gray(self.info.path, self.size), 0]
        it, pos = self.fallback
        if t < pos:
            it, pos = iter_gray(self.info.path, self.size), 0
        for g in it:
            pos += 1
            if pos - 1 == t:
                self.fallback = [it, pos]
                return srgb_to_l(g)
        raise RuntimeError(f"{Path(self.info.path).name}: frame {t} could not be decoded")

    def close(self) -> None:
        self.r.close()


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
    clip_id: str = ""

    @property
    def length(self) -> int:
        return self.src_out - self.src_in


class FrameRenderer:
    """Full-size colour for one frame: original luma plus cached chroma."""

    def __init__(self, s: VideoSettings):
        self.s = s
        self.eps: float | None = None
        self.g = float(np.clip(s.grain, 0.0, 100.0)) / 100.0

    def __call__(self, gray: np.ndarray, ab_small: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
        """gray: float 0..1 HxW. bias: the shot's colour cast (chroma.ShotCasts).
        Returns sRGB float HxWx3."""
        from ..chroma import adjust

        Lfull = srgb_to_l(gray)
        if self.eps is None:
            self.eps = float(np.clip((2.0 * estimate_noise(Lfull)) ** 2, 1.0, 100.0))
        ab = adjust(ab_small, self.s.saturation, self.s.cast, bias)
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
           quiet: bool = False, progress: ProgressFn | None = None, grades=None) -> dict[str, float]:
    """Render pieces in order into one file. audio_source passes a single
    untouched clip's audio through; audio_timeline cuts and joins tracks.
    grades: a grade.GradeBook, applied per frame after colorizing."""
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
            casts = None
            if s.cast:
                from ..chroma import ShotCasts

                casts = ShotCasts(p.analysis, ab)
            for i, g16 in enumerate(iter_range(p.path, p.fps, p.src_in, p.src_out)):
                gray = g16.astype(np.float32) / 65535.0
                rgb = rend(gray, ab[p.src_in + i], casts.at(p.src_in + i) if casts else None)
                if grades is not None:
                    from ..grade import apply as grade_apply

                    f = p.src_in + i
                    g = grades.at(p.clip_id, f)
                    if g is not None:
                        rgb = grade_apply(rgb, g, grades.mask_offset(p.clip_id, f))
                writer.write(quantize(rgb, 16))
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


def clear_cache(root: Path | None = None) -> int:
    """Empty the working folder. Only EpochColor's own subfolders go, so a
    working folder pointed at a whole disk can't take anything else along."""
    root = root or cache_root()
    n = 0
    for part in CACHE_PARTS:
        d = root / part
        if d.is_dir():
            n += sum(1 for _ in d.iterdir())
            shutil.rmtree(d, ignore_errors=True)
    return n
