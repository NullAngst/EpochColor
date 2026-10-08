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
    # film restoration (film.py). deflicker and dust clean the model's copy;
    # the _out switches apply them to the output picture too
    deflicker: float = 0.0  # 0..1 strength
    dust: bool = False
    deflicker_out: bool = False
    dust_out: bool = False
    regrain: float = 0.0  # grain strength in percent at mid grey, 0 = off
    regrain_size: float = 1.0  # grain size in pixels at 1080 lines
    regrain_chroma: float = 0.0  # 0..1 colour grain on top of the mono grain
    paint_spread: float = 0.35  # how far a stroke fills through a flat area, fraction of the short side
    frames: int | None = None  # process only the first N frames
    flow_preset: str = "medium"
    chroma_size: int = 256  # short side of the stored chroma

    @classmethod
    def from_project(cls, d: dict) -> "VideoSettings":
        keys = {"working_size", "grain", "denoise", "stabilize", "shot_threshold", "saturation",
                "chroma_size", "cast", "deflicker", "dust", "deflicker_out", "dust_out", "regrain",
                "regrain_size", "regrain_chroma", "paint_spread"}
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


CACHE_PARTS = ("clips", "photos", "media", "export", "tmp", "match", "audio", "frame")  # all cache_root ever holds


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


def cache_key(path: Path, s: VideoSettings, model_name: str, weights: str | None,
              negative: dict | None = None) -> str:
    st = path.stat()
    ident = {
        "src": str(path.resolve()), "size": st.st_size, "mtime": st.st_mtime_ns,
        "ws": s.working_size, "dn": s.denoise, "frames": s.frames,
        "model": model_name, "weights": str(weights or ""), "cs": s.chroma_size, "v": 2,
    }
    # what the model sees; only added when on, so caches from before stay valid
    if s.deflicker:
        ident["dfl"] = round(float(s.deflicker), 3)
    if s.dust:
        ident["dust"] = True
    if negative:
        ident["neg"] = {k: v for k, v in sorted(negative.items())}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:20]


def analysis_dir(path: Path, s: VideoSettings, model_name: str, weights: str | None = None,
                 negative: dict | None = None) -> Path:
    """weights: the model's cache id; None looks up the installed model's.
    negative: the clip's negative inversion, when it's a negative."""
    if weights is None:
        from ..models import cache_id_for

        weights = cache_id_for(model_name)
    return cache_root() / "clips" / cache_key(Path(path), s, model_name, weights, negative)


def finished_analysis(path: Path, s: VideoSettings, model_name: str,
                      weights: str | None = None, negative: dict | None = None) -> Path | None:
    """The cache folder if this clip is fully analysed with these settings."""
    d = analysis_dir(path, s, model_name, weights, negative)
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


def shot_reference(references: dict | None, a: int, b: int, painted: bool = False) -> dict | None:
    """The reference set on the shot starting at a, if any. "*" is the
    fallback for every shot without its own: what painting has taught,
    which a shot with its own painting doesn't need."""
    if not references:
        return None
    r = references.get(str(a)) or references.get(a)
    if r is None and not painted:
        r = references.get("*")
    if not r:
        return None
    return r if (r.get("path") or (r.get("kind") == "painted" and r.get("sources"))) else None


def wanted_stab(shots, strength: float, hints: dict | None, references: dict | None = None,
                spread: float = 0.35) -> dict[str, str]:
    """What each shot's stabilized chroma has to have been made from: the
    stabilizer strength, the hints painted in that shot and its reference
    image. A shot whose id changes is redone; the rest stay as they are."""
    from ..hintpaint import hints_in, strokes_key
    from ..reference import reference_id
    from .hints_pass import PAINT_VERSION

    out = {}
    for a, b in shots:
        h = hints_in(hints, a, b)
        rid = reference_id(shot_reference(references, a, b, painted=bool(h)))
        paint = f"p{PAINT_VERSION}.{spread}.{strokes_key(h)}" if h else "-"
        out[f"{a}-{b}"] = f"{strength}:{paint}" + (f":ref{rid}" if rid else "")
    return out


def analyze(info: ClipInfo, model: ColorModel, s: VideoSettings | None = None,
            shots: list[tuple[int, int]] | None = None, use_cache: bool = True,
            quiet: bool = False, progress: ProgressFn | None = None,
            hints: dict | None = None, references: dict | None = None,
            only_shots: list | None = None) -> Analysis:
    """Run passes 1 to 4 on a clip. shots, when given, replace detection
    (the GUI passes the shots the user sees and may have edited). hints:
    painted strokes, {clip frame: [stroke, ...]}. references: reference
    images per shot, {str(shot start): {"path", "strength"}}. only_shots:
    work on just these shots (Fill shot); the rest stay as they are.

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
    neg = getattr(info, "negative", None)
    cdir = analysis_dir(info.path, s, model.info.name, getattr(model, "cache_id", None), neg)
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

        from ..film import frame_stats, invert

        prog = _Progress("decode", s.frames or info.frames_estimate, quiet, progress)
        mid = max(0, (s.frames or info.frames_estimate) // 2)
        scores, prev_t, sample, lstats = [], None, None, []
        n = 0
        for g in iter_gray(info.path, (ww, wh), s.frames):
            Lf = srgb_to_l(invert(g, neg) if neg else g)
            t = thumb(Lf)
            scores.append(0.0 if prev_t is None else cut_score(prev_t, t))
            lstats.append([round(v, 3) for v in frame_stats(t)])
            prev_t = t
            if n <= mid:
                sample = Lf  # ends up as the middle frame, or the last if the clip is shorter
            n += 1
            prog(n)
        prog.total = max(1, n)
        rep.timings["decode"] = prog.done()
        if n == 0:
            raise RuntimeError("no frames decoded")
        meta.update({"frames": n, "scores": scores, "model_done": [], "lstats": lstats,
                     "sigma": float(estimate_noise(sample)), "decoded": True})
        meta_path.write_text(json.dumps(meta))
        return n

    # ---------------------------------------------- 1. decode at working size
    if not meta.get("scores") or "sigma" not in meta or "lstats" not in meta:
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
    only = {tuple(x) for x in only_shots} if only_shots else None
    todo = sorted(sh for sh in shots if tuple(sh) not in done and (only is None or tuple(sh) in only))
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
    frames = _LumaStream(info, (ww, wh), neg) if todo else None
    if todo and s.deflicker:
        from ..film import deflicker, deflicker_targets

        lstats = np.asarray(meta["lstats"], np.float32)
        targets = deflicker_targets(lstats, shots, float(info.fps))

        def fetch(t):  # the model's copy, brightness steadied
            return deflicker(frames.get(t), lstats[t], targets[t], float(s.deflicker))
    else:
        def fetch(t):
            return frames.get(t)
    dust_thr = max(8.0, 4.0 * sigma)
    try:
        for a, b in todo:
            cur = fetch(a)
            prev = None
            for t in range(a, b):
                nxt = fetch(t + 1) if t + 1 < b else None
                src = cur
                if s.dust:
                    from ..film import remove_dust

                    src, _ = remove_dust(cur, prev, nxt, flow, dust_thr)
                if s.denoise == 0:
                    dn = src
                else:
                    dn = temporal_denoise(prev, src, nxt, flow, sigma)
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
    want = wanted_stab(shots, s.stabilize, hints, references, s.paint_spread)
    have = meta.get("stab") if isinstance(meta.get("stab"), dict) else {}
    ab_path = cdir / "ab.npy"
    redo = [(a, b) for a, b in shots if (have.get(f"{a}-{b}") != want[f"{a}-{b}"] or not ab_path.exists())
            and (only is None or (a, b) in only)]
    view = meta.setdefault("view", {})
    for a, b in shots:  # what the editor shows for each shot: this colorize from now on
        if only is None or (a, b) in only:
            view[f"{a}-{b}"] = "filled"
    if not redo:
        meta_path.write_text(json.dumps(meta))
        return rep
    if ab_path.exists():
        ab = np.load(ab_path, mmap_mode="r+")
        if ab.shape != (n, ch, cw, 2):
            del ab
            ab = _memmap(ab_path, (n, ch, cw, 2), np.int8)
            redo = [sh for sh in shots if only is None or tuple(sh) in only]
            have = {}
    else:
        ab = _memmap(ab_path, (n, ch, cw, 2), np.int8)
        redo = [sh for sh in shots if only is None or tuple(sh) in only]
    from ..hintpaint import group_by_reach, hints_in, strokes_key
    from .hints_pass import PAINT_VERSION, carry, keyframe_fix

    tol = max(2.0, 3.0 * sigma)
    ref_cache: dict = {}  # a reference's features, kept between its shots and keyframes
    total = sum(b - a for a, b in redo)
    prog = _Progress("stabilize", total, quiet, progress)
    count = 0
    scratch = out = None
    for a, b in redo:
        src = _Slice(ab_raw, a, b)
        shot_hints = hints_in(hints, a, b)
        ref = shot_reference(references, a, b, painted=bool(shot_hints))
        if shot_hints or ref:
            from ..reference import reference_id

            rid = reference_id(ref)
            key = f"p{PAINT_VERSION}{s.paint_spread}" + strokes_key(shot_hints) + (f"r{rid}" if rid else "")
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
                        kf = cdir / f"kf_{f}_{reach}_{strokes_key(group)}_p{PAINT_VERSION}_{s.paint_spread}.npz"
                        used.add(kf.name)
                        if kf.exists():
                            with np.load(kf) as z:
                                T, W = z["T"], z["W"]
                        else:
                            T, W = keyframe_fix(None, info.path, info.fps, f, group, (ww, wh), (cw, ch),
                                                None, s.denoise, spread=s.paint_spread, negative=neg)
                            np.savez(kf, T=T.astype(np.float16), W=W.astype(np.float16))
                        keys.append((f, np.asarray(T, np.float32), np.asarray(W, np.float32), reach))
                if ref:
                    from ..reference import keyframes, reference_fix

                    for f in keyframes(a, b):
                        kf = cdir / f"kfref_{f}_{rid}.npz"
                        used.add(kf.name)
                        if kf.exists():
                            with np.load(kf) as z:
                                T, W = z["T"], z["W"]
                        else:
                            Lw = _working_frame(info, f, (ww, wh), neg, s.denoise)
                            T, W = reference_fix(model, Lw, ref, (cw, ch), cache=ref_cache)
                            np.savez(kf, T=T.astype(np.float16), W=W.astype(np.float16))
                        keys.append((f, np.asarray(T, np.float32), np.asarray(W, np.float32), -1))
                for old_kf in list(cdir.glob("kf_*.npz")) + list(cdir.glob("kfref_*.npz")):
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


def paint_key(s: VideoSettings, shot_hints: dict) -> str:
    from ..hintpaint import strokes_key
    from .hints_pass import PAINT_VERSION

    return f"p{PAINT_VERSION}.{s.paint_spread}.{strokes_key(shot_hints)}"


class _Offset:
    """A shot's own array, indexed by clip frame like the whole-clip ones."""

    def __init__(self, arr, a: int):
        self.arr, self.a = arr, a

    def __getitem__(self, t):
        return self.arr[t - self.a]


class _Zeros:
    def __init__(self, shape):
        self.z = np.zeros(shape, np.float32)

    def __getitem__(self, t):
        return self.z


def paint_shot(info: ClipInfo, s: VideoSettings, model_name: str, weights: str | None,
               shot: tuple[int, int], hints: dict | None, progress: ProgressFn | None = None) -> Path:
    """Color shot: colour one shot from its painted strokes alone, no model.

    Each painted frame's strokes fill their objects (keyframe_fix), the
    fills ride the motion through the shot (carry), and everything nothing
    was painted on stays grey. Fast, since there's no model pass: what you
    see is exactly what your painting does. Fill shot then lets the model
    colour the rest."""
    from ..film import invert
    from ..hintpaint import group_by_reach, hints_in, strokes_key
    from ..media import FrameReader
    from .hints_pass import PAINT_VERSION, carry, keyframe_fix

    a, b = int(shot[0]), int(shot[1])
    neg = getattr(info, "negative", None)
    shot_hints = hints_in(hints, a, b)
    if not shot_hints:
        raise RuntimeError("nothing is painted in this shot yet")
    W, H = info.width, info.height
    ww, wh = working_size(W, H, s.working_size)
    cw, ch = working_size(W, H, min(s.working_size, s.chroma_size))
    cdir = analysis_dir(info.path, s, model_name, weights, neg)
    cdir.mkdir(parents=True, exist_ok=True)
    key = paint_key(s, shot_hints)
    out_path = cdir / f"paint_{a}_{b}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npy"
    meta_path = cdir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        meta = {}
    n = b - a
    if not out_path.exists():
        # the shot's luma at chroma size, for following the motion
        Ls = np.empty((n, ch, cw), np.float32) if n * ch * cw * 4 < 4e8 else \
            _memmap(cdir / "pluma.npy", (n, ch, cw), np.float32)
        r = FrameReader(info.path, info.fps, "gray16le", size=(cw, ch), cache=3)
        try:
            for i in range(n):
                g = r.get(a + i)
                if g is None:
                    raise RuntimeError(f"{Path(info.path).name}: frame {a + i} could not be decoded")
                Ls[i] = srgb_to_l(invert(g, neg))
                if progress and i % 8 == 0:
                    progress("reading the shot", i, n)
        finally:
            r.close()
        keys = []
        frames = sorted(shot_hints)
        for j, f in enumerate(frames):
            if progress:
                progress(f"painted frame {j + 1} of {len(frames)}", j, len(frames))
            for reach, group in group_by_reach(shot_hints[f]).items():
                kf = cdir / f"kf_{f}_{reach}_{strokes_key(group)}_p{PAINT_VERSION}_{s.paint_spread}.npz"
                if kf.exists():
                    with np.load(kf) as z:
                        T, Wt = z["T"], z["W"]
                else:
                    T, Wt = keyframe_fix(None, info.path, info.fps, f, group, (ww, wh), (cw, ch),
                                         None, s.denoise, spread=s.paint_spread, negative=neg)
                    np.savez(kf, T=T.astype(np.float16), W=Wt.astype(np.float16))
                keys.append((f, np.asarray(T, np.float32), np.asarray(Wt, np.float32), reach))
        if progress:
            progress("carrying the paint through the shot", 0, 1)
        buf = np.zeros((n, ch, cw, 2), np.float32) if n * ch * cw * 8 < 5e8 else \
            _memmap(cdir / "pbuf.npy", (n, ch, cw, 2), np.float32)
        sigma = float(meta.get("sigma", 1.0))
        carry(_Offset(Ls, a), _Zeros((ch, cw, 2)), a, b, keys, Flow(s.flow_preset), max(2.0, 3.0 * sigma), buf)
        res = _memmap(out_path.with_suffix(".tmp.npy"), (n, ch, cw, 2), np.int8)
        res[:] = _to_i8(np.asarray(buf))
        res.flush()
        del res, buf, Ls
        for tmp in ("pluma.npy", "pbuf.npy"):
            (cdir / tmp).unlink(missing_ok=True)
        out_path.with_suffix(".tmp.npy").rename(out_path)
    for old in cdir.glob(f"paint_{a}_{b}_*.npy"):  # one painted version per shot
        if old != out_path and not old.name.endswith(".tmp.npy"):
            old.unlink()
    meta.setdefault("painted", {})[f"{a}-{b}"] = {"key": key, "file": out_path.name}
    meta.setdefault("view", {})[f"{a}-{b}"] = "painted"
    meta_path.write_text(json.dumps(meta))
    if progress:
        progress("done", 1, 1)
    return out_path


def _working_frame(info: ClipInfo, f: int, size: tuple[int, int], negative, denoise) -> np.ndarray:
    """One frame's L* at working size, denoised like the model's copy."""
    from ..film import invert
    from ..media import FrameReader

    r = FrameReader(info.path, info.fps, "gray16le", size=size, cache=2)
    try:
        g = r.get(f)
    finally:
        r.close()
    if g is None:
        raise RuntimeError(f"{Path(info.path).name}: frame {f} could not be decoded")
    return denoise_l(srgb_to_l(invert(g, negative)), denoise)


class _LumaStream:
    """Frames of a clip at working size as L*, read in order. Jumps (to the
    next shot still to do) seek; everything else decodes straight on. Only
    the frames in hand are in memory."""

    def __init__(self, info: ClipInfo, size: tuple[int, int], negative: dict | None = None):
        from ..media import FrameReader

        self.info, self.size, self.negative = info, size, negative
        self.r = FrameReader(info.path, info.fps, "gray16le", size=size, cache=3)
        self.fallback = None  # counting decoder, for files whose timestamps don't seek

    def get(self, t: int) -> np.ndarray:
        if self.fallback is None:
            g = self.r.get(t)
            if g is not None:
                from ..film import invert

                return srgb_to_l(invert(g, self.negative))
            self.fallback = [iter_gray(self.info.path, self.size), 0]
        it, pos = self.fallback
        if t < pos:
            it, pos = iter_gray(self.info.path, self.size), 0
        for g in it:
            pos += 1
            if pos - 1 == t:
                self.fallback = [it, pos]
                from ..film import invert

                return srgb_to_l(invert(g, self.negative) if self.negative else g)
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
    negative: dict | None = None  # the clip's negative inversion, when it's a negative

    @property
    def length(self) -> int:
        return self.src_out - self.src_in


class FrameRenderer:
    """Full-size colour for one frame: original luma plus cached chroma."""

    def __init__(self, s: VideoSettings):
        self.s = s
        self.eps: float | None = None
        self.g = float(np.clip(s.grain, 0.0, 100.0)) / 100.0

    def __call__(self, gray: np.ndarray, ab_small: np.ndarray, bias: np.ndarray | None = None,
                 L: np.ndarray | None = None) -> np.ndarray:
        """gray: float 0..1 HxW. bias: the shot's colour cast (chroma.ShotCasts).
        L: the luma already restored (OutputRestore), used instead of gray's.
        Returns sRGB float HxWx3."""
        from ..chroma import adjust

        Lfull = srgb_to_l(gray) if L is None else L
        if self.eps is None:
            self.eps = float(np.clip((2.0 * estimate_noise(Lfull)) ** 2, 1.0, 100.0))
        ab = adjust(ab_small, self.s.saturation, self.s.cast, bias)
        if self.s.guided:
            abf = guided_upsample(ab, Lfull, self.eps)
        else:
            import cv2

            abf = cv2.resize(ab, (Lfull.shape[1], Lfull.shape[0]), interpolation=cv2.INTER_CUBIC)
        Lout = Lfull if self.g >= 1.0 else self.g * Lfull + (1 - self.g) * denoise_l(Lfull, self.s.denoise)
        return lab_to_srgb(Lout, abf)


class OutputRestore:
    """Deflicker and dust on the output picture, when switched on. Works on
    full-size L*, with the frame statistics from the clip's analysis and
    optical flow at working size scaled up."""

    def __init__(self, analysis: Path, s: VideoSettings, fps, size: tuple[int, int]):
        self.s = s
        self.on = bool(s.deflicker_out or s.dust_out)
        self.stats = self.targets = None
        self.shots: list[tuple[int, int]] = []
        try:
            meta = json.loads((Path(analysis) / "meta.json").read_text())
        except (OSError, ValueError):
            meta = {}
        self.shots = sorted(tuple(int(v) for v in k.split("-")) for k in (meta.get("stab") or {}))
        self.sigma = float(meta.get("sigma", 0.0))
        if s.deflicker_out and meta.get("lstats"):
            from ..film import deflicker_targets

            self.stats = np.asarray(meta["lstats"], np.float32)
            self.targets = deflicker_targets(self.stats, self.shots or [(0, len(self.stats))], float(fps))
        self.work = working_size(size[0], size[1], s.working_size)
        self.flow = Flow(s.flow_preset) if s.dust_out else None

    def same_shot(self, a: int, b: int) -> bool:
        for x, y in self.shots:
            if x <= a < y:
                return x <= b < y
        return True

    def _deflicker(self, f: int, L: np.ndarray) -> np.ndarray:
        if self.targets is None or f >= len(self.targets):
            return L
        from ..film import deflicker

        return deflicker(L, self.stats[f], self.targets[f], 1.0)

    def __call__(self, f: int, L: np.ndarray, prev: np.ndarray | None = None,
                 nxt: np.ndarray | None = None) -> np.ndarray:
        """Restored L for frame f. prev and nxt: neighbours' L (unrestored)."""
        if not self.on:
            return L
        L = self._deflicker(f, L)
        if self.flow is not None and prev is not None and nxt is not None \
                and self.same_shot(f, f - 1) and self.same_shot(f, f + 1):
            import cv2

            from ..film import remove_dust, scaled_flow

            prev, nxt = self._deflicker(f - 1, prev), self._deflicker(f + 1, nxt)
            small = [cv2.resize(x, self.work, interpolation=cv2.INTER_AREA) for x in (L, prev, nxt)]
            Fp = scaled_flow(self.flow, small[0], small[1], L.shape)
            Fn = scaled_flow(self.flow, small[0], small[2], L.shape)
            L, _ = remove_dust(L, prev, nxt, None, max(8.0, 4.0 * self.sigma), flows=(Fp, Fn))
        return L


def regrain_seed(clip_id: str, frame: int) -> int:
    import zlib

    return (zlib.crc32(clip_id.encode()) + frame * 7919) & 0x7FFFFFFF


def finish(rgb: np.ndarray, s: VideoSettings, grades, clip_id: str, f: int) -> np.ndarray:
    """After colorizing: the grade, then regrain. Shared by export and the preview's still."""
    if grades is not None:
        from ..grade import apply as grade_apply

        g = grades.at(clip_id, f)
        if g is not None:
            rgb = grade_apply(rgb, g, grades.mask_offset(clip_id, f))
    if s.regrain:
        from ..film import regrain

        rgb = regrain(rgb, s.regrain, s.regrain_size, s.regrain_chroma, regrain_seed(clip_id, f))
    return rgb


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
            from ..film import invert

            restore = OutputRestore(p.analysis, s, p.fps, (p.width, p.height))
            ahead = restore.flow is not None  # dust removal needs the frame after too
            lo = max(0, p.src_in - 1) if ahead else p.src_in
            hi = min(p.src_out + 1, len(ab)) if ahead else p.src_out
            frames = ((lo + k, srgb_to_l(invert(g16, p.negative)))
                      for k, g16 in enumerate(iter_range(p.path, p.fps, lo, hi)))

            def out(f, L, prev, nxt):
                nonlocal done
                L = restore(f, L, prev, nxt)
                rgb = rend(None, ab[f], casts.at(f) if casts else None, L=L)
                writer.write(quantize(finish(rgb, s, grades, p.clip_id, f), 16))
                done += 1
                prog(done)

            if not ahead:
                for f, L in frames:
                    out(f, L, None, None)
            else:
                before = here = None
                for f, L in frames:
                    if here is not None and p.src_in <= here[0] < p.src_out:
                        out(here[0], here[1], before[1] if before else None, L)
                    before, here = here, (f, L)
                if here is not None and p.src_in <= here[0] < p.src_out:
                    out(here[0], here[1], before[1] if before else None, None)
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
                   quiet: bool = False, reference: dict | None = None) -> Analysis:
    """Analyse one clip and render all of it: what the CLI does. reference:
    one reference image for every shot."""
    s = s or VideoSettings()
    rep = analyze(info, model, s, use_cache=use_cache, quiet=quiet,
                  references={"*": reference} if reference else None)
    piece = Piece(info.path, info.fps, info.width, info.height, rep.dir, 0, rep.frames, "clip",
                  getattr(info, "negative", None))
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
