"""The worker process.

Everything slow runs here: proxies, the model, renders. The GUI hands jobs
over and gets progress back through queues, so a long render never freezes
the window and a model crash (a GPU driver fault, an out-of-memory kill)
takes down only this process. The GUI notices, reports it, and starts a
fresh worker for the next job. Finished shots stay cached, so a rerun picks
up where the crash left off.

Jobs and results are plain dicts, so nothing Qt crosses the process line.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

_state = {"progress": None, "notice": None, "notice_kind": "info", "gpu_env": None}


def _gpu_device(pref: str) -> str:
    """Test the GPU in a child process the first time (gpucheck), then set
    up this process to use what passed. Has to run before PyTorch touches
    the GPU, since ROCm reads its variables once."""
    from . import gpucheck

    dev, env, note, fresh = gpucheck.ensure(pref, progress=_state["progress"])
    if _state["gpu_env"] is None:
        os.environ.update(env)
        _state["gpu_env"] = env
        print(f"gpu: {dev} {env or ''}", flush=True)
    if fresh and note:
        _state["notice"] = note
        _state["notice_kind"] = "warning" if dev == "cpu" else "info"
    return dev


def _model(cache: dict, name: str, weights: str | None, device: str):
    key = (name, weights, device)
    if key not in cache:
        from .models import create

        # settle the GPU before anything imports PyTorch: ROCm reads its
        # device variables once, when it starts
        pref = device if name == "hints" else _gpu_device(device)
        m = create(name, weights)
        if m.info.needs_torch:
            from .device import pick_device
            from .resources import torch_threads

            dev, desc = pick_device(pref)
            torch_threads()
            print(f"model {name} on {desc}", flush=True)
            m.load(dev)
        cache.clear()  # one model in memory at a time
        cache[key] = m
    return cache[key]


def photo_result_path(src: str, settings: dict, model: str, strokes: list | None = None,
                      negative: dict | None = None, reference: dict | None = None) -> Path:
    """Where a colorized photo is cached. Saturation, cast removal, grading
    and regrain are applied on top when shown or exported, so they don't
    make a new colorize."""
    from .hintpaint import strokes_key
    from .reference import reference_id
    from .video.pipeline import cache_root

    p = Path(src)
    st = p.stat()
    keys = {k: settings.get(k) for k in ("working_size", "grain", "denoise")}
    d = {"p": str(p.resolve()), "s": st.st_size, "m": st.st_mtime_ns,
         "set": keys, "model": model, "hints": strokes_key(strokes) if strokes else None}
    if negative:  # only when set, so earlier results keep their names
        d["neg"] = dict(sorted(negative.items()))
    if reference_id(reference):
        d["ref"] = reference_id(reference)
    ident = json.dumps(d, sort_keys=True)
    return cache_root() / "photos" / (hashlib.sha256(ident.encode()).hexdigest()[:20] + ".png")


def finish_photo(rgb, settings: dict, grade: dict | None):
    """A colorized photo as shown and exported: cast removal and saturation,
    the grade, then regrain. One place, so the viewer shows what exports."""
    from .chroma import adjust_rgb

    rgb = adjust_rgb(rgb, float(settings.get("saturation", 1.0)), float(settings.get("cast", 0.0)))
    if grade:
        from .grade import apply as grade_apply

        rgb = grade_apply(rgb, grade)
    if float(settings.get("regrain", 0.0) or 0.0) > 0:
        from .film import regrain

        rgb = regrain(rgb, float(settings["regrain"]), float(settings.get("regrain_size", 1.0)),
                      float(settings.get("regrain_chroma", 0.0)), seed=1)
    return rgb


def _photo_settings(settings: dict):
    from .pipeline import PhotoSettings

    return PhotoSettings(working_size=settings.get("working_size", 512),
                         grain=settings.get("grain", 100.0), denoise=settings.get("denoise"))


def _colorize_photo(job: dict, models: dict) -> Path:
    from .imageio import load_image, save_image
    from .pipeline import colorize_photo

    strokes = job.get("strokes") or []
    neg, ref = job.get("negative"), job.get("reference")
    out = photo_result_path(job["path"], job["settings"], job["model"], strokes, neg, ref)
    if out.exists():
        return out
    model = _model(models, job["model"], job.get("weights"), job.get("device", "auto"))
    img, meta = load_image(job["path"])
    if neg:
        from .film import invert_rgb

        img = invert_rgb(img, neg)
    hints = None
    if strokes:
        from .hintpaint import rasterize

        hints = rasterize(strokes, img.shape[1], img.shape[0])
    rgb, _ = colorize_photo(img, model, _photo_settings(job["settings"]), hints, reference=ref)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(out, rgb, bits=16)
    return out


def run_job(kind: str, job: dict, progress, models: dict) -> dict:
    if kind == "import":
        from .media import import_clip

        clip, media = import_clip(job["path"], progress, job.get("shot_threshold", 6.0))
        return {"clip": asdict(clip)}

    if kind == "analyze":
        from .project import AudioInfo, Clip
        from .timeline_export import clip_info
        from .video.pipeline import VideoSettings, analyze

        cd = dict(job["clip"])
        cd["audio"] = [AudioInfo(**a) for a in cd.get("audio", [])]
        clip = Clip(**cd)
        model = _model(models, job["model"], job.get("weights"), job.get("device", "auto"))
        s = VideoSettings.from_project(job["settings"])
        rep = analyze(clip_info(clip), model, s, shots=clip.shots, quiet=True, progress=progress,
                      hints=job.get("hints"), references=job.get("references"))
        return {"clip": clip.id, "dir": str(rep.dir), "timings": rep.timings}

    if kind == "photo":
        progress("colorize", 0, 1)
        out = _colorize_photo(job, models)
        progress("colorize", 1, 1)
        return {"photo": job["photo"], "result": str(out)}

    if kind == "export":
        from .export.detect import Detector
        from .export.plan import ExportSettings
        from .export.writer import ffmpeg_path
        from .project import Project, _data_from_dict
        from .timeline_export import build_export
        from .video.pipeline import cache_root

        p = Project(_data_from_dict(job["project"]))
        es, _ = ExportSettings.from_dict(job["export"])
        det = Detector(ffmpeg_path(), cache_root() / "encoders.json", job.get("vaapi_device"))
        audio_enc = {e for e in det.encoders() if e in ("aac", "libopus", "flac")}
        ej = build_export(p, None, es, job["out"], model_name=job["model"], weights=job.get("weights"),
                          probe=det.check, audio_encoders=audio_enc, vaapi_device=det.vaapi_device)
        from .timeline_export import analysis_state

        need = any(analysis_state(c, ej.settings, ej.model_name, ej.weights, ej.hints.get(c.id),
                                  ej.references.get(c.id)) != "ready"
                   for c in ej.clips)
        model = _model(models, job["model"], job.get("weights"), job.get("device", "auto")) if need else None
        timings = ej.run(model, progress=progress, quiet=True)
        return {"out": job["out"], "timings": timings, "plan": ej.plan.describe()}

    if kind == "export_photos":
        from .imageio import load_image, save_image

        from .grade import apply as grade_apply

        done = []
        items = job["items"]
        sat = float(job["settings"].get("saturation", 1.0))
        for i, item in enumerate(items):
            src, dst = item["src"], item["dst"]
            progress("photos", i, len(items))
            res = _colorize_photo({**job, "path": src, "strokes": item.get("strokes"),
                                   "negative": item.get("negative"), "reference": item.get("reference")}, models)
            rgb, _ = load_image(res)
            rgb = finish_photo(rgb, job["settings"], item.get("grade"))
            _, meta = load_image(src)
            save_image(dst, rgb, bits=job.get("bits"), quality=job.get("quality", 95), meta=meta)
            done.append(dst)
        progress("photos", len(items), len(items))
        return {"files": done}

    if kind == "fetch":
        from .models.manager import download, entry

        if entry(job["model"]) is None:
            raise ValueError(f"{job['model']} needs no download")
        path = download(job["model"], progress, accept_license=job.get("accept_license", False))
        return {"path": str(path), "model": job["model"]}

    if kind == "add_model":
        from .models.manager import add_from_file, check_loads, remove

        mid = add_from_file(job["weights"], job["manifest"])
        progress("checking it loads", 0, 1)
        try:
            check_loads(mid, "cpu")
        except Exception as e:
            remove(mid)
            raise RuntimeError(f"{mid} doesn't load, so it was removed again: {e}") from None
        return {"model": mid}

    if kind == "refresh_catalog":
        from .models.manager import refresh_catalog

        good, bad = refresh_catalog()
        return {"good": good, "skipped": bad}

    if kind == "move_models":
        from .models.manager import move_storage

        return {"dir": str(move_storage(job["dir"]))}

    if kind == "match":
        return _match(job, progress, models)

    if kind == "export_frame":
        return _export_frame(job, progress)

    if kind == "measure_negative":
        return {"params": _measure_negative(job, progress)}

    if kind == "track_mask":
        return _track_mask(job, progress)

    raise ValueError(f"unknown job {kind}")


def _working_l(path: str, frame: int | None, fps, size: int, negative: dict | None = None):
    """L* at the model's working size, from a clip frame or a photo."""
    import cv2
    import numpy as np

    from .color import srgb_to_l
    from .denoise import denoise_l
    from .film import invert, invert_rgb

    if frame is None:
        from .imageio import load_image

        img, _ = load_image(path)
        L = srgb_to_l(invert_rgb(img, negative) if negative else img)
    else:
        from .media import FrameReader

        r = FrameReader(path, fps, "gray16le", cache=2)
        try:
            g = r.get(int(frame))
        finally:
            r.close()
        if g is None:
            raise RuntimeError(f"frame {frame} of {Path(path).name} could not be decoded")
        L = srgb_to_l(invert(g, negative))
    h, w = L.shape
    k = min(1.0, size / min(h, w))
    Lw = cv2.resize(L, (max(8, round(w * k)), max(8, round(h * k))), interpolation=cv2.INTER_AREA)
    return denoise_l(Lw)


def _match(job: dict, progress, models: dict) -> dict:
    """Look for every saved colour in every other shot and photo."""
    from fractions import Fraction

    from .hintpaint import strokes_key
    from .match import exemplar, peaks, similarity
    from .project import new_id

    model = _model(models, job["model"], job.get("weights"), job.get("device", "auto"))
    if not hasattr(model, "features"):
        raise RuntimeError(f"{job['model']} can't match saved colours; pick a network model")
    size = int(job["settings"].get("working_size", 512))
    thr = float(job["settings"].get("match_threshold", 0.75))
    clips = job["clips"]  # id -> {path, fps_num, fps_den, shots}
    photos = job["photos"]  # id -> path

    photo_neg = job.get("photo_neg") or {}

    def where(kind, target, frame):
        if kind == "clip":
            c = clips[target]
            return c["path"], frame, Fraction(c["fps_num"], c["fps_den"]), c.get("negative")
        return photos[target], None, None, photo_neg.get(target)

    colors, ex = [], []
    for col in job["colors"]:
        src = col["source"]
        path, frame, fps, neg = where(src["kind"], src["target"], src.get("frame"))
        L = _working_l(path, frame, fps, size, neg)
        e = exemplar(model, L, [src["stroke"]])
        if e is not None:
            colors.append(col)
            ex.append(e)
            _thumb(L, None, src["stroke"], col["rgb"], match_thumb_path(col["id"]))
    if not ex:
        return {"suggestions": [], "colors": [c["id"] for c in job["colors"]]}

    targets = []
    for cid, c in clips.items():
        for a, b in c["shots"]:
            targets.append(("clip", cid, (a + b - 1) // 2, (a, b)))
    for pid in photos:
        targets.append(("photo", pid, None, None))
    out = []
    for i, (kind, tid, frame, span) in enumerate(targets):
        progress("matching", i, len(targets))
        path, fr, fps, neg = where(kind, tid, frame)
        L = _working_l(path, fr, fps, size, neg)
        sims = similarity(model, L, ex)
        for col, S in zip(colors, sims):
            src = col["source"]
            # skip where the colour came from
            if src["kind"] == kind and src["target"] == tid and (
                    kind == "photo" or span[0] <= int(src.get("frame", -1)) < span[1]):
                continue
            score, pts = peaks(S, thr)
            if pts:
                sid = new_id()
                out.append({"id": sid, "color": col["id"], "kind": kind, "target": tid,
                            "frame": frame, "points": pts, "score": round(score, 3)})
                _thumb(L, pts, None, col["rgb"], match_thumb_path(sid))
    progress("matching", len(targets), len(targets))
    out.sort(key=lambda x: -x["score"])
    return {"suggestions": out, "colors": [c["id"] for c in job["colors"]]}


def _export_frame(job: dict, progress) -> dict:
    """One frame at full size through the export path: inversion, output
    cleanup, colour, grade, regrain."""
    from .color import srgb_to_l
    from .film import invert
    from .grade import GradeBook
    from .imageio import save_image
    from .media import FrameReader
    from .project import Project, _data_from_dict
    from .timeline_export import analysis_state
    from .video.pipeline import FrameRenderer, OutputRestore, VideoSettings, analysis_dir, finish
    import numpy as np

    p = Project(_data_from_dict(job["project"]))
    c = p.d.clips[job["clip"]]
    f = int(job["frame"])
    s = VideoSettings.from_project(p.d.settings)
    if analysis_state(c, s, job["model"], job.get("weights"), p.d.hints.get(c.id),
                      p.d.references.get(c.id)) not in ("ready", "update"):
        raise RuntimeError(f"{c.name} isn't colorized with the current settings; colorize it first")
    d = analysis_dir(Path(c.path), s, job["model"], job.get("weights"), c.neg)
    ab = np.load(d / "ab.npy", mmap_mode="r")
    progress("rendering", 0, 1)
    r = FrameReader(c.path, c.fps, "gray16le", cache=4)
    try:
        def L_at(i):
            g = r.get(i) if 0 <= i < len(ab) else None
            return srgb_to_l(invert(g, c.neg)) if g is not None else None

        L = L_at(f)
        if L is None:
            raise RuntimeError(f"frame {f} of {c.name} could not be decoded")
        rest = OutputRestore(d, s, c.fps, (c.width, c.height))
        if rest.on:
            L = rest(f, L, L_at(f - 1) if rest.flow else None, L_at(f + 1) if rest.flow else None)
    finally:
        r.close()
    bias = None
    if s.cast:
        from .chroma import ShotCasts

        bias = ShotCasts(d, ab).at(f)
    rgb = FrameRenderer(s)(None, np.asarray(ab[f], np.float32), bias, L=L)
    rgb = finish(rgb, s, GradeBook(p.d.grades, {cid: x.shots for cid, x in p.d.clips.items()}), c.id, f)
    out = Path(job["out"])
    save_image(out, rgb, bits=8 if out.suffix.lower() in (".jpg", ".jpeg") else 16)
    progress("rendering", 1, 1)
    return {"out": str(out)}


def _measure_negative(job: dict, progress) -> dict:
    """Film base and levels of a negative: from the photo, or from frames
    spread through the clip (one frame can be all sky or all shadow)."""
    from fractions import Fraction

    from .film import measure_negative

    if job["kind"] == "photo":
        from .imageio import load_image

        img, _ = load_image(job["path"])
        return measure_negative([img])
    from .media import FrameReader

    n = int(job["frames"])
    picks = sorted({int(n * (i + 0.5) / 7) for i in range(7)})
    r = FrameReader(job["path"], Fraction(job["fps_num"], job["fps_den"]), "gray16le", size=None, cache=2)
    samples = []
    try:
        for i, f in enumerate(picks):
            progress("measuring", i, len(picks))
            g = r.get(f)
            if g is not None:
                samples.append(g.astype("float32") / 65535.0)
    finally:
        r.close()
    if not samples:
        raise RuntimeError("no frames could be read to measure the negative")
    return measure_negative(samples)


def match_thumb_path(ident: str) -> Path:
    """A small picture for a saved colour (where it came from) or a match
    (the frame, the found places ringed). Kept out of the project file."""
    from .video.pipeline import cache_root

    return cache_root() / "match" / f"{ident}.png"


def _thumb(L, points, stroke, rgb, out: Path, width: int = 192) -> None:
    """The grey frame with the match points ringed, or the saved stroke drawn,
    in the colour's own colour."""
    import cv2
    import numpy as np

    from .color import lab_to_srgb

    try:
        h, w = L.shape
        k = width / w
        g = lab_to_srgb(cv2.resize(L, (width, max(8, round(h * k))), interpolation=cv2.INTER_AREA),
                        np.zeros((max(8, round(h * k)), width, 2), np.float32))
        img = np.ascontiguousarray((np.clip(g, 0, 1) * 255).astype(np.uint8)[..., ::-1])  # BGR for cv2
        th, tw = img.shape[:2]
        col = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        if stroke is not None:
            from .hintpaint import rasterize

            m = rasterize([stroke], tw, th).mask > 0
            img[m] = (0.45 * img[m] + 0.55 * np.array(col)).astype(np.uint8)
        for x, y in points or []:
            c = (int(round(x * (tw - 1))), int(round(y * (th - 1))))
            r = max(6, int(0.09 * min(tw, th)))
            cv2.circle(img, c, r + 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(img, c, r, col, 2, cv2.LINE_AA)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
    except Exception:  # a picture is a nicety; the match stands without it
        pass


def _track_mask(job: dict, progress) -> dict:
    """Follow a grade's shape mask through its shot along optical flow.
    Returns {frame: [dx, dy]}, offsets of the mask centre as fractions of
    the frame, using the denoised luma from the clip's analysis."""
    import numpy as np

    from .video.temporal import Flow

    d = Path(job["analysis"])
    Ldn = np.load(d / "Ldn.npy", mmap_mode="r")
    a, b, start = int(job["a"]), int(job["b"]), int(job["frame"])
    cx, cy, mw, mh = (float(job["mask"][k]) for k in ("cx", "cy", "w", "h"))
    h, w = Ldn.shape[1:]
    flow = Flow("medium")
    track = {str(start): [0.0, 0.0]}
    for step in (1, -1):
        x, y = cx * w, cy * h
        t = start
        while a <= t + step < b:
            F = flow(np.asarray(Ldn[t + step], np.float32), np.asarray(Ldn[t], np.float32))
            # F pulls frame t onto t+step; the mask region moves by minus its mean
            rx, ry = max(2, int(mw * w / 4)), max(2, int(mh * h / 4))
            y0, y1 = int(max(0, y - ry)), int(min(h, y + ry + 1))
            x0, x1 = int(max(0, x - rx)), int(min(w, x + rx + 1))
            if y1 <= y0 or x1 <= x0:
                break
            region = F[y0:y1, x0:x1].reshape(-1, 2)
            dx, dy = -np.median(region[:, 0]), -np.median(region[:, 1])
            x, y = x + dx, y + dy
            t += step
            track[str(t)] = [round(x / w - cx, 5), round(y / h - cy, 5)]
            progress("tracking", len(track), b - a)
    return {"track": track}


def worker_main(jobs, events, log_name: str | None = "worker.log") -> None:
    from . import __version__
    from .resources import apply_limits, capture_output

    if log_name:
        capture_output(log_name)  # C-level output too: a crash leaves its trace here
    print(f"--- worker {os.getpid()} started {time.ctime()}, EpochColor {__version__}, "
          f"Python {sys.version.split()[0]}", flush=True)
    apply_limits(log=lambda m: print(m, flush=True))
    try:
        from .video.pipeline import sweep_stale

        sweep_stale()
    except Exception:
        pass
    models: dict = {}
    while True:
        item = jobs.get()
        if item is None:
            return
        jid, kind, job = item

        def progress(stage, done, total, _jid=jid):
            events.put(("progress", _jid, str(stage), int(done), int(total)))

        _state["progress"] = progress
        print(f"job {jid} {kind} {job.get('path') or job.get('model') or ''}", flush=True)
        try:
            result = run_job(kind, job, progress, models)
            if _state["notice"]:
                result = dict(result, notice=_state["notice"], notice_kind=_state["notice_kind"])
                _state["notice"] = None
            events.put(("done", jid, result))
        except BaseException as e:  # noqa: BLE001 - everything goes back to the GUI
            from .diskspace import explain, is_full

            msg = explain(e) if is_full(e) else f"{e}"
            tb = traceback.format_exc()
            print(tb, flush=True)
            events.put(("error", jid, msg, tb))
