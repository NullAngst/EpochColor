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


def photo_result_path(src: str, settings: dict, model: str, strokes: list | None = None) -> Path:
    """Where a colorized photo is cached. Saturation and grading are applied
    on top when shown or exported, so they don't make a new colorize."""
    from .hintpaint import strokes_key
    from .video.pipeline import cache_root

    p = Path(src)
    st = p.stat()
    keys = {k: settings.get(k) for k in ("working_size", "grain", "denoise")}
    ident = json.dumps({"p": str(p.resolve()), "s": st.st_size, "m": st.st_mtime_ns,
                        "set": keys, "model": model,
                        "hints": strokes_key(strokes) if strokes else None}, sort_keys=True)
    return cache_root() / "photos" / (hashlib.sha256(ident.encode()).hexdigest()[:20] + ".png")


def _photo_settings(settings: dict):
    from .pipeline import PhotoSettings

    return PhotoSettings(working_size=settings.get("working_size", 512),
                         grain=settings.get("grain", 100.0), denoise=settings.get("denoise"))


def _colorize_photo(job: dict, models: dict) -> Path:
    from .imageio import load_image, save_image
    from .pipeline import colorize_photo

    strokes = job.get("strokes") or []
    out = photo_result_path(job["path"], job["settings"], job["model"], strokes)
    if out.exists():
        return out
    model = _model(models, job["model"], job.get("weights"), job.get("device", "auto"))
    img, meta = load_image(job["path"])
    hints = None
    if strokes:
        from .hintpaint import rasterize

        hints = rasterize(strokes, img.shape[1], img.shape[0])
    rgb, _ = colorize_photo(img, model, _photo_settings(job["settings"]), hints)
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
                      hints=job.get("hints"))
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

        need = any(analysis_state(c, ej.settings, ej.model_name, ej.weights, ej.hints.get(c.id)) != "ready"
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
            res = _colorize_photo({**job, "path": src, "strokes": item.get("strokes")}, models)
            rgb, _ = load_image(res)
            if sat != 1.0:
                from .color import lab_to_srgb, srgb_to_lab

                lab = srgb_to_lab(rgb)
                rgb = lab_to_srgb(lab[..., 0], lab[..., 1:] * sat)
            if item.get("grade"):
                rgb = grade_apply(rgb, item["grade"])
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

    if kind == "track_mask":
        return _track_mask(job, progress)

    raise ValueError(f"unknown job {kind}")


def _working_l(path: str, frame: int | None, fps, size: int):
    """L* at the model's working size, from a clip frame or a photo."""
    import cv2
    import numpy as np

    from .color import srgb_to_l
    from .denoise import denoise_l

    if frame is None:
        from .imageio import load_image

        img, _ = load_image(path)
        L = srgb_to_l(img)
    else:
        from .media import FrameReader

        r = FrameReader(path, fps, "gray16le", cache=2)
        try:
            g = r.get(int(frame))
        finally:
            r.close()
        if g is None:
            raise RuntimeError(f"frame {frame} of {Path(path).name} could not be decoded")
        L = srgb_to_l(g.astype(np.float32) / 65535.0)
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

    def where(kind, target, frame):
        if kind == "clip":
            c = clips[target]
            return c["path"], frame, Fraction(c["fps_num"], c["fps_den"])
        return photos[target], None, None

    colors, ex = [], []
    for col in job["colors"]:
        src = col["source"]
        path, frame, fps = where(src["kind"], src["target"], src.get("frame"))
        e = exemplar(model, _working_l(path, frame, fps, size), [src["stroke"]])
        if e is not None:
            colors.append(col)
            ex.append(e)
    if not ex:
        return {"suggestions": []}

    targets = []
    for cid, c in clips.items():
        for a, b in c["shots"]:
            targets.append(("clip", cid, (a + b - 1) // 2, (a, b)))
    for pid in photos:
        targets.append(("photo", pid, None, None))
    out = []
    for i, (kind, tid, frame, span) in enumerate(targets):
        progress("matching", i, len(targets))
        path, fr, fps = where(kind, tid, frame)
        sims = similarity(model, _working_l(path, fr, fps, size), ex)
        for col, S in zip(colors, sims):
            src = col["source"]
            # skip where the colour came from
            if src["kind"] == kind and src["target"] == tid and (
                    kind == "photo" or span[0] <= int(src.get("frame", -1)) < span[1]):
                continue
            score, pts = peaks(S, thr)
            if pts:
                out.append({"id": new_id(), "color": col["id"], "kind": kind, "target": tid,
                            "frame": frame, "points": pts, "score": round(score, 3)})
    progress("matching", len(targets), len(targets))
    out.sort(key=lambda x: -x["score"])
    return {"suggestions": out}


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
