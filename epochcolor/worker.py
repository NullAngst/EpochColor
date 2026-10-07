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
import traceback
from dataclasses import asdict
from pathlib import Path


def _model(cache: dict, name: str, weights: str | None, device: str):
    key = (name, weights, device)
    if key not in cache:
        from .models import create

        m = create(name, weights)
        if m.info.needs_torch:
            from .device import pick_device

            dev, _ = pick_device(device)
            m.load(dev)
        cache.clear()  # one model in memory at a time
        cache[key] = m
    return cache[key]


def photo_result_path(src: str, settings: dict, model: str) -> Path:
    from .video.pipeline import cache_root

    p = Path(src)
    st = p.stat()
    keys = {k: settings.get(k) for k in ("working_size", "grain", "denoise", "saturation")}
    ident = json.dumps({"p": str(p.resolve()), "s": st.st_size, "m": st.st_mtime_ns,
                        "set": keys, "model": model}, sort_keys=True)
    return cache_root() / "photos" / (hashlib.sha256(ident.encode()).hexdigest()[:20] + ".png")


def _photo_settings(settings: dict):
    from .pipeline import PhotoSettings

    return PhotoSettings(working_size=settings.get("working_size", 512),
                         grain=settings.get("grain", 100.0), denoise=settings.get("denoise"),
                         saturation=settings.get("saturation", 1.0))


def _colorize_photo(job: dict, models: dict) -> Path:
    from .imageio import load_image, save_image
    from .pipeline import colorize_photo

    out = photo_result_path(job["path"], job["settings"], job["model"])
    if out.exists():
        return out
    model = _model(models, job["model"], job.get("weights"), job.get("device", "auto"))
    img, meta = load_image(job["path"])
    rgb, _ = colorize_photo(img, model, _photo_settings(job["settings"]))
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
        rep = analyze(clip_info(clip), model, s, shots=clip.shots, quiet=True, progress=progress)
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
        model = _model(models, job["model"], job.get("weights"), job.get("device", "auto"))
        timings = ej.run(model, progress=progress, quiet=True)
        return {"out": job["out"], "timings": timings, "plan": ej.plan.describe()}

    if kind == "export_photos":
        from .imageio import load_image, save_image

        done = []
        items = job["items"]
        for i, (src, dst) in enumerate(items):
            progress("photos", i, len(items))
            res = _colorize_photo({**job, "path": src}, models)
            rgb, _ = load_image(res)
            _, meta = load_image(src)
            save_image(dst, rgb, bits=job.get("bits"), quality=job.get("quality", 95), meta=meta)
            done.append(dst)
        progress("photos", len(items), len(items))
        return {"files": done}

    if kind == "fetch":
        from .models.weights import CATALOG, fetch

        if job["model"] not in CATALOG:
            raise ValueError(f"{job['model']} needs no download")
        progress("download", 0, 1)
        path = fetch(job["model"], progress=False)
        progress("download", 1, 1)
        return {"path": str(path)}

    raise ValueError(f"unknown job {kind}")


def worker_main(jobs, events) -> None:
    models: dict = {}
    while True:
        item = jobs.get()
        if item is None:
            return
        jid, kind, job = item

        def progress(stage, done, total, _jid=jid):
            events.put(("progress", _jid, str(stage), int(done), int(total)))

        try:
            result = run_job(kind, job, progress, models)
            events.put(("done", jid, result))
        except BaseException as e:  # noqa: BLE001 - everything goes back to the GUI
            events.put(("error", jid, f"{e}", traceback.format_exc()))
