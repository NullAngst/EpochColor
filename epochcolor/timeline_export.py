"""Glue between a project timeline and the render: which clips to analyse,
which pieces to render, how the audio gets there."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .export.plan import AudioStream, ExportPlan, ExportSettings, resolve
from .export.writer import AudioTimeline
from .models import ColorModel
from .project import Clip, Project
from .video.io import ClipInfo
from .video.pipeline import Piece, VideoSettings, analysis_dir, analyze, render


def clip_info(c: Clip) -> ClipInfo:
    info = ClipInfo(path=Path(c.path), width=c.width, height=c.height, fps=c.fps,
                    frames_estimate=c.frames, duration=c.frames / float(c.fps),
                    start_time=c.start_time, codec=c.codec)
    info.negative = c.neg
    return info


def analysis_state(c: Clip, s: VideoSettings, model_name: str, weights: str | None = None,
                   hints: dict | None = None, references: dict | None = None) -> str:
    """"ready" when the cached chroma matches the clip's current shots and
    the stabilizer setting, "partial" when some of the model pass is done,
    "none" otherwise."""
    d = analysis_dir(Path(c.path), s, model_name, weights, c.neg)
    m = d / "meta.json"
    if not m.exists():
        return "none"
    try:
        meta = json.loads(m.read_text())
    except ValueError:
        return "none"
    from .video.pipeline import wanted_stab

    stab = meta.get("stab")
    want = wanted_stab([tuple(x) for x in c.shots], s.stabilize, hints, references, s.paint_spread)
    if isinstance(stab, dict) and all(stab.get(k) == v for k, v in want.items()) and (d / "ab.npy").exists():
        return "ready"
    done = {tuple(x) for x in meta.get("model_done", [])}
    if done and all(tuple(x) in done for x in c.shots):
        return "update"  # model pass done; only hints or the stabilizer need redoing
    return "partial" if meta.get("model_done") else "none"


def shot_states(c: Clip, s: VideoSettings, model_name: str, weights: str | None = None,
                hints: dict | None = None, references: dict | None = None) -> dict:
    """What each shot of a clip shows, keyed "a-b":

    {"state": "none" | "painted" | "colored", "current": bool, "file": path or None}

    painted: Color shot made it from the strokes alone (file is its chroma).
    colored: the full pass (Fill shot or Colorize) did it, from ab.npy.
    current: False when the strokes, reference or settings changed since,
    so it's the last result shown until it's redone."""
    from .hintpaint import hints_in
    from .video.pipeline import paint_key, wanted_stab

    d = analysis_dir(Path(c.path), s, model_name, weights, c.neg)
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    stab = meta.get("stab") if isinstance(meta.get("stab"), dict) else {}
    painted = meta.get("painted") or {}
    view = meta.get("view") or {}
    want = wanted_stab([tuple(x) for x in c.shots], s.stabilize, hints, references, s.paint_spread)
    have_ab = (d / "ab.npy").exists()
    out = {}
    for a, b in c.shots:
        k = f"{a}-{b}"
        p = painted.get(k)
        pfile = d / p["file"] if p and (d / p["file"]).exists() else None
        colored = have_ab and k in stab
        v = view.get(k) or ("colored" if colored else "painted" if pfile else "none")
        if v == "painted" and pfile is not None:
            sh = hints_in(hints, a, b)
            out[k] = {"state": "painted", "current": bool(sh) and p["key"] == paint_key(s, sh), "file": str(pfile)}
        elif colored:
            out[k] = {"state": "colored", "current": stab.get(k) == want[k], "file": str(d / "ab.npy")}
        else:
            out[k] = {"state": "none", "current": True, "file": None}
    return out


def project_audio(p: Project) -> list[AudioStream]:
    """One entry per project audio track, described by the first clip that has it."""
    out = []
    for k in range(p.audio_tracks):
        src = next((c.audio[k] for c in p.d.clips.values() if len(c.audio) > k), None)
        if src is None:
            out.append(AudioStream(k, "unknown", 2))
        else:
            out.append(AudioStream(k, src.codec, src.channels, src.title, src.layout))
    return out


@dataclass
class ExportJob:
    project_pieces: list[tuple[Clip, int, int]]
    plan: ExportPlan
    out: Path
    settings: VideoSettings
    model_name: str
    weights: str | None
    audio_source: tuple[Path, float] | None = None
    audio_timeline: AudioTimeline | None = None
    clips: list[Clip] = field(default_factory=list)

    hints: dict = field(default_factory=dict)
    grades: object = None
    references: dict = field(default_factory=dict)  # clip id -> {str(shot start): reference}

    def run(self, model: ColorModel | None = None, progress=None, quiet: bool = True) -> dict:
        timings = {}
        dirs = {}
        for c in self.clips:
            if analysis_state(c, self.settings, self.model_name, self.weights,
                              self.hints.get(c.id), self.references.get(c.id)) != "ready":
                if model is None:
                    raise RuntimeError(f"{c.name} is not colorized yet")
            if model is None:
                dirs[c.id] = analysis_dir(Path(c.path), self.settings, self.model_name, self.weights, c.neg)
                continue
            rep = analyze(clip_info(c), model, self.settings, shots=c.shots, quiet=quiet,
                          progress=progress, hints=self.hints.get(c.id), references=self.references.get(c.id))
            dirs[c.id] = rep.dir
        pieces = [Piece(Path(c.path), c.fps, c.width, c.height, dirs[c.id], a, b, c.id, c.neg)
                  for c, a, b in self.project_pieces]
        timings.update(render(pieces, self.plan, self.out, self.settings,
                              audio_source=self.audio_source, audio_timeline=self.audio_timeline,
                              quiet=quiet, progress=progress, grades=self.grades))
        return timings


def build_export(p: Project, model: ColorModel | None, es: ExportSettings, out: str | Path,
                 model_name: str | None = None, weights: str | None = None, probe=None,
                 audio_encoders: set[str] | None = None, vaapi_device: str | None = None) -> ExportJob:
    out = Path(out)
    pieces = p.pieces()
    if not pieces:
        raise ValueError("the timeline is empty")
    edited = not p.is_simple()
    audio = project_audio(p)
    es.audio_tracks = [k + 1 for k, on in enumerate(p.d.audio_enabled) if on]
    plan = resolve(es, out, audio, probe=probe, ffmpeg_audio_encoders=audio_encoders,
                   vaapi_device=vaapi_device, edited=edited)
    name = model_name or (model.info.name if model else p.d.settings["model"])
    s = VideoSettings.from_project(p.d.settings)
    clips, seen = [], set()
    for c, _, _ in pieces:
        if c.id not in seen:
            seen.add(c.id)
            clips.append(c)
    from .grade import GradeBook

    job = ExportJob(pieces, plan, out, s, name, weights, clips=clips, hints=dict(p.d.hints),
                    grades=GradeBook(p.d.grades, {cid: c.shots for cid, c in p.d.clips.items()}),
                    references={cid: p.expanded_references(cid) for cid in p.d.clips})
    if plan.audio_maps:
        if edited:
            inputs, index = [], {}
            spans = []
            for c, a, b in pieces:
                if c.path not in index:
                    index[c.path] = len(inputs)
                    inputs.append(c.path)
                fps = float(c.fps)
                spans.append((index[c.path], c.start_time + a / fps, c.start_time + b / fps))
            streams = [len(next(c for c in clips if c.path == path).audio) for path in inputs]
            layouts = [audio[k].layout for k in plan.audio_maps]
            job.audio_timeline = AudioTimeline(inputs, spans, list(plan.audio_maps), layouts, streams)
        else:
            c = pieces[0][0]
            job.audio_source = (Path(c.path), c.start_time)
    return job
