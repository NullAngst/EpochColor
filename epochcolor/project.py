"""The project: clips, the timeline, photos, settings. Plain data, no Qt.

The timeline is one video track: a list of segments, each a frame range
[src_in, src_out) of one clip, played back to back. Every edit is a change
to that list, so splits, trims, ripple deletes and moves are all simple
list operations, and undo is a snapshot of the whole project before each
change. Projects are small JSON, so snapshots cost nothing worth counting.

Frame numbers on the timeline are 0-based and run over the segments in
order. Clip frame numbers are 0-based in decode order.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

FORMAT_VERSION = 1


class ProjectError(ValueError):
    pass


@dataclass
class AudioInfo:
    codec: str
    channels: int
    layout: str = ""
    title: str = ""


@dataclass
class Clip:
    id: str
    path: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    frames: int
    start_time: float = 0.0
    codec: str = ""
    audio: list[AudioInfo] = field(default_factory=list)
    shots: list[list[int]] = field(default_factory=list)  # [start, end) in clip frames
    negative: bool = False

    @property
    def fps(self) -> Fraction:
        return Fraction(self.fps_num, self.fps_den)

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class Segment:
    clip: str
    src_in: int
    src_out: int

    @property
    def length(self) -> int:
        return self.src_out - self.src_in


@dataclass
class Marker:
    frame: int
    note: str = ""


@dataclass
class Photo:
    id: str
    path: str
    negative: bool = False
    strokes: list = field(default_factory=list)  # painted hints, see hintpaint.py
    grade: dict | None = None


def default_settings() -> dict:
    return {
        "model": "siggraph17",
        "working_size": 512,
        "stabilize": 0.9,
        "grain": 100.0,
        "denoise": None,
        "saturation": 1.0,
        "shot_threshold": 6.0,
        "chroma_size": 256,
        "match_mode": "ask",  # saved colours: "ask" marks matches, "auto" paints them
        "match_threshold": 0.75,
    }


@dataclass
class ProjectData:
    clips: dict[str, Clip] = field(default_factory=dict)
    timeline: list[Segment] = field(default_factory=list)
    audio_enabled: list[bool] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    range: list[int] | None = None  # [in, out) on the timeline
    photos: list[Photo] = field(default_factory=list)
    settings: dict = field(default_factory=default_settings)
    export: dict = field(default_factory=dict)
    hints: dict = field(default_factory=dict)  # clip id -> {str(clip frame): [stroke, ...]}
    colors: list = field(default_factory=list)  # saved colours
    suggestions: list = field(default_factory=list)  # saved-colour matches waiting for a click
    grades: dict = field(default_factory=dict)  # clip id -> {str(shot start): track}


class Project:
    def __init__(self, data: ProjectData | None = None, path: str | Path | None = None):
        self.d = data or ProjectData()
        self.path = Path(path) if path else None
        self._undo: list[tuple[str, dict]] = []
        self._redo: list[tuple[str, dict]] = []
        self.dirty = False

    # ------------------------------------------------------------ undo

    def checkpoint(self, label: str) -> None:
        """Call before every change. Undo returns to this state."""
        self._undo.append((label, self._dump()))
        self._redo.clear()
        self.dirty = True

    def commit(self, label: str, before: dict) -> bool:
        """Record a change made in several steps (a drag) as one undo step,
        given the state from before it started. Nothing if nothing changed."""
        if before == self._dump():
            return False
        self._undo.append((label, before))
        self._redo.clear()
        self.dirty = True
        return True

    def undo(self) -> str | None:
        if not self._undo:
            return None
        label, state = self._undo.pop()
        self._redo.append((label, self._dump()))
        self._load(state)
        self.dirty = True
        return label

    def redo(self) -> str | None:
        if not self._redo:
            return None
        label, state = self._redo.pop()
        self._undo.append((label, self._dump()))
        self._load(state)
        self.dirty = True
        return label

    @property
    def undo_label(self) -> str | None:
        return self._undo[-1][0] if self._undo else None

    @property
    def redo_label(self) -> str | None:
        return self._redo[-1][0] if self._redo else None

    def _dump(self) -> dict:
        return copy.deepcopy(asdict(self.d))

    def _load(self, state: dict) -> None:
        self.d = _data_from_dict(copy.deepcopy(state))

    # ---------------------------------------------------------- format

    @property
    def fps(self) -> Fraction | None:
        c = self.first_clip
        return c.fps if c else None

    @property
    def size(self) -> tuple[int, int] | None:
        c = self.first_clip
        return (c.width, c.height) if c else None

    @property
    def first_clip(self) -> Clip | None:
        if self.d.timeline:
            return self.d.clips.get(self.d.timeline[0].clip)
        return next(iter(self.d.clips.values()), None)

    # ----------------------------------------------------------- clips

    def check_clip(self, width: int, height: int, fps: Fraction) -> None:
        """The first clip sets the project format; others have to match."""
        if not self.d.clips:
            return
        pw, ph = self.size
        pf = self.fps
        if (width, height) != (pw, ph) or fps != pf:
            raise ProjectError(
                f"this clip is {width}x{height} at {float(fps):.3f} fps, the project is "
                f"{pw}x{ph} at {float(pf):.3f} fps. Convert it first to a constant frame rate "
                f"at the project resolution; HandBrake does this well."
            )

    def add_clip(self, clip: Clip, append: bool = True) -> Clip:
        self.check_clip(clip.width, clip.height, clip.fps)
        self.checkpoint(f"add {clip.name}")
        self.d.clips[clip.id] = clip
        if not clip.shots and clip.frames:
            clip.shots = [[0, clip.frames]]
        if append and clip.frames:
            self.d.timeline.append(Segment(clip.id, 0, clip.frames))
        self._sync_audio()
        return clip

    def remove_clip(self, clip_id: str) -> None:
        if clip_id not in self.d.clips:
            return
        self.checkpoint(f"remove {self.d.clips[clip_id].name}")
        self.d.timeline = [s for s in self.d.timeline if s.clip != clip_id]
        del self.d.clips[clip_id]
        self.d.hints.pop(clip_id, None)
        self.d.grades.pop(clip_id, None)
        self.d.suggestions = [x for x in self.d.suggestions if x.get("target") != clip_id]
        self._sync_audio()
        self._clamp_after_edit()

    def set_shots(self, clip_id: str, shots: list[list[int]], checkpoint: bool = True) -> None:
        if checkpoint:
            self.checkpoint("shots")
        self.d.clips[clip_id].shots = [list(s) for s in shots]

    def toggle_cut(self, frame: int) -> bool:
        """Add a shot boundary at the playhead, or remove the one that is
        there. Returns True if a cut now exists at that frame."""
        hit = self.locate(frame)
        if hit is None:
            raise ProjectError("no clip under the playhead")
        _, seg, src = hit
        clip = self.d.clips[seg.clip]
        starts = sorted({a for a, _ in clip.shots} | {0})
        self.checkpoint("toggle cut")
        if src in starts and src != 0:
            starts.remove(src)
            now = False
        elif src == 0:
            raise ProjectError("the first frame of a clip always starts a shot")
        else:
            starts.append(src)
            now = True
        starts = sorted(starts)
        clip.shots = [[a, b] for a, b in zip(starts, starts[1:] + [clip.frames])]
        grades = self.d.grades.get(clip.id, {})
        if now:
            # a new cut keeps the look: the new shot starts with the old shot's grade
            prev = max((a for a in starts if a < src), default=0)
            if str(prev) in grades:
                grades[str(src)] = copy.deepcopy(grades[str(prev)])
        else:
            grades.pop(str(src), None)  # merged into the shot before it
        return now

    def _sync_audio(self) -> None:
        n = max((len(c.audio) for c in self.d.clips.values()), default=0)
        cur = self.d.audio_enabled
        self.d.audio_enabled = (cur + [True] * n)[:n]

    @property
    def audio_tracks(self) -> int:
        return len(self.d.audio_enabled)

    def toggle_audio(self, k: int) -> bool:
        self.checkpoint(f"audio track {k + 1}")
        self.d.audio_enabled[k] = not self.d.audio_enabled[k]
        return self.d.audio_enabled[k]

    # -------------------------------------------------------- timeline

    @property
    def duration(self) -> int:
        return sum(s.length for s in self.d.timeline)

    def seg_starts(self) -> list[int]:
        out, t = [], 0
        for s in self.d.timeline:
            out.append(t)
            t += s.length
        return out

    def locate(self, frame: int) -> tuple[int, Segment, int] | None:
        """(segment index, segment, clip frame) at a timeline frame."""
        t = 0
        for i, s in enumerate(self.d.timeline):
            if t <= frame < t + s.length:
                return i, s, s.src_in + (frame - t)
            t += s.length
        return None

    def edit_points(self) -> list[int]:
        pts = self.seg_starts()
        return pts + [self.duration] if pts else []

    def shot_starts(self) -> list[int]:
        """Timeline frames where a shot begins, including segment starts."""
        out = []
        for start, seg in zip(self.seg_starts(), self.d.timeline):
            out.append(start)
            for a, _ in self.d.clips[seg.clip].shots:
                if seg.src_in < a < seg.src_out:
                    out.append(start + a - seg.src_in)
        return sorted(set(out))

    def split(self, frame: int) -> bool:
        hit = self.locate(frame)
        if hit is None:
            return False
        i, seg, src = hit
        if src == seg.src_in:
            return False  # already an edit point
        self.checkpoint("split")
        self.d.timeline[i:i + 1] = [Segment(seg.clip, seg.src_in, src), Segment(seg.clip, src, seg.src_out)]
        return True

    def delete(self, index: int) -> None:
        """Ripple delete: later segments close the gap."""
        if not 0 <= index < len(self.d.timeline):
            return
        start = self.seg_starts()[index]
        length = self.d.timeline[index].length
        self.checkpoint("ripple delete")
        del self.d.timeline[index]
        self.d.markers = [m for m in self.d.markers if not start <= m.frame < start + length]
        for m in self.d.markers:
            if m.frame >= start + length:
                m.frame -= length
        self._clamp_after_edit()

    def trim(self, index: int, src_in: int | None = None, src_out: int | None = None,
             checkpoint: bool = True) -> None:
        seg = self.d.timeline[index]
        clip = self.d.clips[seg.clip]
        new_in = seg.src_in if src_in is None else src_in
        new_out = seg.src_out if src_out is None else src_out
        new_in = max(0, min(new_in, new_out - 1))
        new_out = min(clip.frames, max(new_out, new_in + 1))
        if (new_in, new_out) == (seg.src_in, seg.src_out):
            return
        if checkpoint:
            self.checkpoint("trim")
        seg.src_in, seg.src_out = new_in, new_out
        self._clamp_after_edit()

    def move(self, index: int, delta: int) -> int:
        j = index + delta
        if not (0 <= index < len(self.d.timeline) and 0 <= j < len(self.d.timeline)):
            return index
        self.checkpoint("move")
        tl = self.d.timeline
        tl[index], tl[j] = tl[j], tl[index]
        return j

    def set_in(self, frame: int) -> None:
        self.checkpoint("mark in")
        out = self.d.range[1] if self.d.range else self.duration
        self.d.range = [frame, max(frame + 1, out)]
        self._clamp_after_edit()

    def set_out(self, frame: int) -> None:
        """Out is inclusive of the playhead frame, like any editor."""
        self.checkpoint("mark out")
        start = self.d.range[0] if self.d.range else 0
        self.d.range = [min(start, frame), frame + 1]
        self._clamp_after_edit()

    def clear_range(self) -> None:
        if self.d.range is not None:
            self.checkpoint("clear in/out")
            self.d.range = None

    def export_range(self) -> tuple[int, int]:
        if self.d.range:
            return self.d.range[0], self.d.range[1]
        return 0, self.duration

    def add_marker(self, frame: int, note: str = "") -> None:
        self.checkpoint("marker")
        self.d.markers = [m for m in self.d.markers if m.frame != frame]
        self.d.markers.append(Marker(frame, note))
        self.d.markers.sort(key=lambda m: m.frame)

    def remove_marker(self, frame: int) -> bool:
        if not any(m.frame == frame for m in self.d.markers):
            return False
        self.checkpoint("remove marker")
        self.d.markers = [m for m in self.d.markers if m.frame != frame]
        return True

    def _clamp_after_edit(self) -> None:
        dur = self.duration
        if self.d.range:
            a, b = self.d.range
            a, b = max(0, min(a, dur)), max(0, min(b, dur))
            self.d.range = [a, b] if b > a else None
        self.d.markers = [m for m in self.d.markers if m.frame < dur]

    def pieces(self, start: int | None = None, end: int | None = None) -> list[tuple[Clip, int, int]]:
        """The timeline range as (clip, src_in, src_out) pieces in order."""
        a, b = (start, end) if start is not None else self.export_range()
        out, t = [], 0
        for seg in self.d.timeline:
            s0, s1 = t, t + seg.length
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo:
                out.append((self.d.clips[seg.clip], seg.src_in + lo - s0, seg.src_in + hi - s0))
            t = s1
        return out

    def is_simple(self) -> bool:
        """True when the export is exactly one untouched clip, the only case
        where audio can be passed through without re-encoding."""
        p = self.pieces()
        return len(p) == 1 and p[0][1] == 0 and p[0][2] == p[0][0].frames

    # ---------------------------------------------------------- photos

    def add_photo(self, path: str) -> Photo:
        self.checkpoint("add photo")
        ph = Photo(new_id(), str(path))
        self.d.photos.append(ph)
        return ph

    def remove_photo(self, photo_id: str) -> None:
        self.checkpoint("remove photo")
        self.d.photos = [p for p in self.d.photos if p.id != photo_id]

    # ---------------------------------------------------- hints, colours

    def strokes_at(self, clip_id: str, frame: int) -> list:
        return list(self.d.hints.get(clip_id, {}).get(str(frame), []))

    def set_strokes(self, clip_id: str, frame: int, strokes: list, label: str = "paint") -> None:
        self.checkpoint(label)
        h = self.d.hints.setdefault(clip_id, {})
        if strokes:
            h[str(frame)] = [dict(s) for s in strokes]
        else:
            h.pop(str(frame), None)
            if not h:
                self.d.hints.pop(clip_id, None)

    def hint_frames(self, clip_id: str) -> list[int]:
        return sorted(int(k) for k, v in self.d.hints.get(clip_id, {}).items() if v)

    def set_photo_strokes(self, photo_id: str, strokes: list, label: str = "paint") -> None:
        ph = self.photo(photo_id)
        self.checkpoint(label)
        ph.strokes = [dict(s) for s in strokes]

    def photo(self, photo_id: str) -> Photo:
        for ph in self.d.photos:
            if ph.id == photo_id:
                return ph
        raise ProjectError("no such photo")

    def add_color(self, name: str, rgb, source: dict) -> dict:
        """source: {"kind": "clip"|"photo", "target": id, "frame": n, "stroke": {...}}"""
        self.checkpoint(f"save colour {name}")
        c = {"id": new_id(), "name": name, "rgb": [int(v) for v in rgb], "source": source}
        self.d.colors.append(c)
        return c

    def remove_color(self, color_id: str) -> None:
        self.checkpoint("remove saved colour")
        self.d.colors = [c for c in self.d.colors if c["id"] != color_id]
        self.d.suggestions = [x for x in self.d.suggestions if x.get("color") != color_id]

    def rename_color(self, color_id: str, name: str) -> None:
        self.checkpoint("rename saved colour")
        for c in self.d.colors:
            if c["id"] == color_id:
                c["name"] = name

    def set_suggestions(self, items: list) -> None:
        self.checkpoint("find saved colours")
        self.d.suggestions = list(items)

    def apply_suggestion(self, sug_id: str) -> dict | None:
        from .hintpaint import dabs

        sug = next((x for x in self.d.suggestions if x["id"] == sug_id), None)
        col = next((c for c in self.d.colors if sug and c["id"] == sug["color"]), None)
        if not sug or not col:
            return None
        self.checkpoint(f"apply {col['name']}")
        new = dabs(sug["points"], col["rgb"])
        if sug["kind"] == "clip":
            h = self.d.hints.setdefault(sug["target"], {})
            h[str(sug["frame"])] = h.get(str(sug["frame"]), []) + new
        else:
            ph = self.photo(sug["target"])
            ph.strokes = list(ph.strokes) + new
        self.d.suggestions = [x for x in self.d.suggestions if x["id"] != sug_id]
        return sug

    def dismiss_suggestion(self, sug_id: str) -> None:
        self.checkpoint("dismiss match")
        self.d.suggestions = [x for x in self.d.suggestions if x["id"] != sug_id]

    # ----------------------------------------------------------- grades

    def shot_start(self, clip_id: str, frame: int) -> int:
        start = 0
        for a, b in self.d.clips[clip_id].shots:
            if a <= frame:
                start = a
            if a <= frame < b:
                return a
        return start

    def grade_track(self, clip_id: str, frame: int) -> dict | None:
        return self.d.grades.get(clip_id, {}).get(str(self.shot_start(clip_id, frame)))

    def set_grade_track(self, clip_id: str, frame: int, track: dict | None, label: str = "grade",
                        checkpoint: bool = True) -> None:
        if checkpoint:
            self.checkpoint(label)
        g = self.d.grades.setdefault(clip_id, {})
        key = str(self.shot_start(clip_id, frame))
        if track:
            g[key] = copy.deepcopy(track)
        else:
            g.pop(key, None)

    def set_photo_grade(self, photo_id: str, grade: dict | None, label: str = "grade",
                        checkpoint: bool = True) -> None:
        if checkpoint:
            self.checkpoint(label)
        self.photo(photo_id).grade = copy.deepcopy(grade) if grade else None

    def set_setting(self, key: str, value) -> None:
        if self.d.settings.get(key) == value:
            return
        self.checkpoint(f"set {key}")
        self.d.settings[key] = value

    # ------------------------------------------------------------ disk

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else self.path
        if p is None:
            raise ProjectError("no file name to save to")
        data = {"epochcolor_project": FORMAT_VERSION, **asdict(self.d)}
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(p)  # never leave a half-written project behind
        self.path = p
        self.dirty = False
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        raw = json.loads(Path(path).read_text())
        if raw.get("epochcolor_project") is None:
            raise ProjectError(f"{path} is not an EpochColor project")
        if raw["epochcolor_project"] > FORMAT_VERSION:
            raise ProjectError("this project was saved by a newer EpochColor")
        raw.pop("epochcolor_project")
        return cls(_data_from_dict(raw), path)


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def _data_from_dict(d: dict) -> ProjectData:
    clips = {}
    for cid, c in d.get("clips", {}).items():
        c = dict(c)
        c["audio"] = [AudioInfo(**a) for a in c.get("audio", [])]
        clips[cid] = Clip(**c)
    settings = default_settings()
    settings.update(d.get("settings", {}))
    return ProjectData(
        clips=clips,
        timeline=[Segment(**s) for s in d.get("timeline", [])],
        audio_enabled=list(d.get("audio_enabled", [])),
        markers=[Marker(**m) for m in d.get("markers", [])],
        range=d.get("range"),
        photos=[Photo(**p) for p in d.get("photos", [])],
        settings=settings,
        export=dict(d.get("export", {})),
        hints=dict(d.get("hints", {})),
        colors=list(d.get("colors", [])),
        suggestions=list(d.get("suggestions", [])),
        grades=dict(d.get("grades", {})),
    )
