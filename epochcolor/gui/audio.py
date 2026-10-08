"""Sound while playing the timeline.

The enabled audio tracks of the edited timeline get mixed to one stereo
WAV in the working folder, with the same cut-and-join filter graph the
export uses, by an ffmpeg run in the background (not the worker, so a long
colorize doesn't hold the sound up). Qt plays it, started at the playhead
and nudged back in step if it drifts from the picture.

The mix is redone a moment after an edit; until it's ready, playback is
silent. Shuttling faster than 1x is silent too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, QUrl


def mix_command(project, out: Path) -> list[str] | None:
    """ffmpeg arguments for the timeline's enabled audio as stereo 48 kHz
    WAV, or None when there's nothing to hear."""
    from ..export.writer import AudioTimeline, ffmpeg_path

    tracks = [k for k, on in enumerate(project.d.audio_enabled) if on]
    if not tracks or not project.d.timeline:
        return None
    inputs, index, spans = [], {}, []
    for seg in project.d.timeline:
        c = project.d.clips[seg.clip]
        if c.path not in index:
            index[c.path] = len(inputs)
            inputs.append(c.path)
        fps = float(c.fps)
        spans.append((index[c.path], c.start_time + seg.src_in / fps, c.start_time + seg.src_out / fps))
    streams = [len(next(c for c in project.d.clips.values() if c.path == p).audio) for p in inputs]
    layouts = []
    for k in tracks:
        src = next((c.audio[k] for c in project.d.clips.values() if len(c.audio) > k), None)
        layouts.append((src.layout if src else "") or "stereo")
    tl = AudioTimeline(inputs, spans, tracks, layouts, streams)
    graph, labels = tl.filter_graph(0)
    st = ";".join(f"{lab}aformat=sample_fmts=fltp:channel_layouts=stereo[s{i}]" for i, lab in enumerate(labels))
    if len(labels) > 1:
        mix = "".join(f"[s{i}]" for i in range(len(labels))) + f"amix=inputs={len(labels)}:normalize=0[mix]"
    else:
        mix = "[s0]anull[mix]"
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", f"{graph};{st};{mix}", "-map", "[mix]", "-ac", "2", "-ar", "48000",
            "-c:a", "pcm_s16le", str(out)]
    return cmd


def mix_key(project) -> str | None:
    tracks = [k for k, on in enumerate(project.d.audio_enabled) if on]
    if not tracks or not project.d.timeline:
        return None
    segs = []
    for seg in project.d.timeline:
        c = project.d.clips[seg.clip]
        try:
            st = Path(c.path).stat()
            stamp = (st.st_size, st.st_mtime_ns)
        except OSError:
            stamp = None
        segs.append([c.path, stamp, seg.src_in, seg.src_out])
    return hashlib.sha1(json.dumps([segs, tracks]).encode()).hexdigest()[:16]


class TimelineAudio(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = self.output = None
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

            self.player = QMediaPlayer(self)
            self.output = QAudioOutput(self)
            self.player.setAudioOutput(self.output)
        except Exception:  # no multimedia backend: play silent
            self.player = None
        self.enabled = True
        self.key: str | None = None  # the mix the project needs
        self.ready_key: str | None = None  # the mix the player holds
        self.proc: QProcess | None = None
        self._pending = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(1200)
        self.timer.timeout.connect(self._build)

    def available(self) -> bool:
        return self.player is not None

    def project_changed(self, project) -> None:
        """Call after edits; rebuilds the mix a moment later if it changed."""
        if self.player is None:
            return
        key = mix_key(project)
        if key == self.key:
            return
        self.key = key
        self._pending = project
        self.stop()
        if key is None:
            self.ready_key = None
            self.player.setSource(QUrl())
            return
        self.timer.start()

    def _build(self) -> None:
        from ..video.pipeline import cache_root

        project, key = self._pending, self.key
        if project is None or key is None:
            return
        out = cache_root() / "audio" / f"{key}.wav"
        if out.exists():
            self._ready(key, out)
            return
        try:
            cmd = mix_command(project, out.with_suffix(".tmp.wav"))
        except Exception:
            return
        if cmd is None:
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.proc is not None:
            self.proc.kill()
        self.proc = QProcess(self)
        self.proc.finished.connect(lambda code, _s, k=key, o=out, pr=self.proc: self._built(code, k, o, pr))
        self.proc.start(cmd[0], cmd[1:])

    def _built(self, code: int, key: str, out: Path, proc) -> None:
        if proc is not self.proc:
            return
        self.proc = None
        tmp = out.with_suffix(".tmp.wav")
        if code == 0 and tmp.exists():
            tmp.replace(out)
            if key == self.key:
                self._ready(key, out)
        else:
            tmp.unlink(missing_ok=True)

    def _ready(self, key: str, path: Path) -> None:
        self.ready_key = key
        self.player.setSource(QUrl.fromLocalFile(str(path)))

    def is_ready(self) -> bool:
        return self.player is not None and self.ready_key is not None and self.ready_key == self.key

    def play_from(self, seconds: float) -> None:
        if not (self.enabled and self.is_ready()):
            return
        self.player.setPosition(max(0, int(seconds * 1000)))
        self.player.play()

    def stop(self) -> None:
        if self.player is not None:
            self.player.pause()

    def keep_in_step(self, seconds: float) -> None:
        """Called while playing: put the sound back where the picture is if it wandered."""
        if self.player is None or not self.is_ready() or not self.enabled:
            return
        from PySide6.QtMultimedia import QMediaPlayer

        if self.player.playbackState() != QMediaPlayer.PlayingState:
            return
        if abs(self.player.position() - seconds * 1000) > 150:
            self.player.setPosition(max(0, int(seconds * 1000)))
