"""Feed rendered frames to ffmpeg according to an ExportPlan.

Single pass: frames go straight down a pipe into the final encode.

Two-pass: the encoder needs to see every frame twice, and rendering a frame
is the expensive part, so frames go once into a lossless FFV1 RGB file in
the cache folder, then both passes read that file. It costs disk space for
the length of the export and is deleted afterwards.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .plan import ExportPlan


@dataclass
class AudioTimeline:
    """Audio for an edited timeline: each output track is cut from its
    stream in every piece and joined. Compressed audio frames do not line
    up with video frames, so these tracks are always re-encoded.

    inputs: source files. pieces: (input index, start s, end s) in each
    source's own timestamps. tracks: the source stream index per output
    track. layouts: channel layout per output track, for silence where a
    source has no such stream. streams: per input, how many audio streams.
    """
    inputs: list[str]
    pieces: list[tuple[int, float, float]]
    tracks: list[int]
    layouts: list[str] = field(default_factory=list)
    streams: list[int] = field(default_factory=list)

    def filter_graph(self, first_input: int) -> tuple[str, list[str]]:
        parts, labels = [], []
        for j, k in enumerate(self.tracks):
            layout = (self.layouts[j] if j < len(self.layouts) else "") or "stereo"
            fmt = f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts={layout}"
            ins = []
            for n, (inp, t0, t1) in enumerate(self.pieces):
                lab = f"t{j}p{n}"
                has = k < (self.streams[inp] if inp < len(self.streams) else 1)
                if has:
                    # pad then cap each piece to its exact length, so a source
                    # whose audio runs short cannot pull later pieces out of sync
                    d = t1 - t0
                    parts.append(f"[{first_input + inp}:a:{k}]atrim=start={t0:.6f}:end={t1:.6f},"
                                 f"asetpts=PTS-STARTPTS,{fmt},apad=whole_dur={d:.6f},"
                                 f"atrim=duration={d:.6f}[{lab}]")
                else:
                    parts.append(f"anullsrc=r=48000:cl={layout},atrim=duration={t1 - t0:.6f},"
                                 f"{fmt}[{lab}]")
                ins.append(f"[{lab}]")
            out = f"aout{j}"
            parts.append(f"{''.join(ins)}concat=n={len(ins)}:v=0:a=1[{out}]")
            labels.append(f"[{out}]")
        return ";".join(parts), labels


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found in PATH")
    return exe


def pipe_input(width: int, height: int, fps: Fraction) -> list[str]:
    return ["-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{width}x{height}",
            "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "-"]


def build_command(plan: ExportPlan, video_in: list[str], out: str, source: str | None,
                  start_time: float, duration: float | None, pass_n: int | None = None,
                  stats: str | None = None, ffmpeg: str | None = None,
                  timeline: AudioTimeline | None = None) -> list[str]:
    cmd = [ffmpeg or ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    cmd += plan.global_args + video_in
    want_audio = bool(plan.audio_maps) and pass_n != 1
    with_audio = want_audio and source and timeline is None
    with_timeline = want_audio and timeline is not None and timeline.tracks
    if with_audio:
        cmd += ["-itsoffset", f"{-start_time:.6f}", "-i", str(source)]
    if with_timeline:
        for path in timeline.inputs:
            cmd += ["-i", str(path)]
        graph, labels = timeline.filter_graph(1)
        cmd += ["-filter_complex", graph]
    cmd += ["-map", "0:v:0"]
    if with_audio:
        for idx in plan.audio_maps:
            cmd += ["-map", f"1:a:{idx}"]
    if with_timeline:
        for lab in labels:
            cmd += ["-map", lab]
        with_audio = True
    cmd += ["-vf", plan.filter_chain] + list(plan.video_args)

    x265 = list(plan.x265_params)
    if pass_n:
        if plan.encoder.name == "libx265":
            x265 += [f"pass={pass_n}", f"stats={stats}"]
        else:
            cmd += ["-pass", str(pass_n), "-passlogfile", str(stats)]
    if x265:
        cmd += ["-x265-params", ":".join(x265)]
    if plan.svt_params:
        cmd += ["-svtav1-params", ":".join(plan.svt_params)]

    if with_audio:
        cmd += plan.audio_args
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    if pass_n == 1:
        cmd += ["-an", "-f", "null", "-"]
    else:
        cmd += plan.mux_args + [str(out)]
    return cmd


class VideoWriter:
    def __init__(self, plan: ExportPlan, out: str | Path, width: int, height: int,
                 fps: Fraction, source: str | Path | None, start_time: float,
                 workdir: Path, frames_limit: int | None = None,
                 audio_timeline: AudioTimeline | None = None):
        self.plan, self.out = plan, Path(out)
        self.timeline = audio_timeline
        self.width, self.height, self.fps = width, height, fps
        self.source, self.start_time = source, start_time
        self.workdir = workdir
        workdir.mkdir(parents=True, exist_ok=True)
        self.duration = frames_limit / float(fps) if frames_limit else None
        self.log = workdir / "ffmpeg.log"
        self.count = 0
        self.ffmpeg = ffmpeg_path()
        if plan.two_pass:
            self.inter = workdir / "twopass.mkv"
            cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
            cmd += pipe_input(width, height, fps)
            cmd += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp16le", "-g", "1",
                    "-slices", "16", str(self.inter)]
        else:
            self.inter = None
            cmd = build_command(plan, pipe_input(width, height, fps), str(self.out),
                                str(source) if source else None, start_time, self.duration,
                                ffmpeg=self.ffmpeg, timeline=audio_timeline)
        self.cmd = cmd
        self._logf = open(self.log, "w")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._logf)

    def write(self, rgb16: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(rgb16, dtype="<u2").tobytes())
        except BrokenPipeError:
            self.proc.wait()
            raise RuntimeError(f"ffmpeg stopped: {self._err()}") from None
        self.count += 1

    def _err(self) -> str:
        self._logf.flush()
        try:
            return self.log.read_text(errors="replace").strip()[-2000:]
        except OSError:
            return ""

    def _wait(self, proc: subprocess.Popen, what: str) -> None:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg {what} failed ({rc}): {self._err()}\n"
                               f"command: {' '.join(map(str, proc.args))}")

    def close(self, on_pass=None) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self._wait(self.proc, "encode" if not self.inter else "intermediate")
            if self.inter:
                stats = str(self.workdir / "twopass-stats")
                for n in (1, 2):
                    if on_pass:
                        on_pass(n)
                    cmd = build_command(self.plan, ["-i", str(self.inter)], str(self.out),
                                        str(self.source) if self.source else None,
                                        self.start_time, self.duration, pass_n=n, stats=stats,
                                        ffmpeg=self.ffmpeg, timeline=self.timeline)
                    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stderr=self._logf)
                    self._wait(p, f"pass {n}")
        finally:
            self._logf.close()
            self._cleanup()

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait()
        finally:
            self._logf.close()
            self._cleanup()

    def _cleanup(self) -> None:
        if self.inter and self.inter.exists():
            self.inter.unlink()
        for f in self.workdir.glob("twopass-stats*"):
            f.unlink()
