import shutil
import subprocess

import numpy as np
import pytest

av = pytest.importorskip("av")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from epochcolor.models.base import ColorModel, ModelInfo  # noqa: E402
from epochcolor.video.temporal import (  # noqa: E402
    Flow, cut_scores, detect_shots, stabilize_chroma, thumb,
)


def moving_disc(n=30, h=96, w=160, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    frames, masks = [], []
    for t in range(n):
        m = (y - 48) ** 2 + (x - (40 + 2 * t)) ** 2 < 22 ** 2
        L = np.where(m, 35.0, 75.0) * (1 + 0.04 * np.sin(1.3 * t))  # exposure flicker
        L = L + rng.normal(0, 1.0, L.shape)
        frames.append(L.astype(np.float32))
        masks.append(m)
    return frames, masks


class JitterModel(ColorModel):
    """Right colours, wrong by a random offset each frame, like a per-frame
    network drifting in hue."""

    info = ModelInfo("jitter", "automatic", "test", False, False, False)

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    def predict(self, L, hints=None):
        ab = np.zeros(L.shape + (2,), np.float32)
        dark = L < 55
        ab[dark] = (45.0, 30.0)
        ab[~dark] = (-5.0, -20.0)
        return ab + self.rng.normal(0, 6.0, 2).astype(np.float32)


def test_flicker_is_not_a_cut_but_a_cut_is():
    frames, _ = moving_disc()
    other = [100.0 - f for f in frames]  # a different picture
    seq = frames[:15] + other[15:]
    shots = detect_shots(cut_scores([thumb(f, 64) for f in seq]))
    assert shots == [(0, 15), (15, 30)]


def test_stabilizer_cuts_drift_and_keeps_edges():
    frames, masks = moving_disc()
    model = JitterModel()
    raw = np.stack([model.predict(f) for f in frames])
    out = np.empty_like(raw)
    stabilize_chroma(frames, raw, out, strength=0.9, tol=3.0, flow=Flow("medium"))
    bg = (slice(5, 20), slice(5, 30))  # background, never under the disc
    jitter_raw = raw[:, bg[0], bg[1], 1].mean(axis=(1, 2)).std()
    jitter_out = out[:, bg[0], bg[1], 1].mean(axis=(1, 2)).std()
    assert jitter_out < 0.4 * jitter_raw, (jitter_raw, jitter_out)
    # colour still follows the moving disc
    t = 20
    inside = out[t][masks[t]].mean(axis=0)
    assert inside[0] > 30, inside


def _make_clip(path, n=24, vfr=False):
    w, h = 160, 96
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
    if vfr:
        cmd += ["-vf", "setpts='if(lt(N,8),N,N*2)/24/TB'", "-fps_mode", "vfr"]
    cmd += ["-map", "0", "-map", "1", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
            "-c:a", "flac", "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    frames, _ = moving_disc(n, h, w)
    for f in frames:
        p.stdin.write(np.clip(f * 2.55, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0


@needs_ffmpeg
def test_vfr_is_refused(tmp_path):
    from epochcolor.video.io import InputError, probe

    clip = tmp_path / "vfr.mkv"
    _make_clip(clip, vfr=True)
    with pytest.raises(InputError, match="variable frame rate"):
        probe(clip)


@needs_ffmpeg
def test_video_end_to_end(tmp_path, monkeypatch):
    from epochcolor.color import srgb_to_l
    from epochcolor.export.plan import ExportSettings, resolve
    from epochcolor.video.io import iter_gray, probe
    from epochcolor.video.pipeline import VideoSettings, colorize_video

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    clip = tmp_path / "clip.mkv"
    _make_clip(clip)
    info = probe(clip)
    assert info.fps == 24 and len(info.audio_streams) == 1
    out = tmp_path / "out.mkv"
    plan = resolve(ExportSettings(encoder="software", rf=8, speed="ultrafast"), out, info.audio)
    rep = colorize_video(info, JitterModel(), out, plan, VideoSettings(), quiet=True)
    assert rep.frames == 24 and len(rep.shots) == 1

    with av.open(str(out)) as c:
        vs = c.streams.video[0]
        assert vs.codec_context.name == "hevc"
        assert vs.codec_context.pix_fmt == "yuv420p10le"
        assert len(c.streams.audio) == 1
        rgb = [f.to_ndarray(format="rgb48le").astype(np.float32) / 65535 for f in c.decode(vs)]
    assert len(rgb) == 24
    # colour came through: the disc is warm, the background cool
    f = rgb[10]
    assert f[48, 60, 0] > f[48, 60, 2] and f[10, 150, 2] > f[10, 150, 0]
    # luma survived the trip within encode noise
    src = srgb_to_l(next(iter_gray(clip)))
    from epochcolor.color import srgb_to_lab

    assert abs(float((srgb_to_lab(rgb[0])[..., 0] - src).mean())) < 1.0

    # second run reuses the model pass
    out2 = tmp_path / "out2.mkv"
    plan2 = resolve(ExportSettings(encoder="software", rf=8, speed="ultrafast"), out2, info.audio)
    rep2 = colorize_video(info, JitterModel(), out2, plan2, VideoSettings(), quiet=True)
    assert rep2.cached_shots == 1
