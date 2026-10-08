"""Film restoration and reference images through the real pipeline."""
import shutil
import subprocess

import numpy as np
import pytest

av = pytest.importorskip("av")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

from test_video import JitterModel, moving_disc  # noqa: E402


def write_clip(path, frames8, w=160, h=96):
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p", str(path)],
                         stdin=subprocess.PIPE)
    for f in frames8:
        p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def read_frames(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24").astype(np.float32) / 255 for f in c.decode(video=0)]


@needs_ffmpeg
def test_negative_clip_comes_out_positive(tmp_path, monkeypatch):
    from epochcolor.export.plan import ExportSettings, resolve
    from epochcolor.film import measure_negative
    from epochcolor.video.io import probe
    from epochcolor.video.pipeline import Piece, VideoSettings, analyze, render

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    frames, masks = moving_disc(24)
    neg8 = [np.clip(255 - f * 2.55, 0, 255).astype(np.uint8) for f in frames]  # a plain inverted copy
    clip = tmp_path / "neg.mkv"
    write_clip(clip, neg8)
    info = probe(clip)
    info.negative = measure_negative([n.astype(np.float32) / 255 for n in neg8[:3]])
    s = VideoSettings(working_size=96, chroma_size=96)
    rep = analyze(info, JitterModel(), s, quiet=True)
    out = tmp_path / "out.mkv"
    plan = resolve(ExportSettings(encoder="software", rf=8, speed="ultrafast", container=".mkv"), out, [])
    render([Piece(clip, info.fps, 160, 96, rep.dir, 0, 24, "c", info.negative)], plan, out, s, quiet=True)
    rgb = read_frames(out)[10]
    disc = masks[10]
    # the disc was dark in the positive: dark again, and red (JitterModel colours dark things red)
    assert rgb[disc].mean() < rgb[~disc].mean()
    assert rgb[disc][:, 0].mean() > rgb[disc][:, 2].mean()


@needs_ffmpeg
def test_output_deflicker_dust_and_regrain_render(tmp_path, monkeypatch):
    from epochcolor.export.plan import ExportSettings, resolve
    from epochcolor.video.io import probe
    from epochcolor.video.pipeline import Piece, VideoSettings, analyze, render

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:160]
    frames8 = []
    for t in range(24):
        base = 110 + 40 * np.sin((xx - 2 * t) / 11.0) + 20 * np.cos(yy / 9.0)
        base = base * (1.0 + 0.15 * (1 if t % 2 else -1))  # heavy flicker
        f = np.clip(base, 0, 255)
        if t == 12:
            f[40:46, 70:76] = 255  # a speck of dust
        frames8.append(f.astype(np.uint8))
    clip = tmp_path / "flick.mkv"
    write_clip(clip, frames8)
    info = probe(clip)
    s = VideoSettings(working_size=96, chroma_size=96, deflicker=1.0, dust=True, deflicker_out=True,
                      dust_out=True, regrain=2.0)
    rep = analyze(info, JitterModel(), s, quiet=True)
    out = tmp_path / "out.mkv"
    plan = resolve(ExportSettings(encoder="software", rf=4, speed="ultrafast", container=".mkv"), out, [])
    render([Piece(clip, info.fps, 160, 96, rep.dir, 0, 24, "c")], plan, out, s, quiet=True)
    got = read_frames(out)
    assert len(got) == 24, "read-ahead renders every frame once, in order"
    means = np.array([g.mean() for g in got])
    src_means = np.array([f.mean() / 255 for f in frames8])
    jitter = lambda m: np.abs(np.diff(m)).mean()
    assert jitter(means) < 0.4 * jitter(src_means), "the output flicker is calmed"
    assert got[12][42, 72].mean() < 0.85, "the speck is gone"


class FeatureJitter(JitterModel):
    """JitterModel with features: one-hot brightness bands, so a reference
    matches dark with dark and light with light."""

    def features(self, L):
        import cv2
        import torch

        h, w = L.shape
        s = cv2.resize(L, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
        bands = np.stack([np.exp(-((s - c) / 12.0) ** 2) for c in (15, 35, 55, 75, 95)])
        f = torch.from_numpy(bands.astype(np.float32))
        return f / (f.norm(dim=0, keepdim=True) + 1e-6)


def reference_image(path):
    """A colour photo: dark things blue, light things yellow."""
    import cv2

    img = np.zeros((96, 160, 3), np.uint8)
    img[:, :80] = (50, 80, 190)  # blue about as dark as the disc (L* ~35), RGB
    img[:, 80:] = (215, 185, 70)  # yellow about as light as the background (L* ~76)
    cv2.imwrite(str(path), img[..., ::-1])


@needs_ffmpeg
def test_reference_image_steers_a_shot(tmp_path, monkeypatch):
    from test_video import _make_clip

    from epochcolor.video.io import probe
    from epochcolor.video.pipeline import VideoSettings, analyze

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    clip = tmp_path / "c.mkv"
    _make_clip(clip, n=24)  # dark disc on a light background
    ref = tmp_path / "ref.png"
    reference_image(ref)
    info = probe(clip)
    s = VideoSettings(working_size=96, chroma_size=96)
    m = FeatureJitter()
    rep = analyze(info, m, s, shots=[(0, 24)], quiet=True)
    before = np.load(rep.dir / "ab.npy").astype(np.float32)
    rep = analyze(info, m, s, shots=[(0, 24)], quiet=True,
                  references={"0": {"path": str(ref), "strength": 1.0}})
    after = np.load(rep.dir / "ab.npy").astype(np.float32)
    disc = after[12, 44:52, 52:60].reshape(-1, 2).mean(0)  # disc centre at x = 40 + 24 = 64 of 160 -> 38 of 96
    bg = after[12, 5:12, 80:90].reshape(-1, 2).mean(0)
    assert before[12, 44:52, 52:60, 0].mean() > 25, "the model had the disc red"
    assert disc[1] < -10, f"the disc took the reference's blue: {disc}"
    assert bg[1] > 15, f"the background took its yellow: {bg}"


def test_reference_on_a_photo(tmp_path):
    from epochcolor.pipeline import PhotoSettings, colorize_photo

    ref = tmp_path / "ref.png"
    reference_image(ref)
    frames, masks = moving_disc(1)
    img = np.repeat((frames[0] / 100.0)[..., None], 3, axis=2).astype(np.float32)
    from epochcolor.color import srgb_to_lab

    rgb, _ = colorize_photo(img, FeatureJitter(), PhotoSettings(working_size=96),
                            reference={"path": str(ref), "strength": 1.0})
    lab = srgb_to_lab(rgb)
    assert lab[masks[0]][:, 2].mean() < -5 < 15 < lab[~masks[0]][:, 2].mean()
