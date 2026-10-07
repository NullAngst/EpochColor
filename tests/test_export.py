import shutil
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from epochcolor.export.codecs import ENCODERS
from epochcolor.export.plan import AudioStream, ExportError, ExportSettings, resolve

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
FLAC = [AudioStream(0, "flac", 2, "eng"), AudioStream(1, "pcm_s24le", 6, "")]


def args(plan):
    return " ".join(plan.video_args)


# ------------------------------------------------------------ plan rules


def test_rf_maps_per_encoder():
    cases = {
        "libx265": "-crf 20", "libx264": "-crf 20", "libsvtav1": "-crf 20",
        "hevc_nvenc": "-cq 20", "hevc_qsv": "-global_quality 20", "hevc_vaapi": "-qp 20",
        "hevc_amf": "-qp_i 20",
    }
    for enc, expect in cases.items():
        codec = ENCODERS[enc].codec
        p = resolve(ExportSettings(codec=codec, encoder=enc, rf=20), "o.mkv", [])
        assert expect in args(p), (enc, args(p))
        assert p.quality_text  # always says what RF became


def test_rf0_lossless_where_real_and_warns_elsewhere():
    p = resolve(ExportSettings(encoder="libx264", codec="h264", bits=8, rf=0), "o.mkv", [])
    assert "-qp 0" in args(p) and not p.warnings
    p = resolve(ExportSettings(encoder="libx265", rf=0), "o.mkv", [])
    assert "lossless=1" in p.x265_params and "-crf" not in args(p)
    p = resolve(ExportSettings(encoder="hevc_nvenc", rf=0), "o.mkv", [])
    assert "-tune lossless" in args(p)
    for enc in ("hevc_vaapi", "hevc_qsv", "hevc_amf", "libsvtav1"):
        codec = ENCODERS[enc].codec
        p = resolve(ExportSettings(codec=codec, encoder=enc, rf=0), "o.mkv", [])
        assert any("NOT lossless" in w for w in p.warnings), enc


def test_auto_skips_hardware_without_lossless():
    def only_vaapi(spec, bits, chroma):
        return None if spec.hw in ("vaapi", None) else "missing"

    p = resolve(ExportSettings(rf=18), "o.mkv", [], probe=only_vaapi)
    assert p.encoder.name == "hevc_vaapi"
    p = resolve(ExportSettings(rf=0), "o.mkv", [], probe=only_vaapi)
    assert p.encoder.name == "libx265"
    p = resolve(ExportSettings(rf=18, chroma="444"), "o.mkv", [], probe=only_vaapi)
    assert p.encoder.name == "libx265"
    p = resolve(ExportSettings(rf=18, tune_grain=True), "o.mkv", [], probe=only_vaapi)
    assert p.encoder.name == "libx265"
    with pytest.raises(ExportError, match="no working hardware"):
        resolve(ExportSettings(encoder="hardware", codec="prores"), "o.mov", [], probe=only_vaapi)


def test_container_rules():
    with pytest.raises(ExportError, match="can't go in .mp4"):
        resolve(ExportSettings(codec="prores"), "o.mp4", [])
    with pytest.raises(ExportError, match="can't go in .mov"):
        resolve(ExportSettings(codec="h265"), "o.mov", [])
    p = resolve(ExportSettings(codec="h265", encoder="software"), "o.mp4", [])
    assert "hvc1" in p.mux_args


def test_two_pass_needs_bitrate_and_software():
    with pytest.raises(ExportError, match="needs a bitrate"):
        resolve(ExportSettings(two_pass=True), "o.mkv", [])
    with pytest.raises(ExportError, match="no two-pass"):
        resolve(ExportSettings(encoder="hevc_nvenc", bitrate=4000, rf=None, two_pass=True), "o.mkv", [])
    p = resolve(ExportSettings(bitrate=4000, rf=None, two_pass=True), "o.mkv", [])
    assert p.encoder.name == "libx265" and p.two_pass


def test_software_speed_names_keep_auto_on_software():
    p = resolve(ExportSettings(speed="ultrafast"), "o.mkv", [], probe=lambda *a: None)
    assert p.encoder.name == "libx265"
    p = resolve(ExportSettings(speed="p7"), "o.mkv", [], probe=lambda *a: None)
    assert p.encoder.name == "hevc_nvenc"


def test_audio_copy_and_fallback():
    p = resolve(ExportSettings(encoder="software"), "o.mkv", FLAC)
    assert p.audio_args.count("copy") == 2
    with pytest.raises(ExportError, match="can't be copied into .mov"):
        resolve(ExportSettings(codec="prores"), "o.mov", FLAC)
    p = resolve(ExportSettings(codec="prores", audio_fallback="aac"), "o.mov", FLAC)
    assert p.audio_args[:4] == ["-c:a:0", "aac", "-b:a:0", "192k"]
    assert p.audio_args[4:] == ["-c:a:1", "copy"]  # PCM is fine in MOV
    p = resolve(ExportSettings(codec="prores", audio_codec="aac"), "o.mov", FLAC)
    assert "576k" in p.audio_args  # 6 channels at 96 kbps each
    p = resolve(ExportSettings(encoder="software", audio_fallback="opus"), "o.mp4", FLAC)
    assert p.audio_args[:2] == ["-c:a:0", "copy"] and "libopus" in p.audio_args
    with pytest.raises(ExportError, match="can't hold flac"):
        resolve(ExportSettings(codec="prores", audio_codec="flac"), "o.mov", FLAC)
    p = resolve(ExportSettings(encoder="software", audio_tracks=[2], audio_codec="flac"), "o.mkv", FLAC)
    assert p.audio_maps == [1]
    with pytest.raises(ExportError, match="does not exist"):
        resolve(ExportSettings(audio_tracks=[3]), "o.mkv", FLAC)


def test_presets_roundtrip(tmp_path, monkeypatch):
    from epochcolor.export.presets import BUILTIN, load_preset, save_preset

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for name in BUILTIN:
        s, warn = load_preset(name)
        ext = s.container
        resolve(s, f"o{ext}", [AudioStream(0, "aac", 2)])  # every built-in resolves
        assert not warn
    s, _ = load_preset("share-h264")
    s.rf = 23
    path = save_preset("mine", s, "test")
    s2, _ = load_preset("mine")
    assert s2.rf == 23 and s2.codec == "h264"
    s3, _ = load_preset(str(path))
    assert s3 == s2


# ------------------------------------------------------- real encodes


def frames(n=6, w=96, h=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    out = []
    for t in range(n):
        img = np.stack([(x + t * 3) / w, y / h, 0.5 + 0.3 * np.sin(x / 9.0 + t)], -1)
        img = img + rng.normal(0, 0.02, img.shape)
        out.append((np.clip(img, 0, 1) * 65535).astype(np.uint16))
    return out


def write(tmp_path, s, name, audio_src=None, audio=()):
    from epochcolor.export.writer import VideoWriter

    out = tmp_path / name
    plan = resolve(s, out, list(audio))
    w = VideoWriter(plan, out, 96, 64, Fraction(24), audio_src, 0.0, tmp_path / "work")
    for f in frames():
        w.write(f)
    w.close()
    return out, plan


def decode(path, fmt):
    import av

    with av.open(str(path)) as c:
        return [f.to_ndarray(format=fmt) for f in c.decode(video=0)]


def stream_info(path):
    import av

    with av.open(str(path)) as c:
        v = c.streams.video[0].codec_context
        return v.name, v.pix_fmt, [a.codec_context.name for a in c.streams.audio]


@needs_ffmpeg
@pytest.mark.parametrize("s,name,codec,pix", [
    (ExportSettings(encoder="software", rf=24, speed="ultrafast"), "a.mkv", "hevc", "yuv420p10le"),
    (ExportSettings(encoder="software", rf=24, chroma="422"), "b.mkv", "hevc", "yuv422p10le"),
    (ExportSettings(codec="h264", encoder="software", bits=8, rf=20), "c.mp4", "h264", "yuv420p"),
    (ExportSettings(codec="h264", encoder="software", bits=10, chroma="444", rf=20), "d.mkv", "h264", "yuv444p10le"),
    (ExportSettings(codec="av1", encoder="software", rf=40, speed="12"), "e.mp4", "libdav1d", "yuv420p10le"),
    (ExportSettings(codec="prores"), "f.mov", "prores", "yuv422p10le"),
    # ProRes 4444 is a 12-bit format inside, whatever goes in
    (ExportSettings(codec="prores", chroma="444"), "f2.mov", "prores", "yuv444p12le"),
    (ExportSettings(codec="ffv1", bits=16, chroma="444"), "g.mkv", "ffv1", "yuv444p16le"),
])
def test_software_encoders(tmp_path, s, name, codec, pix):
    out, plan = write(tmp_path, s, name)
    vcodec, vpix, _ = stream_info(out)
    assert vcodec in (codec, "av1") and vpix == pix, (vcodec, vpix)
    assert len(decode(out, "rgb48le")) == 6


@needs_ffmpeg
@pytest.mark.parametrize("s", [
    ExportSettings(codec="h265", encoder="software", rf=0, speed="ultrafast"),
    ExportSettings(codec="h264", encoder="software", bits=10, rf=0, speed="ultrafast"),
])
def test_rf0_is_bit_exact(tmp_path, s):
    """RF 0 must decode to the same YUV as a lossless FFV1 encode of the
    same frames through the same colour conversion."""
    out, _ = write(tmp_path, s, "lossy.mkv")
    ref, _ = write(tmp_path, ExportSettings(codec="ffv1", bits=10, chroma="420"), "ref.mkv")
    a = decode(out, "yuv420p10le")
    b = decode(ref, "yuv420p10le")
    assert len(a) == len(b) == 6
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


@needs_ffmpeg
@pytest.mark.parametrize("codec", ["h265", "h264"])
def test_two_pass(tmp_path, codec):
    s = ExportSettings(codec=codec, encoder="software", bits=8 if codec == "h264" else 10,
                       bitrate=300, rf=None, two_pass=True, speed="ultrafast")
    out, plan = write(tmp_path, s, "tp.mkv")
    assert len(decode(out, "rgb48le")) == 6
    assert not list((tmp_path / "work").glob("twopass*"))  # intermediate cleaned up


@needs_ffmpeg
def test_audio_tracks_copied_and_converted(tmp_path):
    src = tmp_path / "src.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=660:duration=1",
                    "-map", "0", "-map", "1", "-c:a:0", "flac", "-c:a:1", "pcm_s16le",
                    str(src)], check=True)
    aud = [AudioStream(0, "flac", 1), AudioStream(1, "pcm_s16le", 1)]
    out, _ = write(tmp_path, ExportSettings(encoder="software", speed="ultrafast"), "m.mkv", src, aud)
    assert stream_info(out)[2] == ["flac", "pcm_s16le"]
    out, _ = write(tmp_path, ExportSettings(encoder="software", speed="ultrafast",
                                            audio_fallback="aac"), "m.mp4", src, aud)
    assert stream_info(out)[2] == ["flac", "aac"]
    out, _ = write(tmp_path, ExportSettings(encoder="software", speed="ultrafast",
                                            audio_tracks=[2], audio_codec="opus"), "o.mkv", src, aud)
    assert stream_info(out)[2] == ["opus"]
