import shutil

import numpy as np
import pytest

av = pytest.importorskip("av")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def test_edited_timeline_renders_with_joined_audio(tmp_path, monkeypatch):
    from test_video import JitterModel, _make_clip

    from epochcolor.export.plan import ExportError, ExportSettings, resolve
    from epochcolor.media import import_clip
    from epochcolor.project import Project
    from epochcolor.timeline_export import build_export

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    a, b = tmp_path / "a.mkv", tmp_path / "b.mkv"
    _make_clip(a, n=24)
    _make_clip(b, n=24)
    p = Project()
    for f in (a, b):
        c, _ = import_clip(f)
        p.add_clip(c)
    p.split(10)
    p.delete(0)  # drop the first 10 frames of clip a
    p.trim(1, src_out=12)  # keep 12 frames of clip b
    assert p.duration == 14 + 12

    model = JitterModel()
    out = tmp_path / "edit.mkv"
    with pytest.raises(ExportError, match="cut or joined"):
        build_export(p, model, ExportSettings(encoder="software", speed="ultrafast"), out)
    job = build_export(p, model, ExportSettings(encoder="software", speed="ultrafast",
                                                audio_fallback="flac"), out)
    job.run(model, quiet=True)
    with av.open(str(out)) as c:
        assert len(c.streams.audio) == 1
        n = sum(1 for _ in c.decode(video=0))
    with av.open(str(out)) as c:
        samples = sum(f.samples for f in c.decode(audio=0))
        rate = c.streams.audio[0].rate
    assert n == 26
    # joined audio is exactly as long as the joined video
    assert abs(samples / rate - 26 / 24) < 0.03, samples / rate


def test_whole_clip_passes_audio_through(tmp_path, monkeypatch):
    from test_video import JitterModel, _make_clip

    from epochcolor.export.plan import ExportSettings
    from epochcolor.media import import_clip
    from epochcolor.project import Project
    from epochcolor.timeline_export import build_export

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    a = tmp_path / "a.mkv"
    _make_clip(a, n=24)
    p = Project()
    p.add_clip(import_clip(a)[0])
    m = JitterModel()
    job = build_export(p, m, ExportSettings(encoder="software", speed="ultrafast"), tmp_path / "o.mkv")
    assert "copied" in job.plan.describe()
    job.run(m, quiet=True)
    with av.open(str(tmp_path / "o.mkv")) as c:
        assert c.streams.audio[0].codec_context.name == "flac"
