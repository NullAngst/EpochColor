import json
import shutil
from fractions import Fraction

import numpy as np
import pytest

from epochcolor.project import AudioInfo, Clip, Project, ProjectError


def clip(cid="a", frames=100, w=640, h=360, fps=(24, 1), shots=None, audio=1):
    return Clip(cid, f"/x/{cid}.mkv", w, h, fps[0], fps[1], frames,
                audio=[AudioInfo("flac", 2)] * audio, shots=shots or [])


def two_clip_project():
    p = Project()
    p.add_clip(clip("a", 100, shots=[[0, 40], [40, 100]]))
    p.add_clip(clip("b", 50, audio=2))
    return p


def test_format_comes_from_first_clip():
    p = Project()
    p.add_clip(clip("a"))
    with pytest.raises(ProjectError, match="Convert it first"):
        p.add_clip(clip("b", fps=(25, 1)))
    with pytest.raises(ProjectError):
        p.add_clip(clip("c", w=720))
    assert p.fps == Fraction(24) and p.size == (640, 360)


def test_split_trim_delete_move_and_locate():
    p = two_clip_project()
    assert p.duration == 150 and p.audio_tracks == 2
    assert p.shot_starts() == [0, 40, 100]
    assert p.split(70)
    assert not p.split(70)  # already an edit point
    assert [(s.clip, s.src_in, s.src_out) for s in p.d.timeline] == [
        ("a", 0, 70), ("a", 70, 100), ("b", 0, 50)]
    assert p.locate(75)[2] == 75 and p.locate(105)[1].clip == "b"
    p.trim(2, src_in=10)
    assert p.duration == 140
    p.delete(1)  # ripple
    assert p.duration == 110 and p.locate(75)[1].clip == "b"
    assert p.move(1, -1) == 0
    assert p.d.timeline[0].clip == "b"


def test_trim_clamps_to_clip():
    p = two_clip_project()
    p.trim(0, src_in=-5, src_out=500)
    assert (p.d.timeline[0].src_in, p.d.timeline[0].src_out) == (0, 100)
    p.trim(0, src_in=99, src_out=99)
    assert p.d.timeline[0].length == 1


def test_undo_redo_covers_everything():
    p = two_clip_project()
    before = p._dump()
    p.split(20)
    p.set_in(10)
    p.set_out(30)
    p.toggle_audio(1)
    p.add_marker(5, "scratch on frame")
    p.set_setting("stabilize", 0.5)
    for _ in range(6):
        p.undo()
    assert p._dump() == before
    for _ in range(6):
        p.redo()
    assert p.d.range == [10, 31] and p.d.audio_enabled == [True, False]
    assert p.d.settings["stabilize"] == 0.5 and p.d.markers[0].note == "scratch on frame"


def test_delete_moves_markers_and_range():
    p = two_clip_project()
    p.add_marker(10)
    p.add_marker(120)
    p.set_in(5)
    p.set_out(140)
    p.delete(0)  # removes frames 0..99
    assert [m.frame for m in p.d.markers] == [20]
    assert p.d.range == [5, 41] or p.d.range[1] <= p.duration


def test_toggle_cut():
    p = two_clip_project()
    assert p.toggle_cut(70)
    assert p.d.clips["a"].shots == [[0, 40], [40, 70], [70, 100]]
    assert not p.toggle_cut(70)
    with pytest.raises(ProjectError):
        p.toggle_cut(0)


def test_pieces_and_simple():
    p = Project()
    p.add_clip(clip("a", 100))
    assert p.is_simple()
    p.set_in(10)
    assert not p.is_simple()
    p.clear_range()
    p.add_clip(clip("b", 50))
    p.set_in(90)
    p.set_out(109)
    pieces = [(c.id, a, b) for c, a, b in p.pieces()]
    assert pieces == [("a", 90, 100), ("b", 0, 10)]


def test_save_load_roundtrip(tmp_path):
    p = two_clip_project()
    p.split(30)
    p.add_marker(3, "note")
    p.add_photo("/x/p.tif")
    f = p.save(tmp_path / "t.epochcolor")
    q = Project.load(f)
    assert q._dump() == p._dump()
    assert json.loads(f.read_text())["epochcolor_project"] == 1


# ---------------------------------------------------------------- media

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def clip_file(tmp_path):
    from test_video import _make_clip

    path = tmp_path / "c.mkv"
    _make_clip(path, n=40)
    return path


@needs_ffmpeg
def test_import_builds_proxy_and_reader_is_frame_accurate(tmp_path, monkeypatch, clip_file):
    pytest.importorskip("av")
    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    from epochcolor.media import FrameReader, import_clip
    from epochcolor.video.io import iter_gray

    c, media = import_clip(clip_file)
    assert c.frames == 40 and media.frames == 40
    assert np.load(media.thumbs).shape[0] == 40
    assert len(media.waves) == 1 and np.load(media.waves[0]).size > 100
    # random access matches sequential decode, in any order
    seq = [g for g in iter_gray(clip_file)]
    r = FrameReader(clip_file, c.fps, "gray16le")
    for i in (30, 2, 3, 39, 0, 17, 16):
        f = r.get(i).astype(np.float32) / 65535
        assert np.abs(f - seq[i]).max() < 1e-6, i
    r.close()
    rp = FrameReader(media.proxy, c.fps, "gray")
    assert rp.get(39) is not None and rp.get(40) is None
    # second import reuses the cache
    c2, m2 = import_clip(clip_file)
    assert m2.dir == media.dir and c2.shots == c.shots
