import shutil

import numpy as np
import pytest

from epochcolor.hintpaint import dabs, hints_in, rasterize, strokes_key

BLUE = [40, 70, 200]


def test_rasterize_and_key():
    st = [{"rgb": BLUE, "radius": 0.05, "points": [[0.2, 0.5], [0.8, 0.5]]},
          {"neutral": True, "radius": 0.05, "points": [[0.5, 0.2]]}]
    h = rasterize(st, 200, 100)
    assert h.mask[50, 40] == 1 and h.mask[50, 160] == 1 and h.mask[90, 40] == 0
    assert h.ab[50, 100, 1] < -30  # blue: negative b
    assert h.mask[20, 100] == 1 and np.allclose(h.ab[20, 100], 0)  # neutral dab
    assert strokes_key(st) == strokes_key([dict(s) for s in st])
    st2 = [dict(st[0], points=[[0.2, 0.5], [0.81, 0.5]])]
    assert strokes_key(st2) != strokes_key(st[:1])
    assert set(hints_in({"3": st, "9": st, "12": []}, 0, 10)) == {3, 9}
    assert dabs([[0.1, 0.2]], BLUE)[0]["points"] == [[0.1, 0.2]]


class CountingJitter:
    """test_video's model, flickering from frame to frame but giving the same
    answer for the same input, like a real network. Counts its runs."""

    def __init__(self):
        from test_video import JitterModel

        self.info = JitterModel.info
        self.calls = 0

    def predict(self, L, hints=None):
        from test_video import JitterModel

        self.calls += 1
        seed = int(np.round(L[::7, ::7] * 10).sum()) % (2 ** 31)
        return JitterModel(seed).predict(L, hints)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_video_hint_follows_the_object(tmp_path, monkeypatch):
    pytest.importorskip("av")
    from test_video import _make_clip

    from epochcolor.video.io import probe
    from epochcolor.video.pipeline import VideoSettings, analyze

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    clip = tmp_path / "c.mkv"
    _make_clip(clip, n=24)  # dark disc moving right 2 px a frame, 160x96
    info = probe(clip)
    s = VideoSettings(working_size=96, chroma_size=96)
    m = CountingJitter()
    rep = analyze(info, m, s, shots=[(0, 24)], quiet=True)
    before = np.load(rep.dir / "ab.npy")
    calls_model_pass = m.calls

    def disc(t):  # centre as fractions of the frame
        return [(40 + 2 * t) / 160, 48 / 96]

    hints = {5: [{"rgb": BLUE, "radius": 0.08, "points": [disc(5)]}]}
    rep = analyze(info, m, s, shots=[(0, 24)], quiet=True, hints=hints)
    after = np.load(rep.dir / "ab.npy").astype(np.float32)
    assert m.calls - calls_model_pass == 1, "only the keyframe reruns, not the model pass"

    def inside(arr, t):
        h, w = arr.shape[1:3]
        x, y = disc(t)
        return arr[t, int(y * h) - 3:int(y * h) + 3, int(x * w) - 3:int(x * w) + 3].reshape(-1, 2).mean(0)

    from epochcolor.hintpaint import stroke_ab

    target = np.array(stroke_ab({"rgb": BLUE}))
    for t in (0, 5, 12, 20):
        was = inside(before, t)
        now = inside(after, t)
        assert was[0] > 25, "the model had it red"
        # blue now, before and after the keyframe: much closer to the target than it was
        assert np.linalg.norm(now - target) < 0.25 * np.linalg.norm(was - target), (t, was, now, target)
    # the background is left alone
    bg_before = before[:, 5:15, 120:150].astype(np.float32).mean()
    bg_after = after[:, 5:15, 120:150].mean()
    assert abs(bg_before - bg_after) < 3

    # removing the hint goes back without a model pass
    calls = m.calls
    rep = analyze(info, m, s, shots=[(0, 24)], quiet=True, hints={})
    assert m.calls == calls
    assert inside(np.load(rep.dir / "ab.npy").astype(np.float32), 12)[0] > 25
