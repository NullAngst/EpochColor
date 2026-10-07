import time

import numpy as np
import pytest

from epochcolor import grade as G
from epochcolor import scopes


def img(h=90, w=160, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    base = np.stack([x, y, 1 - x], -1).astype(np.float32)
    return np.clip(base * 0.8 + rng.random((h, w, 3), np.float32) * 0.2, 0, 1)


def test_identity_is_exact():
    x = img()
    assert np.array_equal(G.apply(x, G.default_grade()), x)
    assert np.array_equal(G.apply(x, None), x)


def test_primaries_move_the_right_way():
    x = np.full((4, 4, 3), 0.5, np.float32)
    g = G.default_grade()
    g["gain"] = [0.2, 0, 0]
    assert G.apply(x, g)[0, 0, 0] > 0.55 and abs(G.apply(x, g)[0, 0, 1] - 0.5) < 1e-6
    g = G.default_grade()
    g["gamma_m"] = 0.2
    assert G.apply(x, g)[0, 0, 0] > 0.55
    g = G.default_grade()
    g["lift_m"] = 0.1
    y = G.apply(np.zeros((1, 1, 3), np.float32), g)
    assert abs(y[0, 0, 0] - 0.1) < 1e-6
    g = G.default_grade()
    g["contrast"] = 1.5
    assert G.apply(np.full((1, 1, 3), 0.8, np.float32), g)[0, 0, 0] > 0.8


def test_nothing_clips_until_the_end():
    x = np.full((1, 1, 3), 0.6, np.float32)
    up = G.default_grade()
    up["gain_m"] = 1.0  # doubles, past 1.0
    down = G.default_grade()
    down["gain_m"] = -0.5
    assert abs(G.apply(G.apply(x, up), down)[0, 0, 0] - 0.6) < 1e-5


def test_white_balance_neutralizes_a_cast():
    cast = np.array([0.62, 0.55, 0.42], np.float32)
    t, n = G.wb_from_neutral(tuple(cast))
    g = G.default_grade()
    g["temperature"], g["tint"] = t, n
    out = G.apply(cast[None, None], g)[0, 0]
    assert out.max() - out.min() < 0.01, out


def test_curves_and_hue_curves():
    g = G.default_grade()
    g["curves"]["master"] = [[0, 0], [0.5, 0.7], [1, 1]]
    assert G.apply(np.full((1, 1, 3), 0.5, np.float32), g)[0, 0, 0] == pytest.approx(0.7, abs=0.01)
    red = np.array([[[0.8, 0.2, 0.2]]], np.float32)
    g = G.default_grade()
    g["hue_sat"] = [[0.0, 0.0], [0.5, 1.0]]  # desaturate reds only
    out = G.apply(red, g)[0, 0]
    assert out.max() - out.min() < 0.02
    blue = np.array([[[0.2, 0.2, 0.8]]], np.float32)
    out = G.apply(blue, g)[0, 0]
    assert out[2] - out[0] > 0.3  # blue untouched


def test_qualifier_and_mask():
    x = np.zeros((20, 40, 3), np.float32)
    x[:, :20] = [0.7, 0.2, 0.2]  # red half
    x[:, 20:] = [0.2, 0.2, 0.7]  # blue half
    g = G.default_grade()
    g["qualifier"].update(enabled=True, hue=0.0, hue_width=0.1, d_sat=0.0)
    m = []
    out = G.apply(x, g, matte_out=m)
    assert m[0][:, :20].mean() > 0.95 and m[0][:, 20:].mean() < 0.05
    assert np.ptp(out[5, 5]) < 0.02 and np.ptp(out[5, 30]) > 0.4
    g = G.default_grade()
    g["mask"].update(shape="ellipse", cx=0.25, cy=0.5, w=0.3, h=0.6)
    g["qualifier"]["d_lum"] = 0.5
    out = G.apply(x, g)
    assert out[10, 10, 0] > 0.9 and abs(out[10, 30, 2] - 0.7) < 1e-3


def test_keyframes():
    a = G.default_grade()
    b = G.default_grade()
    b["saturation"] = 2.0
    track = {"keys": [{"frame": 0, "ease": "linear", "grade": a}, {"frame": 10, "ease": "linear", "grade": b}]}
    assert G.evaluate(track, 5)["saturation"] == pytest.approx(1.5)
    track["keys"][0]["ease"] = "ease"
    assert G.evaluate(track, 2)["saturation"] < 1.2  # eased start is slow
    assert G.evaluate(track, 50)["saturation"] == 2.0
    # editing between two keys adds one
    c = G.default_grade()
    c["vibrance"] = 0.3
    G.set_value(track, 7, c)
    assert [k["frame"] for k in track["keys"]] == [0, 7, 10]


def test_lut_roundtrip(tmp_path):
    g = G.default_grade()
    g["gain"] = [0.15, 0, -0.1]
    g["saturation"] = 1.3
    g["curves"]["master"] = [[0, 0.03], [0.5, 0.55], [1, 0.97]]
    left = G.export_cube(g, tmp_path / "g.cube")
    assert left == []
    x = img()
    direct = np.clip(G.apply(x, g), 0, 1)
    lut = G.default_grade()
    lut["lut"] = str(tmp_path / "g.cube")
    via = G.apply(x, lut)
    assert np.abs(direct - via).max() < 0.02


def test_lut_skips_the_mask(tmp_path):
    g = G.default_grade()
    g["mask"]["shape"] = "ellipse"
    assert G.export_cube(g, tmp_path / "m.cube")


def test_gradebook_finds_the_shot():
    track = G.new_track()
    track["keys"][0]["grade"]["saturation"] = 0.5
    book = G.GradeBook({"c": {"40": track}}, {"c": [[0, 40], [40, 90]]})
    assert book.at("c", 10) is None
    assert book.at("c", 60)["saturation"] == 0.5


def test_speed_at_preview_size():
    x = img(360, 640)
    g = G.default_grade()
    g["gain"] = [0.1, 0, 0]
    g["saturation"] = 1.2
    g["curves"]["r"] = [[0, 0], [0.5, 0.6], [1, 1]]
    g["qualifier"]["enabled"] = True
    G.apply(x, g)
    t = time.perf_counter()
    G.apply(x, g)
    assert time.perf_counter() - t < 0.25


def test_scopes_shapes():
    x = (img(180, 320) * 255).astype(np.uint8)
    for name, fn in scopes.SCOPES.items():
        out = fn(x)
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3, name
        assert out.max() > 0, name
