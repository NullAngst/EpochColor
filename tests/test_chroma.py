import json

import numpy as np

from epochcolor.chroma import ShotCasts, adjust, adjust_rgb, estimate_cast


def scene(rng, h=60, w=80):
    """A plausible picture's chroma: mostly near-neutral, a red coat, a blue sky."""
    ab = rng.normal(0, 2.0, (h, w, 2)).astype(np.float32)
    ab[10:30, 10:30] = (45, 25)  # red coat
    ab[:8] = (-5, -30)  # sky
    return ab


def test_orange_wash_is_found_and_taken_out():
    rng = np.random.default_rng(0)
    ab = scene(rng)
    baked = ab + np.array([12.0, 30.0], np.float32)  # everything pushed orange
    bias = estimate_cast(baked)
    assert np.allclose(bias, (12, 30), atol=3)
    fixed = adjust(baked, cast=1.0, bias=bias)
    assert np.abs(np.median(fixed[30:, 40:], axis=(0, 1))).max() < 3, "the neutral part is neutral again"
    assert fixed[20, 20, 0] > 35, "the coat stays red"


def test_honest_warm_scene_reads_as_little_cast():
    rng = np.random.default_rng(1)
    ab = scene(rng)
    ab[30:, :40] = (20, 50)  # a big warm sunset area, a quarter of the frame
    assert np.hypot(*estimate_cast(ab)) < 4


def test_adjust_leaves_input_alone_and_scales_after():
    ab = np.full((4, 4, 2), 10, np.int8)
    out = adjust(ab, saturation=2.0, cast=0.5, bias=np.array([4.0, 0.0]))
    assert out.dtype == np.float32 and ab[0, 0, 0] == 10
    assert out[0, 0, 0] == (10 - 2) * 2 and out[0, 0, 1] == 20


def test_adjust_rgb_keeps_luma():
    from epochcolor.color import srgb_to_lab

    rgb = np.dstack([np.full((8, 8), 0.8), np.full((8, 8), 0.55), np.full((8, 8), 0.3)]).astype(np.float32)
    out = adjust_rgb(rgb, cast=1.0)
    assert np.allclose(srgb_to_lab(out)[..., 0], srgb_to_lab(rgb)[..., 0], atol=0.5)
    assert np.abs(srgb_to_lab(out)[..., 1:]).max() < 2  # an all-orange card goes neutral


def test_shot_casts_per_shot(tmp_path):
    ab = np.zeros((20, 10, 10, 2), np.int8)
    ab[:10] = (0, 30)  # first shot washed yellow
    np.save(tmp_path / "ab.npy", ab)
    (tmp_path / "meta.json").write_text(json.dumps({"stab": {"0-10": "x", "10-20": "x"}}))
    sc = ShotCasts(tmp_path)
    assert np.allclose(sc.at(3), (0, 30)) and np.allclose(sc.at(15), (0, 0))
