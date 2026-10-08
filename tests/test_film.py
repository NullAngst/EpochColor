import numpy as np
import pytest

from epochcolor import film
from epochcolor.color import linear_to_srgb, srgb_to_linear


def negative_of(pos):
    """A plausible B&W negative of a positive grey image (0..1 encoded):
    transmittance falls with density, density rises with brightness."""
    D = 0.15 + 1.6 * np.asarray(pos, np.float32)  # base fog 0.15
    T = 10.0 ** (-D)
    neg = linear_to_srgb(T / 10.0 ** -0.15 * 0.9)  # base lets 90% through
    neg[:6] = linear_to_srgb(np.float32(0.9))  # clear rebate along the top
    return neg.astype(np.float32)


def test_negative_round_trip_ranks_and_spreads():
    rng = np.random.default_rng(0)
    pos = np.clip(np.linspace(0.05, 0.95, 200)[None].repeat(120, 0) + rng.normal(0, 0.01, (120, 200)), 0, 1)
    neg = negative_of(pos)
    p = film.measure_negative([neg])
    assert p["base"] == pytest.approx(float(linear_to_srgb(np.float32(0.9))), abs=0.01)
    out = film.invert((neg * 65535).astype(np.uint16), p)
    body = out[10:]
    # brighter in the positive is brighter out, and the range gets used
    assert np.corrcoef(body.ravel(), pos[10:].ravel())[0, 1] > 0.98
    assert body.min() < 0.05 and body.max() > 0.95
    # 8-bit input goes through the same table
    o8 = film.invert((neg * 255).astype(np.uint8), p)
    assert np.abs(o8[10:] - body).mean() < 0.02


def test_deflicker_removes_pulsing_keeps_fade():
    rng = np.random.default_rng(1)
    base = rng.uniform(20, 80, (40, 60)).astype(np.float32)
    frames = []
    for t in range(48):
        fade = 1.0 - 0.4 * t / 47  # a slow fade down
        flick = 1.0 + 0.12 * (1 if t % 2 else -1)  # pulsing every frame
        frames.append(np.clip(base * fade * flick, 0, 100))
    stats = [film.frame_stats(f) for f in frames]
    tgt = film.deflicker_targets(stats, [(0, 48)], fps=24)
    fixed = [film.deflicker(f, s, g) for f, s, g in zip(frames, stats, tgt)]
    means_before = np.array([f.mean() for f in frames])
    means_after = np.array([f.mean() for f in fixed])
    jitter = lambda m: np.abs(np.diff(m, 2)).mean()
    assert jitter(means_after) < 0.2 * jitter(means_before)
    assert means_after[-1] < 0.75 * means_after[0], "the fade is still there"


def test_dust_speck_found_and_filled():
    from epochcolor.video.temporal import Flow

    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:96, 0:160]
    scene = lambda t: (50 + 20 * np.sin((xx - 2 * t) / 9.0) + 10 * np.cos(yy / 7.0)).astype(np.float32)
    prev, cur, nxt = scene(0), scene(1), scene(2)
    dirty = cur.copy()
    dirty[40:44, 70:74] = 98.0  # a white speck
    dirty[10:12, 20:22] = 3.0  # a black one
    out, m = film.remove_dust(dirty, prev, nxt, Flow("medium"), thr=10)
    assert m[41, 71] and m[10, 20]
    assert abs(out[41, 71] - cur[41, 71]) < 6 and abs(out[10, 20] - cur[10, 20]) < 6
    assert m.mean() < 0.01, "the rest of the frame is untouched"


def test_motion_is_not_dust():
    from epochcolor.video.temporal import Flow

    prev = np.full((96, 160), 40, np.float32)
    cur, nxt = prev.copy(), prev.copy()
    cur[30:60, 60:90] = 90  # something appears and stays: not dirt
    nxt[30:60, 60:90] = 90
    _, m = film.remove_dust(cur, prev, nxt, Flow("medium"), thr=10)
    assert not m[45, 75]


def test_regrain_repeats_and_scales():
    rgb = np.full((108, 192, 3), 0.5, np.float32)
    a = film.regrain(rgb, 4, size=1.0, seed=7)
    b = film.regrain(rgb, 4, size=1.0, seed=7)
    assert np.array_equal(a, b)
    assert 0.025 < float((a - rgb).std()) < 0.05
    assert np.allclose(a[..., 0], a[..., 1]), "mono by default"
    c = film.regrain(rgb, 4, size=1.0, chroma=1.0, seed=7)
    assert not np.allclose(c[..., 0], c[..., 1])
    assert film.regrain(rgb, 0) is rgb
