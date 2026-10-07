import numpy as np
import pytest

from epochcolor.color import lab_to_srgb, srgb_to_l, srgb_to_lab
from epochcolor.hints import Hints, empty_hints
from epochcolor.models import create
from epochcolor.pipeline import PhotoSettings, colorize_photo


def scene(h=240, w=360, grain=4.0, seed=0):
    """Grey test photo: light background, dark disc, mid square, plus grain."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    L = np.full((h, w), 75.0, np.float32)
    L[(y - 120) ** 2 + (x - 100) ** 2 < 60 ** 2] = 35.0
    L[60:180, 220:320] = 55.0
    L += rng.normal(0, grain, L.shape).astype(np.float32)
    L = np.clip(L, 0, 100)
    return lab_to_srgb(L, np.zeros((h, w, 2), np.float32)), L


def stroke_hints(h, w, y0, y1, x0, x1, ab):
    hint = empty_hints(h, w)
    hint.mask[y0:y1, x0:x1] = 1.0
    hint.ab[y0:y1, x0:x1] = ab
    return hint


def test_gray_roundtrip_keeps_l():
    _, L = scene()
    img = lab_to_srgb(L, np.zeros(L.shape + (2,), np.float32))
    assert np.abs(srgb_to_l(img) - L).max() < 0.01


def test_gamut_fit_keeps_l():
    L = np.linspace(2, 98, 200, dtype=np.float32)[None, :].repeat(4, 0)
    ab = np.full(L.shape + (2,), 120.0, np.float32)  # far out of gamut
    rgb = lab_to_srgb(L, ab)
    assert rgb.min() >= 0 and rgb.max() <= 1
    assert np.abs(srgb_to_lab(rgb)[..., 0] - L).max() < 0.05


def test_output_luma_is_source_luma():
    img, L = scene()
    h, w = L.shape
    hints = stroke_hints(h, w, 110, 130, 90, 110, (50.0, 30.0))
    rgb, _ = colorize_photo(img, create("hints"), PhotoSettings(working_size=120), hints)
    assert np.abs(srgb_to_lab(rgb)[..., 0] - srgb_to_l(img)).max() < 0.05


def test_hint_fills_object_and_stops_at_edge():
    img, _ = scene(grain=2.0)
    h, w = img.shape[:2]
    red = (50.0, 30.0)
    hints = stroke_hints(h, w, 115, 125, 95, 105, red)
    rgb, _ = colorize_photo(img, create("hints"), PhotoSettings(working_size=240), hints)
    ab = srgb_to_lab(rgb)[..., 1:]
    inside = ab[120, 60]  # far side of the disc, unpainted
    outside = ab[20, 20]  # background
    assert inside[0] > 30, inside
    assert np.hypot(*outside) < 0.4 * np.hypot(*inside), (inside, outside)


def test_grain_slider():
    img, _ = scene(grain=6.0)
    model = create("hints")
    full, _ = colorize_photo(img, model, PhotoSettings(grain=100, working_size=120))
    clean, _ = colorize_photo(img, model, PhotoSettings(grain=0, working_size=120))
    patch = (slice(10, 50), slice(10, 50))
    std_full = srgb_to_lab(full)[patch][..., 0].std()
    std_clean = srgb_to_lab(clean)[patch][..., 0].std()
    assert std_clean < 0.5 * std_full


def test_tinted_scan_is_reduced_to_luminance():
    img, L = scene()
    sepia = np.clip(img * np.array([1.0, 0.9, 0.75], np.float32), 0, 1)
    rgb, rep = colorize_photo(sepia, create("hints"), PhotoSettings(working_size=120))
    assert any("colour" in wmsg for wmsg in rep.warnings)
    assert np.hypot(*srgb_to_lab(rgb)[..., 1:].reshape(-1, 2).T).max() < 1.0


@pytest.mark.parametrize("name", ["eccv16", "siggraph17"])
def test_network_adapters_shapes(tmp_path, name):
    torch = pytest.importorskip("torch")
    from epochcolor.models import zhang

    builder = zhang.build_eccv16 if name == "eccv16" else zhang.build_siggraph17
    wpath = tmp_path / f"{name}.pth"
    torch.save(builder().state_dict(), wpath)
    model = create(name, str(wpath))
    model.load("cpu")
    _, L = scene(h=200, w=300)
    hints = stroke_hints(200, 300, 90, 110, 90, 110, (40.0, 20.0)) if name == "siggraph17" else None
    ab = model.predict(L, hints)
    assert ab.shape == (200, 300, 2) and ab.dtype == np.float32
    assert np.isfinite(ab).all()


def test_rejects_wrong_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from epochcolor.models import zhang

    wpath = tmp_path / "wrong.pth"
    torch.save(zhang.build_eccv16().state_dict(), wpath)
    model = create("siggraph17", str(wpath))
    with pytest.raises(RuntimeError, match="does not fit"):
        model.load("cpu")
