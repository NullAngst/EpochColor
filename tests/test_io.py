import numpy as np
import pytest

from epochcolor.cli import main
from epochcolor.hints import load_hints
from epochcolor.imageio import load_image, save_image


def ramp(h=64, w=96):
    x = np.linspace(0, 1, w, dtype=np.float32)
    img = np.stack([np.tile(x, (h, 1))] * 3, axis=-1)
    img[..., 0] *= 0.9
    return img


# 8-bit output is dithered on purpose, so it can land one step off
@pytest.mark.parametrize("ext,bits,tol", [
    (".png", 8, 2 / 255 + 1e-6),
    (".png", 16, 1 / 65535 + 1e-6),
    (".tif", 16, 1 / 65535 + 1e-6),
    (".tif", 8, 2 / 255 + 1e-6),
])
def test_roundtrip(tmp_path, ext, bits, tol):
    img = ramp()
    p = tmp_path / f"x{ext}"
    save_image(p, img, bits=bits)
    back, meta = load_image(p)
    assert meta.bits == bits
    assert np.abs(back - img).max() <= tol


def test_16bit_png_has_srgb_profile(tmp_path):
    from PIL import Image

    p = tmp_path / "x.png"
    save_image(p, ramp(), bits=16)
    assert b"iCCP" in p.read_bytes()[:200]
    assert Image.open(p).info.get("icc_profile")


def test_jpeg_keeps_exif(tmp_path):
    from PIL import Image

    src = tmp_path / "src.jpg"
    im = Image.fromarray((ramp() * 255).astype(np.uint8))
    exif = im.getexif()
    exif[0x010F] = "TestCam"
    im.save(src, exif=exif.tobytes())
    img, meta = load_image(src)
    out = tmp_path / "out.jpg"
    save_image(out, img, meta=meta)
    assert Image.open(out).getexif().get(0x010F) == "TestCam"


def test_hints_rgba_and_painted_copy(tmp_path):
    import cv2

    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[10:20, 10:20] = (0, 0, 200, 255)  # red in BGRA
    p = tmp_path / "h.png"
    cv2.imwrite(str(p), rgba)
    h = load_hints(p, 40, 60)
    assert h.count == 100 and h.ab[15, 15, 0] > 40

    painted = np.full((40, 60, 3), 128, np.uint8)
    painted[0:5, 0:5] = (200, 0, 0)  # blue in BGR
    p2 = tmp_path / "h2.png"
    cv2.imwrite(str(p2), painted)
    h2 = load_hints(p2, 80, 120)  # scaled to a larger photo
    assert h2.mask.shape == (80, 120) and h2.count == 100 and h2.ab[2, 2, 1] < -40


def test_cli_photo_hints_model(tmp_path):
    import cv2

    src = tmp_path / "bw.png"
    cv2.imwrite(str(src), (ramp()[..., 0] * 255).astype(np.uint8))
    hint = np.zeros((64, 96, 4), np.uint8)
    hint[30:34, 40:60] = (0, 160, 0, 255)
    hp = tmp_path / "hint.png"
    cv2.imwrite(str(hp), hint)
    out = tmp_path / "out.png"
    assert main(["photo", str(src), "-m", "hints", "--hints", str(hp), "-o", str(out)]) == 0
    img, meta = load_image(out)
    assert img.shape == (64, 96, 3) and meta.bits == 8


def test_cli_missing_weights_is_clean_error(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    import cv2

    monkeypatch.setenv("EPOCHCOLOR_MODELS", str(tmp_path / "models"))
    src = tmp_path / "bw.png"
    cv2.imwrite(str(src), (ramp()[..., 0] * 255).astype(np.uint8))
    assert main(["photo", str(src), "-m", "siggraph17"]) == 1
    assert "epochcolor fetch siggraph17" in capsys.readouterr().err
