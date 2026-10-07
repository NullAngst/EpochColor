"""Scopes: waveform, RGB parade, vectorscope, histogram. Plain numpy, each
returns an RGB uint8 image ready to show."""

from __future__ import annotations

import numpy as np

H = 256


def _prep(rgb8: np.ndarray, width: int = 480) -> np.ndarray:
    import cv2

    h, w = rgb8.shape[:2]
    if w > width:
        rgb8 = cv2.resize(rgb8, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return rgb8


def _column_hist(chan: np.ndarray, cols: int) -> np.ndarray:
    """(256 levels x cols) count image of one 8-bit channel."""
    h, w = chan.shape
    xi = (np.arange(w) * cols // w)[None, :].repeat(h, 0)
    flat = xi.ravel() * 256 + chan.ravel().astype(np.int64)
    counts = np.bincount(flat, minlength=cols * 256).reshape(cols, 256).T
    return counts[::-1]  # white at the top


def _tone(counts: np.ndarray, gain: float = 6.0) -> np.ndarray:
    c = counts.astype(np.float32)
    ref = max(1.0, np.percentile(c[c > 0], 99) if (c > 0).any() else 1.0)
    return np.clip(np.log1p(c * gain / ref) / np.log1p(gain), 0, 1)


def _graticule(img: np.ndarray, levels=(0, 0.25, 0.5, 0.75, 1.0)) -> None:
    for lv in levels:
        y = int(round((1 - lv) * (img.shape[0] - 1)))
        img[y, :] = np.maximum(img[y, :], 60)


def waveform(rgb8: np.ndarray) -> np.ndarray:
    x = _prep(rgb8)
    luma = (x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722).astype(np.uint8)
    t = _tone(_column_hist(luma, x.shape[1]))
    img = np.zeros((H, x.shape[1], 3), np.uint8)
    img[..., 1] = (t * 230).astype(np.uint8)
    img[..., 0] = (t * 160).astype(np.uint8)
    img[..., 2] = (t * 130).astype(np.uint8)
    _graticule(img)
    return img


def parade(rgb8: np.ndarray) -> np.ndarray:
    x = _prep(rgb8, 360)
    cols = x.shape[1]
    out = np.zeros((H, cols * 3 + 8, 3), np.uint8)
    for c in range(3):
        t = _tone(_column_hist(x[..., c], cols))
        part = np.zeros((H, cols, 3), np.uint8)
        part[..., c] = (t * 255).astype(np.uint8)
        part[..., [i for i in range(3) if i != c]] = (t * 70).astype(np.uint8)[..., None]
        out[:, c * (cols + 4): c * (cols + 4) + cols] = part
    _graticule(out)
    return out


def vectorscope(rgb8: np.ndarray, size: int = 256) -> np.ndarray:
    x = _prep(rgb8, 320).astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748
    gain = 1.0  # BT.709 75% colour bars land well inside the circle
    u = np.clip(((cb * gain) + 0.5) * (size - 1), 0, size - 1).astype(np.int32)
    v = np.clip(((-cr * gain) + 0.5) * (size - 1), 0, size - 1).astype(np.int32)
    counts = np.bincount((v * size + u).ravel(), minlength=size * size).reshape(size, size)
    t = _tone(counts, 12.0)
    img = np.zeros((size, size, 3), np.uint8)
    img[..., :] = (t * 235).astype(np.uint8)[..., None]
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(xx - c, yy - c)
    ring = (np.abs(d - c * 0.98) < 0.7) | (np.abs(d - c * 0.5) < 0.5)
    img[ring] = np.maximum(img[ring], 55)
    img[c, :] = np.maximum(img[c, :], 40)
    img[:, c] = np.maximum(img[:, c], 40)
    # primary and secondary targets
    for col in ((1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1)):
        rr, gg, bb = (0.75 * k for k in col)
        yv = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
        uu = int(((bb - yv) / 1.8556 * gain + 0.5) * (size - 1))
        vv = int(((-(rr - yv) / 1.5748) * gain + 0.5) * (size - 1))
        if not (3 <= uu < size - 4 and 3 <= vv < size - 4):
            continue
        box = (slice(vv - 3, vv + 4), slice(uu - 3, uu + 4))
        img[box] = np.maximum(img[box], np.array([200 * k + 40 for k in col], np.uint8))
    # skin tone line, about 123 degrees from the cb axis
    for t_ in np.linspace(0, 0.48, 120):
        ang = np.deg2rad(123)
        px = int(c + np.cos(ang) * t_ * size)
        py = int(c - np.sin(ang) * t_ * size)
        if 0 <= px < size and 0 <= py < size:
            img[py, px] = np.maximum(img[py, px], 90)
    return img


def histogram(rgb8: np.ndarray, width: int = 256) -> np.ndarray:
    x = _prep(rgb8)
    img = np.zeros((160, width, 3), np.uint16)
    for c in range(3):
        hist = np.bincount(x[..., c].ravel(), minlength=256).astype(np.float32)
        hist = hist / max(1.0, np.percentile(hist, 99.5))
        cols = np.clip(hist[np.linspace(0, 255, width).astype(int)], 0, 1)
        heights = (cols * 159).astype(int)
        for i, hgt in enumerate(heights):
            img[160 - hgt:, i, c] += 200
    img = np.clip(img, 0, 255).astype(np.uint8)
    img[:, [width // 4, width // 2, 3 * width // 4]] = np.maximum(img[:, [width // 4, width // 2, 3 * width // 4]], 45)
    return img


SCOPES = {"Waveform": waveform, "RGB parade": parade, "Vectorscope": vectorscope, "Histogram": histogram}
