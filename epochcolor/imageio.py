"""Still image load and save.

Load returns float32 sRGB in 0..1 (HxW or HxWx3) plus metadata. Save takes
float32 sRGB HxWx3 and writes PNG, TIFF, JPEG or WebP with the sRGB profile
embedded.
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAW_EXTS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng", ".raf",
    ".orf", ".rw2", ".pef", ".srw", ".3fr", ".iiq", ".erf", ".kdc", ".mrw",
    ".x3f", ".raw", ".rwl",
}
PIL_EXTS = {".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif"}
TIFF_EXTS = {".tif", ".tiff"}
PNG_EXTS = {".png"}
SUPPORTED_IN = RAW_EXTS | PIL_EXTS | TIFF_EXTS | PNG_EXTS


@dataclass
class ImageMeta:
    bits: int = 8
    exif: bytes | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class RawDevelop:
    exposure_ev: float = 0.0
    camera_wb: bool = True
    highlight_recovery: bool = True


def _to_float(arr: np.ndarray) -> tuple[np.ndarray, int]:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0, 8
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0, 16
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0), 32
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float32) / float(info.max), info.bits
    raise ValueError(f"unsupported pixel type {arr.dtype}")


def _drop_alpha(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 2:  # gray + alpha
        return arr[..., 0]
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    return arr


def _load_pil(path: Path, meta: ImageMeta) -> np.ndarray:
    from PIL import Image, ImageCms, ImageOps

    if path.suffix.lower() in {".heic", ".heif"}:
        try:
            import pillow_heif  # type: ignore

            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise RuntimeError("HEIC needs pillow-heif: pip install pillow-heif") from e

    im = Image.open(path)
    exif = im.getexif()
    im = ImageOps.exif_transpose(im)
    if exif:
        exif[0x0112] = 1  # orientation already applied
        meta.exif = exif.tobytes()

    icc = im.info.get("icc_profile")
    if im.mode not in ("L", "RGB"):
        im = im.convert("RGB" if im.mode in ("RGBA", "P", "CMYK", "YCbCr", "LA") else "L")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
        except Exception as e:  # broken profiles are common in old scans
            meta.notes.append(f"embedded colour profile ignored: {e}")
    return np.asarray(im)


def _load_png(path: Path) -> np.ndarray:
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"could not read {path}")
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.shape[2] == 3:
            arr = arr[..., ::-1]
    return np.ascontiguousarray(arr)


def _load_tiff(path: Path) -> np.ndarray:
    import tifffile

    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)  # planar to interleaved
    return _drop_alpha(arr)


def _load_raw(path: Path, dev: RawDevelop) -> np.ndarray:
    try:
        import rawpy
    except ImportError as e:
        raise RuntimeError("camera RAW needs rawpy: pip install rawpy") from e

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=dev.camera_wb,
            use_auto_wb=not dev.camera_wb,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            no_auto_bright=True,
            exp_shift=float(2.0 ** dev.exposure_ev),
            exp_preserve_highlights=0.8,
            highlight_mode=rawpy.HighlightMode.Blend if dev.highlight_recovery else rawpy.HighlightMode.Clip,
        )
    return rgb


def load_image(path: str | Path, raw_develop: RawDevelop | None = None) -> tuple[np.ndarray, ImageMeta]:
    path = Path(path)
    ext = path.suffix.lower()
    meta = ImageMeta(source=str(path))
    if ext in RAW_EXTS:
        arr = _load_raw(path, raw_develop or RawDevelop())
    elif ext in TIFF_EXTS:
        arr = _load_tiff(path)
    elif ext in PNG_EXTS:
        arr = _load_png(path)
    elif ext in PIL_EXTS:
        arr = _load_pil(path, meta)
    else:
        raise RuntimeError(f"unsupported input type {ext}")
    img, bits = _to_float(arr)
    meta.bits = bits
    return img, meta


# ---------------------------------------------------------------- saving

def srgb_icc() -> bytes:
    from PIL import ImageCms

    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def quantize(img: np.ndarray, bits: int, dither: bool = True, seed: int = 0) -> np.ndarray:
    """float 0..1 to uint8/uint16. 8-bit output gets triangular dither so
    smooth colour gradients do not band."""
    maxv = 255.0 if bits == 8 else 65535.0
    x = np.clip(img, 0.0, 1.0) * maxv
    if dither and bits == 8:
        rng = np.random.default_rng(seed)
        x = x + (rng.random(x.shape, dtype=np.float32) - rng.random(x.shape, dtype=np.float32))
    x = np.clip(np.rint(x), 0, maxv)
    return x.astype(np.uint8 if bits == 8 else np.uint16)


def _png_insert_chunk(png: bytes, ctype: bytes, data: bytes) -> bytes:
    """Insert a chunk right after IHDR."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.unpack(">I", png[8:12])[0]
    cut = 8 + 12 + ihdr_len
    chunk = struct.pack(">I", len(data)) + ctype + data
    chunk += struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
    return png[:cut] + chunk + png[cut:]


def save_image(
    path: str | Path,
    img: np.ndarray,
    bits: int | None = None,
    quality: int = 95,
    meta: ImageMeta | None = None,
) -> None:
    from PIL import Image

    path = Path(path)
    ext = path.suffix.lower()
    exif = meta.exif if meta else None
    icc = srgb_icc()
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    if ext in {".jpg", ".jpeg", ".webp"}:
        if bits not in (None, 8):
            raise ValueError(f"{ext} is 8-bit only")
        im = Image.fromarray(quantize(img, 8), "RGB")
        kw = {"quality": int(quality), "icc_profile": icc}
        if exif:
            kw["exif"] = exif
        if ext == ".webp":
            kw["method"] = 6
        else:
            kw["subsampling"] = 0 if quality >= 90 else 2
        im.save(path, **kw)
    elif ext == ".png":
        bits = bits or 8
        if bits == 8:
            im = Image.fromarray(quantize(img, 8), "RGB")
            kw = {"icc_profile": icc}
            if exif:
                kw["exif"] = exif
            im.save(path, **kw)
        elif bits == 16:
            import cv2

            ok, buf = cv2.imencode(".png", quantize(img, 16)[..., ::-1])
            if not ok:
                raise RuntimeError("PNG encode failed")
            iccp = b"sRGB\x00\x00" + zlib.compress(icc)
            path.write_bytes(_png_insert_chunk(buf.tobytes(), b"iCCP", iccp))
        else:
            raise ValueError("PNG takes 8 or 16 bits")
    elif ext in TIFF_EXTS:
        import tifffile

        bits = bits or 16
        if bits not in (8, 16):
            raise ValueError("TIFF takes 8 or 16 bits")
        tifffile.imwrite(
            str(path),
            quantize(img, bits),
            photometric="rgb",
            compression="zlib",
            iccprofile=icc,
        )
    else:
        raise RuntimeError(f"unsupported output type {ext}")
