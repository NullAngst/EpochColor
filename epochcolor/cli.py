"""Command line front end for milestone 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__

OUT_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".webp"}


def _err(msg: str) -> None:
    print(f"epochcolor: {msg}", file=sys.stderr)


def _load_model(name: str, weights: str | None, device_pref: str):
    from .models import create

    model = create(name, weights)
    if model.info.needs_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError(
                f"{name} needs PyTorch. Install the wheel for your GPU (see README), "
                "or use --model hints"
            ) from None
        from .device import pick_device

        dev, desc = pick_device(device_pref)
        print(f"device: {desc}", file=sys.stderr)
        model.load(dev)
    return model


def _find_hint(hints_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".tif", ".tiff", ".webp"):
        p = hints_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def cmd_photo(a: argparse.Namespace) -> int:
    from .hints import load_hints
    from .imageio import RawDevelop, SUPPORTED_IN, load_image, save_image
    from .pipeline import PhotoSettings, colorize_photo

    sources = [Path(p) for p in a.sources]
    for p in sources:
        if not p.is_file():
            _err(f"no such file: {p}")
            return 2
        if p.suffix.lower() not in SUPPORTED_IN:
            _err(f"unsupported input type: {p.name}")
            return 2
    if a.output and len(sources) > 1:
        _err("-o takes one source; use --out-dir for a set")
        return 2
    if a.hints and len(sources) > 1:
        _err("--hints takes one source; use --hints-dir for a set")
        return 2
    if a.output and Path(a.output).suffix.lower() not in OUT_EXTS:
        _err(f"output must end in one of {', '.join(sorted(OUT_EXTS))}")
        return 2

    try:
        model = _load_model(a.model, a.weights, a.device)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        _err(str(e))
        return 1

    settings = PhotoSettings(
        working_size=a.working_size,
        grain=a.grain,
        denoise=a.denoise,
        spread=a.spread,
        guided=not a.no_guided,
        saturation=a.saturation,
    )
    dev = RawDevelop(exposure_ev=a.raw_exposure, camera_wb=not a.raw_auto_wb)
    failed = 0
    for src in sources:
        try:
            img, meta = load_image(src, dev)
            for n in meta.notes:
                print(f"{src.name}: {n}", file=sys.stderr)
            h, w = img.shape[:2]
            hints = None
            hint_path = Path(a.hints) if a.hints else (
                _find_hint(Path(a.hints_dir), src.stem) if a.hints_dir else None
            )
            if hint_path:
                hints = load_hints(hint_path, h, w)
            rgb, rep = colorize_photo(img, model, settings, hints)

            if a.output:
                out = Path(a.output)
            else:
                out_dir = Path(a.out_dir) if a.out_dir else src.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{src.stem}.color.{a.format}"
            bits = a.bits
            if bits is None and out.suffix.lower() in {".png", ".tif", ".tiff"}:
                bits = 16 if meta.bits > 8 else 8
                if out.suffix.lower() in {".tif", ".tiff"}:
                    bits = 16
            save_image(out, rgb, bits=bits, quality=a.quality, meta=meta)

            for wmsg in rep.warnings:
                print(f"{src.name}: warning: {wmsg}", file=sys.stderr)
            t = " ".join(f"{k} {v:.2f}s" for k, v in rep.timings.items())
            hint_txt = f", {rep.hint_pixels} hint px" if rep.hint_pixels else ""
            print(f"{src.name} -> {out}  ({rep.size[0]}x{rep.size[1]}, chroma at "
                  f"{rep.working[0]}x{rep.working[1]}{hint_txt}; {t})")
        except Exception as e:  # keep going through a set
            failed += 1
            _err(f"{src.name}: {e}")
            if a.debug:
                raise
    return 1 if failed else 0


def cmd_fetch(a: argparse.Namespace) -> int:
    from .models.weights import CATALOG, fetch

    names = list(CATALOG) if a.model == "all" else [a.model]
    rc = 0
    for n in names:
        if n not in CATALOG:
            _err(f"nothing to fetch for {n}; known: {', '.join(CATALOG)}")
            return 2
        e = CATALOG[n]
        print(f"{n}: about {e.size_mb} MB, license {e.license}", file=sys.stderr)
        try:
            print(fetch(n))
        except Exception as ex:
            _err(f"{n}: {ex}")
            rc = 1
    return rc


def cmd_models(a: argparse.Namespace) -> int:
    from .models import DEFAULT_MODEL, REGISTRY
    from .models.weights import CATALOG, models_dir

    print(f"weights folder: {models_dir()}")
    for name, cls in REGISTRY.items():
        info = cls.info
        if info.needs_weights:
            e = CATALOG.get(name)
            have = e is not None and (models_dir() / e.filename).is_file()
            state = "ready" if have else "not downloaded"
        else:
            state = "built in"
        star = " (default)" if name == DEFAULT_MODEL else ""
        print(f"  {name:<11} {info.mode:<11} {state:<15} {info.license:<13} {info.description}{star}")
    return 0


def cmd_device(a: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed; only --model hints works")
        return 1
    from .device import pick_device

    dev, desc = pick_device(a.device)
    print(f"torch {torch.__version__}, using {dev}: {desc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="epochcolor", description="Colorize black and white photos.")
    p.add_argument("--version", action="version", version=f"epochcolor {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ph = sub.add_parser("photo", help="colorize one photo or a set")
    ph.add_argument("sources", nargs="+", help="input image(s)")
    ph.add_argument("-o", "--output", help="output file (single source)")
    ph.add_argument("--out-dir", help="output folder (default: next to each source)")
    ph.add_argument("--format", default="png", choices=["png", "tif", "jpg", "webp"],
                    help="output type when -o is not given (default png)")
    ph.add_argument("-m", "--model", default="siggraph17", help="siggraph17, eccv16 or hints")
    ph.add_argument("--weights", help="weights file, instead of the downloaded one")
    ph.add_argument("--hints", help="hint image for a single source")
    ph.add_argument("--hints-dir", help="folder of hint images named after each source")
    ph.add_argument("--grain", type=float, default=100.0,
                    help="percent of original luma kept, 100 = grain untouched (default)")
    ph.add_argument("--denoise", type=float, default=None,
                    help="denoise strength for the model's copy, in L* units (default auto)")
    ph.add_argument("--spread", type=float, default=None,
                    help="how far a hint travels through flat areas, as a fraction of the short side")
    ph.add_argument("--working-size", type=int, default=512,
                    help="short side in pixels for chroma work (default 512)")
    ph.add_argument("--saturation", type=float, default=1.0, help="chroma multiplier (default 1.0)")
    ph.add_argument("--no-guided", action="store_true", help="plain bicubic chroma upscale")
    ph.add_argument("--bits", type=int, choices=[8, 16], help="PNG/TIFF bit depth")
    ph.add_argument("--quality", type=int, default=95, help="JPEG/WebP quality (default 95)")
    ph.add_argument("--raw-exposure", type=float, default=0.0, help="RAW exposure in EV")
    ph.add_argument("--raw-auto-wb", action="store_true", help="RAW auto white balance")
    ph.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, xpu")
    ph.add_argument("--debug", action="store_true", help="full tracebacks")
    ph.set_defaults(func=cmd_photo)

    fe = sub.add_parser("fetch", help="download model weights")
    fe.add_argument("model", help="siggraph17, eccv16 or all")
    fe.set_defaults(func=cmd_fetch)

    mo = sub.add_parser("models", help="list models and weight status")
    mo.set_defaults(func=cmd_models)

    de = sub.add_parser("device", help="show which device PyTorch would use")
    de.add_argument("--device", default="auto")
    de.set_defaults(func=cmd_device)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
