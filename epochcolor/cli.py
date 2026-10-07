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


def cmd_video(a: argparse.Namespace) -> int:
    from .video.io import EncodeSettings, InputError, probe
    from .video.pipeline import VideoSettings, colorize_video

    src = Path(a.source)
    if not src.is_file():
        _err(f"no such file: {src}")
        return 2
    out = Path(a.output) if a.output else src.with_name(f"{src.stem}.color.mkv")
    if out.suffix.lower() not in {".mkv", ".mp4"}:
        _err("output must be .mkv or .mp4 for now")
        return 2
    if out.resolve() == src.resolve():
        _err("output would overwrite the source")
        return 2
    try:
        info = probe(src)
    except InputError as e:
        _err(str(e))
        return 2
    print(f"{src.name}: {info.width}x{info.height} at {float(info.fps):.3f} fps, "
          f"{info.codec}, {len(info.audio_streams)} audio track(s)", file=sys.stderr)
    if a.probe:
        for line in info.audio_streams:
            print(f"  audio {line}")
        return 0
    try:
        model = _load_model(a.model, a.weights, a.device)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        _err(str(e))
        return 1
    if model.info.mode == "propagation":
        _err("video needs a network model for now; painted hints on video come in milestone 5")
        return 2
    vs = VideoSettings(
        working_size=a.working_size, grain=a.grain, denoise=a.denoise,
        stabilize=a.stabilize, shot_threshold=a.shot_threshold, guided=not a.no_guided,
        saturation=a.saturation, frames=a.frames,
    )
    es = EncodeSettings(rf=a.rf, preset=a.preset, tune_grain=a.tune_grain, audio=not a.no_audio)
    try:
        rep = colorize_video(info, model, out, vs, es, use_cache=not a.no_cache)
    except KeyboardInterrupt:
        _err("stopped. Finished shots are cached, run the same command to pick up.")
        return 130
    except Exception as e:
        _err(str(e))
        if a.debug:
            raise
        return 1
    for w in rep.warnings:
        print(f"warning: {w}", file=sys.stderr)
    cached = f", {rep.cached_shots} from cache" if rep.cached_shots else ""
    print(f"{src.name} -> {out}  ({rep.frames} frames, {len(rep.shots)} shots{cached})")
    if a.list_shots:
        fps = info.fps_float
        for i, (s0, s1) in enumerate(rep.shots, 1):
            print(f"  shot {i}: frames {s0}-{s1 - 1}  ({s0 / fps:.2f}s - {s1 / fps:.2f}s)")
    return 0


def cmd_cache(a: argparse.Namespace) -> int:
    from .video.pipeline import cache_root, clear_cache

    if a.action == "clear":
        print(f"removed {clear_cache()} cached clip(s) from {cache_root()}")
    else:
        root = cache_root()
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) if root.exists() else 0
        print(f"{root}: {size / 2**30:.2f} GB")
    return 0


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
    p = argparse.ArgumentParser(prog="epochcolor", description="Colorize black and white photos and film.")
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

    vi = sub.add_parser("video", help="colorize a clip, H.265 10-bit out")
    vi.add_argument("source", help="input clip, constant frame rate")
    vi.add_argument("-o", "--output", help="output .mkv or .mp4 (default: <name>.color.mkv)")
    vi.add_argument("-m", "--model", default="siggraph17", help="siggraph17 or eccv16")
    vi.add_argument("--weights", help="weights file, instead of the downloaded one")
    vi.add_argument("--rf", type=float, default=18.0, help="x265 constant quality, lower is better (default 18)")
    vi.add_argument("--preset", default="medium", help="x265 speed preset (default medium)")
    vi.add_argument("--tune-grain", action="store_true", help="x265 tune=grain, keeps grain at higher RF")
    vi.add_argument("--grain", type=float, default=100.0, help="percent of original luma kept (default 100)")
    vi.add_argument("--denoise", type=float, default=None,
                    help="spatial denoise strength after the temporal pass, L* units (0 turns both off)")
    vi.add_argument("--stabilize", type=float, default=0.9,
                    help="chroma stabilizer, 0 off, 0.9 averages up to 10 frames each way (default)")
    vi.add_argument("--shot-threshold", type=float, default=6.0,
                    help="cut sensitivity in L* units, lower finds more cuts (default 6)")
    vi.add_argument("--working-size", type=int, default=512, help="short side for chroma work")
    vi.add_argument("--saturation", type=float, default=1.0, help="chroma multiplier")
    vi.add_argument("--no-guided", action="store_true", help="plain bicubic chroma upscale")
    vi.add_argument("--frames", type=int, help="only the first N frames, for quick tests")
    vi.add_argument("--no-audio", action="store_true", help="leave the audio out")
    vi.add_argument("--no-cache", action="store_true", help="redo every pass")
    vi.add_argument("--list-shots", action="store_true", help="print the detected shots")
    vi.add_argument("--probe", action="store_true", help="check the clip and stop")
    vi.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, xpu")
    vi.add_argument("--debug", action="store_true", help="full tracebacks")
    vi.set_defaults(func=cmd_video)

    ca = sub.add_parser("cache", help="show or clear the video cache")
    ca.add_argument("action", nargs="?", default="show", choices=["show", "clear"])
    ca.set_defaults(func=cmd_cache)

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
