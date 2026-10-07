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


def _export_settings(a: argparse.Namespace):
    from .export.plan import ExportSettings
    from .export.presets import load_preset

    warnings: list[str] = []
    if a.preset:
        es, warnings = load_preset(a.preset)
    else:
        es = ExportSettings()
    for key in ("codec", "encoder", "bits", "chroma", "speed", "film_grain", "audio_codec",
                "audio_fallback", "audio_bitrate"):
        v = getattr(a, key)
        if v is not None:
            setattr(es, key, v)
    if a.rf is not None:
        es.rf, es.bitrate, es.two_pass = a.rf, None, False
    if a.bitrate is not None:
        es.bitrate, es.rf = a.bitrate, None
    if a.two_pass:
        es.two_pass = True
    if a.tune_grain:
        es.tune_grain = True
    if a.no_audio:
        es.audio_tracks = []
    elif a.audio_tracks is not None:
        if a.audio_tracks.strip().lower() == "none":
            es.audio_tracks = []
        elif a.audio_tracks.strip().lower() == "all":
            es.audio_tracks = None
        else:
            try:
                es.audio_tracks = [int(x) for x in a.audio_tracks.split(",") if x.strip()]
            except ValueError:
                raise ValueError("--audio-tracks takes numbers like 1,3, or all, or none") from None
    return es, warnings


def _ask_audio_fallback(tracks, ext: str) -> str | None:
    from .export.codecs import AUDIO_ENCODE_OK

    choices = AUDIO_ENCODE_OK[ext]
    print(f"These audio tracks can't be copied into {ext} as they are:", file=sys.stderr)
    for t in tracks:
        print(f"  track {t.index + 1}: {t.codec}, {t.channels}ch {t.title}", file=sys.stderr)
    names = {"aac": "AAC", "opus": "Opus", "flac": "FLAC"}
    menu = ", ".join(f"{i}) {names[c]}" for i, c in enumerate(choices, 1))
    while True:
        try:
            ans = input(f"Re-encode them to {menu}? [1] ").strip().lower() or "1"
        except EOFError:
            return None
        if ans.isdigit() and 1 <= int(ans) <= len(choices):
            return choices[int(ans) - 1]
        if ans in choices:
            return ans


def _detector(vaapi_device: str | None = None):
    from .export.detect import Detector
    from .export.writer import ffmpeg_path
    from .video.pipeline import cache_root

    return Detector(ffmpeg_path(), cache_root() / "encoders.json", vaapi_device)


def cmd_video(a: argparse.Namespace) -> int:
    from .export.plan import ExportError, resolve, tracks_needing_choice
    from .export.presets import save_preset
    from .video.io import InputError, probe
    from .video.pipeline import VideoSettings, colorize_video

    src = Path(a.source)
    if not src.is_file():
        _err(f"no such file: {src}")
        return 2
    try:
        es, pwarn = _export_settings(a)
    except Exception as e:
        _err(str(e))
        return 2
    for w in pwarn:
        print(f"warning: {w}", file=sys.stderr)
    out = Path(a.output) if a.output else src.with_name(f"{src.stem}.color{es.container}")
    if out.resolve() == src.resolve():
        _err("output would overwrite the source")
        return 2
    try:
        info = probe(src)
    except InputError as e:
        _err(str(e))
        return 2
    print(f"{src.name}: {info.width}x{info.height} at {float(info.fps):.3f} fps, "
          f"{info.codec}, {len(info.audio)} audio track(s)", file=sys.stderr)
    if a.probe:
        for line in info.audio_streams:
            print(f"  audio {line}")
        return 0

    ext = out.suffix.lower()
    try:
        if es.audio_codec is None and not es.audio_fallback and ext in (".mp4", ".mov"):
            stuck = tracks_needing_choice(es, ext, info.audio)
            if stuck and sys.stdin.isatty():
                es.audio_fallback = _ask_audio_fallback(stuck, ext)
        det = _detector(a.vaapi_device)
        audio_enc = {e for e in det.encoders() if e in ("aac", "libopus", "flac")}
        plan = resolve(es, out, info.audio, probe=det.check, ffmpeg_audio_encoders=audio_enc,
                       vaapi_device=det.vaapi_device)
    except (ExportError, RuntimeError) as e:
        _err(str(e))
        return 2
    print(plan.describe(), file=sys.stderr)
    for w in plan.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if a.save_preset:
        print(f"saved preset {a.save_preset}: {save_preset(a.save_preset, es)}", file=sys.stderr)
    if a.dry_run:
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
    try:
        rep = colorize_video(info, model, out, plan, vs, use_cache=not a.no_cache)
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
    shots = f"{len(rep.shots)} shot" + ("" if len(rep.shots) == 1 else "s")
    print(f"{src.name} -> {out}  ({rep.frames} frames, {shots}{cached})")
    if a.list_shots:
        fps = info.fps_float
        for i, (s0, s1) in enumerate(rep.shots, 1):
            print(f"  shot {i}: frames {s0}-{s1 - 1}  ({s0 / fps:.2f}s - {s1 / fps:.2f}s)")
    return 0


def cmd_encoders(a: argparse.Namespace) -> int:
    try:
        det = _detector(a.vaapi_device)
    except RuntimeError as e:
        _err(str(e))
        return 1
    if a.retest:
        det.clear()
    print(f"ffmpeg: {det.ffmpeg}")
    print(f"VAAPI device: {det.vaapi_device or 'none found'}")
    print(f"  {'encoder':<12} {'codec':<7} {'8-bit':<13} 10-bit")
    for name, codec, r8, r10 in det.table():
        print(f"  {name:<12} {codec:<7} {r8:<13} {r10}")
    return 0


def cmd_presets(a: argparse.Namespace) -> int:
    from .export.presets import list_presets, presets_dir

    print(f"saved presets folder: {presets_dir()}")
    for name, src, desc in list_presets():
        print(f"  {name:<14} {src:<9} {desc}")
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

    vi = sub.add_parser("video", help="colorize a clip")
    vi.add_argument("source", help="input clip, constant frame rate")
    vi.add_argument("-o", "--output", help="output .mkv, .mp4 or .mov (default: <name>.color.mkv)")
    vi.add_argument("-m", "--model", default="siggraph17", help="siggraph17 or eccv16")
    vi.add_argument("--weights", help="weights file, instead of the downloaded one")
    g = vi.add_argument_group("colour")
    g.add_argument("--grain", type=float, default=100.0, help="percent of original luma kept (default 100)")
    g.add_argument("--denoise", type=float, default=None,
                   help="spatial denoise strength after the temporal pass, L* units (0 turns both off)")
    g.add_argument("--stabilize", type=float, default=0.9,
                   help="chroma stabilizer, 0 off, 0.9 averages up to 10 frames each way (default)")
    g.add_argument("--shot-threshold", type=float, default=6.0,
                   help="cut sensitivity in L* units, lower finds more cuts (default 6)")
    g.add_argument("--working-size", type=int, default=512, help="short side for chroma work")
    g.add_argument("--saturation", type=float, default=1.0, help="chroma multiplier")
    g.add_argument("--no-guided", action="store_true", help="plain bicubic chroma upscale")
    e = vi.add_argument_group("export")
    e.add_argument("--preset", help="export preset name or .json path (see `epochcolor presets`)")
    e.add_argument("--save-preset", metavar="NAME", help="save these export settings as a preset")
    e.add_argument("--codec", choices=["h265", "h264", "av1", "prores", "ffv1"], help="default h265")
    e.add_argument("--encoder", help="auto (default), software, hardware, or a name from `epochcolor encoders`")
    e.add_argument("--bits", type=int, choices=[8, 10, 12, 16], help="bit depth (default 10)")
    e.add_argument("--chroma", choices=["420", "422", "444"], help="chroma subsampling (default 420)")
    e.add_argument("--rf", type=float, help="constant quality 0 to 51, lower is better (default 18), 0 = lossless")
    e.add_argument("--bitrate", type=int, help="average bitrate in kbps instead of RF")
    e.add_argument("--two-pass", action="store_true", help="two-pass bitrate, x264 and x265 only")
    e.add_argument("--speed", help="encoder speed preset: medium, slow, p5, 6 and so on")
    e.add_argument("--tune-grain", action="store_true", help="x264/x265 grain tuning")
    e.add_argument("--film-grain", type=int, help="SVT-AV1 grain synthesis level (replaces real grain)")
    e.add_argument("--audio-tracks", help="1-based list like 1,3, or all (default), or none")
    e.add_argument("--audio-codec", choices=["copy", "aac", "opus", "flac"],
                   help="for every track; default copies where the container allows")
    e.add_argument("--audio-fallback", choices=["aac", "opus", "flac"],
                   help="codec for tracks that can't be copied (asked for if not given)")
    e.add_argument("--audio-bitrate", type=int, help="AAC/Opus kbps per track (default by channel count)")
    e.add_argument("--no-audio", action="store_true", help="leave the audio out")
    e.add_argument("--vaapi-device", help="VAAPI render node (default: first /dev/dri/renderD*)")
    e.add_argument("--dry-run", action="store_true", help="show the export plan and stop")
    r = vi.add_argument_group("run")
    r.add_argument("--frames", type=int, help="only the first N frames, for quick tests")
    r.add_argument("--no-cache", action="store_true", help="redo every pass")
    r.add_argument("--list-shots", action="store_true", help="print the detected shots")
    r.add_argument("--probe", action="store_true", help="check the clip and stop")
    r.add_argument("--device", default="auto", help="model device: auto, cpu, cuda, cuda:1, xpu")
    r.add_argument("--debug", action="store_true", help="full tracebacks")
    vi.set_defaults(func=cmd_video)

    en = sub.add_parser("encoders", help="test which video encoders work here")
    en.add_argument("--retest", action="store_true", help="forget cached results and test again")
    en.add_argument("--vaapi-device", help="VAAPI render node")
    en.set_defaults(func=cmd_encoders)

    pr = sub.add_parser("presets", help="list export presets")
    pr.set_defaults(func=cmd_presets)

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
