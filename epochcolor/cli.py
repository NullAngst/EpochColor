"""Command line front end for milestone 1."""

from __future__ import annotations

import argparse
import os
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
                f"{name} needs PyTorch. Run `epochcolor setup-torch` to download the build "
                "for your GPU, or use --model hints"
            ) from None
        from . import gpucheck
        from .device import pick_device
        from .resources import torch_threads

        dev, env, note, fresh = gpucheck.ensure(device_pref, log=lambda m: print(m, file=sys.stderr))
        os.environ.update(env)
        if note and fresh:
            print(f"note: {note}", file=sys.stderr)
        dev, desc = pick_device(dev)
        torch_threads()
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
        saturation=a.saturation, cast=a.cast / 100.0,
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
            if a.negative:
                from .film import invert_rgb, measure_negative

                img = invert_rgb(img, measure_negative([img]))
            ref = {"path": a.reference, "strength": a.reference_strength / 100.0} if a.reference else None
            rgb, rep = colorize_photo(img, model, settings, hints, reference=ref)
            if a.regrain:
                from .film import regrain

                rgb = regrain(rgb, a.regrain, a.grain_size, a.grain_colour / 100.0, seed=1)

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
        saturation=a.saturation, cast=a.cast / 100.0, frames=a.frames, chroma_size=a.chroma_size,
        deflicker=0.0 if a.deflicker == "off" else 1.0, dust=a.dust != "off",
        deflicker_out=a.deflicker == "output", dust_out=a.dust == "output",
        regrain=a.regrain, regrain_size=a.grain_size, regrain_chroma=a.grain_colour / 100.0,
    )
    if a.negative:
        from .worker import _measure_negative

        info.negative = _measure_negative({"kind": "clip", "path": str(src), "fps_num": info.fps.numerator,
                                           "fps_den": info.fps.denominator, "frames": info.frames_estimate},
                                          lambda *x: None)
        print(f"negative: film base {info.negative['base']:.3f}, density {info.negative['dmin']:.2f} to "
              f"{info.negative['dmax']:.2f}", file=sys.stderr)
    ref = {"path": a.reference, "strength": a.reference_strength / 100.0} if a.reference else None
    try:
        rep = colorize_video(info, model, out, plan, vs, use_cache=not a.no_cache, reference=ref)
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


def cmd_render(a: argparse.Namespace) -> int:
    """Export a project saved in the editor: its timeline, hints and grades."""
    from .export.plan import ExportError, ExportSettings
    from .project import Project, ProjectError
    from .timeline_export import analysis_state, build_export

    try:
        p = Project.load(a.project)
    except (OSError, ValueError, ProjectError) as e:
        _err(str(e))
        return 2
    if not a.inout:
        p.d.range = None
    if a.preset:
        from .export.presets import load_preset

        es, _ = load_preset(a.preset)
    else:
        es, _ = ExportSettings.from_dict({k: v for k, v in p.d.export.items() if k != "last_out"})
    if a.audio_fallback:
        es.audio_fallback = a.audio_fallback
    out = Path(a.output) if a.output else Path(p.d.export.get("last_out") or Path(a.project).with_suffix(es.container))
    try:
        det = _detector()
        audio_enc = {e for e in det.encoders() if e in ("aac", "libopus", "flac")}
        job = build_export(p, None, es, out, model_name=p.d.settings["model"], probe=det.check,
                           audio_encoders=audio_enc, vaapi_device=det.vaapi_device)
    except (ExportError, ValueError, RuntimeError) as e:
        _err(str(e))
        return 2
    print(job.plan.describe(), file=sys.stderr)
    need = [c for c in job.clips if analysis_state(c, job.settings, job.model_name, None, job.hints.get(c.id)) != "ready"]
    model = None
    if need:
        print(f"colorizing first: {', '.join(c.name for c in need)}", file=sys.stderr)
        try:
            model = _load_model(p.d.settings["model"], None, a.device)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            _err(str(e))
            return 1
    try:
        job.run(model, quiet=False)
    except Exception as e:
        _err(str(e))
        if a.debug:
            raise
        return 1
    print(f"{Path(a.project).name} -> {out}")
    return 0


def cmd_cache(a: argparse.Namespace) -> int:
    from .diskspace import describe
    from .video.pipeline import cache_root, cache_size, clear_cache, set_cache_root, sweep_stale

    if a.action == "clear":
        print(f"removed {clear_cache()} cached item(s) from {cache_root()}")
    elif a.action == "dir":
        if not a.path:
            print(cache_root())
            return 0
        old = cache_root()
        new = set_cache_root(None if a.path == "default" else a.path)
        print(f"working folder: {new}\n{describe(new)}")
        if old != new and old.exists():
            print(f"the old one is left as it was: {old} ({cache_size(old) / 2**30:.2f} GB). "
                  f"`epochcolor cache clear` before switching empties it.")
    else:
        root = cache_root()
        freed = sweep_stale(root)
        if freed:
            print(f"deleted {freed / 2**30:.1f} GB of luma left by an older version")
        print(f"{root}: {cache_size(root) / 2**30:.2f} GB\n{describe(root)}")
    return 0


def _progress_printer(label: str):
    def show(stage, done, total):
        if total:
            print(f"\r{label}: {stage} {done * 100 // max(1, total)}%   ", end="", file=sys.stderr)
    return show


def cmd_fetch(a: argparse.Namespace) -> int:
    from .models.manager import ModelError, catalog, download, human_size, installed

    ids = [e["id"] for e in catalog()] if a.model == "all" else [a.model]
    if a.model == "all":
        ids = [i for i in ids if next(e for e in catalog() if e["id"] == i).get("license_ok")]
        print("all = every model whose license needs no accepting: " + ", ".join(ids), file=sys.stderr)
    rc = 0
    for mid in ids:
        e = next((x for x in catalog() if x["id"] == mid), None)
        if e is None:
            _err(f"{mid} is not in the catalog; `epochcolor models` lists it")
            return 2
        if mid in installed():
            print(f"{mid}: installed already")
            continue
        print(f"{mid}: about {human_size(e.get('size_mb'))}, license {e['license']}", file=sys.stderr)
        try:
            path = download(mid, _progress_printer(mid), accept_license=a.accept_license)
            print(file=sys.stderr)
            print(path)
        except (ModelError, OSError) as ex:
            print(file=sys.stderr)
            _err(f"{mid}: {ex}")
            rc = 1
    return rc


def cmd_models(a: argparse.Namespace) -> int:
    from .models import DEFAULT_MODEL
    from .models.manager import (
        ModelError, add_from_file, catalog, check_loads, human_size, installed, move_storage,
        refresh_catalog, remove,
    )
    from .models.weights import models_dir

    try:
        if a.action == "refresh":
            good, bad = refresh_catalog()
            print(f"catalog: {good} models" + (f", {bad} skipped (need a newer EpochColor)" if bad else ""))
            return 0
        if a.action == "add":
            if len(a.args) != 2:
                _err("models add WEIGHTS MANIFEST.json")
                return 2
            mid = add_from_file(a.args[0], a.args[1])
            print(f"added {mid}; checking it loads...")
            try:
                check_loads(mid)
            except Exception as e:
                remove(mid)
                _err(f"{mid} doesn't load, removed again: {e}")
                return 1
            print("ok")
            return 0
        if a.action == "remove":
            if len(a.args) != 1:
                _err("models remove ID")
                return 2
            print("removed" if remove(a.args[0]) else f"{a.args[0]} is not installed")
            return 0
        if a.action == "dir":
            if a.args:
                print(f"models now in {move_storage(a.args[0] if a.args[0] != 'default' else None)}")
            else:
                print(models_dir())
            return 0
    except (ModelError, OSError, ValueError) as e:
        _err(str(e))
        return 1

    have = installed()
    print(f"models folder: {models_dir()}")
    print(f"  {'id':<20} {'mode':<10} {'state':<15} {'size':>7}  license")
    print(f"  {'hints':<20} {'paint':<10} {'built in':<15} {'':>7}  GPL-3.0")
    for e in catalog():
        state = "installed" if e["id"] in have else "available"
        star = " (default)" if e["id"] == DEFAULT_MODEL else ""
        lic = e["license"] if len(e["license"]) < 40 else e["license"][:37] + "..."
        print(f"  {e['id']:<20} {e['mode']:<10} {state:<15} {human_size(e.get('size_mb')):>7}  {lic}{star}")
    for mid, man in have.items():
        if not any(e["id"] == mid for e in catalog()):
            print(f"  {mid:<20} {man.get('mode', ''):<10} {'installed':<15} {'':>7}  {man.get('license', '')}")
    return 0


def cmd_device(a: argparse.Namespace) -> int:
    from . import gpucheck
    from .torch_setup import have_torch

    if not have_torch():
        print("PyTorch is not installed; only --model hints works")
        return 1
    if a.test:
        gpucheck.forget()
        r = gpucheck.run_ladder(log=print)
        print(f"\nresult: {r['device']} {r.get('name', '')} {r.get('env') or ''}")
        if r.get("note"):
            print(r["note"])
        print(f"saved to {gpucheck.state_path()}")
        return 0 if r["device"] != "cpu" or not r.get("note") else 2
    dev, env, note, fresh = gpucheck.ensure(a.device, log=print)
    os.environ.update(env)
    import torch

    from .device import pick_device

    dev, desc = pick_device(dev)
    print(f"torch {torch.__version__}, using {dev}: {desc}" + (f" with {env}" if env else ""))
    if note:
        print(note)
    return 0


def cmd_setup_torch(a: argparse.Namespace) -> int:
    from . import torch_setup as ts

    if a.remove:
        print("removed" if ts.remove() else "nothing to remove", ts.torch_dir())
        return 0
    if a.check:
        fam, why = ts.detect()
        print(f"detected: {why}; would install {fam}")
        print(f"installed: {ts.installed_variant() or 'none'} in {ts.active_dir() or ts.torch_dir()}")
        from .diskspace import describe

        print(f"room: {describe(ts.torch_dir().parent)}")
        need = ts.NEED_GB.get(fam)
        if need:
            print(f"a {fam} install needs about {need} GB there while it unpacks")
        return 0
    try:
        ts.install(a.variant, log=print)
    except Exception as e:
        _err(str(e))
        return 1
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    try:
        from .gui.main import main as gui_main
    except ImportError as e:
        _err(f"the GUI needs PySide6: pip install PySide6 ({e})")
        return 1
    return gui_main([sys.argv[0]] + a.files)


def _film_args(p, video: bool) -> None:
    p.add_argument("--negative", action="store_true",
                   help="the source is a black and white negative: measure its film base and invert it first")
    p.add_argument("--reference", metavar="IMAGE",
                   help="a colour photo of the same place, scene or era whose colours guide the result"
                        + (" (used for every shot)" if video else ""))
    p.add_argument("--reference-strength", type=float, default=100.0, metavar="PCT",
                   help="how strongly the reference wins over the model, 0 to 100 (default 100)")
    if video:
        p.add_argument("--deflicker", choices=["off", "model", "output"], default="model",
                       help="steady flickering brightness: for the model's copy only (default), "
                            "the output too, or off")
        p.add_argument("--dust", choices=["off", "model", "output"], default="model",
                       help="remove dirt specks: from the model's copy only (default), the output too, or off")
    p.add_argument("--regrain", type=float, default=0.0, metavar="PCT",
                   help="add synthetic grain after colour, strength in percent at mid grey (default off)")
    p.add_argument("--grain-size", type=float, default=1.0, metavar="PX",
                   help="regrain size in pixels at 1080 lines (default 1.0)")
    p.add_argument("--grain-colour", type=float, default=0.0, metavar="PCT",
                   help="faint colour grain on top of the mono grain, 0 to 100 (default 0)")


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
    ph.add_argument("-m", "--model", default="siggraph17",
                    help="an installed model id (see `epochcolor models`), or hints")
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
    _film_args(ph, video=False)
    ph.add_argument("--cast", type=float, default=0.0,
                    help="remove the model's all-over colour cast, 0 to 100 percent (default 0)")
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
    vi.add_argument("-m", "--model", default="siggraph17", help="an installed model id (see `epochcolor models`)")
    vi.add_argument("--weights", help="weights file, instead of the downloaded one")
    g = vi.add_argument_group("colour")
    g.add_argument("--grain", type=float, default=100.0, help="percent of original luma kept (default 100)")
    g.add_argument("--denoise", type=float, default=None,
                   help="spatial denoise strength after the temporal pass, L* units (0 turns both off)")
    g.add_argument("--stabilize", type=float, default=0.9,
                   help="chroma stabilizer, 0 off, 0.9 averages up to 10 frames each way (default)")
    g.add_argument("--shot-threshold", type=float, default=6.0,
                   help="cut sensitivity in L* units, lower finds more cuts (default 6)")
    g.add_argument("--working-size", type=int, default=512, help="short side the model works at")
    g.add_argument("--chroma-size", type=int, default=256,
                   help="short side of the stored colour (default 256); higher is crisper and bigger")
    g.add_argument("--saturation", type=float, default=1.0, help="chroma multiplier")
    _film_args(g, video=True)
    g.add_argument("--cast", type=float, default=0.0,
                   help="remove the model's all-over colour cast per shot, 0 to 100 percent (default 0)")
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

    ts = sub.add_parser("setup-torch", help="download PyTorch for this machine's GPU")
    ts.add_argument("--variant", default="auto", choices=["auto", "rocm", "cuda", "xpu", "cpu"],
                    help="auto picks from the GPU and driver (default)")
    ts.add_argument("--check", action="store_true", help="show what would be installed and stop")
    ts.add_argument("--remove", action="store_true", help="delete the downloaded PyTorch")
    ts.set_defaults(func=cmd_setup_torch)

    gu = sub.add_parser("gui", help="open the editor")
    gu.add_argument("files", nargs="*", help="a project, clips or photos to open")
    gu.set_defaults(func=cmd_gui)

    pr = sub.add_parser("presets", help="list export presets")
    pr.set_defaults(func=cmd_presets)

    re_ = sub.add_parser("render", help="export a project saved in the editor")
    re_.add_argument("project", help="a .epochcolor project")
    re_.add_argument("-o", "--output", help="output file (default: the project's last export)")
    re_.add_argument("--preset", help="export preset instead of the project's export settings")
    re_.add_argument("--inout", action="store_true", help="only the in/out range")
    re_.add_argument("--audio-fallback", choices=["aac", "opus", "flac"],
                     help="codec for audio that has to be re-encoded")
    re_.add_argument("--device", default="auto")
    re_.add_argument("--debug", action="store_true")
    re_.set_defaults(func=cmd_render)

    ca = sub.add_parser("cache", help="show, move or clear the working folder (the cache)")
    ca.add_argument("action", nargs="?", default="show", choices=["show", "clear", "dir"])
    ca.add_argument("path", nargs="?", help="with dir: the new working folder, or default")
    ca.set_defaults(func=cmd_cache)

    fe = sub.add_parser("fetch", help="download a model from the catalog")
    fe.add_argument("model", help="a model id from `epochcolor models`, or all")
    fe.add_argument("--accept-license", action="store_true",
                    help="accept the model's license terms (read them first: `epochcolor models`)")
    fe.set_defaults(func=cmd_fetch)

    mo = sub.add_parser("models", help="list, refresh, add or remove models")
    mo.add_argument("action", nargs="?", default="list", choices=["list", "refresh", "add", "remove", "dir"],
                    help="list (default), refresh the catalog, add WEIGHTS MANIFEST, remove ID, "
                         "dir [PATH|default] to show or move the models folder")
    mo.add_argument("args", nargs="*")
    mo.set_defaults(func=cmd_models)

    de = sub.add_parser("device", help="show which device PyTorch would use")
    de.add_argument("--device", default="auto")
    de.add_argument("--test", action="store_true",
                    help="test each GPU again in a separate process and print every attempt")
    de.set_defaults(func=cmd_device)
    return p


def main(argv: list[str] | None = None) -> int:
    from .diskspace import explain, is_full

    a = build_parser().parse_args(argv)
    if a.cmd in ("photo", "video", "render"):
        from .resources import Watchdog, apply_limits

        apply_limits()
        Watchdog().start()
    try:
        return a.func(a)
    except OSError as e:
        if not is_full(e):
            raise
        _err(explain(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
