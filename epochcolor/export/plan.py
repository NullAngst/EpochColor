"""Turn export settings into a concrete ffmpeg plan.

ExportSettings is what a preset stores and the user picks. resolve() checks
it against the codec, container and encoder rules, picks an encoder, maps
the quality number, and returns an ExportPlan with the exact arguments plus
a plain description of what it did and any warnings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .codecs import (
    AUDIO_ENCODE_OK, CODEC_LABEL, CODECS, CONTAINERS, ENCODERS, HW_ORDER, SOFTWARE,
    EncoderSpec, audio_copy_ok, container_allows,
)


class ExportError(ValueError):
    pass


@dataclass
class ExportSettings:
    codec: str = "h265"
    encoder: str = "auto"  # auto, software, hardware, or an ffmpeg encoder name
    bits: int = 10
    chroma: str = "420"
    container: str = ".mkv"
    rf: float | None = 18.0  # constant quality; None when bitrate is set
    bitrate: int | None = None  # kbps
    two_pass: bool = False
    speed: str | None = None  # None = the encoder's default
    tune_grain: bool = False
    film_grain: int = 0  # SVT-AV1 grain synthesis level, 0 = off
    audio_tracks: list[int] | None = None  # 1-based, None = all, [] = none
    audio_codec: str | None = None  # copy, aac, opus, flac for every track; None = copy where possible
    audio_fallback: str | None = None  # for tracks that can't be copied when audio_codec is None
    audio_bitrate: int | None = None  # kbps per track, None = by channel count

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> tuple["ExportSettings", list[str]]:
        known = {f.name for f in fields(cls)}
        unknown = [k for k in d if k not in known]
        return cls(**{k: v for k, v in d.items() if k in known}), unknown


@dataclass
class AudioStream:
    index: int  # position among the source's audio streams, 0-based
    codec: str
    channels: int
    title: str = ""


@dataclass
class ExportPlan:
    settings: ExportSettings
    encoder: EncoderSpec
    pix_fmt: str
    global_args: list[str] = field(default_factory=list)
    filter_chain: str = ""
    video_args: list[str] = field(default_factory=list)
    audio_args: list[str] = field(default_factory=list)
    audio_maps: list[int] = field(default_factory=list)
    mux_args: list[str] = field(default_factory=list)
    quality_text: str = ""
    audio_text: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    x265_params: list[str] = field(default_factory=list)
    svt_params: list[str] = field(default_factory=list)

    @property
    def two_pass(self) -> bool:
        return bool(self.settings.two_pass and self.settings.bitrate)

    def describe(self) -> str:
        s = self.settings
        lines = [
            f"video: {CODEC_LABEL[s.codec]} {s.bits}-bit {s.chroma[0]}:{s.chroma[1]}:{s.chroma[2]}"
            f" in {s.container[1:].upper()}, {self.encoder.label}, {self.quality_text}",
        ]
        lines += [f"audio: {t}" for t in self.audio_text] or ["audio: none"]
        return "\n".join(lines)


# ----------------------------------------------------------------- encoder


def _usable(spec: EncoderSpec, s: ExportSettings, probe) -> str | None:
    """None if the encoder can do this job, else the reason it can't."""
    if s.bits not in spec.depths:
        return f"{spec.name} does not do {s.bits}-bit"
    if s.chroma not in spec.chromas:
        return f"{spec.name} does not do {s.chroma[0]}:{s.chroma[1]}:{s.chroma[2]}"
    if s.bitrate and not spec.bitrate:
        return f"{spec.name} has no bitrate mode"
    if s.two_pass and not spec.two_pass:
        return f"{spec.name} has no two-pass mode here"
    if probe is not None:
        why = probe(spec, s.bits, s.chroma)
        if why:
            return why
    return None


def pick_encoder(s: ExportSettings, probe=None) -> tuple[EncoderSpec, list[str]]:
    """probe(spec, bits, chroma) -> None if it works, else a reason. It runs
    the one-frame test encode; tests pass None to skip it."""
    notes: list[str] = []
    choice = s.encoder
    if choice in ENCODERS:
        spec = ENCODERS[choice]
        if spec.codec != s.codec:
            raise ExportError(f"{choice} encodes {CODEC_LABEL[spec.codec]}, not {CODEC_LABEL[s.codec]}")
        why = _usable(spec, s, probe)
        if why:
            raise ExportError(why)
        return spec, notes

    sw = ENCODERS[SOFTWARE[s.codec]]
    if choice == "software":
        why = _usable(sw, s, probe)
        if why:
            raise ExportError(why)
        return sw, notes

    if choice not in ("auto", "hardware"):
        raise ExportError(f"unknown encoder {choice!r}: use auto, software, hardware or a name "
                          f"from `epochcolor encoders`")

    lossless = s.bitrate is None and s.rf == 0
    reasons = []
    for hw in HW_ORDER:
        spec = next((e for e in ENCODERS.values() if e.codec == s.codec and e.hw == hw), None)
        if spec is None:
            continue
        if lossless and not spec.lossless and choice == "auto":
            reasons.append(f"{spec.name} has no lossless mode")
            continue
        if (s.tune_grain or s.film_grain) and choice == "auto":
            reasons.append(f"{spec.name} has no grain tuning")
            continue
        if s.speed and s.speed not in spec.speeds:
            # speed names belong to one encoder family; "slow" means x265
            reasons.append(f"{spec.name} has no speed {s.speed!r}")
            continue
        why = _usable(spec, s, probe)
        if why:
            reasons.append(why)
            continue
        return spec, notes
    if choice == "hardware":
        detail = "; ".join(reasons) or f"no hardware encoder exists for {CODEC_LABEL[s.codec]}"
        raise ExportError(f"no working hardware encoder: {detail}")
    why = _usable(sw, s, probe)
    if why:
        raise ExportError(why)
    return sw, notes


# ----------------------------------------------------------------- quality


def _rate_args(spec: EncoderSpec, s: ExportSettings, plan: ExportPlan, x26x_params: list[str],
               svt_params: list[str]) -> list[str]:
    a: list[str] = []
    n = spec.name
    if s.bitrate:
        kb = f"{int(s.bitrate)}k"
        plan.quality_text = f"average {int(s.bitrate)} kbps" + (", two-pass" if s.two_pass else "")
        if spec.hw == "nvenc":
            a += ["-rc", "vbr", "-b:v", kb, "-maxrate", f"{int(s.bitrate * 1.5)}k"]
        elif spec.hw == "vaapi":
            a += ["-rc_mode", "VBR", "-b:v", kb, "-maxrate", f"{int(s.bitrate * 1.5)}k"]
        elif spec.hw == "amf":
            a += ["-rc", "vbr_peak", "-b:v", kb, "-maxrate", f"{int(s.bitrate * 1.5)}k"]
        else:
            a += ["-b:v", kb]
        return a

    rf = float(s.rf if s.rf is not None else 18.0)
    if rf < 0 or rf > spec.rf_max:
        raise ExportError(f"RF {rf:g} is out of range for {n} (0 to {spec.rf_max})")
    q = int(round(rf))

    if spec.quality_mode == "fixed":
        if n == "ffv1":
            plan.quality_text = "lossless (FFV1 is always lossless, RF ignored)"
        else:
            plan.quality_text = "fixed quality by profile (ProRes ignores RF)"
        return a

    if rf == 0:
        if spec.lossless:
            plan.quality_text = "RF 0, lossless"
            if n == "libx264":
                return ["-qp", "0"]
            if n == "libx265":
                x26x_params.append("lossless=1")
                return a
            if spec.hw == "nvenc":
                return ["-tune", "lossless"]
        plan.warnings.append(f"{n} has no lossless mode; RF 0 runs at its lowest QP and is NOT lossless")
        if spec.hw == "qsv":
            plan.quality_text = "RF 0, lowest ICQ (1), not lossless"
            return ["-global_quality", "1"]
        if spec.hw == "amf":
            plan.quality_text = "RF 0, lowest QP, not lossless"
            return ["-rc", "cqp", "-qp_i", "0", "-qp_p", "0"] + (["-qp_b", "0"] if spec.codec != "av1" else [])
        if spec.hw == "vaapi":
            plan.quality_text = "RF 0, lowest QP, not lossless"
            return ["-rc_mode", "CQP", "-qp", "0"]
        if spec.hw == "nvenc":  # av1_nvenc
            plan.quality_text = "RF 0, constant QP 0, not lossless"
            return ["-rc", "constqp", "-qp", "0"]
        if n == "libsvtav1":
            plan.quality_text = "RF 0, CRF 0 (lowest), not lossless"
            return ["-crf", "0"]

    if spec.hw is None:
        plan.quality_text = f"RF {rf:g} as CRF {rf:g}"
        if n == "libsvtav1":
            return ["-crf", str(q)]
        return ["-crf", f"{rf:g}"]
    if spec.hw == "nvenc":
        plan.quality_text = f"RF {rf:g} as CQ {q}"
        return ["-rc", "vbr", "-cq", str(max(1, q)), "-b:v", "0"]
    if spec.hw == "qsv":
        plan.quality_text = f"RF {rf:g} as ICQ {max(1, q)}"
        return ["-global_quality", str(max(1, q))]
    if spec.hw == "vaapi":
        plan.quality_text = f"RF {rf:g} as QP {q}"
        return ["-rc_mode", "CQP", "-qp", str(q)]
    if spec.hw == "amf":
        plan.quality_text = f"RF {rf:g} as QP {q}"
        return ["-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q)] + (
            ["-qp_b", str(q)] if spec.codec != "av1" else [])
    raise ExportError(f"no quality mapping for {n}")


def _speed_args(spec: EncoderSpec, s: ExportSettings, plan: ExportPlan) -> list[str]:
    sp = s.speed or spec.default_speed
    if not spec.speeds:
        if s.speed:
            plan.warnings.append(f"{spec.name} has no speed presets; --speed ignored")
        return []
    if sp not in spec.speeds:
        raise ExportError(f"speed {sp!r} is not valid for {spec.name}: {', '.join(spec.speeds)}")
    if spec.hw == "amf":
        return ["-quality", sp]
    return ["-preset", sp]


def resolve(s: ExportSettings, out: str | Path, audio: list[AudioStream],
            probe=None, ffmpeg_audio_encoders: set[str] | None = None,
            vaapi_device: str | None = None) -> ExportPlan:
    out = Path(out)
    ext = out.suffix.lower()
    s.container = ext
    if s.codec not in CODECS:
        raise ExportError(f"unknown codec {s.codec!r}: {', '.join(CODECS)}")
    if ext not in CONTAINERS:
        raise ExportError(f"unknown container {ext!r}: .mkv, .mp4 or .mov")
    if not container_allows(ext, s.codec):
        ok = sorted(c for c, v in CONTAINERS.items() if s.codec in v)
        raise ExportError(f"{CODEC_LABEL[s.codec]} can't go in {ext}; use {' or '.join(ok)}")
    if s.chroma not in ("420", "422", "444"):
        raise ExportError("chroma is 420, 422 or 444")
    if s.bitrate is not None and s.bitrate <= 0:
        raise ExportError("bitrate must be above 0")
    if s.two_pass and not s.bitrate:
        raise ExportError("two-pass needs a bitrate; constant quality is one pass by nature")
    if s.codec == "prores" and s.bits != 10:
        s.bits = 10
    if s.codec == "prores" and s.chroma == "420":
        s.chroma = "422"

    spec, notes = pick_encoder(s, probe)
    plan = ExportPlan(settings=s, encoder=spec, pix_fmt=spec.pix_fmt(s.bits, s.chroma))
    plan.warnings += notes
    if s.codec == "h264" and s.bits == 10:
        plan.warnings.append("10-bit H.264 plays on very little hardware; H.265 10-bit is the safer pick")

    # colour conversion: RGB in, BT.709 limited range YUV out
    chain = (f"scale=out_color_matrix=bt709:out_range=tv:"
             f"flags=accurate_rnd+full_chroma_int+full_chroma_inp,format={plan.pix_fmt}")
    if spec.hw == "vaapi":
        plan.global_args += ["-vaapi_device", vaapi_device or "/dev/dri/renderD128"]
        chain += ",hwupload"
    plan.filter_chain = chain

    x26x: list[str] = []
    svt: list[str] = []
    v = ["-c:v", spec.name]
    v += _rate_args(spec, s, plan, x26x, svt)
    v += _speed_args(spec, s, plan)

    if s.tune_grain:
        if spec.name == "libx264":
            v += ["-tune", "grain"]
        elif spec.name == "libx265":
            x26x.append("tune=grain")
        else:
            plan.warnings.append(f"{spec.name} has no grain tuning; --tune-grain ignored")
    if s.film_grain:
        if spec.name == "libsvtav1":
            svt += [f"film-grain={int(s.film_grain)}", "film-grain-denoise=1"]
            plan.warnings.append("film grain synthesis replaces the real grain with a synthetic one")
        else:
            plan.warnings.append("film grain synthesis is SVT-AV1 only; ignored")

    if spec.name == "libx265":
        x26x += ["colorprim=bt709", "transfer=bt709", "colormatrix=bt709", "range=limited",
                 "log-level=error"]
    if spec.name == "libx264":
        v += ["-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709"]
    if spec.name == "prores_ks":
        v += ["-profile:v", "4" if s.chroma == "444" else "3", "-vendor", "apl0"]
    if spec.name == "ffv1":
        v += ["-level", "3", "-g", "1", "-slices", "16", "-slicecrc", "1"]
    if spec.name == "libsvtav1":
        svt.append("enable-overlays=1")

    v += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
          "-color_range", "tv"]
    plan.video_args = v
    plan.x265_params = x26x  # joined by the writer, after any pass params go in
    plan.svt_params = svt

    if ext == ".mp4":
        plan.mux_args += ["-movflags", "+faststart"]
        if s.codec == "h265":
            plan.mux_args += ["-tag:v", "hvc1"]

    _plan_audio(plan, s, ext, audio, ffmpeg_audio_encoders)
    return plan


# ------------------------------------------------------------------- audio


AUDIO_ENCODER = {"aac": "aac", "opus": "libopus", "flac": "flac"}


def default_audio_bitrate(codec: str, channels: int) -> int:
    ch = max(2, channels)
    return {"aac": 96, "opus": 64}.get(codec, 0) * ch


def tracks_needing_choice(s: ExportSettings, ext: str, audio: list[AudioStream]) -> list[AudioStream]:
    """Selected tracks that can't be copied into this container as they are."""
    sel = _selected(s, audio)
    return [a for a in sel if not audio_copy_ok(ext, a.codec)]


def _selected(s: ExportSettings, audio: list[AudioStream]) -> list[AudioStream]:
    if s.audio_tracks is None:
        return list(audio)
    out = []
    for n in s.audio_tracks:
        if n < 1 or n > len(audio):
            raise ExportError(f"audio track {n} does not exist; the source has {len(audio)}")
        out.append(audio[n - 1])
    return out


def _plan_audio(plan: ExportPlan, s: ExportSettings, ext: str, audio: list[AudioStream],
                have: set[str] | None) -> None:
    sel = _selected(s, audio)
    want = s.audio_codec
    if want not in (None, "copy", "aac", "opus", "flac"):
        raise ExportError(f"audio codec {want!r}: copy, aac, opus or flac")
    if want and want != "copy" and want not in AUDIO_ENCODE_OK[ext]:
        raise ExportError(f"{ext} can't hold {want}; use {', '.join(AUDIO_ENCODE_OK[ext])}")
    strict = False
    for j, a in enumerate(sel):
        plan.audio_maps.append(a.index)
        name = f"track {a.index + 1} ({a.codec}, {a.channels}ch{', ' + a.title if a.title else ''})"
        codec = want
        if want in (None, "copy"):
            if audio_copy_ok(ext, a.codec):
                plan.audio_args += [f"-c:a:{j}", "copy"]
                plan.audio_text.append(f"{name}, copied")
                if ext == ".mp4" and a.codec == "flac":
                    strict = True
                continue
            if want == "copy" or not s.audio_fallback:
                raise ExportError(f"{name} can't be copied into {ext}; re-encode it with "
                                  f"--audio-fallback {' or '.join(AUDIO_ENCODE_OK[ext])}, "
                                  f"or use MKV")
            codec = s.audio_fallback
            if codec not in AUDIO_ENCODE_OK[ext]:
                raise ExportError(f"{ext} can't hold {codec}; use {', '.join(AUDIO_ENCODE_OK[ext])}")
        want_here = codec
        enc = AUDIO_ENCODER[want_here]
        if have is not None and enc not in have:
            raise ExportError(f"this ffmpeg has no {enc} encoder")
        plan.audio_args += [f"-c:a:{j}", enc]
        if want_here == "flac":
            plan.audio_text.append(f"{name}, to FLAC")
            if ext == ".mp4":
                strict = True
        else:
            kb = s.audio_bitrate or default_audio_bitrate(want_here, a.channels)
            plan.audio_args += [f"-b:a:{j}", f"{kb}k"]
            label = "AAC" if want_here == "aac" else "Opus"
            plan.audio_text.append(f"{name}, to {label} {kb} kbps")
    if strict:
        plan.audio_args += ["-strict", "experimental"]
