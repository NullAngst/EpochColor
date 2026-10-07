"""What every encoder can do, and how a quality setting maps onto it.

The UI speaks HandBrake: pick a codec, bit depth, chroma, container, then
either an RF (0 to 51, lower is better) or a bitrate. Each encoder takes
that number in its own mode, so the mapping is spelled out per encoder and
reported back, since the same RF gives different sizes on GPU and CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

CODECS = ("h265", "h264", "av1", "prores", "ffv1")
CODEC_LABEL = {"h265": "H.265", "h264": "H.264", "av1": "AV1", "prores": "ProRes", "ffv1": "FFV1"}
HW_ORDER = ("nvenc", "vaapi", "qsv", "amf")

X26X_SPEEDS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium",
               "slow", "slower", "veryslow", "placebo")
NVENC_SPEEDS = tuple(f"p{i}" for i in range(1, 8))
SVT_SPEEDS = tuple(str(i) for i in range(0, 14))
QSV_SPEEDS = ("veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
AMF_SPEEDS = ("speed", "balanced", "quality")


@dataclass(frozen=True)
class EncoderSpec:
    name: str  # ffmpeg encoder name
    codec: str
    hw: str | None  # nvenc, vaapi, qsv, amf, or None for software
    depths: tuple[int, ...]
    chromas: tuple[str, ...]
    quality_mode: str  # what RF becomes: CRF, CQ, ICQ, QP, or "fixed"
    lossless: bool  # has a real lossless mode for RF 0
    bitrate: bool
    two_pass: bool
    speeds: tuple[str, ...] = ()
    default_speed: str | None = None
    grain_tune: bool = False
    rf_max: int = 51

    @property
    def label(self) -> str:
        return f"{self.name} ({self.hw or 'software'})"

    def pix_fmt(self, depth: int, chroma: str) -> str:
        if self.hw:
            return {8: "nv12", 10: "p010le"}[depth]
        if depth == 8:
            return f"yuv{chroma}p"
        return f"yuv{chroma}p{depth}le"


_HW420_8_10 = dict(depths=(8, 10), chromas=("420",))
_HW420_8 = dict(depths=(8,), chromas=("420",))

ENCODERS: dict[str, EncoderSpec] = {e.name: e for e in [
    # software
    EncoderSpec("libx265", "h265", None, (8, 10), ("420", "422", "444"), "CRF", True, True, True,
                X26X_SPEEDS, "medium", True),
    EncoderSpec("libx264", "h264", None, (8, 10), ("420", "422", "444"), "CRF", True, True, True,
                X26X_SPEEDS, "medium", True),
    EncoderSpec("libsvtav1", "av1", None, (8, 10), ("420",), "CRF", False, True, False,
                SVT_SPEEDS, "6", False, rf_max=63),
    EncoderSpec("prores_ks", "prores", None, (10,), ("422", "444"), "fixed", False, False, False),
    EncoderSpec("ffv1", "ffv1", None, (8, 10, 12, 16), ("420", "422", "444"), "fixed", True,
                False, False),
    # NVIDIA
    EncoderSpec("hevc_nvenc", "h265", "nvenc", quality_mode="CQ", lossless=True, bitrate=True,
                two_pass=False, speeds=NVENC_SPEEDS, default_speed="p5", **_HW420_8_10),
    EncoderSpec("h264_nvenc", "h264", "nvenc", quality_mode="CQ", lossless=True, bitrate=True,
                two_pass=False, speeds=NVENC_SPEEDS, default_speed="p5", **_HW420_8),
    EncoderSpec("av1_nvenc", "av1", "nvenc", quality_mode="CQ", lossless=False, bitrate=True,
                two_pass=False, speeds=NVENC_SPEEDS, default_speed="p5", **_HW420_8_10),
    # VAAPI: AMD and Intel on Linux
    EncoderSpec("hevc_vaapi", "h265", "vaapi", quality_mode="QP", lossless=False, bitrate=True,
                two_pass=False, **_HW420_8_10),
    EncoderSpec("h264_vaapi", "h264", "vaapi", quality_mode="QP", lossless=False, bitrate=True,
                two_pass=False, **_HW420_8),
    EncoderSpec("av1_vaapi", "av1", "vaapi", quality_mode="QP", lossless=False, bitrate=True,
                two_pass=False, **_HW420_8_10),
    # Intel Quick Sync
    EncoderSpec("hevc_qsv", "h265", "qsv", quality_mode="ICQ", lossless=False, bitrate=True,
                two_pass=False, speeds=QSV_SPEEDS, default_speed="medium", **_HW420_8_10),
    EncoderSpec("h264_qsv", "h264", "qsv", quality_mode="ICQ", lossless=False, bitrate=True,
                two_pass=False, speeds=QSV_SPEEDS, default_speed="medium", **_HW420_8),
    EncoderSpec("av1_qsv", "av1", "qsv", quality_mode="ICQ", lossless=False, bitrate=True,
                two_pass=False, speeds=QSV_SPEEDS, default_speed="medium", **_HW420_8_10),
    # AMD AMF (mostly Windows)
    EncoderSpec("hevc_amf", "h265", "amf", quality_mode="QP", lossless=False, bitrate=True,
                two_pass=False, speeds=AMF_SPEEDS, default_speed="balanced", **_HW420_8_10),
    EncoderSpec("h264_amf", "h264", "amf", quality_mode="QP", lossless=False, bitrate=True,
                two_pass=False, speeds=AMF_SPEEDS, default_speed="balanced", **_HW420_8),
    EncoderSpec("av1_amf", "av1", "amf", quality_mode="QP", lossless=False, bitrate=True,
                two_pass=False, speeds=AMF_SPEEDS, default_speed="balanced", **_HW420_8_10),
]}

SOFTWARE = {"h265": "libx265", "h264": "libx264", "av1": "libsvtav1", "prores": "prores_ks",
            "ffv1": "ffv1"}

CONTAINERS = {
    ".mkv": set(CODECS),
    ".mp4": {"h264", "h265", "av1"},
    ".mov": {"prores"},
}

# audio that can be copied as is, and what can be encoded, per container
AUDIO_COPY_OK = {
    ".mkv": None,  # anything
    ".mp4": {"aac", "mp3", "ac3", "eac3", "opus", "alac", "flac"},
    ".mov": {"aac", "alac", "mp3", "ac3", "pcm_s16le", "pcm_s24le", "pcm_s16be", "pcm_s24be"},
}
AUDIO_ENCODE_OK = {
    ".mkv": ("aac", "opus", "flac"),
    ".mp4": ("aac", "opus", "flac"),
    ".mov": ("aac",),
}


def encoders_for(codec: str) -> list[EncoderSpec]:
    return [e for e in ENCODERS.values() if e.codec == codec]


def container_allows(ext: str, codec: str) -> bool:
    return codec in CONTAINERS.get(ext, set())


def audio_copy_ok(ext: str, codec_name: str) -> bool:
    ok = AUDIO_COPY_OK.get(ext)
    return True if ok is None else codec_name in ok
