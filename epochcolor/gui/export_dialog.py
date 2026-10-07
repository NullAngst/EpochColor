"""Export dialogs: video (the full HandBrake-style matrix) and photos."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
    QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget,
)

from ..export.codecs import ENCODERS, SOFTWARE, AUDIO_ENCODE_OK
from ..export.plan import ExportError, ExportSettings, resolve, tracks_needing_choice
from ..export.presets import list_presets, load_preset, save_preset

CODEC_NAMES = [("h265", "H.265"), ("h264", "H.264"), ("av1", "AV1"), ("prores", "ProRes"),
               ("ffv1", "FFV1 lossless")]
CONTAINER_FOR = {"h265": ".mkv", "h264": ".mp4", "av1": ".mkv", "prores": ".mov", "ffv1": ".mkv"}


class _DetectThread(QObject):
    done = Signal(object)

    @Slot()
    def run(self):
        try:
            from ..export.detect import Detector
            from ..export.writer import ffmpeg_path
            from ..video.pipeline import cache_root

            det = Detector(ffmpeg_path(), cache_root() / "encoders.json")
            det.table()  # runs and caches every test
            self.done.emit(det)
        except Exception as e:  # no ffmpeg
            self.done.emit(e)


class ExportDialog(QDialog):
    def __init__(self, project, audio, edited: bool, default_out: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        self.setMinimumWidth(560)
        self.project = project
        self.audio = audio
        self.edited = edited
        self.det = None
        self.settings = ExportSettings(**{k: v for k, v in project.d.export.items()
                                          if k in ExportSettings.__dataclass_fields__}) \
            if project.d.export else ExportSettings()

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItem("(current settings)", None)
        for name, src, desc in list_presets():
            self.preset.addItem(f"{name}  ({src})", name)
            self.preset.setItemData(self.preset.count() - 1, desc, 3)
        self.preset.currentIndexChanged.connect(self._load_preset)
        save = QPushButton("Save preset...")
        save.clicked.connect(self._save_preset)
        top.addWidget(QLabel("Preset"))
        top.addWidget(self.preset, 1)
        top.addWidget(save)
        lay.addLayout(top)

        vid = QGroupBox("Video")
        f = QFormLayout(vid)
        self.codec = QComboBox()
        for key, label in CODEC_NAMES:
            self.codec.addItem(label, key)
        self.bits = QComboBox()
        for b in (8, 10, 12, 16):
            self.bits.addItem(f"{b}-bit", b)
        self.chroma = QComboBox()
        for c in ("420", "422", "444"):
            self.chroma.addItem(f"{c[0]}:{c[1]}:{c[2]}", c)
        self.encoder = QComboBox()
        self.speed = QComboBox()
        self.speed.setEditable(True)
        f.addRow("Codec", self.codec)
        f.addRow("Bit depth", self.bits)
        f.addRow("Chroma", self.chroma)
        f.addRow("Encoder", self.encoder)
        f.addRow("Speed", self.speed)

        q = QWidget()
        ql = QHBoxLayout(q)
        ql.setContentsMargins(0, 0, 0, 0)
        self.by_rf = QRadioButton("Constant quality RF")
        self.rf = QDoubleSpinBox()
        self.rf.setRange(0, 63)
        self.rf.setDecimals(1)
        self.by_rate = QRadioButton("Average kbps")
        self.rate = QSpinBox()
        self.rate.setRange(100, 500000)
        self.rate.setSingleStep(500)
        self.two_pass = QCheckBox("Two-pass")
        for w in (self.by_rf, self.rf, self.by_rate, self.rate, self.two_pass):
            ql.addWidget(w)
        f.addRow("Quality", q)
        self.tune = QCheckBox("Grain tuning (x264, x265)")
        self.film = QSpinBox()
        self.film.setRange(0, 50)
        self.film.setSpecialValueText("off")
        f.addRow("", self.tune)
        f.addRow("AV1 film grain synthesis", self.film)
        lay.addWidget(vid)

        aud = QGroupBox("Audio")
        af = QFormLayout(aud)
        self.audio_note = QLabel()
        self.audio_note.setWordWrap(True)
        self.fallback = QComboBox()
        af.addRow(self.audio_note)
        af.addRow("Re-encode to", self.fallback)
        self.audio_bitrate = QSpinBox()
        self.audio_bitrate.setRange(0, 1024)
        self.audio_bitrate.setSpecialValueText("by channels")
        self.audio_bitrate.setSuffix(" kbps")
        af.addRow("AAC/Opus bitrate", self.audio_bitrate)
        lay.addWidget(aud)

        out = QGroupBox("Output")
        of = QFormLayout(out)
        self.whole = QRadioButton("Whole timeline")
        self.inout = QRadioButton("In to out")
        rr = QWidget()
        rl = QHBoxLayout(rr)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.whole)
        rl.addWidget(self.inout)
        rl.addStretch(1)
        self.inout.setEnabled(project.d.range is not None)
        (self.inout if project.d.range else self.whole).setChecked(True)
        of.addRow("Range", rr)
        pr = QWidget()
        pl = QHBoxLayout(pr)
        pl.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit(str(default_out))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        pl.addWidget(self.path, 1)
        pl.addWidget(browse)
        of.addRow("File", pr)
        lay.addWidget(out)

        self.plan_text = QPlainTextEdit()
        self.plan_text.setReadOnly(True)
        self.plan_text.setMaximumHeight(110)
        lay.addWidget(self.plan_text)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)

        self._fill(self.settings)
        for w in (self.codec, self.bits, self.chroma, self.encoder, self.fallback):
            w.currentIndexChanged.connect(self._changed)
        self.speed.currentTextChanged.connect(self._changed)
        for w in (self.rf, self.rate, self.film, self.audio_bitrate):
            w.valueChanged.connect(self._changed)
        for w in (self.by_rf, self.by_rate, self.two_pass, self.tune, self.whole, self.inout):
            w.toggled.connect(self._changed)
        self.path.textChanged.connect(self._changed)
        self.codec.currentIndexChanged.connect(self._codec_changed)

        # hardware tests run in the background; the plan updates when they finish
        self._thread = QThread(self)
        self._det = _DetectThread()
        self._det.moveToThread(self._thread)
        self._thread.started.connect(self._det.run)
        self._det.done.connect(self._detected)
        self._thread.start()
        self._changed()

    # ------------------------------------------------------------ fill

    def _fill(self, s: ExportSettings) -> None:
        widgets = (self.codec, self.bits, self.chroma, self.encoder, self.speed, self.fallback)
        for w in widgets:
            w.blockSignals(True)
        self.codec.setCurrentIndex(max(0, self.codec.findData(s.codec)))
        self.bits.setCurrentIndex(max(0, self.bits.findData(s.bits)))
        self.chroma.setCurrentIndex(max(0, self.chroma.findData(s.chroma)))
        self._fill_encoders(s.encoder)
        self._fill_speeds(s.speed)
        if s.bitrate:
            self.by_rate.setChecked(True)
            self.rate.setValue(int(s.bitrate))
        else:
            self.by_rf.setChecked(True)
            self.rf.setValue(float(s.rf if s.rf is not None else 18))
        self.two_pass.setChecked(bool(s.two_pass))
        self.tune.setChecked(bool(s.tune_grain))
        self.film.setValue(int(s.film_grain or 0))
        self.audio_bitrate.setValue(int(s.audio_bitrate or 0))
        self._fill_fallback(s.audio_fallback or s.audio_codec)
        p = Path(self.path.text())
        self.path.setText(str(p.with_suffix(s.container if s.container else CONTAINER_FOR[s.codec])))
        for w in widgets:
            w.blockSignals(False)

    def _fill_encoders(self, current: str = "auto") -> None:
        codec = self.codec.currentData()
        self.encoder.clear()
        self.encoder.addItem("auto (hardware if it works)", "auto")
        self.encoder.addItem("software", "software")
        self.encoder.addItem("hardware", "hardware")
        for e in ENCODERS.values():
            if e.codec == codec:
                ok = ""
                if self.det is not None:
                    why = self.det.check(e, e.depths[-1] if 10 not in e.depths else 10, e.chromas[0])
                    ok = "" if why is None else "  (not working here)"
                self.encoder.addItem(f"{e.name}{ok}", e.name)
        self.encoder.setCurrentIndex(max(0, self.encoder.findData(current)))

    def _fill_speeds(self, current: str | None) -> None:
        enc = self.encoder.currentData()
        spec = ENCODERS.get(enc) or ENCODERS[SOFTWARE[self.codec.currentData()]]
        self.speed.clear()
        self.speed.addItem("(default)", None)
        for sp in spec.speeds:
            self.speed.addItem(sp, sp)
        if current:
            i = self.speed.findData(current)
            if i < 0:
                self.speed.addItem(current, current)
                i = self.speed.count() - 1
            self.speed.setCurrentIndex(i)

    def _fill_fallback(self, current: str | None) -> None:
        ext = Path(self.path.text()).suffix.lower() if hasattr(self, "path") else ".mkv"
        choices = AUDIO_ENCODE_OK.get(ext, ("aac", "opus", "flac"))
        self.fallback.blockSignals(True)
        self.fallback.clear()
        names = {"aac": "AAC", "opus": "Opus", "flac": "FLAC"}
        for c in choices:
            self.fallback.addItem(names[c], c)
        self.fallback.setCurrentIndex(max(0, self.fallback.findData(current)))
        self.fallback.blockSignals(False)

    # --------------------------------------------------------- reading

    def current(self) -> ExportSettings:
        s = ExportSettings(
            codec=self.codec.currentData(), encoder=self.encoder.currentData() or "auto",
            bits=self.bits.currentData(), chroma=self.chroma.currentData(),
            container=Path(self.path.text()).suffix.lower() or ".mkv",
            rf=None if self.by_rate.isChecked() else float(self.rf.value()),
            bitrate=int(self.rate.value()) if self.by_rate.isChecked() else None,
            two_pass=self.two_pass.isChecked() and self.by_rate.isChecked(),
            speed=self.speed.currentData() if self.speed.currentData() is not None
            else (self.speed.currentText() if self.speed.currentText() not in ("", "(default)") else None),
            tune_grain=self.tune.isChecked(), film_grain=int(self.film.value()),
            audio_fallback=self.fallback.currentData(),
            audio_bitrate=int(self.audio_bitrate.value()) or None,
        )
        return s

    def use_range(self) -> bool:
        return self.inout.isChecked()

    def out_path(self) -> Path:
        return Path(self.path.text()).expanduser()

    # --------------------------------------------------------- events

    def _codec_changed(self) -> None:
        codec = self.codec.currentData()
        self._fill_encoders("auto")
        self._fill_speeds(None)
        if codec == "prores":
            self.bits.setCurrentIndex(self.bits.findData(10))
            if self.chroma.currentData() == "420":
                self.chroma.setCurrentIndex(self.chroma.findData("422"))
        p = Path(self.path.text())
        from ..export.codecs import CONTAINERS

        if codec not in CONTAINERS.get(p.suffix.lower(), set()):
            self.path.setText(str(p.with_suffix(CONTAINER_FOR[codec])))

    def _changed(self, *_):
        if self.sender() is self.encoder:
            self._fill_speeds(self.speed.currentData())
        if self.sender() is self.path:
            self._fill_fallback(self.fallback.currentData())
        self.rf.setEnabled(self.by_rf.isChecked())
        self.rate.setEnabled(self.by_rate.isChecked())
        self.two_pass.setEnabled(self.by_rate.isChecked())
        s = self.current()
        ext = s.container
        try:
            stuck = tracks_needing_choice(s, ext, self.audio, self.edited) if self.audio else []
        except ExportError:
            stuck = []
        on = [a for a in self.audio]
        if not on:
            self.audio_note.setText("No audio tracks are switched on.")
        elif self.edited:
            self.audio_note.setText(
                "The timeline cuts or joins the audio. Compressed audio frames don't line up "
                "with video frames, so every track gets re-encoded:\n" +
                "\n".join(f"  track {a.index + 1}: {a.codec}, {a.channels}ch" for a in stuck))
        elif stuck:
            self.audio_note.setText(
                f"These tracks can't go into {ext} as they are and get re-encoded:\n" +
                "\n".join(f"  track {a.index + 1}: {a.codec}, {a.channels}ch" for a in stuck))
        else:
            self.audio_note.setText("Every track is copied as it is.")
        self.fallback.setEnabled(bool(stuck))
        probe = self.det.check if self.det is not None else None
        try:
            plan = resolve(ExportSettings(**s.to_dict()), self.out_path(), self.audio, probe=probe,
                           edited=self.edited)
            text = plan.describe()
            if plan.warnings:
                text += "\n" + "\n".join(f"warning: {w}" for w in plan.warnings)
            if self.det is None:
                text += "\n(still testing hardware encoders...)"
            self.plan_text.setPlainText(text)
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except (ExportError, ValueError) as e:
            self.plan_text.setPlainText(f"Can't export like this: {e}")
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def _detected(self, det) -> None:
        self._thread.quit()
        if isinstance(det, Exception):
            self.plan_text.setPlainText(f"ffmpeg problem: {det}")
            return
        self.det = det
        self._fill_encoders(self.encoder.currentData() or "auto")
        self._changed()

    def _load_preset(self) -> None:
        name = self.preset.currentData()
        if not name:
            return
        try:
            s, _ = load_preset(name)
        except ExportError as e:
            self.plan_text.setPlainText(str(e))
            return
        self._fill(s)
        self._changed()

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok and name:
            try:
                path = save_preset(name.strip(), self.current())
                self.plan_text.appendPlainText(f"saved {path}")
            except ExportError as e:
                self.plan_text.appendPlainText(str(e))

    def _browse(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "Export to", self.path.text(),
                                           "Video (*.mkv *.mp4 *.mov)")
        if f:
            self.path.setText(f)

    def done(self, r) -> None:
        self._thread.quit()
        self._thread.wait(3000)
        super().done(r)


class PhotoExportDialog(QDialog):
    def __init__(self, count: int, default_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export photos")
        f = QFormLayout(self)
        f.addRow(QLabel(f"{count} photo(s), colorized if they aren't yet."))
        self.format = QComboBox()
        for label, ext in (("PNG", "png"), ("TIFF 16-bit", "tif"), ("JPEG", "jpg"), ("WebP", "webp")):
            self.format.addItem(label, ext)
        self.bits = QComboBox()
        self.bits.addItem("16-bit", 16)
        self.bits.addItem("8-bit", 8)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(95)
        d = QWidget()
        dl = QHBoxLayout(d)
        dl.setContentsMargins(0, 0, 0, 0)
        self.dir = QLineEdit(str(default_dir))
        b = QPushButton("Browse...")
        b.clicked.connect(self._browse)
        dl.addWidget(self.dir, 1)
        dl.addWidget(b)
        f.addRow("Format", self.format)
        f.addRow("PNG bit depth", self.bits)
        f.addRow("JPEG/WebP quality", self.quality)
        f.addRow("Folder", d)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.format.currentIndexChanged.connect(self._fmt)
        self._fmt()

    def _fmt(self) -> None:
        ext = self.format.currentData()
        self.bits.setEnabled(ext == "png")
        self.quality.setEnabled(ext in ("jpg", "webp"))

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Export to", self.dir.text())
        if d:
            self.dir.setText(d)

    def values(self) -> dict:
        ext = self.format.currentData()
        bits = self.bits.currentData() if ext == "png" else (16 if ext == "tif" else None)
        return {"ext": ext, "bits": bits, "quality": self.quality.value(),
                "dir": Path(self.dir.text()).expanduser()}
