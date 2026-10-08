"""EpochColor main window."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from .. import __version__
from ..export.plan import ExportSettings
from ..media import MediaInfo, media_dir
from ..project import Clip, AudioInfo, Project, ProjectError
from ..timeline_export import analysis_state, project_audio, shot_states
from ..video.pipeline import VideoSettings, analysis_dir
from .. import grade as G
from . import theme
from .export_dialog import ExportDialog, PhotoExportDialog
from .grade_panel import GradePanel
from .jobs import JobRunner
from .panels import ColoursPanel, PaintBar, ScopesPanel
from .preview import LayoutPiece, PreviewProvider, to_qimage
from .timeline import MediaCache, Timeline, timecode
from .viewer import Viewer

VIDEO_FILTER = "Video (*.mkv *.mp4 *.mov *.avi *.m4v *.webm *.mpg *.mpeg *.ts *.mxf *.dv);;All files (*)"
PHOTO_FILTER = ("Photos (*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp *.heic *.heif *.dng *.cr2 "
                "*.cr3 *.nef *.arw *.raf *.orf *.rw2 *.pef);;All files (*)")
PROJECT_FILTER = "EpochColor project (*.epochcolor)"

KEYS_HELP = """\
Space        play / pause
J  K  L      shuttle back / stop / forward (press again to go faster)
Left/Right   one frame
Shift+L/R    previous / next shot
Up/Down      previous / next edit point
Home / End   start / end
I  O         mark in / out (Alt+X clears)
S            split at the playhead
Delete       ripple delete the selected segment
Alt+Left/R   move the selected segment
B            add or remove a shot cut at the playhead
M            marker with a note
1  2  3  4    colour / before-after split / original / this frame
Ctrl+F       colorize this frame with the selected model (shows as 4)
P            paint on/off; [ and ] (or Shift+wheel) brush size
             drag paints, right-click removes a stroke, Ctrl+click picks a colour
Wheel        zoom the picture at the pointer; middle-drag pans; Ctrl+0 fits
Ctrl+Shift+C color shot: this shot from its paint alone
Ctrl+Shift+F fill shot: the model colours the rest of this shot
H            show or hide strokes
F            paint frame by frame: strokes colour only their own frame (or a set reach)
N  Shift+N   carry this frame's strokes to the next / previous frame along the motion
Ctrl+Return  apply hints now
Ctrl+R       colorize the selected clip (Ctrl+Shift+R: all)
Ctrl+Alt+C/V copy / paste a shot's grade
Ctrl+E       export video
Ctrl+Z       undo, Ctrl+Shift+Z redo
Ctrl+wheel   zoom the timeline, Shift+Z fits it"""


class Inspector(QWidget):
    settingChanged = Signal(str, object)
    negativeToggled = Signal(bool)
    negativeLevels = Signal()
    referencePick = Signal()
    referencePainted = Signal(str)  # "shot" or "everywhere": use my painting as the reference
    referenceClear = Signal()
    referenceStrength = Signal(float)
    shotAction = Signal(str)  # color_shot, fill_shot, colorize_all
    gotoFrame = Signal(int)  # a painted frame of this shot, in clip frames

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        g = QGroupBox("Selected")
        gl = QVBoxLayout(g)
        self.clip_info = QLabel("Nothing selected")
        self.clip_info.setWordWrap(True)
        self.clip_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        gl.addWidget(self.clip_info)
        row = QHBoxLayout()
        self.colorize_btn = QPushButton("Colorize clip")
        self.colorize_all_btn = QPushButton("Colorize all")
        row.addWidget(self.colorize_btn)
        row.addWidget(self.colorize_all_btn)
        gl.addLayout(row)
        row = QHBoxLayout()
        self.negative = QCheckBox("Negative")
        self.negative.setToolTip("This clip or photo is a black and white negative: invert it before "
                                 "anything else. The film base and levels are measured automatically.")
        self.neg_levels = QPushButton("Levels...")
        self.neg_levels.setToolTip("Film base, black and white points and contrast of the inversion")
        row.addWidget(self.negative)
        row.addStretch(1)
        row.addWidget(self.neg_levels)
        gl.addLayout(row)
        self.negative.clicked.connect(lambda on: self.negativeToggled.emit(bool(on)))
        self.neg_levels.clicked.connect(self.negativeLevels.emit)
        lay.addWidget(g)

        # ---- This shot: where painting turns into colour
        sg = QGroupBox("This shot")
        sl = QVBoxLayout(sg)
        self.shot_info = QLabel("Put the playhead on a clip")
        self.shot_info.setWordWrap(True)
        self.shot_info.setTextFormat(Qt.RichText)
        sl.addWidget(self.shot_info)
        self.painted_frames = QComboBox()
        self.painted_frames.setToolTip("Painted frames in this shot; pick one to go there")
        self.painted_frames.activated.connect(
            lambda i: self.painted_frames.itemData(i) is not None and self.gotoFrame.emit(self.painted_frames.itemData(i)))
        sl.addWidget(self.painted_frames)
        row = QHBoxLayout()
        self.color_shot_btn = QPushButton("Color shot")
        self.color_shot_btn.setToolTip("Colour this shot from your paint alone: each painted object filled out to "
                                       "its edges and carried through the shot, everything else left grey. No model, "
                                       "so it takes seconds. Shows exactly what your painting does. (Ctrl+Shift+C)")
        self.fill_shot_btn = QPushButton("Fill shot")
        self.fill_shot_btn.setToolTip("Let the model colour the rest of this shot. Your paint keeps every object you "
                                      "painted; the model only colours what you didn't. (Ctrl+Shift+F)")
        self.shot_all_btn = QPushButton("Colorize all")
        self.shot_all_btn.setToolTip("Every shot of every clip: painted shots with their paint, the rest by the "
                                     "model, learning from your painting if that's switched on below.")
        for b, name in ((self.color_shot_btn, "color_shot"), (self.fill_shot_btn, "fill_shot"),
                        (self.shot_all_btn, "colorize_all")):
            b.clicked.connect(lambda _=False, n=name: self.shotAction.emit(n))
            row.addWidget(b)
        sl.addLayout(row)
        self.ref_label = QLabel("No reference")
        self.ref_label.setWordWrap(True)
        self.ref_label.setStyleSheet(f"color: {theme.DIM};")
        sl.addWidget(self.ref_label)
        row = QHBoxLayout()
        self.ref_btn = QPushButton("Reference...")
        self.ref_btn.setToolTip("What guides this shot's colours besides your paint: a colour photo of the same "
                                "place or era, or your own painting")
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self.ref_btn)
        menu.addAction("A colour photo...", self.referencePick.emit)
        menu.addAction("My painting in this shot", lambda: self.referencePainted.emit("shot"))
        menu.addAction("My painting everywhere in the project", lambda: self.referencePainted.emit("everywhere"))
        self.ref_btn.setMenu(menu)
        self.ref_clear = QPushButton("Clear")
        self.ref_strength = QSpinBox()
        self.ref_strength.setRange(0, 100)
        self.ref_strength.setValue(100)
        self.ref_strength.setSuffix(" %")
        self.ref_strength.setToolTip("How strongly the reference's colours win over the model's")
        row.addWidget(self.ref_btn, 1)
        row.addWidget(self.ref_strength)
        row.addWidget(self.ref_clear)
        sl.addLayout(row)
        self.ref_clear.clicked.connect(self.referenceClear.emit)
        self.ref_strength.editingFinished.connect(lambda: self.referenceStrength.emit(self.ref_strength.value() / 100))
        row = QHBoxLayout()
        self.teach = QCheckBox("Unpainted shots learn from my painting")
        self.teach.setToolTip("On Colorize all, shots you didn't paint look for the things you painted elsewhere "
                              "(by what the model sees, not by position) and take their colour. The model "
                              "colours whatever doesn't match.")
        self.teach_strength = QSpinBox()
        self.teach_strength.setRange(0, 100)
        self.teach_strength.setSuffix(" %")
        row.addWidget(self.teach, 1)
        row.addWidget(self.teach_strength)
        sl.addLayout(row)
        self.teach.toggled.connect(lambda v: self.settingChanged.emit("teach", bool(v)))
        self.teach_strength.editingFinished.connect(
            lambda: self.settingChanged.emit("teach_strength", round(self.teach_strength.value() / 100, 2)))
        how = QLabel("Paint a few frames, Color shot to see your paint alone, Fill shot to let the model do the "
                     "rest, Colorize all when you're happy. The strip under each shot on the timeline: grey not "
                     "coloured, amber painted only, green coloured, striped out of date.")
        how.setWordWrap(True)
        how.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        sl.addWidget(how)
        lay.addWidget(sg)

        c = QGroupBox("Colour")
        f = QFormLayout(c)
        self.model = QComboBox()
        self.manager_btn = QPushButton("Models...")
        self.manager_btn.setToolTip("Download, add and remove models")
        mrow = QWidget()
        ml = QHBoxLayout(mrow)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.model, 1)
        ml.addWidget(self.manager_btn)
        self.working = QSpinBox()
        self.working.setRange(128, 2048)
        self.working.setSingleStep(64)
        self.working.setSuffix(" px")
        self.chroma = QSpinBox()
        self.chroma.setRange(128, 1024)
        self.chroma.setSingleStep(64)
        self.chroma.setSuffix(" px")
        self.chroma.setToolTip("Short side of the stored colour. Higher keeps edges between objects of "
                               "the same grey crisper, and costs disk: 256 px is about 1 GB per minute, "
                               "512 px about 4 GB.")
        self.stabilize = QDoubleSpinBox()
        self.stabilize.setRange(0, 0.99)
        self.stabilize.setSingleStep(0.05)
        self.grain = QDoubleSpinBox()
        self.grain.setRange(0, 100)
        self.grain.setSuffix(" %")
        self.saturation = QDoubleSpinBox()
        self.saturation.setRange(0, 3)
        self.saturation.setSingleStep(0.05)
        self.cast = QSpinBox()
        self.cast.setRange(0, 100)
        self.cast.setSingleStep(10)
        self.cast.setSuffix(" %")
        self.cast.setToolTip("Takes out the all-over tint a model lays on a whole shot (the orange or sepia "
                             "wash). It measures how far the least colourful parts of each shot lean, and "
                             "shifts the colour back by that much. 100% takes all of it out; a shot that's "
                             "honestly warm, like a sunset, may want less.")
        self.denoise_auto = QCheckBox("auto")
        self.denoise = QDoubleSpinBox()
        self.denoise.setRange(0, 30)
        self.denoise.setSingleStep(0.5)
        dn = QWidget()
        dl = QHBoxLayout(dn)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self.denoise_auto)
        dl.addWidget(self.denoise, 1)
        f.addRow("Model", mrow)
        f.addRow("Working size", self.working)
        f.addRow("Chroma size", self.chroma)
        f.addRow("Stabilize", self.stabilize)
        f.addRow("Grain kept", self.grain)
        f.addRow("Saturation", self.saturation)
        f.addRow("Remove colour cast", self.cast)
        f.addRow("Denoise", dn)
        note = QLabel("Model, working size, chroma size and denoise need a new colorize pass. "
                      "Stabilize reruns a quick pass. Grain, saturation and cast removal apply right away.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        f.addRow(note)
        lay.addWidget(c)

        fg = QGroupBox("Film")
        ff = QFormLayout(fg)
        self.dfl = QCheckBox("Deflicker")
        self.dust = QCheckBox("Remove dust")
        self.dfl_out = QCheckBox("Deflicker")
        self.dust_out = QCheckBox("Remove dust")
        for w, tip in ((self.dfl, "Steady the brightness of the copy the model sees, so flicker doesn't turn "
                                  "into colour flicker. Needs a new colorize pass."),
                       (self.dust, "Clean dirt specks off the copy the model sees. Needs a new colorize pass."),
                       (self.dfl_out, "Steady the brightness of the output picture too. Changes the luma."),
                       (self.dust_out, "Clean dirt specks off the output picture too. Changes the luma; "
                                       "shows on the paused full-size frame and in export.")):
            w.setToolTip(tip)
        r1, r2 = QWidget(), QWidget()
        for r, a, b in ((r1, self.dfl, self.dust), (r2, self.dfl_out, self.dust_out)):
            hl = QHBoxLayout(r)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(a)
            hl.addWidget(b)
            hl.addStretch(1)
        ff.addRow("Model's copy", r1)
        ff.addRow("Output", r2)
        self.regrain = QDoubleSpinBox()
        self.regrain.setRange(0, 20)
        self.regrain.setSingleStep(0.5)
        self.regrain.setSuffix(" %")
        self.regrain.setSpecialValueText("off")
        self.regrain.setToolTip("Synthetic grain after grading, for footage cleaned hard or matched to a look")
        self.grain_size = QDoubleSpinBox()
        self.grain_size.setRange(0.3, 4.0)
        self.grain_size.setSingleStep(0.1)
        self.grain_size.setSuffix(" px")
        self.grain_size.setToolTip("Grain size at 1080 lines, scaled to the frame")
        self.grain_colour = QSpinBox()
        self.grain_colour.setRange(0, 100)
        self.grain_colour.setSuffix(" %")
        self.grain_colour.setToolTip("Faint colour grain on top of the mono grain, like colour film stock")
        ff.addRow("Regrain", self.regrain)
        gr = QWidget()
        gl2 = QHBoxLayout(gr)
        gl2.setContentsMargins(0, 0, 0, 0)
        for w in (self.grain_size, QLabel("colour"), self.grain_colour):
            gl2.addWidget(w)
        ff.addRow("Grain size", gr)
        fnote = QLabel("Cleaning the model's copy needs a new colorize pass. Output cleaning and regrain show "
                       "on the paused full-size frame and in export.")
        fnote.setWordWrap(True)
        fnote.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        ff.addRow(fnote)
        lay.addWidget(fg)
        self.dfl.toggled.connect(lambda v: self.settingChanged.emit("deflicker", 1.0 if v else 0.0))
        self.dust.toggled.connect(lambda v: self.settingChanged.emit("dust", bool(v)))
        self.dfl_out.toggled.connect(lambda v: self.settingChanged.emit("deflicker_out", bool(v)))
        self.dust_out.toggled.connect(lambda v: self.settingChanged.emit("dust_out", bool(v)))

        j = QGroupBox("Jobs")
        jl = QVBoxLayout(j)
        self.job_label = QLabel("Idle")
        self.job_label.setWordWrap(True)
        self.job_bar = QProgressBar()
        self.job_bar.setRange(0, 1)
        self.job_bar.setValue(0)
        self.job_bar.setTextVisible(False)
        self.job_queue = QLabel("")
        self.job_queue.setStyleSheet(f"color: {theme.DIM};")
        self.job_queue.setWordWrap(True)
        row = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_all_btn = QPushButton("Cancel all")
        row.addWidget(self.cancel_btn)
        row.addWidget(self.cancel_all_btn)
        jl.addWidget(self.job_label)
        jl.addWidget(self.job_bar)
        jl.addWidget(self.job_queue)
        jl.addLayout(row)
        lay.addWidget(j)
        lay.addStretch(1)

        self.model.currentIndexChanged.connect(lambda: self.settingChanged.emit("model", self.model.currentData()))
        # spin boxes settle for a moment before a change counts, so holding an
        # arrow is one undo step and one preview refresh, not twenty
        self._pending: dict[str, object] = {}
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(450)
        self._debounce.timeout.connect(self._flush)
        self.working.valueChanged.connect(lambda v: self._queue("working_size", int(v)))
        self.chroma.valueChanged.connect(lambda v: self._queue("chroma_size", int(v)))
        self.stabilize.valueChanged.connect(lambda v: self._queue("stabilize", round(v, 3)))
        self.grain.valueChanged.connect(lambda v: self._queue("grain", float(v)))
        self.saturation.valueChanged.connect(lambda v: self._queue("saturation", round(v, 3)))
        self.cast.valueChanged.connect(lambda v: self._queue("cast", round(v / 100.0, 2)))
        self.regrain.valueChanged.connect(lambda v: self._queue("regrain", round(v, 2)))
        self.grain_size.valueChanged.connect(lambda v: self._queue("regrain_size", round(v, 2)))
        self.grain_colour.valueChanged.connect(lambda v: self._queue("regrain_chroma", round(v / 100.0, 2)))
        self.denoise.valueChanged.connect(lambda v: self._denoise())
        self.denoise_auto.toggled.connect(self._denoise)

    def _queue(self, key: str, value) -> None:
        self._pending[key] = value
        self._debounce.start()

    def _flush(self) -> None:
        pending, self._pending = self._pending, {}
        for k, v in pending.items():
            self.settingChanged.emit(k, v)

    def _denoise(self, *_):
        self.denoise.setEnabled(not self.denoise_auto.isChecked())
        self._queue("denoise", None if self.denoise_auto.isChecked() else round(self.denoise.value(), 2))

    def set_models(self, current: str) -> None:
        from ..models import choices

        self.model.blockSignals(True)
        self.model.clear()
        for mid, label, have in choices():
            self.model.addItem(label if have else f"{label} (not downloaded)", mid)
            if not have:
                item = self.model.model().item(self.model.count() - 1)
                item.setEnabled(False)
        if self.model.findData(current) < 0:
            self.model.addItem(f"{current} (not installed)", current)
        self.model.setCurrentIndex(self.model.findData(current))
        self.model.blockSignals(False)

    def show_settings(self, s: dict) -> None:
        widgets = (self.model, self.working, self.chroma, self.stabilize, self.grain, self.saturation,
                   self.cast, self.denoise, self.denoise_auto, self.dfl, self.dust, self.dfl_out, self.dust_out,
                   self.regrain, self.grain_size, self.grain_colour, self.teach, self.teach_strength)
        for w in widgets:
            w.blockSignals(True)
        if self.model.findData(s.get("model")) < 0 or self.model.count() == 0:
            self.set_models(s.get("model"))
        self.model.setCurrentIndex(max(0, self.model.findData(s.get("model"))))
        self.working.setValue(int(s.get("working_size", 512)))
        self.chroma.setValue(int(s.get("chroma_size", 256)))
        self.stabilize.setValue(float(s.get("stabilize", 0.9)))
        self.grain.setValue(float(s.get("grain", 100)))
        self.saturation.setValue(float(s.get("saturation", 1.0)))
        self.cast.setValue(int(round(float(s.get("cast", 0.0)) * 100)))
        self.teach.setChecked(bool(s.get("teach", False)))
        self.teach_strength.setValue(int(round(float(s.get("teach_strength", 0.8)) * 100)))
        self.dfl.setChecked(float(s.get("deflicker", 0.0) or 0.0) > 0)
        self.dust.setChecked(bool(s.get("dust", False)))
        self.dfl_out.setChecked(bool(s.get("deflicker_out", False)))
        self.dust_out.setChecked(bool(s.get("dust_out", False)))
        self.regrain.setValue(float(s.get("regrain", 0.0) or 0.0))
        self.grain_size.setValue(float(s.get("regrain_size", 1.0)))
        self.grain_colour.setValue(int(round(float(s.get("regrain_chroma", 0.0)) * 100)))
        dn = s.get("denoise")
        self.denoise_auto.setChecked(dn is None)
        self.denoise.setEnabled(dn is not None)
        self.denoise.setValue(float(dn or 2.0))
        for w in widgets:
            w.blockSignals(False)


class MainWindow(QMainWindow):
    def __init__(self, project_path: str | None = None):
        super().__init__()
        self.qs = QSettings("EpochColor", "EpochColor")
        self.project = Project()
        self.media = MediaCache()
        self.media_info: dict[str, MediaInfo] = {}
        self.status: dict[str, str] = {}
        self.shot_status: dict = {}  # clip id -> shot_states()
        self.photo_icons: dict[str, QIcon] = {}
        self.playhead = 0
        self.speed = 0.0
        self._play_t0 = 0.0
        self._play_f0 = 0
        self._importing: dict[int, str] = {}  # job id -> path, for re-imports of known clips
        self._grade_before: dict | None = None  # project state when a run of grade edits began
        self._grade_clip: dict | None = None  # copied grade
        self._show_matte = False
        self._manager = None
        self._photo_raw = None

        self.runner = JobRunner(self)
        self.runner.changed.connect(self._jobs_changed)
        self.runner.finished.connect(self._job_done)
        self.runner.failed.connect(self._job_failed)

        self._build_ui()
        self._build_actions()
        self._start_preview()

        from .audio import TimelineAudio

        self.audio = TimelineAudio(self)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(90_000)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(8)
        self.play_timer.timeout.connect(self._tick)
        self._flow_readers: dict = {}  # proxy readers for carrying strokes, per clip
        self._measuring: dict = {}  # job id -> what a negative measurement is for
        self._frame_jobs: dict = {}  # job id -> (clip id, frame) of a Colorize frame
        self._frame_results: dict = {}  # (clip id, frame) -> (picture, label)
        self.hint_timer = QTimer(self)
        self.hint_timer.setSingleShot(True)
        self.hint_timer.setInterval(900)
        self.hint_timer.timeout.connect(self.apply_hints)
        self.grade_timer = QTimer(self)
        self.grade_timer.setSingleShot(True)
        self.grade_timer.setInterval(500)
        self.grade_timer.timeout.connect(self._commit_grade)

        self.resize(1440, 900)
        if project_path:
            self.open_project(project_path)
        self._refresh()

    # ------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.viewer = Viewer()
        self.viewer.modeChanged.connect(self._mode_changed)
        transport = QWidget()
        tl = QHBoxLayout(transport)
        tl.setContentsMargins(6, 2, 6, 2)
        self.btn = {}
        for key, text, tip in (("start", "|<", "Start (Home)"), ("back", "<", "Back one frame (Left)"),
                               ("play", "Play", "Play / pause (Space)"), ("fwd", ">", "Forward one frame (Right)"),
                               ("end", ">|", "End (End)")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            tl.addWidget(b)
            self.btn[key] = b
        self.tc = QLabel("00:00:00:00")
        self.tc.setStyleSheet("font-family: monospace; font-size: 14px; padding: 0 10px;")
        self.speed_label = QLabel("")
        self.speed_label.setStyleSheet(f"color: {theme.ACCENT};")
        tl.addWidget(self.tc)
        tl.addWidget(self.speed_label)
        tl.addStretch(1)
        self.frame_btn = QToolButton()
        self.frame_btn.setText("Colorize frame")
        self.frame_btn.setToolTip("Run the selected model on this one frame, with what's painted on it, and show "
                                  "it under This frame (Ctrl+F). Nothing else changes.")
        self.frame_btn.clicked.connect(self.colorize_frame)
        tl.addWidget(self.frame_btn)
        self.mode_btns = {}
        for mode, text in (("color", "Colour"), ("split", "Before / after"), ("original", "Original"),
                           ("frame", "This frame")):
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, m=mode: self.viewer.set_mode(m))
            tl.addWidget(b)
            self.mode_btns[mode] = b
        self.mode_btns["color"].setChecked(True)
        self.btn["start"].clicked.connect(lambda: self.seek(0))
        self.btn["end"].clicked.connect(lambda: self.seek(self.project.duration - 1))
        self.btn["back"].clicked.connect(lambda: self.step(-1))
        self.btn["fwd"].clicked.connect(lambda: self.step(1))
        self.btn["play"].clicked.connect(self.toggle_play)

        self.paintbar = PaintBar(self)
        self.paintbar.changed.connect(self._brush_changed)
        self.paintbar.action.connect(self._paint_action)
        self.viewer.strokeFinished.connect(self._stroke_added)
        self.viewer.eraseAt.connect(self._erase_at)
        self.viewer.picked.connect(self._picked)
        self.viewer.suggestionClicked.connect(self._suggestion_clicked)
        self.viewer.brushResized.connect(lambda r: self.paintbar.size.setValue(max(1, round(r * 1000))))
        top = QWidget()
        topl = QVBoxLayout(top)
        topl.setContentsMargins(0, 0, 0, 0)
        topl.setSpacing(0)
        topl.addWidget(self.paintbar)
        topl.addWidget(self.paintbar.options_bar)
        topl.addWidget(self.paintbar.frame_bar)
        topl.addWidget(self.viewer, 1)
        topl.addWidget(transport)

        self.timeline = Timeline(self.project, self.media)
        self.timeline.canvas.seek.connect(self.seek)
        self.timeline.canvas.selected.connect(lambda i: self._refresh_inspector())
        self.timeline.canvas.edited.connect(self._edited)
        self.timeline.canvas.audioToggled.connect(self._toggle_audio)

        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListWidget.IconMode)
        self.filmstrip.setFlow(QListWidget.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setIconSize(QSize(144, 108))
        self.filmstrip.setMovement(QListWidget.Static)
        self.filmstrip.setSelectionMode(QListWidget.ExtendedSelection)
        self.filmstrip.currentRowChanged.connect(self._show_photo)

        self.bottom = QTabWidget()
        self.bottom.addTab(self.timeline, "Timeline")
        self.bottom.addTab(self.filmstrip, "Photos")
        self.bottom.currentChanged.connect(self._bottom_tab)

        split = QSplitter(Qt.Vertical)
        split.addWidget(top)
        split.addWidget(self.bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([600, 260])
        self.setCentralWidget(split)

        self.clip_list = QListWidget()
        self.clip_list.itemDoubleClicked.connect(self._append_clip)
        self.clip_list.setToolTip("Double-click to add the whole clip to the end of the timeline again")
        dock = QDockWidget("Clips", self)
        dock.setObjectName("clips")
        dock.setWidget(self.clip_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.clips_dock = dock

        self.inspector = Inspector()
        self.inspector.settingChanged.connect(self._setting_changed)
        self.inspector.colorize_btn.clicked.connect(self.colorize_selected)
        self.inspector.negativeToggled.connect(self._negative_toggled)
        self.inspector.negativeLevels.connect(self.negative_levels)
        self.inspector.referencePick.connect(self.pick_reference)
        self.inspector.referenceClear.connect(self.clear_reference)
        self.inspector.referencePainted.connect(self._use_painting_reference)
        self.inspector.shotAction.connect(self._shot_action)
        self.inspector.gotoFrame.connect(lambda f: (t := self._paint_target()) and t[0] == "clip"
                                         and self.goto_clip_frame(t[1], f))
        self.inspector.referenceStrength.connect(self._reference_strength)
        self.inspector.colorize_all_btn.clicked.connect(self.colorize_all)
        self.inspector.cancel_btn.clicked.connect(lambda: self.runner.cancel(
            self.runner.current.id if self.runner.current else -1))
        self.inspector.cancel_all_btn.clicked.connect(lambda: self.runner.cancel(None))
        self.inspector.manager_btn.clicked.connect(self.open_model_manager)
        dock = QDockWidget("Inspector", self)
        dock.setObjectName("inspector")
        from PySide6.QtWidgets import QFrame, QScrollArea

        scroll = QScrollArea()
        scroll.setWidget(self.inspector)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # never narrower than what's in it, or the right edge gets cut off
        scroll.setMinimumWidth(self.inspector.minimumSizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width() + 4)
        dock.setWidget(scroll)  # scrolls on a short screen instead of squashing
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.inspector_dock = dock

        self.grade_panel = GradePanel()
        self.grade_panel.changed.connect(self._grade_edited)
        self.grade_panel.keyAction.connect(self._grade_key)
        self.grade_panel.action.connect(self._grade_action)
        gdock = QDockWidget("Grade", self)
        gdock.setObjectName("grade")
        gdock.setWidget(self.grade_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, gdock)
        self.tabifyDockWidget(dock, gdock)
        dock.raise_()
        self.grade_dock = gdock

        self.colours = ColoursPanel()
        self.colours.useColour.connect(self._use_colour)
        self.colours.request.connect(self._colour_request)
        self.colours.useItem.connect(self.use_item)
        self.colours.settingChanged.connect(self._setting_changed)
        cdock = QDockWidget("Items", self)
        cdock.setObjectName("colours")
        cdock.setWidget(self.colours)
        self.addDockWidget(Qt.LeftDockWidgetArea, cdock)
        self.tabifyDockWidget(self.clips_dock, cdock)
        self.colours_dock = cdock
        self.clips_dock.raise_()

        self.scopes = ScopesPanel()
        sdock = QDockWidget("Scopes", self)
        sdock.setObjectName("scopes")
        sdock.setWidget(self.scopes)
        self.addDockWidget(Qt.LeftDockWidgetArea, sdock)
        self.scopes_dock = sdock
        sdock.visibilityChanged.connect(lambda v: v and self.scopes.refresh())

    def _build_actions(self) -> None:
        mb = self.menuBar()

        def act(menu, text, slot, keys=None, tip=None):
            a = QAction(text, self)
            if keys:
                a.setShortcuts([QKeySequence(k) for k in (keys if isinstance(keys, list) else [keys])])
                a.setShortcutContext(Qt.WindowShortcut)
            a.triggered.connect(slot)
            if tip:
                a.setStatusTip(tip)
            menu.addAction(a)
            return a

        m = mb.addMenu("&File")
        act(m, "New project", self.new_project, "Ctrl+N")
        act(m, "Open project...", self.open_project_dialog, "Ctrl+O")
        act(m, "Save project", self.save_project, "Ctrl+S")
        act(m, "Save project as...", self.save_project_as, "Ctrl+Shift+S")
        m.addSeparator()
        act(m, "Add clips...", self.add_clips_dialog, "Ctrl+I")
        act(m, "Add photos...", self.add_photos_dialog, "Ctrl+Shift+I")
        m.addSeparator()
        act(m, "Export video...", self.export_video, "Ctrl+E")
        act(m, "Export photos...", self.export_photos, "Ctrl+Shift+E")
        act(m, "Export current frame...", self.export_frame, "Ctrl+Alt+E")
        m.addSeparator()
        act(m, "Working folder and limits...", self.open_settings)
        m.addSeparator()
        act(m, "Quit", self.close, "Ctrl+Q")

        m = mb.addMenu("&Edit")
        self.undo_act = act(m, "Undo", self.undo, "Ctrl+Z")
        self.redo_act = act(m, "Redo", self.redo, ["Ctrl+Shift+Z", "Ctrl+Y"])
        m.addSeparator()
        act(m, "Split at playhead", self.split, "S")
        act(m, "Ripple delete segment", self.delete_segment, ["Delete", "Backspace"])
        act(m, "Move segment left", lambda: self.move_segment(-1), "Alt+Left")
        act(m, "Move segment right", lambda: self.move_segment(1), "Alt+Right")
        m.addSeparator()
        act(m, "Mark in", self.mark_in, "I")
        act(m, "Mark out", self.mark_out, "O")
        act(m, "Clear in/out", self.clear_range, "Alt+X")
        act(m, "Add marker...", self.add_marker, "M")
        act(m, "Add or remove shot cut", self.toggle_cut, "B")
        m.addSeparator()
        act(m, "Remove clip from project", self.remove_clip)
        act(m, "Remove selected photos", self.remove_photos)
        m.addSeparator()
        act(m, "Copy grade", self.copy_grade, "Ctrl+Alt+C")
        act(m, "Paste grade", self.paste_grade, "Ctrl+Alt+V")
        act(m, "Reset grade", lambda: self._grade_action("reset"))

        m = mb.addMenu("&View")
        act(m, "Colour", lambda: self.viewer.set_mode("color"), "1")
        act(m, "Before / after split", lambda: self.viewer.set_mode("split"), "2")
        act(m, "Original black and white", lambda: self.viewer.set_mode("original"), "3")
        act(m, "This frame (Colorize frame result)", lambda: self.viewer.set_mode("frame"), "4")
        act(m, "Colorize this frame", self.colorize_frame, "Ctrl+F")
        m.addSeparator()
        act(m, "Zoom picture in", lambda: self.viewer.zoom_at(1.25), "Ctrl+Shift+=")
        act(m, "Zoom picture out", lambda: self.viewer.zoom_at(0.8), "Ctrl+Shift+-")
        act(m, "Fit picture", self.viewer.fit, "Ctrl+0")
        m.addSeparator()
        act(m, "Zoom timeline in", lambda: self._zoom(1.5), ["=", "Ctrl+="])
        act(m, "Zoom timeline out", lambda: self._zoom(1 / 1.5), ["-", "Ctrl+-"])
        act(m, "Fit timeline", self._fit, "Shift+Z")
        m.addSeparator()
        act(m, "Paint hints", lambda: self.paintbar.paint.toggle(), "P")
        act(m, "Paint frame by frame", self.toggle_frame_mode, "F")
        act(m, "Carry strokes to the next frame", lambda: self.carry_strokes(1), "N")
        act(m, "Carry strokes to the previous frame", lambda: self.carry_strokes(-1), "Shift+N")
        act(m, "Show strokes", lambda: self.paintbar.show.toggle(), "H")
        act(m, "Smaller brush", lambda: self.paintbar.size.setValue(int(self.paintbar.size.value() / 1.25)), "[")
        act(m, "Bigger brush", lambda: self.paintbar.size.setValue(int(self.paintbar.size.value() * 1.25) + 1), "]")

        m = mb.addMenu("&Playback")
        act(m, "Play / pause", self.toggle_play, "Space")
        act(m, "Shuttle back", lambda: self.shuttle(-1), "J")
        act(m, "Stop", lambda: self.shuttle(0), "K")
        act(m, "Shuttle forward", lambda: self.shuttle(1), "L")
        act(m, "Previous frame", lambda: self.step(-1), "Left")
        act(m, "Next frame", lambda: self.step(1), "Right")
        act(m, "Previous shot", lambda: self.jump(self.project.shot_starts(), -1), "Shift+Left")
        act(m, "Next shot", lambda: self.jump(self.project.shot_starts(), 1), "Shift+Right")
        act(m, "Previous edit", lambda: self.jump(self.project.edit_points(), -1), "Up")
        act(m, "Next edit", lambda: self.jump(self.project.edit_points(), 1), "Down")
        self.sound_act = act(m, "Play sound", self._toggle_sound)
        self.sound_act.setCheckable(True)
        self.sound_act.setChecked(True)
        m.addSeparator()
        act(m, "Go to start", lambda: self.seek(0), "Home")
        act(m, "Go to end", lambda: self.seek(self.project.duration - 1), "End")

        m = mb.addMenu("&Colour")
        act(m, "Colorize selected clip", self.colorize_selected, "Ctrl+R")
        act(m, "Colorize all clips", self.colorize_all, "Ctrl+Shift+R")
        act(m, "Color shot (paint only)", self.color_shot, "Ctrl+Shift+C")
        act(m, "Fill shot (the model does the rest)", self.fill_shot, "Ctrl+Shift+F")
        act(m, "Colorize selected photos", self.colorize_photos, "Ctrl+Alt+R")
        act(m, "Apply hints", self.apply_hints, ["Ctrl+Return", "Ctrl+Enter"])
        act(m, "Find items everywhere", lambda: self._colour_request("find", ""))
        m.addSeparator()
        act(m, "Model manager...", self.open_model_manager)
        act(m, "Install PyTorch...", lambda: self.setup_torch(False))
        act(m, "Download weights for the current model", self.fetch_weights)
        act(m, "Model device...", self.choose_device)
        act(m, "Test the GPU again", self.retest_gpu)

        m = mb.addMenu("&Help")
        act(m, "Keys", self.show_keys, "F1")
        act(m, "About", lambda: QMessageBox.about(
            self, "EpochColor", f"EpochColor {__version__}\nColorize black and white film and photos.\nGPL-3.0-or-later"))

    def _start_preview(self) -> None:
        self.preview_thread = QThread(self)
        self.provider = PreviewProvider()
        self.provider.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.provider.start)
        self.provider.ready.connect(self._frame_ready)
        self.preview_thread.start()

    # ------------------------------------------------------ refreshing

    def _layout(self) -> list[LayoutPiece]:
        s = VideoSettings.from_project(self.project.d.settings)
        model = self.project.d.settings["model"]
        out = []
        for start, seg in zip(self.project.seg_starts(), self.project.d.timeline):
            c = self.project.d.clips[seg.clip]
            mi = self.media_info.get(c.id)
            if mi is None:
                continue
            # each shot shows what it has: the full colorize, the paint-only
            # colour from Color shot, or grey; an out-of-date result stays up
            # until it's redone
            ab = str(analysis_dir(Path(c.path), s, model, negative=c.neg) / "ab.npy")
            shots = tuple((int(k.split("-")[0]), int(k.split("-")[1]), v["state"], v["file"])
                          for k, v in self.shot_status.get(c.id, {}).items())
            out.append(LayoutPiece(start, seg.length, c.id, seg.src_in, str(mi.proxy), c.path, c.fps, ab,
                                   c.neg, shots))
        return out

    def _refresh(self) -> None:
        """Bring every view in line with the project."""
        s = VideoSettings.from_project(self.project.d.settings)
        model = self.project.d.settings["model"]
        refs = {cid: self.project.expanded_references(cid) for cid in self.project.d.clips}
        self.status = {cid: analysis_state(c, s, model, hints=self.project.d.hints.get(cid), references=refs[cid])
                       for cid, c in self.project.d.clips.items()}
        self.shot_status = {cid: shot_states(c, s, model, hints=self.project.d.hints.get(cid), references=refs[cid])
                            for cid, c in self.project.d.clips.items()}
        self.timeline.canvas.status = self.status
        self.timeline.canvas.shot_status = self.shot_status
        self.audio.project_changed(self.project)
        self.timeline.set_project(self.project) if self.timeline.canvas.project is not self.project \
            else self.timeline.refresh()
        self._push_layout()
        self.playhead = max(0, min(self.playhead, max(0, self.project.duration - 1)))
        self.timeline.canvas.playhead = self.playhead
        self._refresh_bin()
        self._refresh_inspector()
        self._refresh_title()
        self._refresh_colours()
        self._update_overlays()
        self._load_grade_panel()
        self.undo_act.setText(f"Undo {self.project.undo_label}" if self.project.undo_label else "Undo")
        self.redo_act.setText(f"Redo {self.project.redo_label}" if self.project.redo_label else "Redo")
        self.undo_act.setEnabled(self.project.undo_label is not None)
        self.redo_act.setEnabled(self.project.redo_label is not None)
        if self.bottom.currentWidget() is self.timeline:
            self._request()
        else:
            self._show_photo(self.filmstrip.currentRow())

    def _push_layout(self) -> None:
        """Send the preview what it needs to draw: pieces, settings, grades."""
        settings = dict(self.project.d.settings)
        settings["_grades"] = G.copy.deepcopy(self.project.d.grades)
        settings["_shots"] = {cid: [list(x) for x in c.shots] for cid, c in self.project.d.clips.items()}
        settings["_matte"] = self._show_matte
        self.provider.layout_sig.emit(self._layout(), settings)

    def _refresh_title(self) -> None:
        name = self.project.path.name if self.project.path else "untitled"
        self.setWindowTitle(f"{'*' if self.project.dirty else ''}{name} - EpochColor")

    def _refresh_bin(self) -> None:
        self.clip_list.clear()
        labels = {"ready": "colorized", "update": "hints changed, apply them", "partial": "partly colorized",
                  "none": "not colorized"}
        for cid, c in self.project.d.clips.items():
            it = QListWidgetItem(f"{c.name}\n{c.width}x{c.height}, {c.frames} frames, "
                                 f"{len(c.shots)} shots, {labels[self.status.get(cid, 'none')]}")
            it.setData(Qt.UserRole, cid)
            self.clip_list.addItem(it)
        # photos
        cur = self.filmstrip.currentRow()
        self.filmstrip.blockSignals(True)
        self.filmstrip.clear()
        from ..worker import photo_result_path

        for ph in self.project.d.photos:
            icon = self.photo_icons.get(ph.id)
            if icon is None:
                icon = self.photo_icons[ph.id] = self._photo_icon(ph.path)
            done = False
            try:
                done = photo_result_path(ph.path, self.project.d.settings, self.project.d.settings["model"],
                                         ph.strokes).exists()
            except OSError:
                pass
            it = QListWidgetItem(icon, Path(ph.path).name + ("" if done else "\n(not colorized)"))
            it.setData(Qt.UserRole, ph.id)
            self.filmstrip.addItem(it)
        self.filmstrip.setCurrentRow(min(max(cur, 0), self.filmstrip.count() - 1))
        self.filmstrip.blockSignals(False)

    def _selected_clip(self) -> Clip | None:
        i = self.timeline.canvas.sel
        if 0 <= i < len(self.project.d.timeline):
            return self.project.d.clips[self.project.d.timeline[i].clip]
        hit = self.project.locate(self.playhead)
        return self.project.d.clips[hit[1].clip] if hit else None

    def _refresh_inspector(self) -> None:
        ins = self.inspector
        if self.bottom.currentWidget() is self.filmstrip:
            ins.colorize_btn.setText("Colorize photo")
            ins.colorize_all_btn.setText("Colorize all photos")
            row = self.filmstrip.currentRow()
            if 0 <= row < len(self.project.d.photos):
                from ..worker import photo_result_path

                ph = self.project.d.photos[row]
                try:
                    done = photo_result_path(ph.path, self.project.d.settings, self.project.d.settings["model"],
                                             ph.strokes, ph.neg, self.project.photo_reference(ph)).exists()
                except OSError:
                    done = False
                state = "colorized" if done else ("hints changed, apply them" if ph.strokes else "not colorized yet")
                ins.clip_info.setText(f"<b>{Path(ph.path).name}</b><br>{Path(ph.path).parent}<br>"
                                      f"{len(ph.strokes)} hint stroke(s)<br>{state}")
            else:
                ins.clip_info.setText("No photo selected")
            ins.show_settings(self.project.d.settings)
            self._refresh_target_widgets()
            return
        ins.colorize_btn.setText("Colorize clip")
        ins.colorize_all_btn.setText("Colorize all")
        c = self._selected_clip()
        if c is None:
            self.inspector.clip_info.setText("Nothing selected")
        else:
            st = {"ready": "colorized", "update": "hints or stabilizer changed: Apply hints (Ctrl+Return)",
                  "partial": "partly colorized, run Colorize to finish",
                  "none": "not colorized yet"}[self.status.get(c.id, "none")]
            nh = len(self.project.hint_frames(c.id))
            st += f"<br>{nh} painted frame(s)" if nh else ""
            audio = ", ".join(f"{a.codec} {a.channels}ch" for a in c.audio) or "none"
            self.inspector.clip_info.setText(
                f"<b>{c.name}</b><br>{c.width}x{c.height} at {float(c.fps):.3f} fps, {c.codec}<br>"
                f"{c.frames} frames, {len(c.shots)} shots<br>audio: {audio}<br>{st}")
        self.inspector.show_settings(self.project.d.settings)
        self._refresh_target_widgets()

    def _target_object(self):
        """(kind, id, the Clip or Photo, clip frame) under the playhead or picked in the filmstrip."""
        t = self._paint_target()
        if t is None:
            return None
        obj = self.project.photo(t[1]) if t[0] == "photo" else self.project.d.clips[t[1]]
        return t[0], t[1], obj, t[2]

    def _target_reference(self, tt) -> dict | None:
        kind, ident, obj, frame = tt
        return obj.reference if kind == "photo" else self.project.reference_at(ident, frame)

    SHOT_STATE_TEXT = {
        ("none", True): "not coloured yet",
        ("painted", True): "<span style='color:#d9a441'>painted only</span> (Color shot): the model hasn't touched it",
        ("painted", False): "<span style='color:#d9a441'>painted only, out of date</span>: the paint changed, "
                            "Color shot again",
        ("colored", True): "<span style='color:#6fbf73'>coloured</span>: your paint plus the model",
        ("colored", False): "<span style='color:#6fbf73'>coloured, out of date</span>: Fill shot (or Colorize all) "
                            "to bring it up to date",
    }

    def _current_shot(self):
        """(clip, shot number from 1, (a, b), clip frame) under the playhead, or None."""
        t = self._paint_target()
        if t is None or t[0] != "clip":
            return None
        c = self.project.d.clips[t[1]]
        for i, (a, b) in enumerate(c.shots):
            if a <= t[2] < b:
                return c, i + 1, (a, b), t[2]
        return None

    def _refresh_target_widgets(self) -> None:
        ins = self.inspector
        tt = self._target_object()
        for w in (ins.negative, ins.neg_levels, ins.ref_btn, ins.ref_clear, ins.ref_strength):
            w.setEnabled(tt is not None)
        shot = self._current_shot()
        for w in (ins.color_shot_btn, ins.fill_shot_btn, ins.painted_frames):
            w.setEnabled(shot is not None)
        ins.painted_frames.clear()
        self.viewer.shot_note = ""
        if tt is None:
            ins.ref_label.setText("No reference")
            ins.shot_info.setText("Put the playhead on a clip, or pick a photo")
            return
        kind, ident, obj, frame = tt
        ins.negative.blockSignals(True)
        ins.negative.setChecked(bool(obj.negative))
        ins.negative.blockSignals(False)
        ins.neg_levels.setEnabled(bool(obj.negative and obj.negative_params))
        if shot is not None:
            c, no, (a, b), f = shot
            st = self.shot_status.get(c.id, {}).get(f"{a}-{b}", {"state": "none", "current": True})
            frames = [k for k in self.project.hint_frames(c.id) if a <= k < b]
            strokes = [x for k in frames for x in self.project.strokes_at(c.id, k)]
            names = {col["id"]: col["name"] for col in self.project.d.colors}
            items = sorted({names[x["item"]] for x in strokes if x.get("item") in names})
            unlabelled = sum(1 for x in strokes if x.get("item") not in names)
            painted = (f"{len(frames)} painted frame(s), {len(strokes)} stroke(s)" if frames
                       else "nothing painted here yet")
            if items:
                painted += "<br>items: " + ", ".join(items) + (f" + {unlabelled} unlabelled" if unlabelled else "")
            ins.shot_info.setText(f"<b>Shot {no} of {len(c.shots)}</b> in {c.name}: frames {a} to {b - 1} "
                                  f"({b - a} frames)<br>{self.SHOT_STATE_TEXT[(st['state'], st['current'])]}"
                                  f"<br>{painted}")
            ins.painted_frames.addItem("Go to a painted frame..." if frames else "No painted frames", None)
            for k in frames:
                ins.painted_frames.addItem(f"frame {k}" + ("  (here)" if k == f else "")
                                           + f"  {len(self.project.strokes_at(c.id, k))} stroke(s)", k)
            ins.fill_shot_btn.setText("Update shot" if st["state"] == "colored" else "Fill shot")
            self.viewer.shot_note = f"shot {no}: " + {"none": "not coloured", "painted": "painted only",
                                                      "colored": "coloured"}[st["state"]] + \
                ("" if st["current"] else ", out of date")
        else:
            ins.shot_info.setText("A photo: paint it, then Colorize photo. Color shot and Fill shot are for "
                                  "video shots.")
        ref = self._target_reference(tt)
        where = "this photo" if kind == "photo" else f"shot {shot[1] if shot else '?'}"
        if ref and ref.get("kind") == "painted":
            what = "your painting in this shot" if ref.get("scope") == "shot" else "your painting, everywhere"
            ins.ref_label.setText(f"Reference for {where}: {what}")
        elif ref:
            ins.ref_label.setText(f"Reference for {where}: {Path(ref['path']).name}")
        else:
            taught = self.project.d.settings.get("teach") and self.project.painted_sources()
            ins.ref_label.setText(f"No reference for {where}" + (" (it learns from your painting on Colorize all)"
                                                                  if taught and kind == "clip" else ""))
        if ref:
            ins.ref_strength.blockSignals(True)
            ins.ref_strength.setValue(int(round(float(ref.get("strength", 1.0)) * 100)))
            ins.ref_strength.blockSignals(False)
        ins.ref_clear.setEnabled(bool(ref))

    # ------------------------------------------------- shots: paint, fill

    def color_shot(self, quiet: bool = False) -> None:
        """Color shot: this shot from its paint alone, no model."""
        shot = self._current_shot()
        if shot is None:
            self.statusBar().showMessage("put the playhead on a shot of a clip first", 5000)
            return
        c, no, (a, b), _ = shot
        frames = [k for k in self.project.hint_frames(c.id) if a <= k < b]
        if not frames:
            self.statusBar().showMessage(f"nothing painted in shot {no} yet: press P and paint on a frame first", 6000)
            return
        n = sum(len(self.project.strokes_at(c.id, k)) for k in frames)
        self.runner.cancel_where(lambda j: j.kind in ("paint_shot", "analyze") and j.payload.get("shot") == [a, b]
                                 and j.payload.get("clip", {}).get("id") == c.id)
        self.runner.submit("paint_shot", f"Color shot {no} of {c.name} from {len(frames)} painted frame(s), {n} stroke(s)",
                           {**self._job_payload(), "clip": asdict(c), "hints": self.project.d.hints.get(c.id),
                            "shot": [a, b]})
        if not quiet:
            self.statusBar().showMessage(f"Color shot {no}: your {n} stroke(s) filling their objects and riding the "
                                         f"motion through {b - a} frames; nothing else gets colour", 8000)

    def fill_shot(self, quiet: bool = False) -> None:
        """Fill shot: the model colours the rest of this shot, the paint keeps what it covers."""
        shot = self._current_shot()
        if shot is None:
            self.statusBar().showMessage("put the playhead on a shot of a clip first", 5000)
            return
        c, no, (a, b), _ = shot
        self.runner.cancel_where(lambda j: j.kind in ("paint_shot", "analyze") and j.payload.get("shot") == [a, b]
                                 and j.payload.get("clip", {}).get("id") == c.id)
        self.runner.submit("analyze", f"Fill shot {no} of {c.name} ({b - a} frames)",
                           {**self._job_payload(), "clip": asdict(c), "hints": self.project.d.hints.get(c.id),
                            "references": self.project.expanded_references(c.id), "only_shots": [[a, b]],
                            "shot": [a, b]})
        if not quiet:
            self.statusBar().showMessage(f"Fill shot {no}: the model colours what you didn't paint; the first time "
                                         f"it runs the model over all {b - a} frames, after that it's quick", 8000)

    def _shot_action(self, name: str) -> None:
        {"color_shot": self.color_shot, "fill_shot": self.fill_shot, "colorize_all": self.colorize_all}[name]()

    def goto_clip_frame(self, clip_id: str, frame: int) -> None:
        for start, seg in zip(self.project.seg_starts(), self.project.d.timeline):
            if seg.clip == clip_id and seg.src_in <= frame < seg.src_out:
                self.seek(start + frame - seg.src_in)
                return
        self.statusBar().showMessage(f"frame {frame} isn't on the timeline", 4000)

    def _use_painting_reference(self, scope: str) -> None:
        tt = self._target_object()
        if tt is None:
            return
        if not self.project.painted_sources():
            self.statusBar().showMessage("paint something first; then your painting can be the reference", 6000)
            return
        self._set_reference(tt, {"kind": "painted", "scope": "shot" if scope == "shot" else "project",
                                 "strength": self.inspector.ref_strength.value() / 100.0})

    # ------------------------------------------------- negatives, references

    def _negative_toggled(self, on: bool) -> None:
        tt = self._target_object()
        if tt is None:
            return
        kind, ident, obj, _ = tt
        if on and not obj.negative_params:
            payload = {"kind": kind, "path": obj.path}
            if kind == "clip":
                payload.update(fps_num=obj.fps_num, fps_den=obj.fps_den, frames=obj.frames)
            job = self.runner.submit("measure_negative", f"Measure negative {Path(obj.path).name}", payload)
            self._measuring[job.id] = (kind, ident)
            self.statusBar().showMessage("measuring the film base and levels...", 4000)
            return
        self.project.set_negative(kind, ident, on)
        self._after_target_change(kind)

    def _after_target_change(self, kind: str) -> None:
        self._refresh()
        if kind == "photo":
            self._show_photo(self.filmstrip.currentRow())

    def negative_levels(self) -> None:
        from .negative_dialog import NegativeDialog

        tt = self._target_object()
        if tt is None or not tt[2].negative_params:
            return
        kind, ident, obj, _ = tt
        before = dict(obj.negative_params)
        self.project.checkpoint("negative levels")

        def live(params):
            obj.negative_params = dict(params)
            self._after_target_change(kind)

        dlg = NegativeDialog(before, self)
        dlg.changed.connect(live)
        dlg.remeasure.connect(lambda: self._negative_remeasure(kind, ident, obj, dlg))
        if dlg.exec() != QDialog.Accepted:
            obj.negative_params = before
            self._after_target_change(kind)

    def _negative_remeasure(self, kind, ident, obj, dlg) -> None:
        payload = {"kind": kind, "path": obj.path}
        if kind == "clip":
            payload.update(fps_num=obj.fps_num, fps_den=obj.fps_den, frames=obj.frames)
        job = self.runner.submit("measure_negative", f"Measure negative {Path(obj.path).name}", payload)
        self._measuring[job.id] = (kind, ident, dlg)

    def pick_reference(self) -> None:
        tt = self._target_object()
        if tt is None:
            return
        f, _ = QFileDialog.getOpenFileName(self, "Reference image: a colour photo", self.qs.value("dir_ref", ""),
                                           "Images (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.bmp *.heic)")
        if not f:
            return
        self.qs.setValue("dir_ref", str(Path(f).parent))
        self._set_reference(tt, {"path": f, "strength": self.inspector.ref_strength.value() / 100.0})

    def _set_reference(self, tt, ref: dict | None) -> None:
        kind, ident, obj, frame = tt
        if kind == "photo":
            self.project.set_photo_reference(ident, ref)
        else:
            self.project.set_reference(ident, frame, ref)
        self._refresh()
        self.apply_hints()  # the reference goes in through the hint pass

    def clear_reference(self) -> None:
        tt = self._target_object()
        if tt is not None and self._target_reference(tt):
            self._set_reference(tt, None)

    def _reference_strength(self, k: float) -> None:
        tt = self._target_object()
        ref = self._target_reference(tt) if tt else None
        if ref and abs(float(ref.get("strength", 1.0)) - k) > 1e-3:
            self._set_reference(tt, dict(ref, strength=round(k, 2)))

    def _edited(self) -> None:
        self._refresh()

    # --------------------------------------------------------- viewer

    def _request(self, playing: bool = False) -> None:
        self.provider.request_sig.emit(self.playhead, playing)
        fps = float(self.project.fps or 24)
        self.tc.setText(f"{timecode(self.playhead, fps)}   {self.playhead} / {self.project.duration}")
        hit = self.project.locate(self.playhead)
        if hit is None:
            self.viewer.clear("Add clips with File > Add clips (Ctrl+I)" if not self.project.d.clips
                              else "The timeline is empty")

    def _frame_ready(self, frame: int, gray, color, full: bool, raw=None) -> None:
        if frame != self.playhead or self.bottom.currentWidget() is not self.timeline:
            return
        self.viewer.set_images(gray, color, full, raw=raw)
        if self.scopes_dock.isVisible() and (full or not self.speed or frame % 4 == 0):
            self.scopes.show_image(color or gray)

    def _mode_changed(self, mode: str) -> None:
        for m, b in self.mode_btns.items():
            b.setChecked(m == mode)

    # ------------------------------------------------------ this frame

    def colorize_frame(self) -> None:
        """The selected model on the frame under the playhead, with that
        frame's paint and the shot's reference, as a temporary picture."""
        t = self._paint_target()
        if t is None or t[0] != "clip":
            self.statusBar().showMessage("Colorize frame works on a clip frame; photos have Colorize photo", 5000)
            return
        cid, f = t[1], t[2]
        c = self.project.d.clips[cid]
        a, b = next(((x, y) for x, y in c.shots if x <= f < y), (f, f + 1))
        refs = self.project.expanded_references(cid)
        from ..video.pipeline import shot_reference

        strokes = self.project.strokes_at(cid, f)
        ref = shot_reference(refs, a, b, painted=bool(self.project.hint_frames(cid) and
                                                       any(a <= k < b for k in self.project.hint_frames(cid))))
        self.runner.cancel_where(lambda j: j.kind == "frame_preview")
        job = self.runner.submit("frame_preview", f"Colorize frame {f} of {c.name}",
                                 {**self._job_payload(), "path": c.path, "fps_num": c.fps_num, "fps_den": c.fps_den,
                                  "frame": f, "negative": c.neg, "strokes": strokes, "reference": ref})
        self._frame_jobs[job.id] = (cid, f)
        self.statusBar().showMessage(f"colorizing frame {f} with {self.project.d.settings['model']}"
                                     + (f" and {len(strokes)} stroke(s)" if strokes else "") + "...", 5000)

    def _frame_view_update(self) -> None:
        """Show this frame's Colorize frame result, if it has one."""
        t = self._paint_target()
        res = self._frame_results.get((t[1], t[2])) if t and t[0] == "clip" else None
        if res and Path(res[0]).exists():
            from PySide6.QtGui import QImage as _QI

            self.viewer.frame_image = _QI(res[0])
            self.viewer.frame_label = res[1]
        else:
            self.viewer.frame_image = None
        self.viewer.update()

    def seek(self, frame: int, playing: bool = False) -> None:
        dur = self.project.duration
        frame = max(0, min(int(frame), max(0, dur - 1)))
        self.playhead = frame
        self.timeline.canvas.playhead = frame
        self.timeline.canvas.ensure_visible(frame)
        self.timeline.sync_scroll()
        if not playing:
            self._refresh_inspector()
            self._update_overlays()
            self._load_grade_panel()
            self._frame_view_update()
        self._request(playing)

    def step(self, n: int) -> None:
        self.shuttle(0)
        self.seek(self.playhead + n)

    def jump(self, points: list[int], direction: int) -> None:
        self.shuttle(0)
        if direction > 0:
            nxt = [p for p in points if p > self.playhead]
            self.seek(nxt[0] if nxt else self.project.duration - 1)
        else:
            prv = [p for p in points if p < self.playhead]
            self.seek(prv[-1] if prv else 0)

    # ------------------------------------------------------- playback

    def _toggle_sound(self) -> None:
        self.audio.enabled = self.sound_act.isChecked()
        if not self.audio.enabled:
            self.audio.stop()
        elif self.speed == 1:
            self.audio.play_from(self.playhead / float(self.project.fps or 24))

    def toggle_play(self) -> None:
        self.shuttle(0 if self.speed else 1, absolute=True)

    def shuttle(self, direction: int, absolute: bool = False) -> None:
        if absolute or direction == 0:
            self.speed = float(direction)
        elif direction > 0:
            self.speed = 1.0 if self.speed <= 0 else min(8.0, self.speed * 2)
        else:
            self.speed = -1.0 if self.speed >= 0 else max(-8.0, self.speed * 2)
        if self.speed and self.project.duration:
            if self.speed > 0 and self.playhead >= self.project.duration - 1:
                self.playhead = 0
            self._play_t0 = time.monotonic()
            self._play_f0 = self.playhead
            self.play_timer.start()
            if self.speed == 1:
                self.audio.play_from(self.playhead / float(self.project.fps or 24))
            else:
                self.audio.stop()  # shuttling: silent
            self.btn["play"].setText("Pause")
            self.speed_label.setText("" if self.speed == 1 else f"{self.speed:+g}x")
        else:
            self.speed = 0.0
            self.play_timer.stop()
            self.audio.stop()
            self.btn["play"].setText("Play")
            self.speed_label.setText("")
            self.seek(self.playhead)  # settle: full-size still

    def _tick(self) -> None:
        fps = float(self.project.fps or 24)
        f = self._play_f0 + int((time.monotonic() - self._play_t0) * fps * self.speed)
        end = self.project.duration - 1
        if f >= end or f <= 0:
            self.seek(max(0, min(f, end)), playing=True)
            self.shuttle(0)
            return
        if f != self.playhead:
            self.seek(f, playing=True)
            if self.speed == 1 and f % 24 == 0:
                self.audio.keep_in_step(f / fps)

    # ------------------------------------------------------------ edits

    def _do(self, fn, *a):
        try:
            r = fn(*a)
        except ProjectError as e:
            self.statusBar().showMessage(str(e), 6000)
            return None
        self._refresh()
        return r

    def undo(self) -> None:
        label = self.project.undo()
        if label:
            self.statusBar().showMessage(f"undid {label}", 3000)
        self._refresh()

    def redo(self) -> None:
        label = self.project.redo()
        if label:
            self.statusBar().showMessage(f"redid {label}", 3000)
        self._refresh()

    def split(self) -> None:
        self._do(self.project.split, self.playhead)

    def delete_segment(self) -> None:
        i = self.timeline.canvas.sel
        if not 0 <= i < len(self.project.d.timeline):
            hit = self.project.locate(self.playhead)
            if not hit:
                return
            i = hit[0]
        self._do(self.project.delete, i)
        self.timeline.canvas.sel = min(i, len(self.project.d.timeline) - 1)
        self._refresh()

    def move_segment(self, delta: int) -> None:
        i = self.timeline.canvas.sel
        if 0 <= i < len(self.project.d.timeline):
            self.timeline.canvas.sel = self._do(self.project.move, i, delta)

    def mark_in(self) -> None:
        self._do(self.project.set_in, self.playhead)

    def mark_out(self) -> None:
        self._do(self.project.set_out, self.playhead)

    def clear_range(self) -> None:
        self._do(self.project.clear_range)

    def add_marker(self) -> None:
        if not self.project.duration:
            return
        note, ok = QInputDialog.getText(self, "Marker", f"Note for frame {self.playhead}:")
        if ok:
            self._do(self.project.add_marker, self.playhead, note)

    def toggle_cut(self) -> None:
        now = self._do(self.project.toggle_cut, self.playhead)
        if now is not None:
            what = "shot cut added" if now else "shot cut removed"
            self.statusBar().showMessage(f"{what}; colorize the clip again to use it", 5000)

    def _toggle_audio(self, k: int) -> None:
        self._do(self.project.toggle_audio, k)

    def _setting_changed(self, key: str, value) -> None:
        self.project.set_setting(key, value)
        self._refresh()

    def remove_clip(self) -> None:
        c = self._selected_clip()
        if c and QMessageBox.question(self, "Remove clip", f"Remove {c.name} and every segment of it?") \
                == QMessageBox.Yes:
            self._do(self.project.remove_clip, c.id)

    def _append_clip(self, item) -> None:
        cid = item.data(Qt.UserRole)
        c = self.project.d.clips.get(cid)
        if c:
            from ..project import Segment

            self.project.checkpoint(f"add {c.name}")
            self.project.d.timeline.append(Segment(c.id, 0, c.frames))
            self._refresh()

    def _zoom(self, k: float) -> None:
        cv = self.timeline.canvas
        anchor = self.playhead
        px = cv.x_of(anchor)
        cv.zoom = float(np.clip(cv.zoom * k, 0.01, 40.0))
        cv.offset = max(0.0, anchor - (px - 96) / cv.zoom)
        self.timeline.sync_scroll()
        cv.update()

    def _fit(self) -> None:
        self.timeline.canvas.fit()
        self.timeline.sync_scroll()

    # ----------------------------------------------------------- media

    def add_clips_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add clips", self.qs.value("dir_clips", ""), VIDEO_FILTER)
        if files:
            self.qs.setValue("dir_clips", str(Path(files[0]).parent))
            self.add_clips(files)

    def add_clips(self, files) -> None:
        for f in files:
            job = self.runner.submit("import", f"Import {Path(f).name}",
                                     {"path": str(f), "shot_threshold": self.project.d.settings["shot_threshold"]})
            self._importing[job.id] = "new"

    def add_photos_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos", self.qs.value("dir_photos", ""), PHOTO_FILTER)
        if files:
            self.qs.setValue("dir_photos", str(Path(files[0]).parent))
            self.add_photos(files)

    def add_photos(self, files) -> None:
        for f in files:
            self.project.add_photo(str(Path(f).resolve()))
        self.bottom.setCurrentWidget(self.filmstrip)
        self._refresh()

    def remove_photos(self) -> None:
        ids = [it.data(Qt.UserRole) for it in self.filmstrip.selectedItems()]
        for pid in ids:
            self.project.remove_photo(pid)
        self._refresh()

    def _photo_icon(self, path: str) -> QIcon:
        try:
            arr = self._photo_display(path, 288)
            return QIcon(QPixmap.fromImage(to_qimage(arr)))
        except Exception:
            return QIcon()

    @staticmethod
    def _photo_display(path: str, long_side: int = 1600) -> np.ndarray:
        import cv2

        from ..imageio import load_image

        img, _ = load_image(path)
        h, w = img.shape[:2]
        k = min(1.0, long_side / max(h, w))
        if k < 1:
            img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
        return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)

    def _show_photo(self, row: int) -> None:
        if self.bottom.currentWidget() is not self.filmstrip:
            return
        if not 0 <= row < len(self.project.d.photos):
            self.viewer.clear("Add photos with File > Add photos (Ctrl+Shift+I)")
            return
        from ..color import lab_to_srgb, srgb_to_l, srgb_to_lab
        from ..worker import photo_result_path

        ph = self.project.d.photos[row]
        try:
            src = self._photo_display(ph.path)
            if ph.neg:  # a negative: show the positive the model gets
                from ..film import invert_rgb

                pos = invert_rgb(src.astype(np.float32) / 255.0, ph.neg)
                src = (np.clip(pos[..., 0] if pos.ndim == 3 else pos, 0, 1) * 255 + 0.5).astype(np.uint8)
            if src.ndim == 3:  # show what the model sees: luminance only
                gray = srgb_to_l(src.astype(np.float32) / 255)
                src = (lab_to_srgb(gray, np.zeros(gray.shape + (2,), np.float32))[..., 0] * 255 + 0.5).astype(np.uint8)
            gimg = to_qimage(src)
            st, model = self.project.d.settings, self.project.d.settings["model"]
            pref = self.project.photo_reference(ph)
            res = photo_result_path(ph.path, st, model, ph.strokes, ph.neg, pref)
            if not res.exists() and ph.strokes:
                # hints not applied yet: show the last unhinted result meanwhile
                res = photo_result_path(ph.path, st, model, None, ph.neg, pref)
            cimg = raw = None
            if res.exists():
                rgb = self._photo_display(str(res)).astype(np.float32) / 255.0
                from ..chroma import adjust_rgb
                from ..worker import finish_photo

                adj = adjust_rgb(rgb, float(st.get("saturation", 1.0)), float(st.get("cast", 0.0)))
                raw = to_qimage((np.clip(adj, 0, 1) * 255 + 0.5).astype(np.uint8))
                cimg = raw
                if self._show_matte and ph.grade and not G.is_identity(ph.grade):
                    matte = []
                    G.apply(adj, ph.grade, matte_out=matte)
                    m = matte[0] if matte and matte[0] is not None else np.ones(rgb.shape[:2], np.float32)
                    cimg = to_qimage((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8))
                elif (ph.grade and not G.is_identity(ph.grade)) or float(st.get("regrain", 0) or 0) > 0:
                    out = finish_photo(rgb, st, ph.grade)  # the same steps as export
                    cimg = to_qimage((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
            self.viewer.set_images(gimg, cimg, True, Path(ph.path).name, raw=raw)
            if self.scopes_dock.isVisible():
                self.scopes.show_image(cimg or gimg)
            self._refresh_inspector()
            self._update_overlays()
            self._load_grade_panel()
        except Exception as e:
            self.viewer.clear(f"Can't show {Path(ph.path).name}: {e}")

    def _bottom_tab(self, i: int) -> None:
        self.shuttle(0)
        self._refresh_inspector()
        if self.bottom.currentWidget() is self.filmstrip:
            if self.filmstrip.currentRow() < 0 and self.filmstrip.count():
                self.filmstrip.setCurrentRow(0)
            self._show_photo(self.filmstrip.currentRow())
        else:
            self._request()

    # ------------------------------------------------------------ jobs

    def _job_payload(self) -> dict:
        return {"model": self.project.d.settings["model"], "settings": dict(self.project.d.settings),
                "device": self.qs.value("device", "auto")}

    def colorize_selected(self) -> None:
        if self.bottom.currentWidget() is self.filmstrip:
            return self.colorize_photos()
        c = self._selected_clip()
        if c is None:
            self.statusBar().showMessage("select a clip on the timeline first", 4000)
            return
        self._colorize([c])

    def colorize_all(self) -> None:
        if self.bottom.currentWidget() is self.filmstrip:
            self.filmstrip.selectAll()
            return self.colorize_photos()
        self._colorize([c for cid, c in self.project.d.clips.items() if self.status.get(cid) != "ready"])

    def _colorize(self, clips) -> None:
        busy = {j.payload.get("clip", {}).get("id") for j in self.runner.all_jobs() if j.kind == "analyze"}
        for c in clips:
            if c.id in busy:
                continue
            self.runner.submit("analyze", f"Colorize {c.name}",
                               {**self._job_payload(), "clip": asdict(c), "hints": self.project.d.hints.get(c.id),
                            "references": self.project.expanded_references(c.id)})

    def _photo_payload(self, ph) -> dict:
        return {"photo": ph.id, "path": ph.path, "strokes": ph.strokes, "negative": ph.neg,
                "reference": self.project.photo_reference(ph)}

    def colorize_photos(self) -> None:
        rows = sorted({self.filmstrip.row(it) for it in self.filmstrip.selectedItems()}) or \
            ([self.filmstrip.currentRow()] if self.filmstrip.currentRow() >= 0 else [])
        for r in rows:
            ph = self.project.d.photos[r]
            self.runner.submit("photo", f"Colorize {Path(ph.path).name}",
                               {**self._job_payload(), **self._photo_payload(ph)})

    def fetch_weights(self) -> None:
        from ..models.manager import entry, installed

        mid = self.project.d.settings["model"]
        e = entry(mid)
        if mid in installed():
            self.statusBar().showMessage(f"{mid} is downloaded already", 4000)
            return
        if e is None or not e.get("license_ok"):
            self.open_model_manager()  # shows the license before downloading
            return
        self.runner.submit("fetch", f"Download {mid} weights", {"model": mid})

    def setup_torch(self, first_run: bool) -> None:
        from .torch_dialog import TorchDialog

        dlg = TorchDialog(first_run, self)
        dlg.exec()
        if first_run and dlg.dont_ask.isChecked():
            self.qs.setValue("torch_dont_ask", True)
        if dlg.ok:
            self.runner.shutdown()  # the next job starts a worker that sees the new PyTorch
            self.statusBar().showMessage("PyTorch installed", 6000)

    def first_run_checks(self) -> None:
        from ..torch_setup import have_torch

        if not have_torch() and not self.qs.value("torch_dont_ask", False, type=bool):
            self.setup_torch(True)

    def open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        dlg = SettingsDialog(self)
        if dlg.exec() != SettingsDialog.Accepted:
            return
        self.runner.shutdown()  # the next job starts a worker with the new limits and folder
        if dlg.changed_root:
            from ..video.pipeline import cache_root

            self.statusBar().showMessage(f"working folder: {cache_root()}", 8000)
            for c in self.project.d.clips.values():
                if Path(c.path).exists() and MediaInfo.load(media_dir(c.path)) is None:
                    job = self.runner.submit("import", f"Rebuild proxy for {c.name}",
                                             {"path": c.path, "shot_threshold": self.project.d.settings["shot_threshold"]})
                    self._importing[job.id] = "rebuild"
            self._refresh()

    def retest_gpu(self) -> None:
        from ..gpucheck import forget

        forget()
        self.runner.shutdown()
        self.statusBar().showMessage("the GPU gets tested again the next time a model runs", 8000)

    def choose_device(self) -> None:
        cur = self.qs.value("device", "auto")
        dev, ok = QInputDialog.getText(self, "Model device",
                                       "auto, cpu, cuda, cuda:1 or xpu.\nROCm GPUs show up as cuda.", text=cur)
        if ok and dev.strip():
            self.qs.setValue("device", dev.strip())

    def _jobs_changed(self) -> None:
        cur = self.runner.current
        ins = self.inspector
        if cur is None:
            ins.job_label.setText("Idle")
            ins.job_bar.setRange(0, 1)
            ins.job_bar.setValue(0)
            ins.job_bar.setTextVisible(False)
        else:
            ins.job_label.setText(f"{cur.label}: {cur.stage}")
            ins.job_bar.setTextVisible(True)
            ins.job_bar.setRange(0, max(1, cur.total))
            ins.job_bar.setValue(min(cur.done, max(1, cur.total)))
        waiting = [j.label for j in self.runner.pending]
        ins.job_queue.setText(("Waiting: " + ", ".join(waiting)) if waiting else "")
        ins.cancel_btn.setEnabled(cur is not None)
        ins.cancel_all_btn.setEnabled(self.runner.busy())

    def _job_done(self, job, result: dict) -> None:
        if self._manager is not None:
            self._manager.reload()
        if result.get("notice"):
            if result.get("notice_kind") == "warning":  # fell back to the CPU: worth a box
                box = QMessageBox(QMessageBox.Warning, "GPU", result["notice"], parent=self)
                box.setModal(False)
                box.show()  # not exec(): the job's result still gets handled below
            else:  # all is well, just say what got picked
                self.statusBar().showMessage(result["notice"], 20000)
        if job.kind == "import":
            cd = dict(result["clip"])
            cd["audio"] = [AudioInfo(**a) for a in cd.get("audio", [])]
            clip = Clip(**cd)
            existing = next((c for c in self.project.d.clips.values() if c.path == clip.path), None)
            if existing is not None and self._importing.get(job.id) == "rebuild":
                self._load_media(existing)
            else:
                try:
                    self.project.add_clip(clip)
                    self._load_media(clip)
                except ProjectError as e:
                    QMessageBox.warning(self, "Can't add clip", f"{clip.name}: {e}")
            self._importing.pop(job.id, None)
            if self.project.duration and self.timeline.canvas.zoom == 2.0:
                self._fit()
        elif job.kind == "export":
            self.statusBar().showMessage(f"done: exported {result['out']}", 10000)
        elif job.kind == "export_frame":
            self.statusBar().showMessage(f"done: frame saved to {result['out']}", 10000)
        elif job.kind == "export_photos":
            self.statusBar().showMessage(f"done: exported {len(result['files'])} photo(s)", 10000)
        elif job.kind == "fetch":
            self.statusBar().showMessage(f"done: weights saved to {result.get('path')}", 10000)
            self.inspector.set_models(self.project.d.settings["model"])
        elif job.kind in ("add_model", "refresh_catalog", "move_models"):
            self.inspector.set_models(self.project.d.settings["model"])
            msg = {"add_model": f"added {result.get('model')}",
                   "refresh_catalog": f"catalog: {result.get('good')} models",
                   "move_models": f"models now in {result.get('dir')}"}[job.kind]
            self.statusBar().showMessage(f"done: {msg}", 8000)
            self.runner.shutdown()  # a fresh worker sees the new models folder
        elif job.kind == "match":
            self._matches_found(result)
        elif job.kind == "frame_preview":
            where = self._frame_jobs.pop(job.id, None)
            if where:
                self._frame_results[where] = (result["path"], result["label"])
                self._frame_view_update()
                if self._paint_target() == ("clip", where[0], where[1]):
                    self.viewer.set_mode("frame")
                self.statusBar().showMessage(f"{result['label']}. 4 or This frame shows it, 1 goes back", 8000)
        elif job.kind == "measure_negative":
            target = self._measuring.pop(job.id, None)
            if target:
                kind, ident = target[0], target[1]
                if len(target) == 3:  # Auto in the levels dialog
                    target[2].set_params(result["params"])
                else:
                    self.project.set_negative(kind, ident, True, result["params"])
                    self._after_target_change(kind)
                    self.statusBar().showMessage("negative: inverted; Levels... adjusts it", 6000)
        elif job.kind == "track_mask":
            self._track_done(job, result)
        else:
            self.statusBar().showMessage(f"done: {job.label}", 5000)
        self._refresh()

    def _job_failed(self, job, msg: str, tb: str) -> None:
        self._importing.pop(job.id, None)
        if self._measuring.pop(job.id, None):
            self._refresh_target_widgets()
        if msg == "cancelled":
            self.statusBar().showMessage(f"cancelled: {job.label}", 5000)
            self._refresh()
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Job failed")
        box.setText(f"{job.label} failed:\n\n{msg}")
        if tb:
            box.setDetailedText(tb)
        box.show()
        self._refresh()

    def _load_media(self, clip: Clip) -> bool:
        mi = MediaInfo.load(media_dir(clip.path)) if Path(clip.path).exists() else None
        if mi is None:
            return False
        self.media_info[clip.id] = mi
        self.media.add(clip.id, mi)
        return True

    # ------------------------------------------------------- painting

    def _paint_target(self):
        """("clip", clip_id, clip frame) or ("photo", photo_id, None) for what's in the viewer."""
        if self.bottom.currentWidget() is self.filmstrip:
            row = self.filmstrip.currentRow()
            if 0 <= row < len(self.project.d.photos):
                return "photo", self.project.d.photos[row].id, None
            return None
        hit = self.project.locate(self.playhead)
        if hit is None:
            return None
        return "clip", hit[1].clip, hit[2]

    def _target_strokes(self, t) -> list:
        if t is None:
            return []
        if t[0] == "photo":
            return list(self.project.photo(t[1]).strokes)
        return self.project.strokes_at(t[1], t[2])

    def _set_target_strokes(self, t, strokes: list, label: str) -> None:
        if t[0] == "photo":
            self.project.set_photo_strokes(t[1], strokes, label)
        else:
            self.project.set_strokes(t[1], t[2], strokes, label)

    def _brush_changed(self) -> None:
        pb = self.paintbar
        self.viewer.paint_mode = pb.paint.isChecked()
        self.viewer.erase_mode = pb.erase.isChecked()
        self.viewer.show_strokes = pb.show.isChecked()  # H hides them, painting or not
        self.viewer.brush = {"rgb": list(pb.rgb), "neutral": pb.neutral.isChecked(), "radius": pb.radius()}
        if pb.item_id and not pb.neutral.isChecked():
            self.viewer.brush["item"] = pb.item_id
        reach = pb.stroke_reach()
        if reach is not None:
            self.viewer.brush["reach"] = reach
        self.viewer.frame_note = "" if reach is None else (
            "frame by frame: this frame only" if reach == 0 else f"frame by frame: reach ±{reach}")
        self._update_overlays()
        self.viewer.update()

    def _stroke_added(self, stroke: dict) -> None:
        t = self._paint_target()
        if t is None:
            return
        if t[0] == "photo":
            stroke.pop("reach", None)  # one frame anyway
        item = next((c for c in self.project.d.colors if c["id"] == stroke.get("item")), None)
        if item is not None and not item.get("source"):
            # an item's first stroke is what Find looks for elsewhere
            item["source"] = {"kind": t[0], "target": t[1], "frame": t[2],
                              "stroke": {k: v for k, v in stroke.items() if k != "item"}}
        self._set_target_strokes(t, self._target_strokes(t) + [stroke], "paint")
        self._after_paint()

    def _erase_at(self, x: float, y: float) -> None:
        t = self._paint_target()
        strokes = self._target_strokes(t)
        best, dist = None, 1e9
        for i, st in enumerate(strokes):
            for px, py in st["points"]:
                d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5 - st.get("radius", 0.02)
                if d < dist:
                    best, dist = i, d
        if best is not None and dist < 0.03:
            strokes.pop(best)
            self._set_target_strokes(t, strokes, "erase stroke")
            self._after_paint()

    def _after_paint(self) -> None:
        self._refresh()
        if self.paintbar.auto.isChecked():
            self.hint_timer.start()

    def _paint_action(self, name: str) -> None:
        t = self._paint_target()
        if name == "apply":
            self.apply_hints()
        elif name == "clear" and t and self._target_strokes(t):
            self._set_target_strokes(t, [], "clear strokes")
            self._after_paint()
        elif name == "save_colour":
            self.save_colour()
        elif name in ("carry_next", "carry_prev"):
            self.carry_strokes(1 if name == "carry_next" else -1)

    def toggle_frame_mode(self) -> None:
        pb = self.paintbar
        pb.frame_mode.toggle()
        if pb.frame_mode.isChecked() and not pb.paint.isChecked():
            pb.paint.setChecked(True)

    def _gray_frame(self, clip_id: str, frame: int):
        """A proxy frame as float L-ish (0..100) for optical flow, or None."""
        from fractions import Fraction

        from ..media import FrameReader

        mi = self.media_info.get(clip_id)
        if mi is None:
            return None
        r = self._flow_readers.get(clip_id)
        if r is None or r.path != str(mi.proxy):
            c = self.project.d.clips[clip_id]
            r = self._flow_readers[clip_id] = FrameReader(mi.proxy, Fraction(c.fps_num, c.fps_den), "gray", cache=4)
            r.path = str(mi.proxy)
        g = r.get(frame)
        return None if g is None else g.astype(np.float32) / 2.55

    def carry_strokes(self, step: int) -> None:
        """Move this frame's strokes onto the next (or previous) frame along the
        optical flow between them, then go there: the frame-by-frame loop is
        paint, carry, touch up, carry."""
        from ..hintpaint import move_strokes
        from ..video.temporal import Flow

        t = self._paint_target()
        if t is None or t[0] != "clip":
            self.statusBar().showMessage("carrying strokes works on video frames", 4000)
            return
        cid, f = t[1], t[2]
        strokes = self.project.strokes_at(cid, f)
        if not strokes:
            self.statusBar().showMessage("nothing painted on this frame to carry", 4000)
            return
        g = f + step
        c = self.project.d.clips[cid]
        if not (0 <= g < c.frames) or self.project.shot_start(cid, g) != self.project.shot_start(cid, f):
            self.statusBar().showMessage("that's across a cut; strokes don't carry into another shot", 5000)
            return
        La, Lb = self._gray_frame(cid, f), self._gray_frame(cid, g)
        if La is None or Lb is None:
            self.statusBar().showMessage("no proxy frame to track with yet", 4000)
            return
        moved = move_strokes(strokes, Flow("medium")(La, Lb))
        for st in moved:
            st["from"] = f
            if self.paintbar.frame_mode.isChecked():
                st["reach"] = self.paintbar.reach.value()
        keep = [st for st in self.project.strokes_at(cid, g) if st.get("from") != f]
        self.project.set_strokes(cid, g, keep + moved, "carry strokes")
        self.step(step)
        if self._paint_target() != ("clip", cid, g):
            self.statusBar().showMessage(f"carried to frame {g}, which isn't next on the timeline", 5000)
        self._after_paint()

    def apply_hints(self) -> None:
        """Rerun the hint pass for what's in the viewer: the clip's changed
        shots (the model pass stays cached), or the photo."""
        self.hint_timer.stop()
        t = self._paint_target()
        if t is None:
            return
        if t[0] == "photo":
            ph = self.project.photo(t[1])
            self.runner.cancel_where(lambda j: j.kind == "photo" and j.payload.get("photo") == ph.id)
            self.runner.submit("photo", f"Hints on {Path(ph.path).name}",
                               {**self._job_payload(), **self._photo_payload(ph)})
            return
        # a video shot: paint-only colour while it hasn't been through the
        # model (seconds), or update its full colour if it has (the model
        # pass is cached, so that's quick too)
        shot = self._current_shot()
        if shot is None:
            return
        c, no, (a, b), _ = shot
        st = self.shot_status.get(c.id, {}).get(f"{a}-{b}", {"state": "none"})
        has_paint = any(a <= k < b for k in self.project.hint_frames(c.id))
        if st["state"] == "colored":
            self.fill_shot(quiet=True)
        elif has_paint:
            self.color_shot(quiet=True)
        elif st["state"] == "painted":  # the last stroke was erased: back to nothing
            self.statusBar().showMessage(f"shot {no} has no paint left; its old Color shot result stays until "
                                         "you Fill it or paint again", 6000)

    def _update_overlays(self) -> None:
        t = self._paint_target()
        strokes = self._target_strokes(t)
        sugs = []
        if t is not None:
            cols = {c["id"]: c for c in self.project.d.colors}
            for m in self.project.d.suggestions:
                c = cols.get(m["color"])
                if not c or m["kind"] != t[0] or m["target"] != t[1]:
                    continue
                if t[0] == "clip":
                    shot = self.project.shot_start(t[1], t[2])
                    if self.project.shot_start(t[1], m["frame"]) != shot:
                        continue
                for x, y in m["points"]:
                    sugs.append((x, y, c["rgb"], c["name"], m["id"]))
        mask = None
        g = self.grade_panel.grade if self.grade_panel.isEnabled() else None
        if g and g["mask"]["shape"] != "none" and self.grade_dock.isVisible():
            mask = dict(g["mask"])
            if t and t[0] == "clip":
                dx, dy = G.GradeBook(self.project.d.grades,
                                     {cid: c.shots for cid, c in self.project.d.clips.items()}).mask_offset(t[1], t[2])
                mask["_dx"], mask["_dy"] = dx, dy
        ghosts = []
        pb = self.paintbar
        if t is not None and t[0] == "clip" and pb.frame_mode.isChecked() and pb.onion.isChecked():
            shot = self.project.shot_start(t[1], t[2])
            earlier = [k for k in self.project.hint_frames(t[1])
                       if k < t[2] and self.project.shot_start(t[1], k) == shot]
            if earlier:
                ghosts = self.project.strokes_at(t[1], earlier[-1])
        self.viewer.set_overlays(strokes, mask, sugs, ghosts)

    def _picked(self, x: float, y: float, rgb) -> None:
        mode = self.viewer.pick_mode
        self.viewer.pick_mode = None
        if rgb is None:
            self.statusBar().showMessage("nothing to sample: colorize first", 4000)
            return
        if mode == "neutral":
            t, n = G.wb_from_neutral(rgb)
            self.grade_panel.set_fields(temperature=round(t, 4), tint=round(n, 4))
        elif mode == "hue":
            import colorsys

            h, sat, v = colorsys.rgb_to_hsv(*rgb)
            self.grade_panel.set_fields(**{"qualifier.hue": round(h, 4), "qualifier.enabled": True})
        else:  # Ctrl+click while painting: brush colour
            self.paintbar.set_rgb([int(round(c * 255)) for c in rgb])
        self.viewer.update()

    # -------------------------------------------------- items (saved colours)

    def save_colour(self) -> None:
        t = self._paint_target()
        strokes = [st for st in self._target_strokes(t) if not st.get("neutral")]
        if not strokes:
            self.statusBar().showMessage("paint a stroke first; its colour is what gets saved", 5000)
            return
        st = strokes[-1]
        name, ok = QInputDialog.getText(self, "Make item", "What did you paint? (\"Anna's coat\"):")
        if not ok or not name.strip():
            return
        source = {"kind": t[0], "target": t[1], "frame": t[2],
                  "stroke": {k: v for k, v in st.items() if k != "item"}}
        c = self.project.add_color(name.strip(), st["rgb"], source)
        st["item"] = c["id"]  # that stroke now belongs to it
        self._set_target_strokes(t, self._target_strokes(t), "label stroke")
        self.paintbar.set_rgb(c["rgb"], c["id"], c["name"])
        self.colours_dock.raise_()
        self._refresh()

    def _use_colour(self, rgb) -> None:
        self.paintbar.set_rgb(rgb)
        if not self.paintbar.paint.isChecked():
            self.paintbar.paint.setChecked(True)

    def _names(self) -> dict:
        out = {("clip", cid): c.name for cid, c in self.project.d.clips.items()}
        out.update({("photo", ph.id): Path(ph.path).name for ph in self.project.d.photos})
        return out

    def _refresh_colours(self) -> None:
        from ..worker import match_thumb_path

        usage = {c["id"]: self.project.item_usage(c["id"]) for c in self.project.d.colors}
        self.colours.show_data(self.project.d.colors, self.project.d.suggestions, self._names(),
                               self.project.d.settings, thumbs=match_thumb_path, usage=usage,
                               active=self.paintbar.item_id)

    def use_item(self, ident: str) -> None:
        """Paint with an item: its colour, and new strokes carry its name."""
        c = next((c for c in self.project.d.colors if c["id"] == ident), None)
        if c is None:
            return
        self.paintbar.set_rgb(c["rgb"], c["id"], c["name"])
        if not self.paintbar.paint.isChecked():
            self.paintbar.paint.setChecked(True)
        self.statusBar().showMessage(f"painting {c['name']}: new strokes belong to it", 5000)
        self._refresh_colours()

    def new_item(self, name: str | None = None, rgb=None) -> dict | None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        if name is None:
            name, ok = QInputDialog.getText(self, "New item", "What is it? (\"Anna's coat\", \"the sky\")")
            if not ok or not name.strip():
                return None
        if rgb is None:
            col = QColorDialog.getColor(QColor(*self.paintbar.rgb), self, f"Colour of {name.strip()}")
            if not col.isValid():
                return None
            rgb = [col.red(), col.green(), col.blue()]
        c = self.project.add_color(name.strip(), rgb, None)
        self.colours_dock.raise_()
        self.use_item(c["id"])
        self._refresh()
        return c

    def recolour_item(self, ident: str, rgb=None) -> None:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        c = next((c for c in self.project.d.colors if c["id"] == ident), None)
        if c is None:
            return
        if rgb is None:
            col = QColorDialog.getColor(QColor(*c["rgb"]), self, f"New colour for {c['name']}")
            if not col.isValid():
                return
            rgb = [col.red(), col.green(), col.blue()]
        n = self.project.recolor_item(ident, rgb)
        if self.paintbar.item_id == ident:
            self.paintbar.set_rgb(c["rgb"], ident, c["name"])
        self.statusBar().showMessage(f"{c['name']}: {n} stroke{'s' if n != 1 else ''} recoloured", 6000)
        self._after_paint()

    def delete_item(self, ident: str, with_strokes: bool | None = None) -> None:
        c = next((c for c in self.project.d.colors if c["id"] == ident), None)
        if c is None:
            return
        k, _ = self.project.item_usage(ident)
        if with_strokes is None:
            with_strokes = False
            if k:
                r = QMessageBox.question(
                    self, "Delete item", f"Delete {c['name']}'s {k} stroke{'s' if k != 1 else ''} too?\n\n"
                    "Yes deletes the paint as well. No keeps the paint, without a name.",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                if r == QMessageBox.Cancel:
                    return
                with_strokes = r == QMessageBox.Yes
        self.project.remove_color(ident, with_strokes)
        if self.paintbar.item_id == ident:
            self.paintbar.set_rgb(self.paintbar.rgb)
        if with_strokes and k:
            self._after_paint()
        else:
            self._refresh()

    def _colour_request(self, action: str, ident: str) -> None:
        p = self.project
        if action == "rename":
            c = next((c for c in p.d.colors if c["id"] == ident), None)
            if c:
                name, ok = QInputDialog.getText(self, "Rename colour", "Name:", text=c["name"])
                if ok and name.strip():
                    p.rename_color(ident, name.strip())
        elif action == "delete":
            self.delete_item(ident)
            return
        elif action == "new_item":
            self.new_item()
            return
        elif action == "recolour":
            self.recolour_item(ident)
            return
        elif action == "save_colour":
            self.save_colour()
            return
        elif action == "find":
            cols = [c for c in p.d.colors if c.get("source") and (c["id"] == ident or not ident)]
            if not cols:
                self.statusBar().showMessage("paint an item first: Find looks for what its first stroke covers",
                                             6000)
                return
            clips = {cid: {"path": c.path, "fps_num": c.fps_num, "fps_den": c.fps_den, "shots": c.shots,
                           "negative": c.neg}
                     for cid, c in p.d.clips.items()}
            photos = {ph.id: ph.path for ph in p.d.photos}
            photo_neg = {ph.id: ph.neg for ph in p.d.photos if ph.neg}
            label = f"Find {cols[0]['name']}" if ident and cols else "Find items"
            self.colours_dock.show()
            self.colours_dock.raise_()
            self.runner.submit("match", label,
                               {**self._job_payload(), "colors": cols, "clips": clips, "photos": photos,
                                "photo_neg": photo_neg})
            self.statusBar().showMessage("looking through every shot and photo...", 4000)
            return
        elif action == "goto":
            self._goto_match(ident)
            return
        elif action == "apply":
            sug = p.apply_suggestion(ident)
            if sug:
                name = next((c["name"] for c in p.d.colors if c["id"] == sug["color"]), "colour")
                self.statusBar().showMessage(f"used {name} there; the hint pass runs next", 5000)
            self._after_paint()
            if not self.paintbar.auto.isChecked():
                self.apply_hints()
            return
        elif action == "dismiss":
            p.dismiss_suggestion(ident)
        elif action == "apply_all":
            for m in list(p.d.suggestions):
                p.apply_suggestion(m["id"])
            self._after_paint()
            if not self.paintbar.auto.isChecked():
                self.apply_hints()
            return
        elif action == "dismiss_all":
            if p.d.suggestions:
                p.set_suggestions([])
        self._refresh()

    def _matches_found(self, result: dict) -> None:
        items = result.get("suggestions", [])
        searched = set(result.get("colors") or [c["id"] for c in self.project.d.colors])
        # a search for one colour replaces only that colour's matches
        keep = [m for m in self.project.d.suggestions if m["color"] not in searched]
        self.project.set_suggestions(keep + items)
        self.colours_dock.show()
        self.colours_dock.raise_()
        if self.project.d.settings.get("match_mode") == "auto" and items:
            for m in items:
                self.project.apply_suggestion(m["id"])
            self.statusBar().showMessage(f"used {len(items)} match(es); the hint pass runs next", 8000)
            self._after_paint()
            return
        if items:
            self.statusBar().showMessage(f"{len(items)} match(es) found. Check each in the Colours panel: "
                                         "Use or Skip, or click the marker on the picture.", 10000)
        else:
            self.statusBar().showMessage("no matches. Slide toward More matches and find again.", 10000)
        self._refresh()

    def _suggestion_clicked(self, ident: str, accept: bool) -> None:
        self._colour_request("apply" if accept else "dismiss", ident)

    def _goto_match(self, ident: str) -> None:
        m = next((x for x in self.project.d.suggestions if x["id"] == ident), None)
        if not m:
            return
        if m["kind"] == "photo":
            self.bottom.setCurrentWidget(self.filmstrip)
            for i, ph in enumerate(self.project.d.photos):
                if ph.id == m["target"]:
                    self.filmstrip.setCurrentRow(i)
            return
        self.bottom.setCurrentWidget(self.timeline)
        for start, seg in zip(self.project.seg_starts(), self.project.d.timeline):
            if seg.clip == m["target"] and seg.src_in <= m["frame"] < seg.src_out:
                self.seek(start + m["frame"] - seg.src_in)
                return
        self.statusBar().showMessage("that frame isn't on the timeline", 4000)

    # ------------------------------------------------------- grading

    def _grade_target(self):
        """(kind, id, clip frame, track) for the grade panel."""
        t = self._paint_target()
        if t is None:
            return None
        if t[0] == "photo":
            return "photo", t[1], None, None
        return "clip", t[1], t[2], self.project.grade_track(t[1], t[2])

    def _load_grade_panel(self) -> None:
        if self._grade_before is not None:
            return  # mid-edit: don't fight the user's drag
        gt = self._grade_target()
        if gt is None:
            self.grade_panel.disable("Nothing to grade: put the playhead on a clip, or pick a photo.")
            return
        kind, ident, frame, track = gt
        if kind == "photo":
            ph = self.project.photo(ident)
            self.grade_panel.load(ph.grade, f"Grading photo {Path(ph.path).name}", "Photos have one grade",
                                  keys_enabled=False)
            return
        c = self.project.d.clips[ident]
        start = self.project.shot_start(ident, frame)
        shot_no = next((i + 1 for i, (a, b) in enumerate(c.shots) if a == start), "?")
        g = G.evaluate(track, frame) if track else G.default_grade()
        keys = (track or {}).get("keys", [])
        k = G.key_at(track, frame) if track else None
        if len(keys) <= 1:
            info = "Static grade: one setting for the whole shot. Add key to animate it."
        else:
            info = f"{len(keys)} keyframes. " + (f"On the key at frame {frame}." if k else
                                                 "Between keys: an edit here adds a key.")
        before = [kk for kk in sorted(keys, key=lambda x: x["frame"]) if kk["frame"] <= frame]
        ease = bool(before and before[-1].get("ease") == "ease")
        self.grade_panel.load(g, f"Grading {c.name}, shot {shot_no} (frames {start} on), frame {frame}",
                              info, ease, keys_enabled=True)

    def _grade_edited(self, g: dict) -> None:
        gt = self._grade_target()
        if gt is None:
            return
        if self._grade_before is None:
            self._grade_before = self.project._dump()
        kind, ident, frame, track = gt
        if kind == "photo":
            self.project.set_photo_grade(ident, None if G.is_identity(g) else g, checkpoint=False)
            self._show_photo(self.filmstrip.currentRow())
        else:
            track = G.copy.deepcopy(track) if track else G.new_track(None, frame)
            G.set_value(track, frame, g)
            self.project.set_grade_track(ident, frame, track, checkpoint=False)
            self._push_layout()
            self._request()
        self._update_overlays()
        self.grade_timer.start()

    def _commit_grade(self) -> None:
        if self._grade_before is not None:
            before, self._grade_before = self._grade_before, None
            self.project.commit("grade", before)
            self._refresh()

    def _grade_key(self, action: str) -> None:
        self._commit_grade()
        gt = self._grade_target()
        if gt is None or gt[0] != "clip":
            return
        _, cid, frame, track = gt
        track = G.copy.deepcopy(track) if track else G.new_track(None, frame)
        keys = track["keys"]
        if action == "add":
            if G.key_at(track, frame) is None:
                keys.append({"frame": frame, "ease": "linear", "grade": G.evaluate(track, frame)})
                keys.sort(key=lambda k: k["frame"])
                self.project.set_grade_track(cid, frame, track, "add grade key")
        elif action == "delete":
            k = G.key_at(track, frame)
            if k is not None and len(keys) > 1:
                keys.remove(k)
                self.project.set_grade_track(cid, frame, track, "delete grade key")
        elif action == "ease":
            before = [k for k in sorted(keys, key=lambda x: x["frame"]) if k["frame"] <= frame]
            if before:
                before[-1]["ease"] = "ease" if self.grade_panel.ease.isChecked() else "linear"
                self.project.set_grade_track(cid, frame, track, "ease")
        elif action in ("prev", "next"):
            frames = sorted(k["frame"] for k in keys)
            tgt = [f for f in frames if (f < frame if action == "prev" else f > frame)]
            if tgt:
                f = tgt[-1] if action == "prev" else tgt[0]
                self.seek(self.playhead + (f - frame))
            return
        self._refresh()

    def _grade_action(self, action: str) -> None:
        gt = self._grade_target()
        if action == "matte":
            self._show_matte = self.grade_panel.show_matte.isChecked()
            self._push_layout()
            self._refresh()
            return
        if gt is None:
            return
        kind, ident, frame, track = gt
        if action == "pick_neutral":
            self.viewer.pick_mode = "neutral"
            self.viewer.update()
        elif action == "pick_hue":
            self.viewer.pick_mode = "hue"
            self.viewer.update()
        elif action == "reset":
            self._commit_grade()
            if kind == "photo":
                self.project.set_photo_grade(ident, None, "reset grade")
            else:
                self.project.set_grade_track(ident, frame, None, "reset grade")
            self._refresh()
        elif action == "load_lut":
            f, _ = QFileDialog.getOpenFileName(self, "Load LUT", self.qs.value("dir_lut", ""), "LUT (*.cube)")
            if f:
                self.qs.setValue("dir_lut", str(Path(f).parent))
                try:
                    G.load_cube(f)
                except (OSError, ValueError) as e:
                    QMessageBox.warning(self, "LUT", str(e))
                    return
                self.grade_panel.set_fields(lut=f)
                self.grade_panel.lut_label.setText(f)
        elif action == "clear_lut":
            self.grade_panel.set_fields(lut=None)
        elif action == "export_lut":
            f, _ = QFileDialog.getSaveFileName(self, "Export grade as LUT", self.qs.value("dir_lut", ""), "LUT (*.cube)")
            if f:
                if not f.endswith(".cube"):
                    f += ".cube"
                left = G.export_cube(self.grade_panel.read(), f)
                msg = f"saved {f}"
                if left:
                    msg += "; a LUT can't hold " + ", ".join(left)
                self.statusBar().showMessage(msg, 10000)
        elif action == "track":
            if kind != "clip":
                return
            c = self.project.d.clips[ident]
            if self.status.get(ident) not in ("ready", "update"):
                self.statusBar().showMessage("colorize the clip first; tracking uses its analysis", 6000)
                return
            g = self.grade_panel.read()
            if g["mask"]["shape"] == "none":
                self.statusBar().showMessage("pick a mask shape first", 4000)
                return
            s = VideoSettings.from_project(self.project.d.settings)
            start = self.project.shot_start(ident, frame)
            end = next(b for a, b in c.shots if a == start)
            self.runner.submit("track_mask", f"Track mask in {c.name}",
                               {"analysis": str(analysis_dir(Path(c.path), s, self.project.d.settings["model"],
                                                             negative=c.neg)),
                                "a": start, "b": end, "frame": frame, "mask": g["mask"], "clip": ident})
        elif action == "clear_track":
            if kind == "clip" and track and track.get("track"):
                t2 = G.copy.deepcopy(track)
                t2.pop("track", None)
                self.project.set_grade_track(ident, frame, t2, "clear mask track")
                self._refresh()

    def _track_done(self, job, result: dict) -> None:
        cid, frame = job.payload["clip"], job.payload["frame"]
        track = G.copy.deepcopy(self.project.grade_track(cid, frame)) or G.new_track(self.grade_panel.read(), frame)
        track["track"] = result["track"]
        self.project.set_grade_track(cid, frame, track, "track mask")
        self.statusBar().showMessage(f"mask tracked over {len(result['track'])} frames", 5000)

    def copy_grade(self) -> None:
        self._grade_clip = self.grade_panel.read() if self.grade_panel.isEnabled() else None
        if self._grade_clip:
            self.statusBar().showMessage("grade copied", 3000)

    def paste_grade(self) -> None:
        gt = self._grade_target()
        if gt is None or self._grade_clip is None:
            return
        self._commit_grade()
        kind, ident, frame, _ = gt
        if kind == "photo":
            self.project.set_photo_grade(ident, self._grade_clip, "paste grade")
        else:
            start = self.project.shot_start(ident, frame)
            self.project.set_grade_track(ident, frame, G.new_track(self._grade_clip, start), "paste grade")
        self._refresh()

    # ------------------------------------------------------ models

    def open_model_manager(self) -> None:
        from .model_manager import ModelManager

        dlg = ModelManager(self.project.d.settings["model"], self)
        dlg.submit.connect(lambda kind, label, payload: self.runner.submit(kind, label, payload))
        self._manager = dlg
        dlg.exec()
        self._manager = None
        self.inspector.set_models(self.project.d.settings["model"])
        if dlg.chosen and dlg.chosen != self.project.d.settings["model"]:
            self._setting_changed("model", dlg.chosen)

    # ---------------------------------------------------------- export

    def export_video(self) -> None:
        if not self.project.duration:
            QMessageBox.information(self, "Export", "The timeline is empty.")
            return
        audio = [a for a, on in zip(project_audio(self.project), self.project.d.audio_enabled) if on]
        default = self.project.d.export.get("last_out") or str(
            Path(self.project.first_clip.path).with_name(Path(self.project.first_clip.path).stem + ".color.mkv"))
        dlg = ExportDialog(self.project, audio, edited=False, default_out=Path(default), parent=self)
        dlg.edited = not self._is_simple(dlg)
        dlg.inout.toggled.connect(lambda _: setattr(dlg, "edited", not self._is_simple(dlg)))
        dlg._changed()
        if dlg.exec() != QDialog.Accepted:
            return
        es = dlg.current()
        out = dlg.out_path()
        self.project.checkpoint("export settings")
        self.project.d.export = {**es.to_dict(), "last_out": str(out)}
        data = asdict(self.project.d)
        if not dlg.use_range():
            data["range"] = None
        payload = {**self._job_payload(), "project": data,
                   "export": {k: v for k, v in es.to_dict().items()}, "out": str(out)}
        self.runner.submit("export", f"Export {out.name}", payload)
        self._refresh()

    def _is_simple(self, dlg) -> bool:
        from ..project import Project as P, _data_from_dict

        data = asdict(self.project.d)
        if not dlg.use_range():
            data["range"] = None
        return P(_data_from_dict(data)).is_simple()

    def export_frame(self) -> None:
        """The frame under the playhead at full size, as export renders it."""
        t = self._paint_target()
        if t is None or t[0] != "clip":
            self.statusBar().showMessage("put the playhead on a clip first (photos: File > Export photos)", 5000)
            return
        if self.status.get(t[1]) not in ("ready", "update"):
            self.statusBar().showMessage("colorize this clip first", 5000)
            return
        c = self.project.d.clips[t[1]]
        f, _ = QFileDialog.getSaveFileName(self, "Export frame", str(Path(self.qs.value("dir_frame", "")) /
                                                                    f"{Path(c.path).stem}_{t[2]:06d}.png"),
                                           "PNG 16-bit (*.png);;TIFF 16-bit (*.tif *.tiff);;JPEG (*.jpg)")
        if not f:
            return
        self.qs.setValue("dir_frame", str(Path(f).parent))
        self.runner.submit("export_frame", f"Export frame {t[2]} of {c.name}",
                           {**self._job_payload(), "project": asdict(self.project.d), "clip": c.id,
                            "frame": t[2], "out": f})

    def export_photos(self) -> None:
        photos = [self.project.d.photos[self.filmstrip.row(it)] for it in self.filmstrip.selectedItems()] \
            or list(self.project.d.photos)
        if not photos:
            QMessageBox.information(self, "Export photos", "There are no photos in the project.")
            return
        dlg = PhotoExportDialog(len(photos), Path(photos[0].path).parent / "colorized", self)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        v["dir"].mkdir(parents=True, exist_ok=True)
        items = [{"src": ph.path, "dst": str(v["dir"] / f"{Path(ph.path).stem}.{v['ext']}"),
                  "strokes": ph.strokes, "grade": ph.grade, "negative": ph.neg,
                  "reference": self.project.photo_reference(ph)}
                 for ph in photos]
        self.runner.submit("export_photos", f"Export {len(items)} photo(s)",
                           {**self._job_payload(), "items": items, "bits": v["bits"], "quality": v["quality"]})

    # -------------------------------------------------------- projects

    def _confirm_discard(self) -> bool:
        if not self.project.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save the project first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self._set_project(Project())

    def _set_project(self, p: Project) -> None:
        self.shuttle(0)
        self.project = p
        self.media = MediaCache()
        self.media_info = {}
        self.photo_icons = {}
        self.timeline.canvas.media = self.media
        self.timeline.set_project(p)
        self.playhead = 0
        for c in p.d.clips.values():
            if not Path(c.path).exists():
                self.statusBar().showMessage(f"missing source: {c.path}", 10000)
                continue
            if not self._load_media(c):
                job = self.runner.submit("import", f"Rebuild proxy for {c.name}",
                                         {"path": c.path, "shot_threshold": p.d.settings["shot_threshold"]})
                self._importing[job.id] = "rebuild"
        self._refresh()
        self._fit()

    def open_project_dialog(self) -> None:
        if not self._confirm_discard():
            return
        f, _ = QFileDialog.getOpenFileName(self, "Open project", self.qs.value("dir_project", ""), PROJECT_FILTER)
        if f:
            self.open_project(f)

    def _autosave(self) -> None:
        from .. import autosave

        if self.project.dirty and (self.project.d.clips or self.project.d.photos):
            try:
                autosave.write(self.project)
            except OSError:
                pass  # a full disk shouldn't interrupt the work; the next try may succeed

    def _offer_recovery(self, path) -> Project | None:
        """The autosaved project, if there is one and the user wants it back."""
        from .. import autosave

        found = autosave.find(path)
        if not found:
            return None
        side, when = found
        what = Path(path).name if path else "a project that was never saved"
        r = QMessageBox.question(
            self, "Recover changes",
            f"There are unsaved changes to {what} from {time.strftime('%Y-%m-%d %H:%M', time.localtime(when))}; "
            "EpochColor didn't close normally. Recover them?",
            QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            autosave.clear(path)
            return None
        try:
            return autosave.recover(side, path)
        except (OSError, ValueError, ProjectError) as e:
            QMessageBox.warning(self, "Can't recover", str(e))
            return None

    def check_recovery(self) -> None:
        """At start, with nothing opened: an unsaved project left by a crash."""
        if self.project.path is None and not self.project.d.clips and not self.project.d.photos:
            p = self._offer_recovery(None)
            if p is not None:
                self._set_project(p)

    def open_project(self, path: str) -> None:
        try:
            p = Project.load(path)
        except (OSError, ValueError, ProjectError) as e:
            QMessageBox.warning(self, "Can't open", f"{path}:\n{e}")
            return
        self.qs.setValue("dir_project", str(Path(path).parent))
        p = self._offer_recovery(path) or p
        self._set_project(p)

    def save_project(self) -> bool:
        from .. import autosave

        if self.project.path is None:
            return self.save_project_as()
        try:
            self.project.save()
        except OSError as e:
            QMessageBox.warning(self, "Can't save", str(e))
            return False
        autosave.clear(self.project.path)
        autosave.clear(None)
        self._refresh_title()
        self.statusBar().showMessage(f"saved {self.project.path}", 3000)
        return True

    def save_project_as(self) -> bool:
        f, _ = QFileDialog.getSaveFileName(self, "Save project", self.qs.value("dir_project", ""), PROJECT_FILTER)
        if not f:
            return False
        if not f.endswith(".epochcolor"):
            f += ".epochcolor"
        self.project.path = Path(f)
        return self.save_project()

    def show_keys(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Keys")
        lay = QVBoxLayout(dlg)
        t = QPlainTextEdit(KEYS_HELP)
        t.setReadOnly(True)
        t.setStyleSheet("font-family: monospace;")
        t.setMinimumSize(520, 420)
        lay.addWidget(t)
        dlg.exec()

    def closeEvent(self, ev) -> None:
        if not self._confirm_discard():
            ev.ignore()
            return
        self.project.dirty = False  # saved or discarded: either way nothing to recover
        if self.runner.busy() and QMessageBox.question(
                self, "Jobs running", "Jobs are still running. Stop them and quit?") != QMessageBox.Yes:
            ev.ignore()
            return
        self.shutdown()
        ev.accept()

    def shutdown(self) -> None:
        from PySide6.QtCore import QMetaObject

        from .. import autosave

        self.play_timer.stop()
        self.autosave_timer.stop()
        self.audio.stop()
        if self.audio.proc is not None:
            self.audio.proc.kill()
        if not self.project.dirty:  # saved, or the changes were discarded on purpose
            autosave.clear(self.project.path)
        self.runner.shutdown()
        if self.preview_thread.isRunning():
            QMetaObject.invokeMethod(self.provider, "stop", Qt.BlockingQueuedConnection)
            self.preview_thread.quit()
            self.preview_thread.wait(2000)


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("EpochColor")
    theme.apply(app)
    project = next((a for a in argv[1:] if a.endswith(".epochcolor")), None)
    w = MainWindow(project)
    media = [a for a in argv[1:] if not a.endswith(".epochcolor") and Path(a).is_file()]
    if media:
        from ..imageio import SUPPORTED_IN

        photos = [m for m in media if Path(m).suffix.lower() in SUPPORTED_IN]
        clips = [m for m in media if m not in photos]
        if clips:
            w.add_clips(clips)
        if photos:
            w.add_photos(photos)
    w.show()
    # Ctrl+C in a terminal or a SIGTERM from the session closes cleanly, so
    # the worker process and its queues go with it
    import signal

    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    ticker = QTimer()
    ticker.start(300)
    ticker.timeout.connect(lambda: None)  # lets Python run its signal handlers
    app.aboutToQuit.connect(w.shutdown)
    if not os.environ.get("EPOCHCOLOR_NO_FIRST_RUN"):
        QTimer.singleShot(400, w.first_run_checks)
        QTimer.singleShot(600, w.check_recovery)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
