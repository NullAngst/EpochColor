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
from ..timeline_export import analysis_state, project_audio
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
1  2  3      colour / before-after split / original
P            paint hints on/off; [ and ] (or the wheel) brush size
             drag paints, right-click removes a stroke, Ctrl+click picks a colour
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
        lay.addWidget(g)

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
                   self.cast, self.denoise, self.denoise_auto)
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

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(8)
        self.play_timer.timeout.connect(self._tick)
        self._flow_readers: dict = {}  # proxy readers for carrying strokes, per clip
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
        self.mode_btns = {}
        for mode, text in (("color", "Colour"), ("split", "Before / after"), ("original", "Original")):
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
        self.inspector.colorize_all_btn.clicked.connect(self.colorize_all)
        self.inspector.cancel_btn.clicked.connect(lambda: self.runner.cancel(
            self.runner.current.id if self.runner.current else -1))
        self.inspector.cancel_all_btn.clicked.connect(lambda: self.runner.cancel(None))
        self.inspector.manager_btn.clicked.connect(self.open_model_manager)
        dock = QDockWidget("Inspector", self)
        dock.setObjectName("inspector")
        dock.setWidget(self.inspector)
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
        self.colours.settingChanged.connect(self._setting_changed)
        cdock = QDockWidget("Colours", self)
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
        act(m, "Go to start", lambda: self.seek(0), "Home")
        act(m, "Go to end", lambda: self.seek(self.project.duration - 1), "End")

        m = mb.addMenu("&Colour")
        act(m, "Colorize selected clip", self.colorize_selected, "Ctrl+R")
        act(m, "Colorize all clips", self.colorize_all, "Ctrl+Shift+R")
        act(m, "Colorize selected photos", self.colorize_photos, "Ctrl+Alt+R")
        act(m, "Apply hints", self.apply_hints, ["Ctrl+Return", "Ctrl+Enter"])
        act(m, "Find saved colours everywhere", lambda: self._colour_request("find", ""))
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
            ab = None
            if self.status.get(c.id) in ("ready", "update"):
                # "update": hints or the stabilizer changed; show the last result until it reruns
                ab = str(analysis_dir(Path(c.path), s, model) / "ab.npy")
            out.append(LayoutPiece(start, seg.length, c.id, seg.src_in, str(mi.proxy), c.path, c.fps, ab))
        return out

    def _refresh(self) -> None:
        """Bring every view in line with the project."""
        s = VideoSettings.from_project(self.project.d.settings)
        model = self.project.d.settings["model"]
        self.status = {cid: analysis_state(c, s, model, hints=self.project.d.hints.get(cid))
                       for cid, c in self.project.d.clips.items()}
        self.timeline.canvas.status = self.status
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
                    done = photo_result_path(ph.path, self.project.d.settings,
                                             self.project.d.settings["model"], ph.strokes).exists()
                except OSError:
                    done = False
                state = "colorized" if done else ("hints changed, apply them" if ph.strokes else "not colorized yet")
                ins.clip_info.setText(f"<b>{Path(ph.path).name}</b><br>{Path(ph.path).parent}<br>"
                                      f"{len(ph.strokes)} hint stroke(s)<br>{state}")
            else:
                ins.clip_info.setText("No photo selected")
            ins.show_settings(self.project.d.settings)
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
            self.btn["play"].setText("Pause")
            self.speed_label.setText("" if self.speed == 1 else f"{self.speed:+g}x")
        else:
            self.speed = 0.0
            self.play_timer.stop()
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
            if src.ndim == 3:  # show what the model sees: luminance only
                gray = srgb_to_l(src.astype(np.float32) / 255)
                src = (lab_to_srgb(gray, np.zeros(gray.shape + (2,), np.float32))[..., 0] * 255 + 0.5).astype(np.uint8)
            gimg = to_qimage(src)
            res = photo_result_path(ph.path, self.project.d.settings, self.project.d.settings["model"], ph.strokes)
            if not res.exists() and ph.strokes:
                # hints not applied yet: show the last unhinted result meanwhile
                res = photo_result_path(ph.path, self.project.d.settings, self.project.d.settings["model"])
            cimg = raw = None
            if res.exists():
                rgb = self._photo_display(str(res)).astype(np.float32) / 255.0
                from ..chroma import adjust_rgb

                rgb = adjust_rgb(rgb, float(self.project.d.settings.get("saturation", 1.0)),
                                 float(self.project.d.settings.get("cast", 0.0)))
                raw = to_qimage((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
                cimg = raw
                if ph.grade and not G.is_identity(ph.grade):
                    matte = [] if self._show_matte else None
                    out = G.apply(rgb, ph.grade, matte_out=matte)
                    if self._show_matte:
                        m = matte[0] if matte and matte[0] is not None else np.ones(rgb.shape[:2], np.float32)
                        cimg = to_qimage((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8))
                    else:
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
                               {**self._job_payload(), "clip": asdict(c), "hints": self.project.d.hints.get(c.id)})

    def colorize_photos(self) -> None:
        rows = sorted({self.filmstrip.row(it) for it in self.filmstrip.selectedItems()}) or \
            ([self.filmstrip.currentRow()] if self.filmstrip.currentRow() >= 0 else [])
        for r in rows:
            ph = self.project.d.photos[r]
            self.runner.submit("photo", f"Colorize {Path(ph.path).name}",
                               {**self._job_payload(), "photo": ph.id, "path": ph.path, "strokes": ph.strokes})

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
        elif job.kind == "track_mask":
            self._track_done(job, result)
        else:
            self.statusBar().showMessage(f"done: {job.label}", 5000)
        self._refresh()

    def _job_failed(self, job, msg: str, tb: str) -> None:
        self._importing.pop(job.id, None)
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
        self.viewer.show_strokes = pb.show.isChecked() or pb.paint.isChecked()
        self.viewer.brush = {"rgb": list(pb.rgb), "neutral": pb.neutral.isChecked(), "radius": pb.radius()}
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
                               {**self._job_payload(), "photo": ph.id, "path": ph.path, "strokes": ph.strokes})
            return
        c = self.project.d.clips[t[1]]
        if self.status.get(c.id) == "none":
            self.statusBar().showMessage(f"{c.name} isn't colorized yet; hints go in with its first colorize pass", 6000)
        self.runner.cancel_where(lambda j: j.kind == "analyze" and j.payload.get("clip", {}).get("id") == c.id,
                                 running=False)
        self.runner.submit("analyze", f"Hints on {c.name}",
                           {**self._job_payload(), "clip": asdict(c), "hints": self.project.d.hints.get(c.id)})

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

    # -------------------------------------------------- saved colours

    def save_colour(self) -> None:
        t = self._paint_target()
        strokes = [st for st in self._target_strokes(t) if not st.get("neutral")]
        if not strokes:
            self.statusBar().showMessage("paint a stroke first; its colour is what gets saved", 5000)
            return
        st = strokes[-1]
        name, ok = QInputDialog.getText(self, "Save colour", "Name (\"Anna's coat, navy\"):")
        if not ok or not name.strip():
            return
        source = {"kind": t[0], "target": t[1], "frame": t[2], "stroke": st}
        self.project.add_color(name.strip(), st["rgb"], source)
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

        self.colours.show_data(self.project.d.colors, self.project.d.suggestions, self._names(),
                               self.project.d.settings, thumbs=match_thumb_path)

    def _colour_request(self, action: str, ident: str) -> None:
        p = self.project
        if action == "rename":
            c = next((c for c in p.d.colors if c["id"] == ident), None)
            if c:
                name, ok = QInputDialog.getText(self, "Rename colour", "Name:", text=c["name"])
                if ok and name.strip():
                    p.rename_color(ident, name.strip())
        elif action == "delete":
            p.remove_color(ident)
        elif action == "save_colour":
            self.save_colour()
            return
        elif action == "find":
            if not p.d.colors:
                self.statusBar().showMessage("save a colour first: paint a stroke, then Save last stroke", 5000)
                return
            cols = [c for c in p.d.colors if c["id"] == ident] if ident else list(p.d.colors)
            clips = {cid: {"path": c.path, "fps_num": c.fps_num, "fps_den": c.fps_den, "shots": c.shots}
                     for cid, c in p.d.clips.items()}
            photos = {ph.id: ph.path for ph in p.d.photos}
            label = f"Find {cols[0]['name']}" if ident and cols else "Find saved colours"
            self.colours_dock.show()
            self.colours_dock.raise_()
            self.runner.submit("match", label,
                               {**self._job_payload(), "colors": cols, "clips": clips, "photos": photos})
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
                               {"analysis": str(analysis_dir(Path(c.path), s, self.project.d.settings["model"])),
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
                  "strokes": ph.strokes, "grade": ph.grade} for ph in photos]
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

    def open_project(self, path: str) -> None:
        try:
            p = Project.load(path)
        except (OSError, ValueError, ProjectError) as e:
            QMessageBox.warning(self, "Can't open", f"{path}:\n{e}")
            return
        self.qs.setValue("dir_project", str(Path(path).parent))
        self._set_project(p)

    def save_project(self) -> bool:
        if self.project.path is None:
            return self.save_project_as()
        try:
            self.project.save()
        except OSError as e:
            QMessageBox.warning(self, "Can't save", str(e))
            return False
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
        if self.runner.busy() and QMessageBox.question(
                self, "Jobs running", "Jobs are still running. Stop them and quit?") != QMessageBox.Yes:
            ev.ignore()
            return
        self.shutdown()
        ev.accept()

    def shutdown(self) -> None:
        from PySide6.QtCore import QMetaObject

        self.play_timer.stop()
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
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
