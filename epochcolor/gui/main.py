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
from . import theme
from .export_dialog import ExportDialog, PhotoExportDialog
from .jobs import JobRunner
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
Ctrl+R       colorize the selected clip (Ctrl+Shift+R: all)
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
        self.model.addItem("siggraph17 (hints, default)", "siggraph17")
        self.model.addItem("eccv16 (automatic)", "eccv16")
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
        self.denoise_auto = QCheckBox("auto")
        self.denoise = QDoubleSpinBox()
        self.denoise.setRange(0, 30)
        self.denoise.setSingleStep(0.5)
        dn = QWidget()
        dl = QHBoxLayout(dn)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self.denoise_auto)
        dl.addWidget(self.denoise, 1)
        f.addRow("Model", self.model)
        f.addRow("Working size", self.working)
        f.addRow("Chroma size", self.chroma)
        f.addRow("Stabilize", self.stabilize)
        f.addRow("Grain kept", self.grain)
        f.addRow("Saturation", self.saturation)
        f.addRow("Denoise", dn)
        note = QLabel("Model, working size, chroma size and denoise need a new colorize pass. "
                      "Stabilize reruns a quick pass. Grain and saturation apply right away.")
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

    def show_settings(self, s: dict) -> None:
        widgets = (self.model, self.working, self.chroma, self.stabilize, self.grain, self.saturation,
                   self.denoise, self.denoise_auto)
        for w in widgets:
            w.blockSignals(True)
        self.model.setCurrentIndex(max(0, self.model.findData(s.get("model"))))
        self.working.setValue(int(s.get("working_size", 512)))
        self.chroma.setValue(int(s.get("chroma_size", 256)))
        self.stabilize.setValue(float(s.get("stabilize", 0.9)))
        self.grain.setValue(float(s.get("grain", 100)))
        self.saturation.setValue(float(s.get("saturation", 1.0)))
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

        top = QWidget()
        topl = QVBoxLayout(top)
        topl.setContentsMargins(0, 0, 0, 0)
        topl.setSpacing(0)
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
        dock = QDockWidget("Inspector", self)
        dock.setObjectName("inspector")
        dock.setWidget(self.inspector)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.inspector_dock = dock

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

        m = mb.addMenu("&View")
        act(m, "Colour", lambda: self.viewer.set_mode("color"), "1")
        act(m, "Before / after split", lambda: self.viewer.set_mode("split"), "2")
        act(m, "Original black and white", lambda: self.viewer.set_mode("original"), "3")
        m.addSeparator()
        act(m, "Zoom timeline in", lambda: self._zoom(1.5), ["=", "Ctrl+="])
        act(m, "Zoom timeline out", lambda: self._zoom(1 / 1.5), ["-", "Ctrl+-"])
        act(m, "Fit timeline", self._fit, "Shift+Z")

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
        m.addSeparator()
        act(m, "Install PyTorch...", lambda: self.setup_torch(False))
        act(m, "Download weights for the current model", self.fetch_weights)
        act(m, "Model device...", self.choose_device)

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
            if self.status.get(c.id) == "ready":
                ab = str(analysis_dir(Path(c.path), s, model) / "ab.npy")
            out.append(LayoutPiece(start, seg.length, c.id, seg.src_in, str(mi.proxy), c.path, c.fps, ab))
        return out

    def _refresh(self) -> None:
        """Bring every view in line with the project."""
        s = VideoSettings.from_project(self.project.d.settings)
        model = self.project.d.settings["model"]
        self.status = {cid: analysis_state(c, s, model) for cid, c in self.project.d.clips.items()}
        self.timeline.canvas.status = self.status
        self.timeline.set_project(self.project) if self.timeline.canvas.project is not self.project \
            else self.timeline.refresh()
        self.provider.layout_sig.emit(self._layout(), dict(self.project.d.settings))
        self.playhead = max(0, min(self.playhead, max(0, self.project.duration - 1)))
        self.timeline.canvas.playhead = self.playhead
        self._refresh_bin()
        self._refresh_inspector()
        self._refresh_title()
        self.undo_act.setText(f"Undo {self.project.undo_label}" if self.project.undo_label else "Undo")
        self.redo_act.setText(f"Redo {self.project.redo_label}" if self.project.redo_label else "Redo")
        self.undo_act.setEnabled(self.project.undo_label is not None)
        self.redo_act.setEnabled(self.project.redo_label is not None)
        if self.bottom.currentWidget() is self.timeline:
            self._request()
        else:
            self._show_photo(self.filmstrip.currentRow())

    def _refresh_title(self) -> None:
        name = self.project.path.name if self.project.path else "untitled"
        self.setWindowTitle(f"{'*' if self.project.dirty else ''}{name} - EpochColor")

    def _refresh_bin(self) -> None:
        self.clip_list.clear()
        labels = {"ready": "colorized", "partial": "partly colorized", "none": "not colorized"}
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
                done = photo_result_path(ph.path, self.project.d.settings, self.project.d.settings["model"]).exists()
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
                                             self.project.d.settings["model"]).exists()
                except OSError:
                    done = False
                ins.clip_info.setText(f"<b>{Path(ph.path).name}</b><br>{Path(ph.path).parent}<br>"
                                      f"{'colorized' if done else 'not colorized yet'}")
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
            st = {"ready": "colorized", "partial": "partly colorized, run Colorize to finish",
                  "none": "not colorized yet"}[self.status.get(c.id, "none")]
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

    def _frame_ready(self, frame: int, gray, color, full: bool) -> None:
        if frame != self.playhead or self.bottom.currentWidget() is not self.timeline:
            return
        self.viewer.set_images(gray, color, full)

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
        from ..color import srgb_to_l
        from ..worker import photo_result_path

        ph = self.project.d.photos[row]
        try:
            src = self._photo_display(ph.path)
            if src.ndim == 3:  # show what the model sees: luminance only
                gray = srgb_to_l(src.astype(np.float32) / 255)
                from ..color import lab_to_srgb

                src = (lab_to_srgb(gray, np.zeros(gray.shape + (2,), np.float32))[..., 0] * 255 + 0.5).astype(np.uint8)
            gimg = to_qimage(src)
            res = photo_result_path(ph.path, self.project.d.settings, self.project.d.settings["model"])
            cimg = to_qimage(self._photo_display(str(res))) if res.exists() else None
            self.viewer.set_images(gimg, cimg, True, Path(ph.path).name)
            self._refresh_inspector()
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
            self.runner.submit("analyze", f"Colorize {c.name}", {**self._job_payload(), "clip": asdict(c)})

    def colorize_photos(self) -> None:
        rows = sorted({self.filmstrip.row(it) for it in self.filmstrip.selectedItems()}) or \
            ([self.filmstrip.currentRow()] if self.filmstrip.currentRow() >= 0 else [])
        for r in rows:
            ph = self.project.d.photos[r]
            self.runner.submit("photo", f"Colorize {Path(ph.path).name}",
                               {**self._job_payload(), "photo": ph.id, "path": ph.path})

    def fetch_weights(self) -> None:
        self.runner.submit("fetch", f"Download {self.project.d.settings['model']} weights",
                           {"model": self.project.d.settings["model"]})

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
        items = [(ph.path, str(v["dir"] / f"{Path(ph.path).stem}.{v['ext']}")) for ph in photos]
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
