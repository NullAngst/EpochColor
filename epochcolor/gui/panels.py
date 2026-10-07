"""Smaller docks: scopes, saved colours with their matches, and the paint bar."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSizePolicy, QSlider, QToolBar, QToolButton, QVBoxLayout, QWidget,
)

from .. import scopes
from . import theme
from .preview import to_qimage


def swatch(rgb, size: int = 16) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return QIcon(pm)


def qimage_to_rgb8(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    return arr[:, : w * 3].reshape(h, w, 3).copy()


class ScopesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.kind = QComboBox()
        self.kind.addItems(list(scopes.SCOPES))
        self.view = QLabel("No picture")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(200, 150)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setStyleSheet("background: #111111;")
        lay.addWidget(self.kind)
        lay.addWidget(self.view, 1)
        self.kind.currentIndexChanged.connect(lambda _: self.refresh())
        self._img = None

    def show_image(self, img: QImage | None) -> None:
        self._img = img
        if self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        if self._img is None or self._img.isNull():
            self.view.setText("No picture")
            return
        rgb = qimage_to_rgb8(self._img)
        out = scopes.SCOPES[self.kind.currentText()](rgb)
        pm = QPixmap.fromImage(to_qimage(out))
        self.view.setPixmap(pm.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.refresh()


class ColoursPanel(QWidget):
    """Saved colours and the matches found for them."""
    useColour = Signal(list)
    request = Signal(str, str)  # action, id: rename, delete, find, goto, apply, dismiss, apply_all
    settingChanged = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(QLabel("Saved colours"))
        self.colours = QListWidget()
        self.colours.itemDoubleClicked.connect(self._use)
        self.colours.setToolTip("Double-click to paint with it")
        lay.addWidget(self.colours, 1)
        row = QHBoxLayout()
        for action, text in (("use", "Paint with"), ("rename", "Rename"), ("delete", "Delete")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, a=action: self._colour_action(a))
            row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("Mark matches, wait for a click", "ask")
        self.mode.addItem("Paint matches straight away", "auto")
        self.mode.currentIndexChanged.connect(lambda _: self.settingChanged.emit("match_mode", self.mode.currentData()))
        row.addWidget(self.mode, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Match threshold"))
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0.3, 0.99)
        self.thr.setSingleStep(0.02)
        self.thr.setToolTip("How alike a place has to look. Lower finds more, including wrong things.")
        self.thr.editingFinished.connect(lambda: self.settingChanged.emit("match_threshold", round(self.thr.value(), 3)))
        row.addWidget(self.thr)
        find = QPushButton("Find in every shot and photo")
        find.clicked.connect(lambda: self.request.emit("find", ""))
        row.addWidget(find)
        lay.addLayout(row)
        lay.addWidget(QLabel("Matches"))
        self.matches = QListWidget()
        self.matches.itemDoubleClicked.connect(lambda it: self.request.emit("goto", it.data(Qt.UserRole)))
        self.matches.setToolTip("Double-click to go there")
        lay.addWidget(self.matches, 1)
        row = QHBoxLayout()
        for action, text in (("apply", "Apply"), ("dismiss", "Dismiss"), ("apply_all", "Apply all")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, a=action: self._match_action(a))
            row.addWidget(b)
        lay.addLayout(row)
        note = QLabel("Matching compares what the model 'sees' under your stroke with every shot and photo. "
                      "It finds the same object well and look-alikes too, so check before applying.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        lay.addWidget(note)
        self._colours = []

    def show_data(self, colours: list, matches: list, names: dict, settings: dict) -> None:
        self._colours = colours
        cur = self.colours.currentRow()
        self.colours.clear()
        for c in colours:
            it = QListWidgetItem(swatch(c["rgb"]), c["name"])
            it.setData(Qt.UserRole, c["id"])
            self.colours.addItem(it)
        if 0 <= cur < self.colours.count():
            self.colours.setCurrentRow(cur)
        by_id = {c["id"]: c for c in colours}
        self.matches.clear()
        for m in matches:
            c = by_id.get(m["color"])
            if not c:
                continue
            where = names.get((m["kind"], m["target"]), m["target"])
            if m["kind"] == "clip":
                where += f", frame {m['frame']}"
            it = QListWidgetItem(swatch(c["rgb"]), f"{c['name']}  ->  {where}  ({m['score']:.2f})")
            it.setData(Qt.UserRole, m["id"])
            self.matches.addItem(it)
        for w, v in ((self.mode, settings.get("match_mode", "ask")),):
            w.blockSignals(True)
            w.setCurrentIndex(max(0, w.findData(v)))
            w.blockSignals(False)
        self.thr.blockSignals(True)
        self.thr.setValue(float(settings.get("match_threshold", 0.75)))
        self.thr.blockSignals(False)

    def _use(self, it) -> None:
        c = next((c for c in self._colours if c["id"] == it.data(Qt.UserRole)), None)
        if c:
            self.useColour.emit(c["rgb"])

    def _colour_action(self, a: str) -> None:
        it = self.colours.currentItem()
        if it is None:
            return
        if a == "use":
            self._use(it)
        else:
            self.request.emit(a, it.data(Qt.UserRole))

    def _match_action(self, a: str) -> None:
        if a == "apply_all":
            self.request.emit("apply_all", "")
            return
        it = self.matches.currentItem()
        if it is not None:
            self.request.emit(a, it.data(Qt.UserRole))


class PaintBar(QToolBar):
    """Brush controls above the viewer."""
    changed = Signal()
    action = Signal(str)  # clear, apply, save_colour

    def __init__(self, parent=None):
        super().__init__("Paint", parent)
        self.setObjectName("paintbar")
        self.rgb = [40, 70, 160]
        self.paint = QToolButton()
        self.paint.setText("Paint hints")
        self.paint.setCheckable(True)
        self.paint.setToolTip("Paint colour hints on the picture (P)")
        self.colour = QToolButton()
        self.colour.setToolTip("Brush colour. Ctrl+click on the picture picks one from it.")
        self.colour.clicked.connect(self._choose)
        self.neutral = QToolButton()
        self.neutral.setText("Grey")
        self.neutral.setCheckable(True)
        self.neutral.setToolTip("Paint 'no colour here': a white shirt, a grey wall")
        self.erase = QToolButton()
        self.erase.setText("Erase")
        self.erase.setCheckable(True)
        self.erase.setToolTip("Click a stroke to remove it (right-click does this too)")
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(3, 120)
        self.size.setValue(20)
        self.size.setFixedWidth(110)
        self.size.setToolTip("Brush size ([ and ], or the mouse wheel while painting)")
        self.show = QCheckBox("Show strokes")
        self.show.setChecked(True)
        self.auto = QCheckBox("Apply as I paint")
        self.auto.setChecked(True)
        self.auto.setToolTip("Re-run the hint pass a moment after each stroke")
        for w in (self.paint, self.colour, self.neutral, self.erase):
            self.addWidget(w)
        self.addWidget(QLabel("  Size "))
        self.addWidget(self.size)
        self.addSeparator()
        self.addWidget(self.show)
        self.addWidget(self.auto)
        self.addSeparator()
        for name, text, tip in (("apply", "Apply hints", "Run the hint pass now (Ctrl+Return)"),
                                ("clear", "Clear frame", "Remove every stroke on this frame"),
                                ("save_colour", "Save colour...", "Name the last stroke's colour to reuse it")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, n=name: self.action.emit(n))
            self.addWidget(b)
        for w in (self.paint, self.neutral, self.erase, self.show):
            w.toggled.connect(lambda _: self.changed.emit())
        self.size.valueChanged.connect(lambda _: self.changed.emit())
        self._update_swatch()

    def _update_swatch(self) -> None:
        self.colour.setIcon(swatch(self.rgb, 18))
        self.colour.setText("")

    def set_rgb(self, rgb) -> None:
        self.rgb = [int(v) for v in rgb]
        self.neutral.setChecked(False)
        self._update_swatch()
        self.changed.emit()

    def _choose(self) -> None:
        c = QColorDialog.getColor(QColor(*self.rgb), self, "Brush colour")
        if c.isValid():
            self.set_rgb([c.red(), c.green(), c.blue()])

    def radius(self) -> float:
        return self.size.value() / 1000.0
