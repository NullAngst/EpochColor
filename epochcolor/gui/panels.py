"""Smaller docks: scopes, saved colours with their matches, and the paint bar."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QSizePolicy, QSlider, QSpinBox, QToolBar, QToolButton, QVBoxLayout,
    QWidget,
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


HOW_MATCHING_WORKS = (
    "<b>1.</b> Paint a stroke on something whose colour you know: a coat, a car, a wall.<br>"
    "<b>2.</b> <b>Save last stroke</b> gives it a name.<br>"
    "<b>3.</b> <b>Find</b> looks for the same thing in the other shots and photos. Each match below "
    "shows where it found it. <b>Use</b> paints the colour there, <b>Skip</b> drops it. On the picture, "
    "click a marker to use it, right-click to skip it.")


def _thumb_icon(path, rgb, size=(144, 81)) -> QIcon:
    from pathlib import Path

    if path and Path(path).exists():
        pm = QPixmap(str(path))
        if not pm.isNull():
            return QIcon(pm.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation))
    return swatch(rgb, size[1])


def confidence(score: float, threshold: float) -> str:
    if score >= threshold + 0.10:
        return "strong match"
    if score >= threshold + 0.04:
        return "likely"
    return "possible, check it"


class ColoursPanel(QWidget):
    """Saved colours and the matches found for them."""
    useColour = Signal(list)
    # rename, delete, find (id, "" for all), goto, apply, dismiss, apply_all, dismiss_all, save_colour
    request = Signal(str, str)
    settingChanged = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import QSize

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        how = QLabel(HOW_MATCHING_WORKS)
        how.setWordWrap(True)
        how.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        lay.addWidget(how)

        lay.addWidget(QLabel("<b>Saved colours</b>"))
        lists_css = (f"QListWidget::item {{ padding: 3px; }}"
                     f"QListWidget::item:selected {{ background: #3a3a3a; color: {theme.TEXT};"
                     f" border-left: 3px solid {theme.ACCENT}; }}")
        self.colours = QListWidget()
        self.colours.setStyleSheet(lists_css)
        self.colours.setIconSize(QSize(96, 54))
        self.colours.itemDoubleClicked.connect(self._use)
        self.colours.setToolTip("Double-click to paint with it")
        lay.addWidget(self.colours, 1)
        row = QHBoxLayout()
        self.save_btn = QPushButton("Save last stroke...")
        self.save_btn.clicked.connect(lambda: self.request.emit("save_colour", ""))
        self.find_one = QPushButton("Find this one")
        self.find_one.clicked.connect(lambda: self._colour_action("find"))
        self.find_all = QPushButton("Find all")
        self.find_all.clicked.connect(lambda: self.request.emit("find", ""))
        for b in (self.save_btn, self.find_one, self.find_all):
            row.addWidget(b)
        lay.addLayout(row)
        row = QHBoxLayout()
        for action, text in (("use", "Paint with it"), ("rename", "Rename"), ("delete", "Delete")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, a=action: self._colour_action(a))
            row.addWidget(b)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("More matches"))
        self.strict = QSlider(Qt.Horizontal)
        self.strict.setRange(50, 95)
        self.strict.setToolTip("How alike a place has to look to count as a match. Left finds more, "
                               "including things that only look similar; right finds fewer, closer ones.")
        self.strict.sliderReleased.connect(
            lambda: self.settingChanged.emit("match_threshold", round(self.strict.value() / 100.0, 2)))
        row.addWidget(self.strict, 1)
        row.addWidget(QLabel("Closer matches"))
        lay.addLayout(row)
        self.mode = QComboBox()
        self.mode.addItem("Ask me about each match", "ask")
        self.mode.addItem("Use every match straight away", "auto")
        self.mode.currentIndexChanged.connect(lambda _: self.settingChanged.emit("match_mode", self.mode.currentData()))
        lay.addWidget(self.mode)

        self.matches_label = QLabel("<b>Matches</b>")
        lay.addWidget(self.matches_label)
        self.matches = QListWidget()
        self.matches.setStyleSheet(lists_css)
        self.matches.setIconSize(QSize(144, 81))
        self.matches.itemDoubleClicked.connect(lambda it: it.data(Qt.UserRole) and
                                               self.request.emit("goto", it.data(Qt.UserRole)))
        self.matches.setToolTip("Double-click to go there")
        lay.addWidget(self.matches, 2)
        row = QHBoxLayout()
        for action, text, tip in (("apply", "Use", "Paint this colour where it was found"),
                                  ("dismiss", "Skip", "Drop this match"),
                                  ("goto", "Go to", "Show the frame or photo"),
                                  ("apply_all", "Use all", "Paint every match in the list"),
                                  ("dismiss_all", "Skip all", "Clear the list")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, a=action: self._match_action(a))
            row.addWidget(b)
        lay.addLayout(row)
        self._colours = []
        self._threshold = 0.75

    def show_data(self, colours: list, matches: list, names: dict, settings: dict, thumbs=None) -> None:
        """thumbs: id -> picture path, for saved colours and matches."""
        thumbs = thumbs or (lambda ident: None)
        self._colours = colours
        self._threshold = float(settings.get("match_threshold", 0.75))
        counts: dict[str, int] = {}
        for m in matches:
            counts[m["color"]] = counts.get(m["color"], 0) + 1
        cur = self.colours.currentRow()
        self.colours.clear()
        for c in colours:
            src = c.get("source", {})
            where = names.get((src.get("kind"), src.get("target")), "?")
            if src.get("kind") == "clip":
                where += f", frame {src.get('frame')}"
            n = counts.get(c["id"], 0)
            text = f"{c['name']}\nfrom {where}" + (f"\n{n} match{'es' if n != 1 else ''} waiting" if n else "")
            it = QListWidgetItem(_thumb_icon(thumbs(c["id"]), c["rgb"], (96, 54)), text)
            it.setData(Qt.UserRole, c["id"])
            self.colours.addItem(it)
        if not colours:
            it = QListWidgetItem("No saved colours yet. Paint a stroke, then Save last stroke.")
            it.setFlags(Qt.NoItemFlags)
            self.colours.addItem(it)
        elif 0 <= cur < self.colours.count():
            self.colours.setCurrentRow(cur)
        elif self.colours.count():
            self.colours.setCurrentRow(0)
        for b in (self.find_one, self.find_all):
            b.setEnabled(bool(colours))

        by_id = {c["id"]: c for c in colours}
        mcur = self.matches.currentRow()
        self.matches.clear()
        for m in matches:
            c = by_id.get(m["color"])
            if not c:
                continue
            where = names.get((m["kind"], m["target"]), m["target"])
            if m["kind"] == "clip":
                where += f", frame {m['frame']}"
            text = f"{c['name']}?\n{where}\n{confidence(m['score'], self._threshold)} ({m['score']:.2f})"
            it = QListWidgetItem(_thumb_icon(thumbs(m["id"]), c["rgb"]), text)
            it.setData(Qt.UserRole, m["id"])
            self.matches.addItem(it)
        self.matches_label.setText(f"<b>Matches</b> ({len(matches)} waiting)" if matches else "<b>Matches</b>")
        if not matches:
            it = QListWidgetItem("Nothing waiting. Find looks for your saved colours; if it finds nothing, "
                                 "slide toward More matches and find again.")
            it.setFlags(Qt.NoItemFlags)
            self.matches.addItem(it)
        elif self.matches.count():
            self.matches.setCurrentRow(min(max(0, mcur), self.matches.count() - 1))
        self.mode.blockSignals(True)
        self.mode.setCurrentIndex(max(0, self.mode.findData(settings.get("match_mode", "ask"))))
        self.mode.blockSignals(False)
        self.strict.blockSignals(True)
        self.strict.setValue(int(round(self._threshold * 100)))
        self.strict.blockSignals(False)

    def _use(self, it) -> None:
        c = next((c for c in self._colours if c["id"] == it.data(Qt.UserRole)), None)
        if c:
            self.useColour.emit(c["rgb"])

    def _colour_action(self, a: str) -> None:
        it = self.colours.currentItem()
        if it is None or not it.data(Qt.UserRole):
            return
        if a == "use":
            self._use(it)
        else:
            self.request.emit(a, it.data(Qt.UserRole))

    def _match_action(self, a: str) -> None:
        if a in ("apply_all", "dismiss_all"):
            self.request.emit(a, "")
            return
        it = self.matches.currentItem()
        if it is not None and it.data(Qt.UserRole):
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
        self.size.setFixedWidth(80)
        self.size.setToolTip("Brush size ([ and ], or the mouse wheel while painting)")
        self.show = QCheckBox("Show strokes")
        self.show.setChecked(True)
        self.auto = QCheckBox("Apply as I paint")
        self.auto.setChecked(True)
        self.auto.setToolTip("Re-run the hint pass a moment after each stroke")
        # paint by frame
        self.frame_mode = QToolButton()
        self.frame_mode.setText("Frame by frame")
        self.frame_mode.setCheckable(True)
        self.frame_mode.setToolTip(
            "Off: a stroke follows its object through the whole shot.\n"
            "On: a stroke only colours the frames within its reach, so you can go through a shot "
            "frame by frame. Carry (N, Shift+N) moves this frame's strokes onto the next or "
            "previous frame along the motion, ready to touch up. (F)")
        self.reach = QSpinBox()
        self.reach.setRange(0, 240)
        self.reach.setPrefix("reach ±")
        self.reach.setSuffix(" fr")
        self.reach.setSpecialValueText("this frame only")
        self.reach.setToolTip("How many frames either side a new stroke colours. 0 is this frame only.")
        self.onion = QCheckBox("Onion skin")
        self.onion.setChecked(True)
        self.onion.setToolTip("Show the nearest earlier painted frame's strokes faintly, to trace over")
        self.carry_prev = QToolButton()
        self.carry_prev.setText("<- Carry")
        self.carry_prev.setToolTip("Move this frame's strokes onto the previous frame along the motion, "
                                   "and go there (Shift+N)")
        self.carry_next = QToolButton()
        self.carry_next.setText("Carry ->")
        self.carry_next.setToolTip("Move this frame's strokes onto the next frame along the motion, "
                                   "and go there (N)")
        self.carry_prev.clicked.connect(lambda: self.action.emit("carry_prev"))
        self.carry_next.clicked.connect(lambda: self.action.emit("carry_next"))
        # two rows, so nothing hides behind the overflow arrow when the viewer
        # is narrow: the brush here, the options and frame by frame in
        # options_bar, which shows while painting
        self.options_bar = QToolBar("Paint options", parent)
        self.options_bar.setObjectName("paintoptions")
        for w in (self.paint, self.colour, self.neutral, self.erase):
            self.addWidget(w)
        self.addWidget(QLabel(" Size "))
        self.addWidget(self.size)
        self.addSeparator()
        self.addWidget(self.frame_mode)
        self.addSeparator()
        for name, text, tip in (("apply", "Apply hints", "Run the hint pass now (Ctrl+Return)"),
                                ("clear", "Clear frame", "Remove every stroke on this frame")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, n=name: self.action.emit(n))
            self.addWidget(b)
        ob = self.options_bar
        ob.addWidget(self.show)
        ob.addWidget(self.auto)
        save = QToolButton()
        save.setText("Save colour...")
        save.setToolTip("Name the last stroke's colour to reuse it and find it elsewhere")
        save.clicked.connect(lambda: self.action.emit("save_colour"))
        ob.addWidget(save)
        self.frame_bar = QToolBar("Frame by frame", parent)
        self.frame_bar.setObjectName("framebyframe")
        self.frame_label = QLabel(" Frame by frame: ")
        for w in (self.frame_label, self.reach, self.carry_prev, self.carry_next, self.onion):
            self.frame_bar.addWidget(w)
        self.paint.toggled.connect(self.options_bar.setVisible)
        self.options_bar.setVisible(False)
        self.frame_bar.setVisible(False)
        for w in (self.paint, self.neutral, self.erase, self.show, self.frame_mode, self.onion):
            w.toggled.connect(lambda _: self.changed.emit())
        self.size.valueChanged.connect(lambda _: self.changed.emit())
        self.reach.valueChanged.connect(lambda _: self.changed.emit())
        self.frame_mode.toggled.connect(self._frame_widgets)
        self._frame_widgets(False)
        self._update_swatch()

    def _frame_widgets(self, on: bool) -> None:
        self.frame_bar.setVisible(on)

    def stroke_reach(self) -> int | None:
        """reach for new strokes: None (the whole shot) unless frame by frame is on."""
        return self.reach.value() if self.frame_mode.isChecked() else None

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
