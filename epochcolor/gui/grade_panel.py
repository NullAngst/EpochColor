"""The grade panel: wheels, sliders, curves, qualifier, shape mask, LUT,
keyframes. It edits one grade dict and says when it changed; the main window
decides where that grade lives (a shot's track at the playhead, or a photo).
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QConicalGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget,
)

from .. import grade as G
from . import theme

PHI = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]  # red, green, blue around the wheel


def rgb_from_puck(u: float, v: float, scale: float) -> list[float]:
    out = [scale * (u * math.cos(p) + v * math.sin(p)) for p in PHI]
    m = sum(out) / 3
    return [o - m for o in out]


def puck_from_rgb(rgb, scale: float) -> tuple[float, float]:
    if scale == 0:
        return 0.0, 0.0
    u = sum(c * math.cos(p) for c, p in zip(rgb, PHI)) * 2 / 3 / scale
    v = sum(c * math.sin(p) for c, p in zip(rgb, PHI)) * 2 / 3 / scale
    return u, v


class Wheel(QWidget):
    """A colour wheel with a puck for the colour push and a slider under it
    for the master level. Double-click resets."""
    changed = Signal()

    def __init__(self, title: str, scale: float, master_range: float, parent=None):
        super().__init__(parent)
        self.title, self.scale = title, scale
        self.u = self.v = 0.0
        self._drag = False
        self.setMinimumSize(118, 150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 118, 0, 0)
        lay.setSpacing(1)
        self.master = QSlider(Qt.Horizontal)
        self.master.setRange(-1000, 1000)
        self.master_range = master_range
        self.master.valueChanged.connect(lambda _: self.changed.emit())
        self.master.setToolTip(f"{title} master level. Double-click the wheel to reset both.")
        lay.addWidget(self.master)

    def value(self) -> tuple[list[float], float]:
        return rgb_from_puck(self.u, self.v, self.scale), self.master.value() / 1000 * self.master_range

    def set_value(self, rgb, master: float) -> None:
        self.u, self.v = puck_from_rgb(rgb, self.scale)
        self.master.blockSignals(True)
        self.master.setValue(int(round(master / self.master_range * 1000)))
        self.master.blockSignals(False)
        self.update()

    def _geom(self):
        size = min(self.width(), 112)
        r = size / 2 - 8
        c = QPointF(self.width() / 2, 6 + size / 2)
        return c, r

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c, r = self._geom()
        g = QConicalGradient(c, 0)
        for i in range(7):
            # Qt goes counter-clockwise from 3 o'clock; match PHI (red at 0)
            col = QColor.fromHsvF((i / 6) % 1.0, 0.55, 0.55)
            g.setColorAt(i / 6, col)
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor(theme.LINE), 1))
        p.drawEllipse(c, r, r)
        rg = QRadialGradient(c, r)
        rg.setColorAt(0, QColor(70, 70, 70, 255))
        rg.setColorAt(1, QColor(70, 70, 70, 0))
        p.setBrush(QBrush(rg))
        p.setPen(Qt.NoPen)
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(110, 110, 110), 1))
        p.drawLine(QPointF(c.x() - 5, c.y()), QPointF(c.x() + 5, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - 5), QPointF(c.x(), c.y() + 5))
        pk = QPointF(c.x() + self.u * r, c.y() - self.v * r)
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.setBrush(QColor(theme.ACCENT))
        p.drawEllipse(pk, 5, 5)
        p.setPen(QColor(theme.TEXT))
        rgb, m = self.value()
        p.drawText(QRectF(0, 0, self.width(), 14), Qt.AlignLeft, self.title)

    def _set_from(self, pos) -> None:
        c, r = self._geom()
        u = (pos.x() - c.x()) / r
        v = -(pos.y() - c.y()) / r
        d = math.hypot(u, v)
        if d > 1:
            u, v = u / d, v / d
        self.u, self.v = u, v
        self.update()
        self.changed.emit()

    def mousePressEvent(self, ev) -> None:
        c, r = self._geom()
        if math.hypot(ev.position().x() - c.x(), ev.position().y() - c.y()) <= r + 6:
            self._drag = True
            self._set_from(ev.position())

    def mouseMoveEvent(self, ev) -> None:
        if self._drag:
            self._set_from(ev.position())

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = False

    def mouseDoubleClickEvent(self, ev) -> None:
        self.u = self.v = 0.0
        self.master.setValue(0)
        self.update()
        self.changed.emit()


class Slider(QWidget):
    """Label, slider and value, for one float. Double-click the label resets."""
    changed = Signal()

    def __init__(self, label: str, lo: float, hi: float, default: float, fmt: str = "{:.2f}", parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.default, self.fmt = lo, hi, default, fmt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.label.setMinimumWidth(78)
        self.label.setToolTip("Double-click to reset")
        self.label.mouseDoubleClickEvent = lambda e: self.set_value(self.default, emit=True)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.val = QLabel()
        self.val.setMinimumWidth(44)
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.val)
        self.slider.valueChanged.connect(self._moved)
        self.set_value(default)

    def value(self) -> float:
        return self.lo + (self.hi - self.lo) * self.slider.value() / 1000

    def set_value(self, v: float, emit: bool = False) -> None:
        self.slider.blockSignals(not emit)
        self.slider.setValue(int(round((float(v) - self.lo) / (self.hi - self.lo) * 1000)))
        self.slider.blockSignals(False)
        self.val.setText(self.fmt.format(self.value()))

    def _moved(self, _):
        self.val.setText(self.fmt.format(self.value()))
        self.changed.emit()


class CurveEditor(QWidget):
    """Curves: master, R, G, B, hue vs saturation, hue vs hue. Drag points,
    double-click to add one, right-click a point to remove it."""
    changed = Signal()
    MODES = [("master", "Master"), ("r", "Red"), ("g", "Green"), ("b", "Blue"),
             ("hue_sat", "Hue vs sat"), ("hue_hue", "Hue vs hue")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grade = G.default_grade()
        self.mode = "master"
        self._drag = None
        self.setMinimumHeight(170)

    def _points(self) -> list:
        if self.mode in ("hue_sat", "hue_hue"):
            return self.grade[self.mode]
        return self.grade["curves"][self.mode]

    def _set_points(self, pts) -> None:
        pts = sorted([[float(x), float(y)] for x, y in pts])
        if self.mode in ("hue_sat", "hue_hue"):
            self.grade[self.mode] = pts
        else:
            self.grade["curves"][self.mode] = pts

    def _yrange(self):
        return {"hue_sat": (0.0, 2.0), "hue_hue": (-0.5, 0.5)}.get(self.mode, (0.0, 1.0))

    def _rect(self) -> QRectF:
        return QRectF(6, 6, self.width() - 12, self.height() - 12)

    def _to_px(self, x, y) -> QPointF:
        r = self._rect()
        lo, hi = self._yrange()
        return QPointF(r.left() + x * r.width(), r.bottom() - (y - lo) / (hi - lo) * r.height())

    def _from_px(self, pos) -> tuple[float, float]:
        r = self._rect()
        lo, hi = self._yrange()
        x = min(1.0, max(0.0, (pos.x() - r.left()) / r.width()))
        y = min(hi, max(lo, lo + (r.bottom() - pos.y()) / r.height() * (hi - lo)))
        return x, y

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._rect()
        p.fillRect(r, QColor("#1b1b1b"))
        hue = self.mode in ("hue_sat", "hue_hue")
        if hue:
            for i in range(60):
                p.fillRect(QRectF(r.left() + r.width() * i / 60, r.bottom() - 6, r.width() / 60 + 1, 6),
                           QColor.fromHsvF(i / 60, 0.7, 0.7))
        p.setPen(QPen(QColor(theme.LINE), 1))
        for i in range(1, 4):
            p.drawLine(QPointF(r.left() + r.width() * i / 4, r.top()), QPointF(r.left() + r.width() * i / 4, r.bottom()))
            p.drawLine(QPointF(r.left(), r.top() + r.height() * i / 4), QPointF(r.right(), r.top() + r.height() * i / 4))
        col = {"r": "#e06060", "g": "#60c060", "b": "#6090e0"}.get(self.mode, "#dddddd")
        pts = self._points()
        if hue:
            lut = G._hue_lut(json.dumps(pts), 1.0 if self.mode == "hue_sat" else 0.0)
            xs = np.linspace(0, 1, len(lut), endpoint=False)
        else:
            lut = G._curve_lut(json.dumps(pts))
            xs = np.linspace(0, 1, len(lut))
        path = QPainterPath(self._to_px(xs[0], lut[0]))
        for x, y in zip(xs[::4], lut[::4]):
            path.lineTo(self._to_px(x, y))
        p.setPen(QPen(QColor(col), 1.6))
        p.drawPath(path)
        p.setBrush(QColor(col))
        for x, y in pts:
            p.drawEllipse(self._to_px(x, y), 4, 4)
        p.setPen(QColor(theme.DIM))
        hint = "double-click adds a point, right-click removes" if pts or not hue else \
            "double-click to add points along the hue range"
        p.drawText(r.adjusted(4, 2, -4, -2), Qt.AlignTop | Qt.AlignLeft, hint)

    def _hit(self, pos) -> int | None:
        for i, (x, y) in enumerate(self._points()):
            q = self._to_px(x, y)
            if abs(q.x() - pos.x()) < 7 and abs(q.y() - pos.y()) < 7:
                return i
        return None

    def mousePressEvent(self, ev) -> None:
        i = self._hit(ev.position())
        if ev.button() == Qt.RightButton and i is not None:
            pts = list(self._points())
            linear = self.mode not in ("hue_sat", "hue_hue")
            if not (linear and len(pts) <= 2):
                pts.pop(i)
                self._set_points(pts)
                self.update()
                self.changed.emit()
            return
        if ev.button() == Qt.LeftButton and i is not None:
            self._drag = i

    def mouseMoveEvent(self, ev) -> None:
        if self._drag is None:
            return
        pts = [list(p) for p in self._points()]
        x, y = self._from_px(ev.position())
        pts[self._drag] = [x, y]
        self._set_points(pts)
        # keep dragging the same point after the sort
        self._drag = next(i for i, p in enumerate(self._points()) if p == [x, y])
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = None

    def mouseDoubleClickEvent(self, ev) -> None:
        if self._hit(ev.position()) is not None:
            return
        x, y = self._from_px(ev.position())
        pts = [list(p) for p in self._points()]
        if self.mode in ("hue_sat", "hue_hue") and not pts:
            base = 1.0 if self.mode == "hue_sat" else 0.0
            pts = [[i / 6, base] for i in range(6)]  # a flat ring of points to pull on
        else:
            pts.append([x, y])
        self._set_points(pts)
        self.update()
        self.changed.emit()


class GradePanel(QScrollArea):
    """Emits changed(grade) on every edit. Keyframe buttons emit keyAction."""
    changed = Signal(dict)
    keyAction = Signal(str)  # add, delete, prev, next, ease
    action = Signal(str)  # pick_neutral, pick_hue, track, clear_track, load_lut, export_lut, reset, copy, paste

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 6, 6, 6)
        self.grade = G.default_grade()
        self._loading = False

        self.target = QLabel("Nothing to grade")
        self.target.setWordWrap(True)
        self.target.setStyleSheet(f"color: {theme.DIM};")
        lay.addWidget(self.target)

        kf = QGroupBox("Keyframes")
        kl = QGridLayout(kf)
        self.key_info = QLabel("Static grade")
        self.key_info.setWordWrap(True)
        kl.addWidget(self.key_info, 0, 0, 1, 4)
        for i, (name, text, tip) in enumerate((("prev", "<", "Previous keyframe"),
                                               ("add", "Add key", "Key the grade at the playhead"),
                                               ("delete", "Delete key", "Remove the keyframe at the playhead"),
                                               ("next", ">", "Next keyframe"))):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, n=name: self.keyAction.emit(n))
            kl.addWidget(b, 1, i)
        self.ease = QCheckBox("Ease out of this key (instead of linear)")
        self.ease.toggled.connect(lambda _: None if self._loading else self.keyAction.emit("ease"))
        kl.addWidget(self.ease, 2, 0, 1, 4)
        lay.addWidget(kf)

        wb = QGroupBox("Primaries")
        wl = QVBoxLayout(wb)
        wheels = QGridLayout()
        self.wheels = {
            "lift": Wheel("Lift", 0.15, 0.2), "gamma": Wheel("Gamma", 0.25, 0.4),
            "gain": Wheel("Gain", 0.35, 0.6), "offset": Wheel("Offset", 0.15, 0.2),
        }
        for i, w in enumerate(self.wheels.values()):
            wheels.addWidget(w, i // 2, i % 2)
            w.changed.connect(self._edited)
        wl.addLayout(wheels)
        self.sl = {
            "temperature": Slider("Temperature", -1, 1, 0.0), "tint": Slider("Tint", -1, 1, 0.0),
            "contrast": Slider("Contrast", 0.5, 1.8, 1.0), "pivot": Slider("Pivot", 0.1, 0.9, 0.435),
            "saturation": Slider("Saturation", 0, 2.5, 1.0), "vibrance": Slider("Vibrance", -1, 1, 0.0),
        }
        for s in self.sl.values():
            wl.addWidget(s)
            s.changed.connect(self._edited)
        row = QHBoxLayout()
        pick = QPushButton("Pick neutral")
        pick.setToolTip("Click something in the picture that should be grey; temperature and tint follow")
        pick.clicked.connect(lambda: self.action.emit("pick_neutral"))
        reset = QPushButton("Reset grade")
        reset.clicked.connect(lambda: self.action.emit("reset"))
        row.addWidget(pick)
        row.addWidget(reset)
        wl.addLayout(row)
        lay.addWidget(wb)

        cb = QGroupBox("Curves")
        cl = QVBoxLayout(cb)
        self.curve_mode = QComboBox()
        for key, label in CurveEditor.MODES:
            self.curve_mode.addItem(label, key)
        self.curves = CurveEditor()
        self.curve_mode.currentIndexChanged.connect(self._curve_mode)
        self.curves.changed.connect(self._curves_edited)
        reset_curve = QPushButton("Reset this curve")
        reset_curve.clicked.connect(self._reset_curve)
        top = QHBoxLayout()
        top.addWidget(self.curve_mode, 1)
        top.addWidget(reset_curve)
        cl.addLayout(top)
        cl.addWidget(self.curves)
        lay.addWidget(cb)

        qb = QGroupBox("Secondary: HSL qualifier")
        ql = QVBoxLayout(qb)
        self.q_on = QCheckBox("Isolate a colour")
        self.q_on.toggled.connect(self._edited)
        self.show_matte = QCheckBox("Show the matte")
        self.show_matte.toggled.connect(lambda _: self.action.emit("matte"))
        qr = QHBoxLayout()
        qr.addWidget(self.q_on)
        qr.addWidget(self.show_matte)
        ql.addLayout(qr)
        pick_hue = QPushButton("Pick the colour to isolate")
        pick_hue.clicked.connect(lambda: self.action.emit("pick_hue"))
        ql.addWidget(pick_hue)
        self.q = {
            "hue": Slider("Hue", 0, 1, 0.0, "{:.2f}"), "hue_width": Slider("Hue width", 0.01, 0.5, 0.08),
            "sat_lo": Slider("Sat from", 0, 1, 0.1), "sat_hi": Slider("Sat to", 0, 1, 1.0),
            "lum_lo": Slider("Lum from", 0, 1, 0.0), "lum_hi": Slider("Lum to", 0, 1, 1.0),
            "soft": Slider("Softness", 0.005, 0.3, 0.05),
            "d_hue": Slider("Shift hue", -0.5, 0.5, 0.0), "d_sat": Slider("Saturation", 0, 2.5, 1.0),
            "d_lum": Slider("Lightness", -0.8, 0.8, 0.0),
        }
        for k, s in self.q.items():
            if k == "d_hue":
                sep = QLabel("Inside the matte:")
                sep.setStyleSheet(f"color: {theme.DIM};")
                ql.addWidget(sep)
            ql.addWidget(s)
            s.changed.connect(self._edited)
        lay.addWidget(qb)

        mb = QGroupBox("Secondary: shape mask")
        ml = QVBoxLayout(mb)
        self.shape = QComboBox()
        for v, t in (("none", "No mask"), ("ellipse", "Ellipse"), ("rectangle", "Rectangle")):
            self.shape.addItem(t, v)
        self.shape.currentIndexChanged.connect(self._edited)
        self.invert = QCheckBox("Invert")
        self.invert.toggled.connect(self._edited)
        mr = QHBoxLayout()
        mr.addWidget(self.shape, 1)
        mr.addWidget(self.invert)
        ml.addLayout(mr)
        self.m = {
            "cx": Slider("Centre x", 0, 1, 0.5), "cy": Slider("Centre y", 0, 1, 0.5),
            "w": Slider("Width", 0.02, 1.5, 0.3), "h": Slider("Height", 0.02, 1.5, 0.3),
            "angle": Slider("Angle", -90, 90, 0.0, "{:.0f}"), "feather": Slider("Feather", 0.0, 1.0, 0.1),
        }
        for s in self.m.values():
            ml.addWidget(s)
            s.changed.connect(self._edited)
        tr = QHBoxLayout()
        track = QPushButton("Track through shot")
        track.setToolTip("Follow the mask's centre along the shot's motion (needs the clip colorized)")
        track.clicked.connect(lambda: self.action.emit("track"))
        untrack = QPushButton("Clear track")
        untrack.clicked.connect(lambda: self.action.emit("clear_track"))
        tr.addWidget(track)
        tr.addWidget(untrack)
        ml.addLayout(tr)
        note = QLabel("With no qualifier, the corrections under 'Inside the matte' apply inside the shape.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.DIM}; font-size: 11px;")
        ml.addWidget(note)
        lay.addWidget(mb)

        lb = QGroupBox("LUT")
        ll = QVBoxLayout(lb)
        self.lut_label = QLabel("No LUT")
        self.lut_label.setWordWrap(True)
        ll.addWidget(self.lut_label)
        self.lut_mix = Slider("Mix", 0, 1, 1.0)
        self.lut_mix.changed.connect(self._edited)
        ll.addWidget(self.lut_mix)
        lr = QHBoxLayout()
        for name, text in (("load_lut", "Load .cube..."), ("clear_lut", "Clear"), ("export_lut", "Export grade as .cube...")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, n=name: self.action.emit(n))
            lr.addWidget(b)
        ll.addLayout(lr)
        lay.addWidget(lb)
        lay.addStretch(1)
        self.setEnabled(False)

    # ------------------------------------------------------------- load

    def load(self, g: dict | None, target: str, key_info: str = "Static grade", ease: bool = False,
             keys_enabled: bool = True) -> None:
        self._loading = True
        self.setEnabled(True)
        self.grade = G.merged(g)
        g = self.grade
        self.target.setText(target)
        self.key_info.setText(key_info)
        self.ease.setChecked(ease)
        for k in ("lift", "gamma", "gain", "offset"):
            self.wheels[k].set_value(g[k], g[k + "_m"])
        for k, s in self.sl.items():
            s.set_value(g[k])
        self.curves.grade = copy.deepcopy(g)
        self.curves.update()
        q = g["qualifier"]
        self.q_on.setChecked(bool(q["enabled"]))
        for k, s in self.q.items():
            s.set_value(q[k])
        m = g["mask"]
        self.shape.setCurrentIndex(max(0, self.shape.findData(m["shape"])))
        self.invert.setChecked(bool(m["invert"]))
        for k, s in self.m.items():
            s.set_value(m[k])
        self.lut_label.setText(g["lut"] or "No LUT")
        self.lut_mix.set_value(g["lut_mix"])
        for w in self.findChildren(QPushButton):
            if w.text() in ("<", ">", "Add key", "Delete key"):
                w.setEnabled(keys_enabled)
        self.ease.setEnabled(keys_enabled)
        self._loading = False

    def disable(self, why: str) -> None:
        self.target.setText(why)
        self.setEnabled(False)

    # ---------------------------------------------------------- editing

    def read(self) -> dict:
        g = copy.deepcopy(self.grade)
        for k, w in self.wheels.items():
            rgb, master = w.value()
            g[k] = [round(v, 5) for v in rgb]
            g[k + "_m"] = round(master, 5)
        for k, s in self.sl.items():
            g[k] = round(s.value(), 4)
        g["curves"] = copy.deepcopy(self.curves.grade["curves"])
        g["hue_sat"] = copy.deepcopy(self.curves.grade["hue_sat"])
        g["hue_hue"] = copy.deepcopy(self.curves.grade["hue_hue"])
        g["qualifier"]["enabled"] = self.q_on.isChecked()
        for k, s in self.q.items():
            g["qualifier"][k] = round(s.value(), 4)
        g["mask"]["shape"] = self.shape.currentData()
        g["mask"]["invert"] = self.invert.isChecked()
        for k, s in self.m.items():
            g["mask"][k] = round(s.value(), 4)
        g["lut_mix"] = round(self.lut_mix.value(), 3)
        return g

    def _edited(self, *_):
        if self._loading:
            return
        self.grade = self.read()
        self.changed.emit(self.grade)

    def _curves_edited(self):
        if self._loading:
            return
        self._edited()

    def _curve_mode(self, _):
        self.curves.mode = self.curve_mode.currentData()
        self.curves.update()

    def _reset_curve(self):
        mode = self.curves.mode
        if mode in ("hue_sat", "hue_hue"):
            self.curves.grade[mode] = []
        else:
            self.curves.grade["curves"][mode] = [[0, 0], [1, 1]]
        self.curves.update()
        self._edited()

    def set_fields(self, **kw) -> None:
        """Change values from outside (pickers), then emit as an edit."""
        g = self.read()
        for k, v in kw.items():
            if "." in k:
                a, b = k.split(".")
                g[a][b] = v
            else:
                g[k] = v
        self.load(g, self.target.text(), self.key_info.text(), self.ease.isChecked(), self.ease.isEnabled())
        self.grade = g
        self.changed.emit(g)
