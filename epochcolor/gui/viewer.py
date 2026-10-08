"""The picture: colour, before/after split, or the original black and white.

Painting goes through QPainter, on a QOpenGLWidget when the platform gives
us a GL context and a plain widget when it does not (remote sessions, some
VMs). The drawing code is the same either way.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from . import theme


def _base():
    if os.environ.get("EPOCHCOLOR_NO_GL") or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return QWidget
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget

        return QOpenGLWidget
    except ImportError:
        return QWidget


class Viewer(_base()):
    MODES = ("color", "split", "original", "frame")
    modeChanged = Signal(str)
    strokeFinished = Signal(dict)  # a painted stroke, coordinates as fractions of the frame
    eraseAt = Signal(float, float)
    picked = Signal(float, float, object)  # x, y, (r, g, b) 0..1 from the ungraded colour, or None
    suggestionClicked = Signal(str, bool)  # a match marker: its id, True to use it, False to skip
    brushResized = Signal(float)  # Shift+wheel changed the brush radius

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gray: QImage | None = None
        self.color: QImage | None = None
        self.full = False
        self.mode = "color"
        self.split = 0.5
        self.message = "Add a clip or a photo to start"
        self.note = ""
        self._drag = False
        # painting
        self.paint_mode = False
        self.erase_mode = False
        self.brush = {"rgb": [40, 70, 160], "neutral": False, "radius": 0.02}
        self.strokes: list[dict] = []
        self.show_strokes = True
        self._live: list | None = None
        self._mouse: QPointF | None = None
        self.pick_mode: str | None = None  # "colour", "neutral" or "hue": next click samples
        self.raw: QImage | None = None  # ungraded colour, for sampling
        # overlays
        self.mask_outline: dict | None = None  # the grade's shape mask
        self.suggestions: list[tuple] = []  # (x, y, rgb, label, id)
        self.ghosts: list[dict] = []  # onion skin: a nearby frame's strokes, drawn faint
        self.frame_note = ""  # "frame by frame, this frame only" and the like
        # zoom and pan: zoom 1 fits the picture; pan moves its centre, in widget pixels
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._panning: QPointF | None = None
        # "This frame": the selected model run on the current frame alone (Colorize frame)
        self.frame_image: QImage | None = None
        self.frame_label = ""
        self.shot_note = ""  # the shot's state, shown in the label line
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)

    def set_mode(self, mode: str) -> None:
        if mode in self.MODES and mode != self.mode:
            self.mode = mode
            self.modeChanged.emit(mode)
            self.update()

    def set_images(self, gray: QImage | None, color: QImage | None, full: bool = False,
                   note: str = "", raw: QImage | None = None) -> None:
        self.gray, self.color, self.full, self.note = gray, color, full, note
        self.raw = raw if raw is not None else color
        self.update()

    def set_overlays(self, strokes=None, mask=None, suggestions=None, ghosts=None) -> None:
        self.ghosts = list(ghosts or [])
        if strokes is not None:
            self.strokes = strokes
        self.mask_outline = mask
        if suggestions is not None:
            self.suggestions = suggestions
        self.update()

    def clear(self, message: str = "") -> None:
        self.gray = self.color = None
        self.message = message
        self.update()

    def _target(self, img: QImage) -> QRectF:
        w, h = self.width(), self.height()
        iw, ih = img.width(), img.height()
        k = min(w / iw, h / ih) * self.zoom
        tw, th = iw * k, ih * k
        cx, cy = w / 2 + self.pan.x(), h / 2 + self.pan.y()
        return QRectF(cx - tw / 2, cy - th / 2, tw, th)

    def zoom_at(self, factor: float, at: QPointF | None = None) -> None:
        """Zoom by factor, keeping the picture point under `at` (default the centre) still."""
        base = self.gray or self.color
        if base is None:  # no picture yet: take the zoom, there's nothing to keep still
            self.zoom = float(min(16.0, max(1.0, self.zoom * factor)))
            self.pan = QPointF(0, 0)
            self.update()
            return
        at = at or QPointF(self.width() / 2, self.height() / 2)
        before = self._target(base)
        nx = (at.x() - before.left()) / before.width()
        ny = (at.y() - before.top()) / before.height()
        self.zoom = float(min(16.0, max(1.0, self.zoom * factor)))
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        else:
            after = self._target(base)
            self.pan += QPointF(at.x() - (after.left() + nx * after.width()),
                                at.y() - (after.top() + ny * after.height()))
            self._clamp_pan()
        self.update()

    def fit(self) -> None:
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.update()

    def _clamp_pan(self) -> None:
        base = self.gray or self.color
        if base is None:
            return
        r = self._target(base)
        w, h = self.width(), self.height()
        # keep the picture covering at least a quarter of the view each way
        dx = min(0.0, r.right() - w * 0.25) or max(0.0, r.left() - w * 0.75)
        dy = min(0.0, r.bottom() - h * 0.25) or max(0.0, r.top() - h * 0.75)
        self.pan -= QPointF(dx, dy)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#141414"))
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        base = self.gray or self.color
        if base is None:
            p.setPen(QColor(theme.DIM))
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        rect = self._target(base)
        show_color = self.color is not None
        if self.mode == "frame":
            if self.frame_image is not None:
                p.drawImage(rect, self.frame_image)
            else:
                p.drawImage(rect, self.gray or self.color)
        elif self.mode == "original" or not show_color:
            p.drawImage(rect, self.gray or self.color)
        elif self.mode == "color":
            p.drawImage(rect, self.color)
        else:
            x = rect.left() + rect.width() * self.split
            p.drawImage(rect, self.gray)
            src_w = self.color.width() * self.split
            p.drawImage(QRectF(x, rect.top(), rect.right() - x, rect.height()), self.color,
                        QRectF(src_w, 0, self.color.width() - src_w, self.color.height()))
            p.setPen(QPen(QColor(theme.ACCENT), 1.5))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            vis = rect.intersected(QRectF(self.rect()))
            self._label(p, QPointF(vis.left() + 8, vis.top() + 8), "before")
            self._label(p, QPointF(vis.right() - 48, vis.top() + 8), "after")
        self._paint_overlays(p, rect)
        tags = []
        if self.paint_mode:
            tags.append("painting: drag to paint, right-click a stroke to remove it, Ctrl+click picks a colour")
            if self.frame_note:
                tags.append(self.frame_note)
        if self.pick_mode:
            tags.append({"colour": "click to pick a colour", "neutral": "click something that should be grey",
                         "hue": "click the colour to isolate"}[self.pick_mode])
        if self.mode == "frame":
            tags.append(self.frame_label if self.frame_image is not None
                        else "this frame: not colorized yet, press Colorize frame (Ctrl+F)")
        elif not show_color:
            tags.append("not colorized")
        elif self.mode == "original":
            tags.append("original")
        if self.shot_note and self.mode != "frame":
            tags.append(self.shot_note)
        if self.zoom > 1.0:
            tags.append(f"zoom {self.zoom * 100:.0f}% (Ctrl+0 fits)")
        tags.append("full size" if self.full else "proxy")
        if self.note:
            tags.append(self.note)
        vis = rect.intersected(QRectF(self.rect()))  # zoomed in, the picture runs past the edges
        self._label(p, QPointF(vis.left() + 8, vis.bottom() - 26), "  ·  ".join(tags))

    def _paint_overlays(self, p: QPainter, rect: QRectF) -> None:
        short = min(rect.width(), rect.height())

        def pt(x, y):
            return QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height())

        p.save()
        p.setClipRect(rect)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self.show_strokes and self.ghosts:
            for st in self.ghosts:
                pts = st.get("points") or []
                if not pts:
                    continue
                width = max(2.0, 2 * float(st.get("radius", 0.02)) * short)
                r, g, b = st.get("rgb", [150, 150, 150]) if not st.get("neutral") else (150, 150, 150)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(int(r), int(g), int(b), 70), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                path = QPainterPath(pt(*pts[0]))
                for x, y in pts[1:]:
                    path.lineTo(pt(x, y))
                if len(pts) == 1:
                    path.addEllipse(pt(*pts[0]), width / 2, width / 2)
                p.drawPath(path)
        strokes = list(self.strokes) if self.show_strokes else []
        if self._live:
            strokes.append({**self.brush, "points": self._live})
        for st in strokes:
            pts = st.get("points") or []
            if not pts:
                continue
            width = max(2.0, 2 * float(st.get("radius", 0.02)) * short)
            if st.get("neutral"):
                col = QColor(150, 150, 150, 170)
            else:
                r, g, b = st.get("rgb", [128, 128, 128])
                col = QColor(int(r), int(g), int(b), 175)
            path = QPainterPath(pt(*pts[0]))
            for x, y in pts[1:]:
                path.lineTo(pt(x, y))
            if len(pts) == 1:
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                p.drawEllipse(pt(*pts[0]), width / 2, width / 2)
            else:
                pen = QPen(col, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
            if st.get("neutral"):
                p.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                for x, y in pts[:: max(1, len(pts) // 6)]:
                    p.drawEllipse(pt(x, y), width / 2, width / 2)
        if self.mask_outline and self.mask_outline.get("shape", "none") != "none":
            m = self.mask_outline
            p.setPen(QPen(QColor(theme.ACCENT), 1.2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            c = pt(m["cx"] + m.get("_dx", 0.0), m["cy"] + m.get("_dy", 0.0))
            p.translate(c)
            p.rotate(m.get("angle", 0.0))
            rw, rh = m["w"] * rect.width() / 2, m["h"] * rect.height() / 2
            if m["shape"] == "ellipse":
                p.drawEllipse(QPointF(0, 0), rw, rh)
            else:
                p.drawRect(QRectF(-rw, -rh, 2 * rw, 2 * rh))
            p.resetTransform()
        for x, y, rgb, label, _ in self.suggestions:
            c = pt(x, y)
            p.setPen(QPen(QColor(255, 255, 255, 230), 2.5))
            p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), 210))
            p.drawEllipse(c, 10, 10)
            p.setPen(QPen(QColor(0, 0, 0, 160), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 12, 12)
            text = f"{label}?  click: use  ·  right-click: skip"
            tw = p.fontMetrics().horizontalAdvance(text) + 16
            x = c.x() + 15 if c.x() + 15 + tw <= rect.right() else c.x() - 15 - tw
            self._label(p, QPointF(x, c.y() - 11), text)
        if self.paint_mode and self._mouse is not None and rect.contains(self._mouse):
            rad = max(2.0, float(self.brush["radius"]) * short)
            p.setPen(QPen(QColor(255, 255, 255, 200), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self._mouse, rad, rad)
            p.setPen(QPen(QColor(0, 0, 0, 160), 1))
            p.drawEllipse(self._mouse, rad + 1, rad + 1)
        p.restore()

    def _norm(self, pos: QPointF) -> tuple[float, float] | None:
        base = self.gray or self.color
        if base is None:
            return None
        r = self._target(base)
        x = (pos.x() - r.left()) / r.width()
        y = (pos.y() - r.top()) / r.height()
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return x, y

    def sample(self, x: float, y: float):
        img = self.raw or self.color
        if img is None:
            return None
        px = img.pixelColor(min(img.width() - 1, int(x * img.width())), min(img.height() - 1, int(y * img.height())))
        return (px.redF(), px.greenF(), px.blueF())

    def _label(self, p: QPainter, at: QPointF, text: str) -> None:
        f = QFont(self.font())
        f.setPointSizeF(max(7.5, f.pointSizeF() * 0.85))
        p.setFont(f)
        fm = p.fontMetrics()
        r = QRectF(at.x(), at.y(), fm.horizontalAdvance(text) + 12, fm.height() + 6)
        p.fillRect(r, QColor(0, 0, 0, 150))
        p.setPen(QColor("#e8e8e8"))
        p.drawText(r, Qt.AlignCenter, text)

    def _near_divider(self, x: float) -> bool:
        base = self.gray or self.color
        if self.mode != "split" or base is None or self.color is None:
            return False
        r = self._target(base)
        return abs(x - (r.left() + r.width() * self.split)) < 8

    def _marker_at(self, pos: QPointF):
        base = self.gray or self.color
        if base is None or not self.suggestions:
            return None
        r = self._target(base)
        for x, y, _, _, ident in self.suggestions:
            c = QPointF(r.left() + x * r.width(), r.top() + y * r.height())
            if (c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2 <= 13 ** 2:
                return ident
        return None

    def mousePressEvent(self, ev) -> None:
        pos = ev.position()
        if ev.button() == Qt.MiddleButton:
            self._panning = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        n = self._norm(pos)
        hit = self._marker_at(pos)
        if hit is not None and ev.button() in (Qt.LeftButton, Qt.RightButton) and not self.pick_mode:
            self.suggestionClicked.emit(hit, ev.button() == Qt.LeftButton)
            return
        if self.pick_mode and ev.button() == Qt.LeftButton:
            if n:
                self.picked.emit(n[0], n[1], self.sample(*n))
            return
        if self.paint_mode and n:
            if ev.button() == Qt.RightButton or self.erase_mode:
                self.eraseAt.emit(*n)
                return
            if ev.button() == Qt.LeftButton and ev.modifiers() & Qt.ControlModifier:
                self.picked.emit(n[0], n[1], self.sample(*n))
                return
            if ev.button() == Qt.LeftButton:
                self._live = [list(n)]
                self.update()
                return
        if ev.button() == Qt.LeftButton and self.mode == "split":
            self._drag = True
            self._move_split(pos.x())

    def mouseMoveEvent(self, ev) -> None:
        pos = ev.position()
        if self._panning is not None:
            self.pan += pos - self._panning
            self._panning = pos
            self._clamp_pan()
            self.update()
            return
        self._mouse = pos
        if self._live is not None:
            n = self._norm(pos)
            if n:
                lx, ly = self._live[-1]
                if abs(n[0] - lx) + abs(n[1] - ly) > 0.003:
                    self._live.append(list(n))
            self.update()
            return
        if self.paint_mode:
            self.update()
            self.setCursor(Qt.CrossCursor)
            return
        if self.pick_mode:
            self.setCursor(Qt.CrossCursor)
            return
        if self._drag:
            self._move_split(pos.x())
        self.setCursor(Qt.SplitHCursor if (self._drag or self._near_divider(pos.x()))
                       else Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.MiddleButton and self._panning is not None:
            self._panning = None
            self.setCursor(Qt.CrossCursor if self.paint_mode else Qt.ArrowCursor)
            return
        if self._live is not None:
            pts, self._live = self._live, None
            self.strokeFinished.emit({**self.brush, "points": pts})
            self.update()
        self._drag = False

    def leaveEvent(self, ev) -> None:
        self._mouse = None
        self.update()

    def wheelEvent(self, ev) -> None:
        up = ev.angleDelta().y() > 0
        if self.paint_mode and ev.modifiers() & Qt.ShiftModifier:  # Shift+wheel: brush size
            k = 1.15 if up else 1 / 1.15
            self.brush["radius"] = float(min(0.2, max(0.001, self.brush["radius"] * k)))
            self.brushResized.emit(self.brush["radius"])
            self.update()
            ev.accept()
            return
        if self.gray is None and self.color is None:
            super().wheelEvent(ev)
            return
        self.zoom_at(1.25 if up else 0.8, ev.position())  # the wheel zooms at the pointer
        ev.accept()

    def _move_split(self, x: float) -> None:
        base = self.gray or self.color
        if base is None:
            return
        r = self._target(base)
        self.split = min(1.0, max(0.0, (x - r.left()) / max(1.0, r.width())))
        self.update()
