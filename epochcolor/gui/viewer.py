"""The picture: colour, before/after split, or the original black and white.

Painting goes through QPainter, on a QOpenGLWidget when the platform gives
us a GL context and a plain widget when it does not (remote sessions, some
VMs). The drawing code is the same either way.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
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
    MODES = ("color", "split", "original")
    modeChanged = Signal(str)

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
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)

    def set_mode(self, mode: str) -> None:
        if mode in self.MODES and mode != self.mode:
            self.mode = mode
            self.modeChanged.emit(mode)
            self.update()

    def set_images(self, gray: QImage | None, color: QImage | None, full: bool = False,
                   note: str = "") -> None:
        self.gray, self.color, self.full, self.note = gray, color, full, note
        self.update()

    def clear(self, message: str = "") -> None:
        self.gray = self.color = None
        self.message = message
        self.update()

    def _target(self, img: QImage) -> QRectF:
        w, h = self.width(), self.height()
        iw, ih = img.width(), img.height()
        k = min(w / iw, h / ih)
        tw, th = iw * k, ih * k
        return QRectF((w - tw) / 2, (h - th) / 2, tw, th)

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
        if self.mode == "original" or not show_color:
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
            self._label(p, QPointF(rect.left() + 8, rect.top() + 8), "before")
            self._label(p, QPointF(rect.right() - 48, rect.top() + 8), "after")
        tags = []
        if not show_color:
            tags.append("not colorized")
        elif self.mode == "original":
            tags.append("original")
        tags.append("full size" if self.full else "proxy")
        if self.note:
            tags.append(self.note)
        self._label(p, QPointF(rect.left() + 8, rect.bottom() - 26), "  ·  ".join(tags))

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

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self.mode == "split":
            self._drag = True
            self._move_split(ev.position().x())

    def mouseMoveEvent(self, ev) -> None:
        if self._drag:
            self._move_split(ev.position().x())
        self.setCursor(Qt.SplitHCursor if (self._drag or self._near_divider(ev.position().x()))
                       else Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = False

    def _move_split(self, x: float) -> None:
        base = self.gray or self.color
        if base is None:
            return
        r = self._target(base)
        self.split = min(1.0, max(0.0, (x - r.left()) / max(1.0, r.width())))
        self.update()
