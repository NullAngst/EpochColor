"""Frames for the viewer, made off the GUI thread.

While scrubbing or playing, frames come from the proxy: 360p grey, plus the
clip's cached chroma when it has been colorized, through the fast preview
conversion. Once the playhead rests for a moment, the same frame is
rendered at full size from the source through the real pipeline and
replaces the proxy frame, so what you judge when paused is what exports.

Requests coalesce: only the newest one is ever worked on.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QImage


@dataclass(frozen=True)
class LayoutPiece:
    start: int
    length: int
    clip_id: str
    src_in: int
    proxy: str
    source: str
    fps: Fraction
    ab: str | None  # ab.npy of a finished analysis, None if not colorized


def to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class PreviewProvider(QObject):
    # frame, grey QImage, colour QImage or None, full size, ungraded colour
    ready = Signal(int, object, object, bool, object)
    request_sig = Signal(int, bool)
    layout_sig = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.layout: list[LayoutPiece] = []
        self.settings: dict = {}
        self.readers: dict[str, object] = {}
        self.sources: dict[str, object] = {}
        self.ab: dict[str, np.ndarray] = {}
        self.cache: OrderedDict[int, tuple] = OrderedDict()
        self.want: tuple[int, bool] | None = None
        self.last_shown = -1
        self.request_sig.connect(self._request)
        self.layout_sig.connect(self._set_layout)
        self._busy = False
        self.still_timer = None
        self.book = None
        self.show_matte = False

    # runs in the provider thread from here on

    @Slot()
    def start(self) -> None:
        self.still_timer = QTimer()
        self.still_timer.setSingleShot(True)
        self.still_timer.setInterval(250)
        self.still_timer.timeout.connect(self._render_still)

    @Slot()
    def stop(self) -> None:
        """Release timers and files from inside the provider thread."""
        if self.still_timer:
            self.still_timer.stop()
            self.still_timer.deleteLater()
            self.still_timer = None
        for d in (self.readers, self.sources):
            for r in d.values():
                r.close()
            d.clear()
        self.ab.clear()

    @Slot(object, object)
    def _set_layout(self, layout, settings) -> None:
        from ..grade import GradeBook

        self.layout = list(layout)
        self.settings = dict(settings)
        self.book = GradeBook(self.settings.pop("_grades", {}), self.settings.pop("_shots", {}))
        self.show_matte = bool(self.settings.pop("_matte", False))
        self.cache.clear()
        live = {p.clip_id for p in self.layout}
        for d in (self.readers, self.sources):
            for k in [k for k in d if k not in live]:
                d.pop(k).close()
        self.ab = {k: v for k, v in self.ab.items() if k in live}
        for p in self.layout:
            if p.ab is None:
                self.ab.pop(p.clip_id, None)
            elif p.clip_id not in self.ab:
                try:
                    self.ab[p.clip_id] = np.load(p.ab, mmap_mode="r")
                except (OSError, ValueError):
                    pass

    @Slot(int, bool)
    def _request(self, frame: int, playing: bool) -> None:
        self.want = (frame, playing)
        if self.still_timer:
            self.still_timer.stop()
        if not self._busy:
            self._busy = True
            QTimer.singleShot(0, self._work)

    def _locate(self, frame: int):
        for p in self.layout:
            if p.start <= frame < p.start + p.length:
                return p, p.src_in + frame - p.start
        return None, None

    def _work(self) -> None:
        try:
            while self.want is not None:
                frame, playing = self.want
                self.want = None
                if frame in self.cache:
                    g, c, raw = self.cache[frame]
                    self.cache.move_to_end(frame)
                else:
                    g, c, raw = self._proxy_frame(frame)
                    if g is not None:
                        self.cache[frame] = (g, c, raw)
                        if len(self.cache) > 240:
                            self.cache.popitem(last=False)
                if g is not None:
                    self.ready.emit(frame, g, c, False, raw)
                    self.last_shown = frame
                if not playing and self.want is None and self.still_timer:
                    self.still_timer.start()
        finally:
            self._busy = False

    def _proxy_frame(self, frame: int):
        from ..color import gray8_to_l, lab_to_srgb8_fast
        from ..media import FrameReader

        p, src = self._locate(frame)
        if p is None:
            return None, None, None
        r = self.readers.get(p.clip_id)
        if r is None:
            r = self.readers[p.clip_id] = FrameReader(p.proxy, p.fps, "gray", cache=48)
        g = r.get(src)
        if g is None:
            return None, None, None
        gray = to_qimage(g)
        ab = self.ab.get(p.clip_id)
        if ab is None or src >= len(ab):
            return gray, None, None
        import cv2

        h, w = g.shape
        abw = cv2.resize(np.asarray(ab[src], np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        sat = float(self.settings.get("saturation", 1.0))
        if sat != 1.0:
            abw *= sat
        rgb8 = lab_to_srgb8_fast(gray8_to_l(g), abw)
        raw = to_qimage(rgb8)
        return gray, self._graded(rgb8.astype(np.float32) / 255.0, p.clip_id, src, raw), raw

    def _graded(self, rgb: np.ndarray, clip_id: str, src: int, raw):
        """The grade on top, or the qualifier matte when asked to show it."""
        g = self.book.at(clip_id, src) if self.book else None
        if g is None:
            if self.show_matte:
                return to_qimage(np.full(rgb.shape[:2], 255, np.uint8))
            return raw
        from ..grade import apply as grade_apply

        matte = [] if self.show_matte else None
        out = grade_apply(rgb, g, self.book.mask_offset(clip_id, src), matte_out=matte)
        if self.show_matte:
            m = matte[0] if matte and matte[0] is not None else np.ones(rgb.shape[:2], np.float32)
            return to_qimage((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8))
        return to_qimage((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))

    def _render_still(self) -> None:
        """Full size from the source, through the export renderer."""
        if self.want is not None:
            return
        frame = self.last_shown
        p, src = self._locate(frame)
        if p is None:
            return
        from ..media import FrameReader
        from ..video.pipeline import FrameRenderer, VideoSettings

        try:
            r = self.sources.get(p.clip_id)
            if r is None:
                r = self.sources[p.clip_id] = FrameReader(p.source, p.fps, "gray16le", cache=4)
            g16 = r.get(src)
            if g16 is None or self.want is not None:
                return
            gray = g16.astype(np.float32) / 65535.0
            gimg = to_qimage((gray * 255.0 + 0.5).astype(np.uint8))
            ab = self.ab.get(p.clip_id)
            cimg = raw = None
            if ab is not None and src < len(ab):
                s = VideoSettings.from_project(self.settings)
                rgb = FrameRenderer(s)(gray, np.asarray(ab[src], np.float32))
                raw = to_qimage((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8))
                cimg = self._graded(rgb, p.clip_id, src, raw)
            if self.want is None and self.last_shown == frame:
                self.ready.emit(frame, gimg, cimg, True, raw)
        except Exception:  # a still is a nicety; the proxy frame stays up
            pass
