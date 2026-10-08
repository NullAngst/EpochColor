"""The timeline: one video track, one row per audio track.

Painted directly rather than through QGraphicsView: a single track of
back-to-back segments needs no scene graph, and painting it in one pass
keeps it simple and fast at any zoom.

Mouse: click to move the playhead, drag on the ruler to scrub, click a
segment to select it, drag a segment's edge to trim it, drag its middle to
move it, click an audio track's name to switch it on or off. Ctrl+wheel
zooms around the cursor, the wheel scrolls.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QScrollBar, QToolTip, QVBoxLayout, QWidget

from ..project import Project
from . import theme

HEAD_W = 96
RULER_H = 22
VIDEO_H = 58
AUDIO_H = 30
EDGE_GRAB = 6


def timecode(frame: int, fps: float) -> str:
    if fps <= 0:
        return str(frame)
    fr = int(round(fps))
    s, f = divmod(frame, fr)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


class MediaCache:
    """Thumbnails and waveforms per clip, loaded once."""

    def __init__(self):
        self.thumbs: dict[str, tuple[list[QImage], int]] = {}
        self.waves: dict[str, list[np.ndarray | None]] = {}

    def add(self, clip_id: str, media) -> None:
        from .preview import to_qimage

        arr = np.load(media.thumbs)
        self.thumbs[clip_id] = ([to_qimage(t) for t in arr], media.thumb_stride)
        ws = []
        for w in media.waves:
            try:
                ws.append(np.load(w).astype(np.float32) if w and w.exists() else None)
            except (OSError, ValueError):
                ws.append(None)
        self.waves[clip_id] = ws


class TimelineCanvas(QWidget):
    seek = Signal(int)
    selected = Signal(int)
    edited = Signal()
    audioToggled = Signal(int)

    def __init__(self, project: Project, media: MediaCache, parent=None):
        super().__init__(parent)
        self.project = project
        self.media = media
        self.playhead = 0
        self.sel = -1
        self.zoom = 2.0  # px per frame
        self.offset = 0.0  # first visible frame
        self.status: dict[str, str] = {}  # clip id -> none/partial/ready
        self.shot_status: dict = {}  # clip id -> {"a-b": {"state", "current"}}, see timeline_export.shot_states
        self._drag = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMinimumHeight(RULER_H + VIDEO_H + AUDIO_H + 8)

    # ------------------------------------------------------- geometry

    def rows_height(self) -> int:
        return RULER_H + VIDEO_H + AUDIO_H * self.project.audio_tracks + 6

    def x_of(self, frame: float) -> float:
        return HEAD_W + (frame - self.offset) * self.zoom

    def frame_at(self, x: float) -> int:
        f = int(round((x - HEAD_W) / self.zoom + self.offset))
        return max(0, min(f, max(0, self.project.duration - 1)))

    def visible_frames(self) -> float:
        return max(1.0, (self.width() - HEAD_W) / self.zoom)

    def fit(self) -> None:
        dur = max(1, self.project.duration)
        self.zoom = max(0.02, (self.width() - HEAD_W - 16) / dur)
        self.offset = 0
        self.update()

    def ensure_visible(self, frame: int) -> None:
        vis = self.visible_frames()
        if frame < self.offset:
            self.offset = max(0, frame - vis * 0.1)
        elif frame > self.offset + vis * 0.95:
            self.offset = frame - vis * 0.5
        self.update()

    def segment_at(self, x: float, y: float):
        """(index, part) where part is 'in', 'out' or 'body'."""
        if not (RULER_H <= y < RULER_H + VIDEO_H):
            return None
        for i, start in enumerate(self.project.seg_starts()):
            seg = self.project.d.timeline[i]
            x0, x1 = self.x_of(start), self.x_of(start + seg.length)
            if x0 - EDGE_GRAB <= x <= x1 + EDGE_GRAB:
                if abs(x - x0) <= EDGE_GRAB and x1 - x0 > 3 * EDGE_GRAB:
                    return i, "in"
                if abs(x - x1) <= EDGE_GRAB and x1 - x0 > 3 * EDGE_GRAB:
                    return i, "out"
                if x0 <= x <= x1:
                    return i, "body"
        return None

    # --------------------------------------------------------- paint

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        fps = float(self.project.fps or 24)
        dur = self.project.duration
        w = self.width()
        self._paint_ruler(p, fps, dur, w)
        y = RULER_H
        self._paint_video(p, y)
        y += VIDEO_H
        for k in range(self.project.audio_tracks):
            self._paint_audio(p, k, y, fps)
            y += AUDIO_H
        # in/out shading over all rows
        if self.project.d.range:
            a, b = self.project.d.range
            shade = QColor(0, 0, 0, 110)
            if a > 0:
                p.fillRect(QRectF(HEAD_W, RULER_H, max(0, self.x_of(a) - HEAD_W), y - RULER_H), shade)
            xe = self.x_of(b)
            if xe < w:
                p.fillRect(QRectF(xe, RULER_H, w - xe, y - RULER_H), shade)
        # headers
        p.fillRect(QRectF(0, 0, HEAD_W, self.height()), QColor(theme.PANEL))
        p.setPen(QColor(theme.DIM))
        p.drawText(QRectF(8, RULER_H, HEAD_W - 8, VIDEO_H), Qt.AlignVCenter, "Video")
        yy = RULER_H + VIDEO_H
        for k, on in enumerate(self.project.d.audio_enabled):
            r = QRectF(6, yy + 6, 14, 14)
            p.setPen(QPen(QColor(theme.TEXT if on else theme.DIM), 1))
            p.setBrush(QColor(theme.ACCENT) if on else Qt.NoBrush)
            p.drawRect(r)
            p.setPen(QColor(theme.TEXT if on else theme.DIM))
            p.drawText(QRectF(26, yy, HEAD_W - 26, AUDIO_H), Qt.AlignVCenter, f"Audio {k + 1}")
            yy += AUDIO_H
        p.setPen(QColor(theme.LINE))
        p.drawLine(HEAD_W, 0, HEAD_W, self.height())
        # playhead
        x = self.x_of(self.playhead)
        if x >= HEAD_W:
            p.setPen(QPen(QColor(theme.PLAYHEAD), 1.5))
            p.drawLine(QPointF(x, 0), QPointF(x, y))
            p.setBrush(QColor(theme.PLAYHEAD))
            p.drawPolygon(QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 7)]))

    def _paint_ruler(self, p: QPainter, fps: float, dur: int, w: int) -> None:
        p.fillRect(QRectF(0, 0, w, RULER_H), QColor(theme.PANEL))
        # pick a tick step that keeps labels ~90px apart
        steps = [1, 2, 5, 10, 12, 24, 48, 120, 240, 480, 1440, 2880, 7200, 14400, 43200]
        steps = sorted(set(steps + [int(fps), int(fps) * 5, int(fps) * 10, int(fps) * 30, int(fps) * 60]))
        step = next((s for s in steps if s * self.zoom >= 90), steps[-1])
        f = int(self.offset // step) * step
        f_end = self.offset + self.visible_frames()
        p.setFont(QFont(self.font().family(), 8))
        while f <= f_end:
            x = self.x_of(f)
            if x >= HEAD_W:
                p.setPen(QColor(theme.LINE))
                p.drawLine(QPointF(x, RULER_H - 7), QPointF(x, RULER_H))
                p.setPen(QColor(theme.DIM))
                p.drawText(QPointF(x + 3, 13), timecode(f, fps))
            f += step
        if self.project.d.range:
            a, b = self.project.d.range
            xa, xb = max(HEAD_W, self.x_of(a)), self.x_of(b)
            if xb > HEAD_W:
                p.fillRect(QRectF(xa, RULER_H - 5, max(1.0, xb - xa), 5), QColor(theme.INOUT))
        for m in self.project.d.markers:
            x = self.x_of(m.frame)
            if x >= HEAD_W:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(theme.MARKER))
                p.drawPolygon(QPolygonF([QPointF(x - 4, RULER_H - 10), QPointF(x + 4, RULER_H - 10),
                                         QPointF(x, RULER_H - 3)]))

    def _paint_video(self, p: QPainter, y: float) -> None:
        clip_ids = list(self.project.d.clips)
        clip_rect = QRectF(HEAD_W, y, self.width() - HEAD_W, VIDEO_H)
        p.save()
        p.setClipRect(clip_rect)
        for i, (start, seg) in enumerate(zip(self.project.seg_starts(), self.project.d.timeline)):
            x0, x1 = self.x_of(start), self.x_of(start + seg.length)
            if x1 < HEAD_W or x0 > self.width():
                continue
            r = QRectF(x0, y + 2, max(1.0, x1 - x0), VIDEO_H - 4)
            ci = clip_ids.index(seg.clip) if seg.clip in clip_ids else 0
            p.fillRect(r, QColor(theme.CLIP_COLORS[ci % len(theme.CLIP_COLORS)]))
            self._paint_thumbs(p, r, seg, start)
            # label strip
            name = self.project.d.clips[seg.clip].name
            state = self.status.get(seg.clip, "none")
            dot = {"ready": "#6fbf73", "update": "#6fa8d9", "partial": theme.ACCENT, "none": "#777777"}.get(state, "#777777")
            p.fillRect(QRectF(r.left(), r.top(), r.width(), 14), QColor(0, 0, 0, 140))
            p.setBrush(QColor(dot))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(r.left() + 7, r.top() + 7), 3, 3)
            p.setPen(QColor("#e6e6e6"))
            p.setFont(QFont(self.font().family(), 8))
            p.drawText(QRectF(r.left() + 13, r.top(), r.width() - 15, 14), Qt.AlignVCenter, name)
            # each shot's state: a strip along the bottom. Grey: not coloured;
            # amber: painted only (Color shot); green: coloured. Striped: out
            # of date (the paint or settings changed since), the last result
            # still shows until it's redone
            clip = self.project.d.clips[seg.clip]
            states = self.shot_status.get(seg.clip, {})
            for a, b in clip.shots:
                lo, hi = max(a, seg.src_in), min(b, seg.src_out)
                if hi <= lo:
                    continue
                st = states.get(f"{a}-{b}", {"state": "none", "current": True})
                col = {"painted": QColor("#d9a441"), "colored": QColor("#6fbf73")}.get(st["state"], QColor("#5a5a5a"))
                sr = QRectF(self.x_of(start + lo - seg.src_in), r.bottom() - 4,
                            max(1.0, self.x_of(start + hi - seg.src_in) - self.x_of(start + lo - seg.src_in)), 4)
                if st.get("current", True):
                    p.fillRect(sr, col)
                else:
                    p.fillRect(sr, QColor(col.red(), col.green(), col.blue(), 90))
                    p.fillRect(sr, QBrush(col, Qt.BDiagPattern))
            # shot cuts inside the segment
            p.setPen(QPen(QColor(theme.CUT), 1))
            for a, _ in clip.shots:
                if seg.src_in < a < seg.src_out:
                    xc = self.x_of(start + a - seg.src_in)
                    p.drawLine(QPointF(xc, r.top() + 14), QPointF(xc, r.bottom()))
            # painted hint frames: dots; grade keyframes: diamonds
            p.setPen(Qt.NoPen)
            for f in self.project.hint_frames(seg.clip):
                if seg.src_in <= f < seg.src_out:
                    xc = self.x_of(start + f - seg.src_in + 0.5)
                    p.setBrush(QColor("#e8e8e8"))
                    p.drawEllipse(QPointF(xc, r.bottom() - 9), 3.5, 3.5)
            for track in self.project.d.grades.get(seg.clip, {}).values():
                keys = track.get("keys", [])
                if len(keys) < 2:
                    continue
                for k in keys:
                    f = k["frame"]
                    if seg.src_in <= f < seg.src_out:
                        xc = self.x_of(start + f - seg.src_in + 0.5)
                        y0 = r.bottom() - 16
                        p.setBrush(QColor(theme.ACCENT))
                        p.drawPolygon(QPolygonF([QPointF(xc, y0 - 4), QPointF(xc + 4, y0), QPointF(xc, y0 + 4),
                                                 QPointF(xc - 4, y0)]))
            border = QPen(QColor(theme.ACCENT), 2) if i == self.sel else QPen(QColor("#111111"), 1)
            p.setPen(border)
            p.setBrush(Qt.NoBrush)
            p.drawRect(r.adjusted(0.5, 0.5, -0.5, -0.5))
        p.restore()

    def _paint_thumbs(self, p: QPainter, r: QRectF, seg, start: int) -> None:
        th = self.media.thumbs.get(seg.clip)
        if not th:
            return
        imgs, stride = th
        if not imgs:
            return
        tw0, th0 = imgs[0].width(), imgs[0].height()
        h = r.height() - 14
        tw = tw0 * h / th0
        x = r.left()
        if x < HEAD_W:  # skip whole tiles hidden under the header
            x += ((HEAD_W - x) // tw) * tw
        while x < min(r.right(), self.width()):
            f = int((x + tw / 2 - HEAD_W) / self.zoom + self.offset)
            src = seg.src_in + max(0, min(seg.length - 1, f - start))
            img = imgs[min(len(imgs) - 1, src // stride)]
            target = QRectF(x, r.top() + 14, min(tw, r.right() - x), h)
            p.drawImage(target, img, QRectF(0, 0, target.width() / tw * tw0, th0))
            x += tw

    def _paint_audio(self, p: QPainter, k: int, y: float, fps: float) -> None:
        on = self.project.d.audio_enabled[k]
        p.save()
        p.setClipRect(QRectF(HEAD_W, y, self.width() - HEAD_W, AUDIO_H))
        p.fillRect(QRectF(HEAD_W, y, self.width(), AUDIO_H), QColor("#1e1e1e"))
        mid = y + AUDIO_H / 2
        color = QColor("#7d9cb5" if on else "#4a4a4a")
        for start, seg in zip(self.project.seg_starts(), self.project.d.timeline):
            x0, x1 = self.x_of(start), self.x_of(start + seg.length)
            if x1 < HEAD_W or x0 > self.width():
                continue
            p.fillRect(QRectF(x0, y + 2, x1 - x0, AUDIO_H - 4), QColor("#262626"))
            waves = self.media.waves.get(seg.clip) or []
            wave = waves[k] if k < len(waves) else None
            if wave is None or wave.size == 0:
                continue
            path = QPainterPath()
            xs = np.arange(max(x0, HEAD_W), min(x1, self.width()), 1.5)
            if xs.size == 0:
                continue
            frames = seg.src_in + (xs - x0) / self.zoom
            # peak over the frames under each step, from the 100 Hz envelope
            t0 = np.clip((frames / fps * 100).astype(int), 0, wave.size - 1)
            t1 = np.clip(((frames + 1.5 / self.zoom) / fps * 100).astype(int) + 1, 1, wave.size)
            peaks = np.array([wave[a:max(a + 1, b)].max() for a, b in zip(t0, t1)])
            amp = np.sqrt(peaks) * (AUDIO_H / 2 - 3)
            path.moveTo(xs[0], mid)
            for x, a in zip(xs, amp):
                path.lineTo(x, mid - a)
            for x, a in zip(xs[::-1], amp[::-1]):
                path.lineTo(x, mid + a)
            path.closeSubpath()
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawPath(path)
        p.setPen(QColor(theme.LINE))
        p.drawLine(QPointF(HEAD_W, y + AUDIO_H - 0.5), QPointF(self.width(), y + AUDIO_H - 0.5))
        p.restore()

    # --------------------------------------------------------- mouse

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.LeftButton:
            return
        x, y = ev.position().x(), ev.position().y()
        if x < HEAD_W:
            k = int((y - RULER_H - VIDEO_H) // AUDIO_H)
            if y >= RULER_H + VIDEO_H and 0 <= k < self.project.audio_tracks:
                self.audioToggled.emit(k)
            return
        hit = self.segment_at(x, y)
        if hit and hit[1] in ("in", "out"):
            i, part = hit
            seg = self.project.d.timeline[i]
            self._before = self.project._dump()
            self._drag = ("trim", i, part, x, seg.src_in, seg.src_out)
            self.sel = i
            self.selected.emit(i)
            return
        if hit:
            self.sel = hit[0]
            self.selected.emit(hit[0])
            self._drag = ("move", hit[0], x)
        else:
            self._drag = ("scrub",)
        self.seek.emit(self.frame_at(x))
        self.update()

    def mouseMoveEvent(self, ev) -> None:
        x, y = ev.position().x(), ev.position().y()
        d = self._drag
        if d is None:
            hit = self.segment_at(x, y)
            self.setCursor(Qt.SizeHorCursor if hit and hit[1] != "body" else Qt.ArrowCursor)
            for m in self.project.d.markers:
                if abs(self.x_of(m.frame) - x) < 5 and y < RULER_H and m.note:
                    QToolTip.showText(ev.globalPosition().toPoint(), m.note, self)
            return
        if d[0] == "scrub":
            self.seek.emit(self.frame_at(x))
        elif d[0] == "trim":
            _, i, part, x0, s_in, s_out = d
            delta = int(round((x - x0) / self.zoom))
            if part == "in":
                self.project.trim(i, src_in=s_in + delta, checkpoint=False)
            else:
                self.project.trim(i, src_out=s_out + delta, checkpoint=False)
            self.edited.emit()
        elif d[0] == "move":
            self.seek.emit(self.frame_at(x))
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        d = self._drag
        self._drag = None
        if d and d[0] == "move":
            # dropped over another segment: move there
            target = self.segment_at(ev.position().x(), RULER_H + 5)
            if target and target[0] != d[1]:
                i, j = d[1], target[0]
                self.project.checkpoint("move")
                tl = self.project.d.timeline
                tl.insert(j, tl.pop(i))
                self.sel = j
                self.selected.emit(j)
                self.edited.emit()
        elif d and d[0] == "trim":
            self.project.commit("trim", self._before)
            self.edited.emit()
        self.update()

    def wheelEvent(self, ev) -> None:
        dy = ev.angleDelta().y() or ev.angleDelta().x()
        if ev.modifiers() & Qt.ControlModifier:
            x = ev.position().x()
            anchor = (x - HEAD_W) / self.zoom + self.offset
            self.zoom = float(np.clip(self.zoom * (1.25 if dy > 0 else 0.8), 0.01, 40.0))
            self.offset = max(0.0, anchor - (x - HEAD_W) / self.zoom)
        else:
            self.offset = max(0.0, self.offset - dy / self.zoom * 0.5)
        if hasattr(self.parent(), "sync_scroll"):
            self.parent().sync_scroll()
        self.update()


class Timeline(QWidget):
    """Canvas plus a horizontal scroll bar."""

    def __init__(self, project: Project, media: MediaCache, parent=None):
        super().__init__(parent)
        self.canvas = TimelineCanvas(project, media, self)
        self.bar = QScrollBar(Qt.Horizontal, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.bar)
        self.bar.valueChanged.connect(self._scrolled)

    def sync_scroll(self) -> None:
        c = self.canvas
        dur = c.project.duration
        vis = int(c.visible_frames())
        self.bar.blockSignals(True)
        self.bar.setRange(0, max(0, dur - vis // 2))
        self.bar.setPageStep(max(1, vis))
        self.bar.setValue(int(c.offset))
        self.bar.blockSignals(False)

    def _scrolled(self, v: int) -> None:
        self.canvas.offset = float(v)
        self.canvas.update()

    def set_project(self, project: Project) -> None:
        self.canvas.project = project
        self.canvas.sel = -1
        self.refresh()

    def refresh(self) -> None:
        self.canvas.setMinimumHeight(self.canvas.rows_height())
        self.sync_scroll()
        self.canvas.update()
