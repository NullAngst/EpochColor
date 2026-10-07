"""Neutral grey theme.

Colour work needs a neutral surround: a tinted UI shifts how you judge the
picture next to it, the same reason grading suites are grey. Every surface
here is a true grey (R=G=B). Colour shows up only in small accents
(selection, playhead, in/out) where it can't bias the picture.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

BG = "#232323"
PANEL = "#2b2b2b"
RAISED = "#333333"
LINE = "#3d3d3d"
TEXT = "#d6d6d6"
DIM = "#8c8c8c"
ACCENT = "#d9a441"  # selection, focus
PLAYHEAD = "#e0453a"
INOUT = "#4a90c2"
CUT = "#e6d36a"
MARKER = "#7fb86a"

# per-clip hues on the timeline, kept dark so thumbnails stay readable
CLIP_COLORS = ["#4b5d73", "#5d4b73", "#4b7360", "#73654b", "#734b55", "#4b6b73"]


def apply(app) -> None:
    app.setStyle("Fusion")
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(BG))
    p.setColor(QPalette.WindowText, c(TEXT))
    p.setColor(QPalette.Base, c("#1d1d1d"))
    p.setColor(QPalette.AlternateBase, c(PANEL))
    p.setColor(QPalette.ToolTipBase, c(RAISED))
    p.setColor(QPalette.ToolTipText, c(TEXT))
    p.setColor(QPalette.Text, c(TEXT))
    p.setColor(QPalette.Button, c(RAISED))
    p.setColor(QPalette.ButtonText, c(TEXT))
    p.setColor(QPalette.BrightText, c("#ffffff"))
    p.setColor(QPalette.Highlight, c(ACCENT))
    p.setColor(QPalette.HighlightedText, c("#111111"))
    p.setColor(QPalette.PlaceholderText, c(DIM))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c("#666666"))
    app.setPalette(p)
    app.setStyleSheet(f"""
        QMainWindow::separator {{ background: {LINE}; width: 1px; height: 1px; }}
        QDockWidget::title {{ background: {PANEL}; padding: 4px 8px; color: {DIM}; }}
        QGroupBox {{ border: 1px solid {LINE}; border-radius: 3px; margin-top: 14px; padding-top: 6px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; color: {DIM}; }}
        QToolButton:checked {{ background: #444444; border: 1px solid {ACCENT}; }}
        QProgressBar {{ border: 1px solid {LINE}; border-radius: 2px; text-align: center; height: 14px; }}
        QProgressBar::chunk {{ background: #5a5a5a; }}
        QStatusBar {{ color: {DIM}; }}
    """)
