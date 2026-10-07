"""Working folder and resource limits."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .. import resources as R
from ..diskspace import describe
from ..models.weights import read_settings, write_setting
from ..video.pipeline import cache_root, cache_size, clear_cache, default_cache_root, set_cache_root


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Working folder and limits")
        self.setMinimumWidth(620)
        self.changed_root = False
        self.old_root = cache_root()
        st = read_settings()
        lay = QVBoxLayout(self)
        intro = QLabel("The working folder holds everything EpochColor makes while it works: proxies, the "
                       "colour analysis (about 0.7 MB per frame at the default chroma size, so roughly 1 GB "
                       "per minute of 24 fps footage), colorized photos and export scratch files. Put it on "
                       "a disk with room to spare. It can all be deleted; it gets rebuilt when needed.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        f = QFormLayout()
        row = QHBoxLayout()
        self.path = QLineEdit(str(self.old_root))
        browse = QPushButton("Browse...")
        default = QPushButton("Default")
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        row.addWidget(default)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        f.addRow("Working folder", w)
        self.info = QLabel()
        self.info.setWordWrap(True)
        f.addRow("", self.info)
        self.empty_btn = QPushButton("Empty the current working folder")
        f.addRow("", self.empty_btn)
        if os.environ.get("EPOCHCOLOR_CACHE"):
            self.path.setEnabled(False)
            browse.setEnabled(False)
            default.setEnabled(False)
            f.addRow("", QLabel("EPOCHCOLOR_CACHE is set, and it wins over this setting."))

        n = os.cpu_count() or 2
        self.threads = QSpinBox()
        self.threads.setRange(1, n)
        self.threads.setValue(R.cpu_threads())
        f.addRow("CPU threads", self.threads)
        tot = (R.meminfo() or (16 * R.GB, 0))[0]
        self.mem = QDoubleSpinBox()
        self.mem.setRange(1, max(2, tot / R.GB))
        self.mem.setDecimals(1)
        self.mem.setSuffix(" GB")
        self.mem.setValue(R.memory_limit(tot) / R.GB)
        f.addRow("Memory limit", self.mem)
        self.nice = QSpinBox()
        self.nice.setRange(0, 19)
        self.nice.setValue(R.niceness())
        f.addRow("Priority (nice)", self.nice)
        lay.addLayout(f)
        note = QLabel(f"The worker is stopped if it goes over the memory limit, or if free memory drops "
                      f"under {R.low_water(tot) / R.GB:.1f} GB, before the system starts swapping. "
                      f"Higher nice means lower priority; 10 keeps the desktop responsive. "
                      f"Defaults: {max(1, n - 1)} threads, {tot * 0.75 / R.GB:.1f} GB, nice 10.")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        browse.clicked.connect(self._browse)
        default.clicked.connect(lambda: self.path.setText(str(default_cache_root())))
        self.empty_btn.clicked.connect(self._empty)
        self.path.textChanged.connect(self._update)
        self._st = st
        self._update()

    def _update(self) -> None:
        p = Path(self.path.text().strip() or default_cache_root()).expanduser()
        try:
            self.info.setText(f"{describe(p)}. Using {cache_size(self.old_root) / R.GB:.2f} GB now "
                              f"in {self.old_root}.")
        except OSError as e:
            self.info.setText(f"can't read {p}: {e}")

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Working folder", self.path.text())
        if d:
            self.path.setText(d)

    def _empty(self) -> None:
        size = cache_size(self.old_root) / R.GB
        if QMessageBox.question(self, "Empty working folder",
                                f"Delete {size:.2f} GB of cached proxies, analysis and photos in "
                                f"{self.old_root}?\n\nClips will need colorizing again.") == QMessageBox.Yes:
            if self.parent() is not None and hasattr(self.parent(), "runner"):
                self.parent().runner.shutdown()
            clear_cache(self.old_root)
            self._update()

    def _accept(self) -> None:
        text = self.path.text().strip()
        new = Path(text).expanduser() if text else default_cache_root()
        if not os.environ.get("EPOCHCOLOR_CACHE") and new.absolute() != self.old_root.absolute():
            try:
                set_cache_root(None if new == default_cache_root() else new)
            except OSError as e:
                QMessageBox.warning(self, "Working folder", f"Can't use {new}: {e}")
                return
            self.changed_root = True
        write_setting("threads", self.threads.value() if self.threads.value() != max(1, (os.cpu_count() or 2) - 1)
                      else None)
        tot = (R.meminfo() or (16 * R.GB, 0))[0]
        lim = round(self.mem.value(), 1)
        write_setting("memory_limit_gb", None if abs(lim - tot * 0.75 / R.GB) < 0.05 else lim)
        write_setting("nice", None if self.nice.value() == 10 else self.nice.value())
        self.accept()
