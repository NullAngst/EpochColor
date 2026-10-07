"""Download PyTorch from inside the editor: the same installer as
`epochcolor setup-torch`, run as a child process with its output shown."""

from __future__ import annotations

import sys

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QPlainTextEdit,
    QVBoxLayout,
)

from .. import torch_setup as ts


class TorchDialog(QDialog):
    def __init__(self, first_run: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install PyTorch")
        self.setMinimumWidth(580)
        self.ok = False
        lay = QVBoxLayout(self)
        fam, why = ts.detect()
        have = ts.installed_variant()
        intro = ("The colorization models need PyTorch, which isn't bundled since the GPU builds run to "
                 "several gigabytes. It downloads once, for the GPU in this machine, into "
                 f"{ts.torch_dir()}.")
        if have:
            intro += f"\n\nInstalled now: {have}. Installing again replaces it."
        elif ts.have_torch():
            intro += "\n\nA PyTorch is already importable (your own install), and that one is used."
        text = QLabel(intro)
        text.setWordWrap(True)
        lay.addWidget(text)
        f = QFormLayout()
        f.addRow("Detected", QLabel(why))
        from ..diskspace import describe

        room = QLabel(f"{describe(ts.torch_dir().parent)}. The install needs about "
                      f"{ts.NEED_GB.get(fam, 12)} GB there while it unpacks.")
        room.setWordWrap(True)
        f.addRow("Room", room)
        self.variant = QComboBox()
        for key, label in (("auto", f"auto ({fam})"), ("rocm", "ROCm (AMD)"), ("cuda", "CUDA (NVIDIA)"),
                           ("xpu", "XPU (Intel)"), ("cpu", "CPU only")):
            self.variant.addItem(f"{label}, {ts.SIZES.get(fam if key == 'auto' else key, '')}", key)
        f.addRow("Build", self.variant)
        lay.addLayout(f)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.log.setMinimumHeight(220)
        lay.addWidget(self.log)
        self.dont_ask = QCheckBox("Don't ask again on start (it stays in the Colour menu)")
        self.dont_ask.setVisible(first_run)
        lay.addWidget(self.dont_ask)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Close)
        self.buttons.button(QDialogButtonBox.Ok).setText("Download and install")
        self.buttons.accepted.connect(self._start)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.proc: QProcess | None = None

    def _start(self) -> None:
        if self.proc is not None:
            return
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.variant.setEnabled(False)
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.start(sys.executable, ["-s", "-m", "epochcolor", "setup-torch",
                                         "--variant", self.variant.currentData()])

    def _read(self) -> None:
        data = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        self.log.appendPlainText(data.rstrip())

    def _done(self, code: int, status) -> None:
        self.proc = None
        if code == 0:
            ts.activate()
            self.ok = True
            self.log.appendPlainText("\nPyTorch is ready. No restart needed.")
            self.buttons.button(QDialogButtonBox.Ok).setVisible(False)
        else:
            self.log.appendPlainText(f"\nFailed (exit {code}). Nothing was changed.")
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
            self.variant.setEnabled(True)

    def reject(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.waitForFinished(3000)
        super().reject()
