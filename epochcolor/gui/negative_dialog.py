"""Levels for a black and white negative's inversion."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QVBoxLayout,
)

from . import theme


class NegativeDialog(QDialog):
    changed = Signal(dict)  # live, as the numbers move
    remeasure = Signal()

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Negative")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        intro = QLabel("The film base is the clearest part of the film, the rebate between frames. It becomes "
                       "black. Shadows and highlights set where the density range starts and ends; contrast "
                       "bends the tones in between. The picture updates as you go.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {theme.DIM};")
        lay.addWidget(intro)
        f = QFormLayout()
        self.base = self._spin(0.01, 1.0, 0.005, 4)
        self.dmin = self._spin(0.0, 3.0, 0.01, 3)
        self.dmax = self._spin(0.05, 4.0, 0.01, 3)
        self.gamma = self._spin(0.3, 3.0, 0.05, 2)
        f.addRow("Film base", self.base)
        f.addRow("Shadows (density)", self.dmin)
        f.addRow("Highlights (density)", self.dmax)
        f.addRow("Contrast", self.gamma)
        lay.addLayout(f)
        auto = QPushButton("Auto")
        auto.setToolTip("Measure again from the picture")
        auto.clicked.connect(self.remeasure.emit)
        lay.addWidget(auto, 0, Qt.AlignLeft)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.set_params(params, emit=False)
        for w in (self.base, self.dmin, self.dmax, self.gamma):
            w.valueChanged.connect(lambda _=0: self.changed.emit(self.params()))

    @staticmethod
    def _spin(lo, hi, step, dec) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(dec)
        s.setKeyboardTracking(False)
        return s

    def params(self) -> dict:
        dmin, dmax = self.dmin.value(), self.dmax.value()
        return {"base": round(self.base.value(), 5), "dmin": round(dmin, 4),
                "dmax": round(max(dmax, dmin + 0.05), 4), "gamma": round(self.gamma.value(), 3)}

    def set_params(self, p: dict, emit: bool = True) -> None:
        for w, k, d in ((self.base, "base", 1.0), (self.dmin, "dmin", 0.0), (self.dmax, "dmax", 2.0),
                        (self.gamma, "gamma", 1.0)):
            w.blockSignals(True)
            w.setValue(float(p.get(k, d)))
            w.blockSignals(False)
        if emit:
            self.changed.emit(self.params())
