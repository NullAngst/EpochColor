"""The model manager dialog: catalog, downloads, add from file, storage."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout,
)

from ..models import manager as M
from ..models.weights import models_dir
from . import theme


class ModelManager(QDialog):
    """Downloads and installs go through the main window's job runner (the
    worker process), so they show progress there and can be cancelled."""
    submit = Signal(str, str, dict)  # kind, label, payload

    def __init__(self, current_model: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model manager")
        self.resize(900, 560)
        self.current_model = current_model
        lay = QVBoxLayout(self)
        self.where = QLabel()
        lay.addWidget(self.where)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Model", "Mode", "Size", "VRAM", "License", "State"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._selected)
        lay.addWidget(self.table, 2)
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(True)
        self.details.setMaximumHeight(130)
        lay.addWidget(self.details)
        row = QHBoxLayout()
        self.download_btn = QPushButton("Download")
        self.remove_btn = QPushButton("Remove")
        self.use_btn = QPushButton("Use for this project")
        add = QPushButton("Add from file...")
        refresh = QPushButton("Refresh catalog")
        move = QPushButton("Storage folder...")
        for b in (self.download_btn, self.remove_btn, self.use_btn):
            row.addWidget(b)
        row.addStretch(1)
        for b in (add, refresh, move):
            row.addWidget(b)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.download_btn.clicked.connect(self._download)
        self.remove_btn.clicked.connect(self._remove)
        self.use_btn.clicked.connect(self._use)
        add.clicked.connect(self._add)
        refresh.clicked.connect(lambda: self.submit.emit("refresh_catalog", "Refresh model catalog", {}))
        move.clicked.connect(self._move)
        self.chosen: str | None = None
        self.reload()

    def reload(self) -> None:
        self.where.setText(f"Models are stored in {models_dir()}")
        have = M.installed()
        rows = list(M.catalog())
        for mid, man in have.items():
            if not any(e["id"] == mid for e in rows):
                rows.append(man)
        sel = self._current_id()
        self.rows = rows
        self.table.setRowCount(len(rows))
        for i, e in enumerate(rows):
            state = "installed" if e["id"] in have else "not downloaded"
            if e["id"] == self.current_model:
                state += ", in use"
            cells = [e.get("name", e["id"]), e.get("mode", ""), M.human_size(e.get("size_mb")),
                     f"{e['vram_gb']} GB" if e.get("vram_gb") else "?", e.get("license", ""), state]
            for j, text in enumerate(cells):
                it = QTableWidgetItem(str(text))
                it.setData(Qt.UserRole, e["id"])
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i, e in enumerate(rows):
            if e["id"] == sel:
                self.table.selectRow(i)
        self._selected()

    def _current_id(self) -> str | None:
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        return it.data(Qt.UserRole) if it else None

    def _entry(self):
        mid = self._current_id()
        return next((e for e in self.rows if e["id"] == mid), None) if mid else None

    def _selected(self) -> None:
        e = self._entry()
        have = M.installed()
        self.download_btn.setEnabled(bool(e) and e["id"] not in have and e.get("source", {}).get("type") != "local")
        self.remove_btn.setEnabled(bool(e) and e["id"] in have)
        self.use_btn.setEnabled(bool(e) and e["id"] in have)
        if not e:
            self.details.setHtml(f"<span style='color:{theme.DIM}'>Pick a model.</span>")
            return
        lic = e.get("license", "")
        url = e.get("license_url")
        lic_html = f"<a href='{url}'>{lic}</a>" if url else lic
        warn = "" if e.get("license_ok", True) else \
            "<br><b>Read the license before downloading</b>: you'll be asked to accept it."
        self.details.setHtml(f"<b>{e.get('name', e['id'])}</b> ({e['id']}, {e.get('architecture')})<br>"
                             f"{e.get('description', '')}<br>License: {lic_html}{warn}")

    def _download(self) -> None:
        e = self._entry()
        if not e:
            return
        accept = False
        if not e.get("license_ok", True):
            r = QMessageBox.question(
                self, "License",
                f"{e['name']}\n\nLicense: {e['license']}\n{e.get('license_url', '')}\n\n"
                "Have you read the license, and do you accept it for how you'll use this model?")
            if r != QMessageBox.Yes:
                return
            accept = True
        self.submit.emit("fetch", f"Download {e['id']}", {"model": e["id"], "accept_license": accept})

    def _remove(self) -> None:
        e = self._entry()
        if e and QMessageBox.question(self, "Remove", f"Delete the {e['id']} weights from disk?") == QMessageBox.Yes:
            M.remove(e["id"])
            self.reload()

    def _use(self) -> None:
        e = self._entry()
        if e:
            self.chosen = e["id"]
            self.current_model = e["id"]
            self.reload()

    def _add(self) -> None:
        w, _ = QFileDialog.getOpenFileName(self, "Weights file", "", "Weights (*.pth *.pt *.bin *.safetensors *.ckpt)")
        if not w:
            return
        m, _ = QFileDialog.getOpenFileName(self, "Manifest for those weights", str(Path(w).parent),
                                           "Manifest (*.json)")
        if not m:
            QMessageBox.information(self, "Manifest needed", M.MANIFEST_HELP)
            return
        try:
            json.loads(Path(m).read_text())
        except (OSError, ValueError) as ex:
            QMessageBox.warning(self, "Manifest", f"Can't read {m}: {ex}")
            return
        self.submit.emit("add_model", f"Add {Path(w).name}", {"weights": w, "manifest": m})

    def _move(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Store models in", str(models_dir()))
        if d:
            self.submit.emit("move_models", "Move models", {"dir": d})
