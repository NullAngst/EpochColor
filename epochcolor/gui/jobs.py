"""GUI side of the worker process: a job queue with progress signals.

Jobs run one at a time in order. Cancel kills the worker outright, since a
model pass can sit in one GPU call for seconds and has no safe point to
stop at; whatever finished before the kill stays cached.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass
class Job:
    id: int
    kind: str
    label: str
    payload: dict
    stage: str = "waiting"
    done: int = 0
    total: int = 0


class JobRunner(QObject):
    changed = Signal()  # queue or progress changed
    finished = Signal(object, dict)  # Job, result
    failed = Signal(object, str, str)  # Job, message, traceback

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.jobs_q = None
        self.events_q = None
        self.pending: list[Job] = []
        self.current: Job | None = None
        self._ids = itertools.count(1)
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._poll)

    # ---------------------------------------------------------- public

    def submit(self, kind: str, label: str, payload: dict) -> Job:
        job = Job(next(self._ids), kind, label, payload)
        self.pending.append(job)
        self.changed.emit()
        self._next()
        return job

    def cancel(self, job_id: int | None = None) -> None:
        """Cancel one job, or with None the running job and everything queued."""
        if job_id is None:
            dropped = self.pending
            self.pending = []
            for j in dropped:
                self.failed.emit(j, "cancelled", "")
            if self.current:
                self._kill("cancelled")
        elif self.current and self.current.id == job_id:
            self._kill("cancelled")
        else:
            for j in [j for j in self.pending if j.id == job_id]:
                self.pending.remove(j)
                self.failed.emit(j, "cancelled", "")
        self.changed.emit()
        self._next()

    def cancel_where(self, pred, running: bool = False) -> None:
        """Drop queued jobs that match (a newer request replaces them). With
        running=True a matching running job is stopped too."""
        drop = [j for j in self.pending if pred(j)]
        for j in drop:
            self.pending.remove(j)
        if running and self.current and pred(self.current):
            self._kill("replaced by a newer request")
        if drop:
            self.changed.emit()

    def busy(self) -> bool:
        return bool(self.current or self.pending)

    def all_jobs(self) -> list[Job]:
        return ([self.current] if self.current else []) + list(self.pending)

    def shutdown(self) -> None:
        self.timer.stop()
        if self.proc and self.proc.is_alive():
            try:
                self.jobs_q.put(None)
                self.proc.join(1.0)
            except Exception:
                pass
            if self.proc.is_alive():
                self.proc.kill()
        self.proc = None

    # --------------------------------------------------------- private

    def _ensure(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        from ..worker import worker_main

        self.jobs_q = self.ctx.Queue()
        self.events_q = self.ctx.Queue()
        self.proc = self.ctx.Process(target=worker_main, args=(self.jobs_q, self.events_q),
                                     daemon=True, name="epochcolor-worker")
        self.proc.start()
        self.timer.start()

    def _next(self) -> None:
        if self.current or not self.pending:
            return
        self._ensure()
        self.current = self.pending.pop(0)
        self.current.stage = "starting"
        self.jobs_q.put((self.current.id, self.current.kind, self.current.payload))
        self.changed.emit()

    def _kill(self, why: str) -> None:
        job = self.current
        self.current = None
        if self.proc is not None:
            self.proc.kill()
            self.proc.join(2.0)
            self.proc = None
        if job:
            self.failed.emit(job, why, "")

    def _poll(self) -> None:
        if self.events_q is None:
            return
        got = False
        while True:
            try:
                ev = self.events_q.get_nowait()
            except Exception:
                break
            got = True
            kind, jid = ev[0], ev[1]
            job = self.current if (self.current and self.current.id == jid) else None
            if job is None:
                continue  # left over from a killed job
            if kind == "progress":
                job.stage, job.done, job.total = ev[2], ev[3], ev[4]
            elif kind == "done":
                self.current = None
                self.finished.emit(job, ev[2])
                self._next()
            elif kind == "error":
                self.current = None
                self.failed.emit(job, ev[2], ev[3])
                self._next()
        if self.current and self.proc is not None and not self.proc.is_alive():
            code = self.proc.exitcode
            self.proc = None
            self._kill(f"the worker process died (exit code {code}); finished shots are cached, "
                       f"run it again to pick up")
            self._next()
            got = True
        if got:
            self.changed.emit()
