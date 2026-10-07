"""GUI end to end, headless: import through the worker process, colour from
a known source, edit, play, export through the worker, photos."""

import os
import shutil
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("av")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class Oracle:
    """Writes the true chroma into the analysis cache under the GUI's model
    name, so the viewer has real colour to show without real weights."""

    def __init__(self, truth):
        from epochcolor.models.base import ModelInfo

        self.info = ModelInfo("siggraph17", "automatic", "oracle", False, False, False)
        self.truth, self.i = truth, 0

    def predict(self, L, hints=None):
        import cv2

        from epochcolor.color import srgb_to_lab

        h, w = L.shape
        f = cv2.resize(self.truth[min(self.i, len(self.truth) - 1)], (w, h)).astype(np.float32) / 255
        self.i += 1
        return srgb_to_lab(f)[..., 1:]


def make_clip(path, n=24, w=320, h=180):
    import subprocess

    import cv2

    color = []
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                          "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                          "-map", "0", "-map", "1", "-c:v", "libx264", "-crf", "12",
                          "-pix_fmt", "yuv420p", "-c:a", "flac", "-shortest", str(path)],
                         stdin=subprocess.PIPE)
    for t in range(n):
        bgr = np.full((h, w, 3), (200, 160, 110), np.uint8)
        cv2.rectangle(bgr, (40 + 2 * t, 50), (140 + 2 * t, 150), (40, 40, 180), -1)
        color.append(bgr[..., ::-1].copy())
        p.stdin.write(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return np.stack(color)


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from epochcolor.gui import theme
    from epochcolor.gui.main import MainWindow
    from epochcolor.models.zhang import build_eccv16, build_siggraph17

    monkeypatch.setenv("EPOCHCOLOR_CACHE", str(tmp_path / "cache"))
    models = tmp_path / "models"
    models.mkdir()
    torch.save(build_siggraph17().state_dict(), models / "siggraph17-df00044c.pth")
    torch.save(build_eccv16().state_dict(), models / "colorization_release_v2-9b330a0b.pth")
    monkeypatch.setenv("EPOCHCOLOR_MODELS", str(models))
    app = QApplication.instance() or QApplication([])
    theme.apply(app)
    w = MainWindow()
    w.failures = []
    w.runner.failed.connect(lambda job, msg, tb: w.failures.append((job.label, msg, tb)))
    w.resize(1200, 800)
    w.show()
    yield app, w
    w.project.dirty = False
    w.shutdown()
    w.close()


def pump(app, sec):
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.01)


def wait(app, w, limit=120):
    t = time.time()
    while w.runner.busy() and time.time() - t < limit:
        pump(app, 0.05)
    assert not w.runner.busy(), "jobs did not finish"
    assert not w.failures, w.failures


def test_gui_end_to_end(tmp_path, window):
    from epochcolor.timeline_export import clip_info
    from epochcolor.video.pipeline import VideoSettings, analyze

    app, w = window
    truth = make_clip(tmp_path / "a.mkv")
    w.add_clips([str(tmp_path / "a.mkv")])
    wait(app, w)
    p = w.project
    assert p.duration == 24 and len(w.media_info) == 1

    c = next(iter(p.d.clips.values()))
    analyze(clip_info(c), Oracle(truth), VideoSettings.from_project(p.d.settings), shots=c.shots, quiet=True)
    w._refresh()
    assert w.status[c.id] == "ready"

    got = {}
    w.provider.ready.connect(lambda f, g, col, full: got.update(frame=f, color=col, full=full))
    w.seek(12)
    pump(app, 1.5)
    assert got.get("frame") == 12 and got.get("color") is not None and got.get("full")
    img = got["color"]
    px = img.pixelColor(int(img.width() * (65 / 320)), int(img.height() * 0.5))
    assert px.red() > px.blue() + 40  # the red block is red

    # edits and undo through the window
    w.split()
    w.timeline.canvas.sel = 0
    w.delete_segment()
    assert p.duration == 12
    w.undo()
    assert p.duration == 24
    w.redo()
    w.toggle_play()
    pump(app, 0.3)
    w.toggle_play()

    # export the edited timeline through the worker
    from dataclasses import asdict

    from epochcolor.export.plan import ExportSettings

    out = tmp_path / "out.mkv"
    es = ExportSettings(encoder="software", speed="ultrafast", rf=20, audio_fallback="flac")
    w.runner.submit("export", "export", {**w._job_payload(), "project": asdict(p.d),
                                         "export": es.to_dict(), "out": str(out)})
    wait(app, w)
    import av

    with av.open(str(out)) as cont:
        assert sum(1 for _ in cont.decode(video=0)) == 12
        assert len(cont.streams.audio) == 1

    # a photo through the worker with the automatic model
    import cv2

    cv2.imwrite(str(tmp_path / "ph.png"), cv2.cvtColor(truth[0][..., ::-1], cv2.COLOR_RGB2GRAY))
    w.add_photos([str(tmp_path / "ph.png")])
    w._setting_changed("model", "eccv16")
    w.filmstrip.setCurrentRow(0)
    w.colorize_photos()
    wait(app, w)
    w._refresh()
    assert "not colorized" not in w.filmstrip.item(0).text()


def test_worker_crash_is_reported(window):
    app, w = window
    w.failures.clear()
    job = w.runner.submit("import", "import", {"path": "/nonexistent.mkv"})
    t = time.time()
    while w.runner.busy() and time.time() - t < 60:
        pump(app, 0.05)
    assert w.failures and "nonexistent" in w.failures[0][1] or "cannot open" in w.failures[0][1]
    w.failures.clear()
    # a dead worker is noticed and replaced
    w.runner._ensure()
    w.runner.proc.kill()
    w.runner.proc.join()
    job = w.runner.submit("fetch", "fetch", {"model": "hints"})
    t = time.time()
    while w.runner.busy() and time.time() - t < 60:
        pump(app, 0.05)
    assert w.failures and "needs no download" in w.failures[0][1]
