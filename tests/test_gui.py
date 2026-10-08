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


def test_paint_grade_colours_and_models(tmp_path, window):
    import av
    from dataclasses import asdict

    from epochcolor import grade as G
    from epochcolor.export.plan import ExportSettings
    from epochcolor.timeline_export import clip_info
    from epochcolor.video.pipeline import VideoSettings, analysis_dir, analyze

    app, w = window
    truth = make_clip(tmp_path / "a.mkv")
    w.add_clips([str(tmp_path / "a.mkv")])
    wait(app, w)
    p = w.project
    c = next(iter(p.d.clips.values()))
    analyze(clip_info(c), Oracle(truth), VideoSettings.from_project(p.d.settings), shots=c.shots, quiet=True)
    w._refresh()
    assert w.status[c.id] == "ready"

    # ---- paint a hint through the viewer; it's applied through the worker
    w.seek(12)
    w.paintbar.paint.setChecked(True)
    w.paintbar.set_rgb([30, 160, 60])
    w.viewer.strokeFinished.emit({"rgb": [30, 160, 60], "neutral": False, "radius": 0.05,
                                  "points": [[0.35, 0.5], [0.4, 0.5]]})
    assert p.hint_frames(c.id) == [12]
    assert w.status[c.id] == "update"
    w.apply_hints()
    wait(app, w)
    w._refresh()
    assert w.status[c.id] == "ready"
    d = analysis_dir(c.path, VideoSettings.from_project(p.d.settings), p.d.settings["model"])
    assert list(d.glob("hint_*.npy")), "the hinted shot was cached"
    w.undo()  # undo the stroke
    assert p.hint_frames(c.id) == []
    w.redo()

    # ---- erase it again with a right-click near the stroke
    w._erase_at(0.37, 0.5)
    assert p.hint_frames(c.id) == []
    w.undo()

    # ---- grade: edit through the panel, it lands on the shot's track
    w.seek(5)
    g = w.grade_panel.read()
    g["gain"] = [0.0, 0.0, 0.4]  # push blue
    w.grade_panel.changed.emit(g)
    w._commit_grade()
    track = p.grade_track(c.id, 5)
    assert track and track["keys"][0]["grade"]["gain"][2] == 0.4
    w.undo()
    assert p.grade_track(c.id, 5) is None
    w.redo()
    # keyframes: a second key changes saturation over time
    w.seek(20)
    w._grade_key("add")
    g = w.grade_panel.read()
    g["saturation"] = 0.0
    w.grade_panel.changed.emit(g)
    w._commit_grade()
    t = p.grade_track(c.id, 20)
    assert [k["frame"] for k in t["keys"]] == [5, 20]
    assert G.evaluate(t, 12)["saturation"] == pytest.approx(0.5, abs=0.05)
    # copy, paste onto the same shot is a static grade
    w.copy_grade()
    w.paste_grade()
    assert len(p.grade_track(c.id, 20)["keys"]) == 1

    # ---- the preview shows the grade
    got = {}
    w.provider.ready.connect(lambda f, gi, col, full, raw: got.update(f=f, col=col, raw=raw, full=full))
    w.seek(6)
    pump(app, 1.5)
    assert got.get("col") is not None and got.get("raw") is not None
    px_graded = got["col"].pixelColor(10, 10)
    px_raw = got["raw"].pixelColor(10, 10)
    assert px_graded != px_raw

    # ---- export carries the grade
    out = tmp_path / "graded.mkv"
    es = ExportSettings(encoder="software", speed="ultrafast", rf=16)
    w.runner.submit("export", "export", {**w._job_payload(), "project": asdict(p.d),
                                         "export": es.to_dict(), "out": str(out)})
    wait(app, w)
    with av.open(str(out)) as cont:
        frames = [f.to_ndarray(format="rgb24") for f in cont.decode(video=0)]
    static = p.grade_track(c.id, 20)["keys"][0]["grade"]
    assert static["saturation"] == pytest.approx(0.0, abs=1e-6)
    f0 = frames[3].astype(int)
    assert np.abs(f0[..., 0] - f0[..., 2]).max() < 25  # saturation 0: grey

    # ---- saved colours: save, find (worker, random siggraph17 features)
    w.seek(12)
    w.paintbar.set_rgb([200, 30, 30])
    w.viewer.strokeFinished.emit({"rgb": [200, 30, 30], "neutral": False, "radius": 0.05, "points": [[0.3, 0.5]]})
    from PySide6.QtWidgets import QInputDialog

    orig = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("red block", True))
    try:
        w.save_colour()
    finally:
        QInputDialog.getText = orig
    assert p.d.colors and p.d.colors[0]["name"] == "red block"
    p.set_setting("match_threshold", 0.3)
    w._colour_request("find", "")
    wait(app, w)
    assert isinstance(p.d.suggestions, list)  # random features: count means nothing, plumbing does
    from epochcolor.worker import match_thumb_path

    col = p.d.colors[0]
    assert match_thumb_path(col["id"]).exists(), "the saved colour got its picture"
    assert "red block" in w.colours.colours.item(0).text()

    # matches merge per colour, show up in the panel, and a marker click uses one
    w.seek(12)
    sug = {"id": "s1", "color": col["id"], "kind": "clip", "target": c.id, "frame": 12,
           "points": [[0.6, 0.5]], "score": 0.9}
    other = {"id": "s2", "color": "someone-else", "kind": "clip", "target": c.id, "frame": 3,
             "points": [[0.2, 0.2]], "score": 0.8}
    p.d.suggestions = [other]
    w._matches_found({"suggestions": [sug], "colors": [col["id"]]})
    assert {m["id"] for m in p.d.suggestions} == {"s1", "s2"}, "a one-colour search keeps the others"
    assert any("strong match" in w.colours.matches.item(i).text() for i in range(w.colours.matches.count()))
    assert any(x[4] == "s1" for x in w.viewer.suggestions), "the marker is on the picture"
    before = len(p.strokes_at(c.id, 12))
    w.viewer.suggestionClicked.emit("s1", True)
    assert len(p.strokes_at(c.id, 12)) == before + 1
    assert [m["id"] for m in p.d.suggestions] == ["s2"]
    w._colour_request("dismiss_all", "")
    assert p.d.suggestions == []
    wait(app, w)

    # ---- photo: paint, colorize with hints, grade on display
    import cv2

    cv2.imwrite(str(tmp_path / "ph.png"), cv2.cvtColor(truth[0][..., ::-1], cv2.COLOR_RGB2GRAY))
    w.add_photos([str(tmp_path / "ph.png")])
    w.filmstrip.setCurrentRow(0)
    w.viewer.strokeFinished.emit({"rgb": [30, 160, 60], "neutral": False, "radius": 0.05, "points": [[0.5, 0.5]]})
    ph = p.d.photos[0]
    assert len(ph.strokes) == 1
    w.apply_hints()
    wait(app, w)
    from epochcolor.worker import photo_result_path

    assert photo_result_path(ph.path, p.d.settings, p.d.settings["model"], ph.strokes).exists()
    g = w.grade_panel.read()
    g["saturation"] = 1.5
    w.grade_panel.changed.emit(g)
    w._commit_grade()
    assert p.d.photos[0].grade["saturation"] == 1.5

    # ---- model manager opens and lists the catalog
    from epochcolor.gui.model_manager import ModelManager

    mm = ModelManager(p.d.settings["model"], w)
    ids = [mm.table.item(i, 0).data(0x0100) for i in range(mm.table.rowCount())]
    assert "ddcolor-tiny" in ids and "siggraph17" in ids
    states = [mm.table.item(i, 5).text() for i in range(mm.table.rowCount())]
    assert any("installed" in s_ for s_ in states)
    mm.close()


def test_frame_by_frame_painting_and_cast(tmp_path, window):
    from epochcolor.timeline_export import clip_info
    from epochcolor.video.pipeline import VideoSettings, analysis_dir, analyze

    app, w = window
    truth = make_clip(tmp_path / "a.mkv")
    w.add_clips([str(tmp_path / "a.mkv")])
    wait(app, w)
    p = w.project
    c = next(iter(p.d.clips.values()))
    analyze(clip_info(c), Oracle(truth), VideoSettings.from_project(p.d.settings), shots=c.shots, quiet=True)
    w._refresh()

    # frame by frame: new strokes get reach 0 (this frame only)
    w.seek(5)
    w.toggle_frame_mode()
    assert w.paintbar.paint.isChecked() and w.viewer.brush.get("reach") == 0
    # the red box's centre on frame 5 is at x = 90 + 2*5 = 100 of 320
    w.viewer.strokeFinished.emit({**w.viewer.brush, "points": [[100 / 320, 0.55]]})
    assert p.strokes_at(c.id, 5)[0]["reach"] == 0

    # carry: the stroke lands on frame 6, moved with the box, and the playhead follows
    w.carry_strokes(1)
    moved = p.strokes_at(c.id, 6)
    assert len(moved) == 1 and moved[0]["from"] == 5
    assert moved[0]["points"][0][0] == pytest.approx(102 / 320, abs=1.5 / 320)
    assert w._paint_target() == ("clip", c.id, 6)
    # onion skin on frame 6 shows frame 5's stroke
    assert w.viewer.ghosts and w.viewer.ghosts[0]["points"] == [[100 / 320, 0.55]]
    # carrying from 5 again replaces what it put on 6 instead of stacking copies
    w.seek(5)
    w.carry_strokes(1)
    assert len(p.strokes_at(c.id, 6)) == 1

    # the hint pass runs through the worker with frame-limited strokes
    w.apply_hints()
    wait(app, w)
    d = analysis_dir(c.path, VideoSettings.from_project(p.d.settings), p.d.settings["model"])
    assert list(d.glob("protect_*.npy")), "frame-limited fixes are kept apart from the stabilizer"

    # cast removal is a live setting: the preview changes without a new pass
    got = {}
    w.provider.ready.connect(lambda f, gi, col, full, raw: got.update(raw=raw))
    w.seek(15)
    pump(app, 1.0)
    before = got["raw"].pixelColor(300, 20)
    w._setting_changed("cast", 1.0)
    got.clear()
    w.seek(16)
    w.seek(15)
    pump(app, 1.0)
    assert not w.runner.busy()
    assert got["raw"].pixelColor(300, 20) != before


def test_negative_reference_frame_export_and_audio(tmp_path, window):
    import cv2

    from epochcolor.gui.audio import mix_command
    from epochcolor.timeline_export import clip_info
    from epochcolor.video.pipeline import VideoSettings, analyze

    app, w = window
    truth = make_clip(tmp_path / "a.mkv")
    w.add_clips([str(tmp_path / "a.mkv")])
    wait(app, w)
    p = w.project
    c = next(iter(p.d.clips.values()))
    s = VideoSettings.from_project(p.d.settings)
    assert s.deflicker == 1.0 and s.dust, "a new project cleans the model's copy"
    analyze(clip_info(c), Oracle(truth), s, shots=c.shots, quiet=True)
    w._refresh()
    assert w.status[c.id] == "ready"

    # a reference image on the shot goes in through the hint pass
    w.seek(10)
    ref = tmp_path / "ref.png"
    img = np.zeros((90, 160, 3), np.uint8)
    img[:] = (60, 120, 200)
    cv2.imwrite(str(ref), img)
    w._set_reference(w._target_object(), {"path": str(ref), "strength": 0.8})
    assert p.reference_at(c.id, 10)["strength"] == 0.8
    assert "ref.png" in w.inspector.ref_label.text()
    wait(app, w)
    w._refresh()
    assert w.status[c.id] == "ready"
    w.clear_reference()
    wait(app, w)

    # export the frame under the playhead, as export renders it
    out = tmp_path / "frame.png"
    w.runner.submit("export_frame", "frame", {**w._job_payload(), "project": __import__("dataclasses").asdict(p.d),
                                              "clip": c.id, "frame": 10, "out": str(out)})
    wait(app, w)
    got = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert got is not None and got.shape[:2] == (180, 320) and got.dtype == np.uint16

    # the timeline's sound mixes to a WAV for playback
    wav = tmp_path / "mix.wav"
    import subprocess

    subprocess.run(mix_command(p, wav), check=True)
    import av

    with av.open(str(wav)) as cont:
        a = cont.streams.audio[0]
        assert a.rate == 48000 and a.channels == 2
        assert abs(float(cont.duration / 1e6) - c.frames / 24) < 0.1
    p.d.audio_enabled = [False] * len(p.d.audio_enabled)
    assert mix_command(p, wav) is None, "nothing enabled, nothing to play"

    # marking a negative measures it in the worker and inverts what the model sees
    w.inspector.negative.setChecked(True)
    w._negative_toggled(True)
    wait(app, w)
    assert c.negative and c.neg and "base" in c.neg
    w._refresh()
    assert w.status[c.id] == "none", "a negative is a different picture: colorize again"
    w._negative_toggled(False)
    assert not c.negative and c.negative_params, "turning it off keeps the measured numbers"


def test_autosave_recovers(tmp_path, monkeypatch):
    from epochcolor import autosave
    from epochcolor.project import Project

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    proj = Project()
    proj.path = tmp_path / "film.epochcolor"
    proj.save()
    proj.set_setting("cast", 0.5)  # an unsaved change
    autosave.write(proj)
    found = autosave.find(proj.path)
    assert found
    back = autosave.recover(found[0], proj.path)
    assert back.d.settings["cast"] == 0.5 and back.dirty and back.path == proj.path
    proj.save()  # saved afterwards: nothing left to recover
    import os
    import time

    later = time.time() + 5
    os.utime(proj.path, (later, later))
    assert autosave.find(proj.path) is None
    autosave.clear(proj.path)
    assert not autosave.path_for(proj.path).exists()


def test_items_shot_states_and_this_frame(tmp_path, window):
    from epochcolor.timeline_export import clip_info

    app, w = window
    make_clip(tmp_path / "a.mkv")
    w.add_clips([str(tmp_path / "a.mkv")])
    wait(app, w)
    p = w.project
    c = next(iter(p.d.clips.values()))
    a, b = c.shots[0]
    key = f"{a}-{b}"
    w.seek(5)
    w._refresh()
    assert w.shot_status[c.id][key]["state"] == "none"
    assert "not coloured" in w.inspector.shot_info.text()

    # zoom and fit
    w.viewer.zoom_at(2.0)
    assert w.viewer.zoom == pytest.approx(2.0)
    w.viewer.fit()
    assert w.viewer.zoom == 1.0 and w.viewer.pan.x() == 0 and w.viewer.pan.y() == 0

    # a new item is the brush; its strokes carry it, and its first stroke is what Find looks for
    item = w.new_item("red box", [200, 30, 30])
    assert w.paintbar.item_id == item["id"] and w.paintbar.paint.isChecked()
    assert w.viewer.brush["item"] == item["id"] and "red box" in w.paintbar.item_label.text()
    w.hint_timer.setInterval(10_000)  # this test presses the buttons itself
    w.viewer.strokeFinished.emit({**w.viewer.brush, "points": [[100 / 320, 0.55]]})
    st = p.strokes_at(c.id, 5)[0]
    assert st["item"] == item["id"] and item["source"]["frame"] == 5 and "item" not in item["source"]["stroke"]
    assert p.item_usage(item["id"]) == (1, 1)
    assert "painted on 1 frame" in w.colours.colours.item(0).text()
    assert "red box" in w.inspector.shot_info.text()
    w.hint_timer.stop()

    # Color shot: paint only, no model
    w.color_shot()
    wait(app, w)
    w._refresh()
    assert w.shot_status[c.id][key] == {**w.shot_status[c.id][key], "state": "painted", "current": True}
    assert "painted only" in w.inspector.shot_info.text()

    # Fill shot: the model does the rest
    w.fill_shot()
    wait(app, w)
    w._refresh()
    assert w.shot_status[c.id][key]["state"] == "colored" and w.shot_status[c.id][key]["current"]
    assert w.inspector.fill_shot_btn.text() == "Update shot"

    # recolour the item: every stroke of it changes, and the shot is out of date until updated
    w.recolour_item(item["id"], [30, 160, 60])
    w.hint_timer.stop()
    assert p.strokes_at(c.id, 5)[0]["rgb"] == [30, 160, 60]
    w._refresh()
    assert not w.shot_status[c.id][key]["current"]
    w.apply_hints()  # a coloured shot updates
    wait(app, w)
    w._refresh()
    assert w.shot_status[c.id][key]["current"]

    # Colorize frame: a temporary picture of the model on this frame, shown as This frame
    w.colorize_frame()
    wait(app, w)
    pump(app, 0.2)
    assert w.viewer.mode == "frame" and w.viewer.frame_image is not None
    w.seek(6)
    assert w.viewer.frame_image is None, "frame 6 has no result of its own"
    w.seek(5)
    assert w.viewer.frame_image is not None

    # an item nobody painted is skipped by Find
    sky = w.new_item("sky", [120, 160, 220])
    w._colour_request("find", sky["id"])
    assert not w.runner.busy()

    # deleting an item can keep its paint, unlabelled
    w.delete_item(item["id"], with_strokes=False)
    assert "item" not in p.strokes_at(c.id, 5)[0] and all(x["id"] != item["id"] for x in p.d.colors)
    w.delete_item(sky["id"])
    assert w.paintbar.item_id is None
    assert clip_info(c)  # still a sane clip
