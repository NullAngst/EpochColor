import hashlib
import http.server
import json
import threading
from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from epochcolor.models import manager as M  # noqa: E402


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "srv"
    root.mkdir()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_head(self):  # honour Range so resume can be tested
            rng = self.headers.get("Range")
            path = self.translate_path(self.path)
            if not rng:
                return super().send_head()
            start = int(rng.split("=")[1].split("-")[0])
            data = open(path, "rb").read()[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            import io

            return io.BytesIO(data)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(root)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setenv("EPOCHCOLOR_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return tmp_path / "models"


def ddcolor_tiny_file(path):
    from epochcolor.models.ddcolor import DDColorAdapter

    net = DDColorAdapter(params={"model_size": "tiny", "input_size": 64}).build()
    torch.save({"params": net.state_dict()}, path)  # wrapped the way DDColor saves


def entries(url, sha, license_ok=True):
    return [{"id": "test-model", "name": "Test", "architecture": "eccv16", "mode": "automatic",
             "license": "BSD-2-Clause" if license_ok else "research only", "license_ok": license_ok,
             "params": {}, "size_mb": 1,
             "source": {"type": "url", "url": url, "filename": "w.pth", "sha256": sha}}]


def test_download_resume_and_verify(server, models, monkeypatch, tmp_path):
    root, base = server
    from epochcolor.models.zhang import build_eccv16

    torch.save(build_eccv16().state_dict(), root / "w.pth")
    data = (root / "w.pth").read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(M, "_bundled", lambda: entries(f"{base}/w.pth", sha))
    # half a file left over from an interrupted download
    (models / "test-model").mkdir(parents=True)
    (models / "test-model" / "w.pth.part").write_bytes(data[: len(data) // 2])
    seen = []
    path = M.download("test-model", lambda s, d, t: seen.append(s))
    assert path.read_bytes() == data and "download" in seen
    man = M.installed()["test-model"]
    assert man["sha256"] == sha
    from epochcolor.models import create

    m = create("test-model")
    m.load("cpu")
    assert m.predict(np.full((40, 64), 50, np.float32)).shape == (40, 64, 2)
    assert M.remove("test-model") and "test-model" not in M.installed()


def test_bad_hash_is_refused(server, models, monkeypatch):
    root, base = server
    (root / "w.pth").write_bytes(b"not the right file")
    monkeypatch.setattr(M, "_bundled", lambda: entries(f"{base}/w.pth", "0" * 64))
    with pytest.raises(M.ModelError, match="SHA-256"):
        M.download("test-model")
    assert not list((models / "test-model").glob("w.pth*"))


def test_license_needs_accepting(server, models, monkeypatch):
    root, base = server
    (root / "w.pth").write_bytes(b"x")
    sha = hashlib.sha256(b"x").hexdigest()
    monkeypatch.setattr(M, "_bundled", lambda: entries(f"{base}/w.pth", sha, license_ok=False))
    with pytest.raises(M.ModelError, match="accept"):
        M.download("test-model")
    M.download("test-model", accept_license=True)
    assert M.installed()["test-model"]["license_accepted"] is True


def test_huggingface_source_uses_published_hash(server, models, monkeypatch):
    root, base = server
    (root / "model.safetensors").write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    e = entries("", "")[0]
    e["source"] = {"type": "huggingface", "repo": "someone/model"}
    monkeypatch.setattr(M, "_bundled", lambda: [e])
    monkeypatch.setattr(M, "_hf_file", lambda repo: (f"{base}/model.safetensors", "model.safetensors", sha, 7))
    assert M.download("test-model").read_bytes() == b"weights"


def test_add_from_file_and_wrong_size(models, tmp_path):
    w = tmp_path / "mine.pth"
    ddcolor_tiny_file(w)
    man = {"id": "my-ddcolor", "name": "Mine", "architecture": "ddcolor", "mode": "automatic",
           "license": "CC-BY-NC-4.0", "params": {"model_size": "tiny", "input_size": 64}}
    mid = M.add_from_file(w, man)
    M.check_loads(mid)
    from epochcolor.models import create

    m = create(mid)
    m.load("cpu")
    out = m.predict(np.full((48, 80), 40, np.float32))
    assert out.shape == (48, 80, 2) and np.isfinite(out).all()
    assert m.features(np.full((48, 80), 40, np.float32)).shape[0] == 384  # ConvNeXt-T stage 3
    with pytest.raises(M.ModelError, match="installed already"):
        M.add_from_file(w, man)
    bad = dict(man, id="wrong-size", params={"model_size": "large", "input_size": 64})
    M.add_from_file(w, bad)
    with pytest.raises(RuntimeError, match="does not fit DDColor-large"):
        M.check_loads("wrong-size")
    with pytest.raises(M.ModelError, match="no adapter"):
        M.add_from_file(w, dict(man, id="x", architecture="bigcolor"))


def test_catalog_refresh_and_storage(models, monkeypatch, tmp_path, server):
    root, base = server
    good = entries("http://x/w.pth", "a" * 64)[0]
    bad = dict(good, id="future", architecture="something-new")
    (root / "catalog.json").write_text(json.dumps({"version": 1, "models": [good, bad]}))
    monkeypatch.setattr(M, "CATALOG_URL", f"{base}/catalog.json")
    assert M.refresh_catalog() == (1, 1)
    assert any(e["id"] == "test-model" for e in M.catalog())
    assert any(e["id"] == "ddcolor-tiny" for e in M.catalog())  # bundled entries stay
    monkeypatch.delenv("EPOCHCOLOR_MODELS")
    from epochcolor.models.weights import models_dir

    new = M.move_storage(tmp_path / "elsewhere")
    assert models_dir() == new


def test_matching_plumbing():
    """Exemplar, similarity and peaks with a hand-made feature extractor
    (local brightness and texture), so the test checks the matching code
    itself rather than what a random network happens to compute."""
    import cv2

    from epochcolor.match import exemplar, peaks, similarity

    class Feat:
        def features(self, L):
            mean = cv2.blur(L, (5, 5))
            std = np.sqrt(np.maximum(cv2.blur(L * L, (5, 5)) - mean * mean, 0))
            f = np.stack([mean / 50 - 1, std / 10, np.ones_like(L) * 0.1])[:, ::4, ::4]
            t = torch.from_numpy(np.ascontiguousarray(f)).float()
            return t / (t.norm(dim=0, keepdim=True) + 1e-6)

    rng = np.random.default_rng(0)
    L = np.full((128, 192), 50, np.float32)
    L[60:94, 40:76] = 20 + 60 * rng.random((34, 36))  # a textured patch
    L[10:40, 130:170] = 20 + 60 * rng.random((30, 40))  # and another like it
    stroke = {"rgb": [200, 40, 40], "radius": 0.05, "points": [[0.3, 0.6]]}
    m = Feat()
    e = exemplar(m, L, [stroke])
    assert e is not None and abs(np.linalg.norm(e) - 1) < 1e-4
    S = similarity(m, L, [e])[0]
    best, pts = peaks(S, threshold=0.9)
    xs = sorted(round(p[0], 1) for p in pts)
    assert len(pts) >= 2 and any(abs(p[0] - 0.3) < 0.12 for p in pts) and any(p[0] > 0.6 for p in pts), pts
    assert all(not (0.0 < p[0] < 0.18) for p in pts)  # the flat background isn't a match
    assert peaks(S, threshold=1.5)[1] == []


def test_deoldify_adds_loads_and_refuses_the_wrong_variant(models, tmp_path):
    """DeOldify's generator, rebuilt without fastai, round-trips a checkpoint
    saved the way DeOldify saves it (an old-style state dict, no BatchNorm
    step counters), and a deep checkpoint can't pass as wide."""
    from epochcolor.models import create
    from epochcolor.models.deoldify import DeOldifyAdapter

    net = DeOldifyAdapter(params={"variant": "deep"}).build()
    sd = {k: v for k, v in net.state_dict().items() if not k.endswith("num_batches_tracked")}
    w = tmp_path / "ColorizeArtistic_gen.pth"
    torch.save(sd, w)
    man = {"id": "my-deoldify", "name": "Mine", "architecture": "deoldify", "mode": "automatic",
           "license": "MIT", "params": {"variant": "deep", "render_size": 96}}
    mid = M.add_from_file(w, man)
    M.check_loads(mid)
    m = create(mid)
    m.load("cpu")
    out = m.predict(np.full((48, 80), 40, np.float32))
    assert out.shape == (48, 80, 2) and np.isfinite(out).all()
    assert m.features(np.full((48, 80), 40, np.float32)).shape[0] == 256  # ResNet-34 layer3
    M.add_from_file(w, dict(man, id="as-wide", params={"variant": "wide", "render_size": 96}))
    with pytest.raises(RuntimeError, match="does not fit DeOldify-wide"):
        M.check_loads("as-wide")


def test_catalog_has_deoldify_with_full_hashes():
    ids = {e["id"]: e for e in M.catalog()}
    for mid in ("deoldify-video", "deoldify-stable", "deoldify-artistic"):
        e = ids[mid]
        assert e["architecture"] == "deoldify" and len(e["source"]["sha256"]) == 64
        assert M.validate_entry(e) == []


class Boom:
    """Pickles to a call of os.system: what a hostile weights file does."""

    def __reduce__(self):
        import os

        return (os.system, ("touch PWNED",))


@pytest.mark.parametrize("legacy", [True, False])
def test_old_checkpoints_with_training_leftovers_load_safely(tmp_path, monkeypatch, legacy):
    """DeOldify's files hold fastai leftovers (slice, functools.partial, the
    optimizer class) beside the weights; PyTorch's safe loader refuses them.
    The fallback keeps the weights and turns everything else inert, so the
    planted os.system call never runs."""
    import functools

    from epochcolor.models.weights import Inert, load_state_dict

    monkeypatch.chdir(tmp_path)
    sd = {"layers.0.weight": torch.randn(3, 3), "layers.0.bias": torch.zeros(3)}
    blob = {"model": sd, "opt": {"opt_func": functools.partial(torch.optim.Adam, betas=(0.9, 0.99)),
                                 "lr": slice(1e-3, 1e-2), "evil": Boom()}}
    f = tmp_path / "gen.pth"
    torch.save(blob, f, _use_new_zipfile_serialization=not legacy)
    got = load_state_dict(f)
    assert set(got) == set(sd) and torch.equal(got["layers.0.weight"], sd["layers.0.weight"])
    assert not (tmp_path / "PWNED").exists(), "nothing from the pickle ran"
    # and what came back for the leftovers is inert
    import pickle

    raw = torch.load(str(f), map_location="cpu", weights_only=False,
                     pickle_module=__import__("epochcolor.models.weights", fromlist=["x"])._inert_pickle())
    assert isinstance(raw["opt"]["evil"], Inert) and isinstance(raw["opt"]["opt_func"], Inert)
    assert raw["opt"]["lr"] == slice(1e-3, 1e-2)
    assert not (tmp_path / "PWNED").exists()
