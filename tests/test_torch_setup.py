import sys

import pytest

from epochcolor import torch_setup as ts


def test_pick_index_newest_first(monkeypatch):
    seen = []

    def has(name):
        seen.append(name)
        return name in ("rocm7.1", "rocm6.4")

    monkeypatch.setattr(ts, "_index_has", has)
    assert ts.pick_index("rocm") == "rocm7.1"
    assert seen[:4] == ["rocm7.4", "rocm7.3", "rocm7.2", "rocm7.1"]


def test_cuda_respects_driver(monkeypatch):
    monkeypatch.setattr(ts, "_index_has", lambda n: True)
    monkeypatch.setattr(ts, "nvidia_driver", lambda: 572)
    assert ts.pick_index("cuda") == "cu128"  # 580 needed for cu130, 575 for cu129
    monkeypatch.setattr(ts, "nvidia_driver", lambda: 400)
    with pytest.raises(RuntimeError, match="older than any CUDA build"):
        ts.pick_index("cuda")


def test_detect(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, "gpu_vendors", lambda: {"amd"})
    monkeypatch.setattr(ts, "KFD", tmp_path / "kfd")
    fam, why = ts.detect()
    assert fam == "cpu" and "kfd" in why
    (tmp_path / "kfd").touch()
    assert ts.detect()[0] == "rocm"


def test_activate_appends_last(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    d = ts.torch_dir()
    assert not ts.activate()
    (d / "torch").mkdir(parents=True)
    before = list(sys.path)
    try:
        assert ts.activate()
        assert sys.path[-1] == str(d) and sys.path[:-1] == before
        assert not ts.activate()  # once only
    finally:
        sys.path[:] = before
