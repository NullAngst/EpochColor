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


def test_dir_is_per_python(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert ts.torch_dir().name == f"torch-{sys.implementation.cache_tag}"


def test_legacy_folder_only_for_matching_python(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    legacy = tmp_path / "epochcolor" / "torch" / "torch"
    legacy.mkdir(parents=True)
    (legacy / "_C.cpython-399-x86_64-linux-gnu.so").touch()  # some other Python
    assert ts.active_dir() is None
    (legacy / f"_C.{sys.implementation.cache_tag}-x86_64-linux-gnu.so").touch()
    assert ts.active_dir() == legacy.parent


def test_install_unpacks_beside_target(monkeypatch, tmp_path):
    """pip must not unpack into /tmp (tmpfs on many distros)."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(ts, "pick_index", lambda fam: "cpu")
    seen = {}

    class FakeProc:
        def __init__(self, cmd, env=None, **kw):
            seen["cmd"], seen["env"] = cmd, env
            target = cmd[cmd.index("--target") + 1]
            from pathlib import Path

            (Path(target) / "torch").mkdir(parents=True)
            self.stdout = iter(["ok\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(ts.subprocess, "Popen", FakeProc)
    final = ts.install("cpu", log=lambda *_: None)
    assert seen["env"]["TMPDIR"].startswith(str(tmp_path))
    assert "--no-cache-dir" in seen["cmd"]
    assert (final / "torch").is_dir() and not (final.parent / "pip-tmp").exists()


def test_install_refuses_without_room(monkeypatch, tmp_path):
    from epochcolor import diskspace

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(ts, "pick_index", lambda fam: "rocm7.1")
    monkeypatch.setattr(diskspace, "free_bytes", lambda p: 3 * diskspace.GB)
    with pytest.raises(diskspace.DiskFull, match="needs about 20.0 GB"):
        ts.install("rocm", log=lambda *_: None)
