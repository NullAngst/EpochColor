import errno
import subprocess
import sys
from pathlib import Path

from epochcolor import diskspace


def test_explain_names_folder(tmp_path):
    e = OSError(errno.ENOSPC, "No space left on device", str(tmp_path / "x.part"))
    assert diskspace.is_full(e)
    msg = diskspace.explain(e)
    assert "x.part" in msg and "GB free" in msg


def test_describe_missing_path_uses_parent(tmp_path):
    assert "GB free" in diskspace.describe(tmp_path / "not" / "yet")


def test_download_checks_room(monkeypatch, tmp_path):
    import pytest

    from epochcolor.models import manager as M

    monkeypatch.setenv("EPOCHCOLOR_MODELS", str(tmp_path))
    monkeypatch.setattr(diskspace, "free_bytes", lambda p: 10)
    with pytest.raises(diskspace.DiskFull):
        M.download("siggraph17")


def test_launcher_runs_cli():
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "epochcolor.py"), "--version"],
                       capture_output=True, text=True, cwd=root, timeout=60)
    assert r.returncode == 0 and "epochcolor" in r.stdout
