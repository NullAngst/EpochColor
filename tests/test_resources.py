import os
import sys

import pytest

from epochcolor import gpucheck, resources


def test_meminfo_and_rss():
    m = resources.meminfo()
    if m is None:
        pytest.skip("no /proc/meminfo")
    total, avail = m
    assert total > avail > 0
    assert resources.tree_rss(os.getpid()) > 10 * 1024 * 1024


def test_over_limit(monkeypatch):
    G = resources.GB
    monkeypatch.setattr(resources, "meminfo", lambda: (32 * G, 20 * G))
    monkeypatch.setattr(resources, "tree_rss", lambda pid: 4 * G)
    assert resources.over_limit(1) is None
    monkeypatch.setattr(resources, "tree_rss", lambda pid: 25 * G)  # over 75% of 32
    assert "over its limit" in resources.over_limit(1)
    monkeypatch.setattr(resources, "tree_rss", lambda pid: 10 * G)
    monkeypatch.setattr(resources, "meminfo", lambda: (32 * G, int(0.5 * G)))
    assert "free memory" in resources.over_limit(1)


def test_describe_exit():
    assert "segmentation fault" in resources.describe_exit(-11)
    assert "out-of-memory" in resources.describe_exit(-9)


def test_capture_output_catches_c_level_writes(tmp_path):
    import subprocess

    code = ("import os; from epochcolor.resources import capture_output; "
            "p = capture_output('t.log'); os.write(2, b'(null): from C\\n'); print(p)")
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stderr == "" and r.stdout == ""
    assert "(null): from C" in (tmp_path / "epochcolor" / "logs" / "t.log").read_text()


def test_gfx_names_and_overrides():
    assert gpucheck.gfx_name(110001) == "gfx1101"
    assert gpucheck.gfx_name(90010) == "gfx90a"
    assert gpucheck.override_for("gfx1031") == "10.3.0"
    assert gpucheck.override_for("gfx1030") is None
    assert gpucheck.override_for("gfx1103") == "11.0.0"


def _node(root, i, ver, mem):
    d = root / str(i)
    (d / "mem_banks" / "0").mkdir(parents=True)
    (d / "properties").write_text(f"cpu_cores_count 0\nsimd_count 64\ngfx_target_version {ver}\n")
    (d / "mem_banks" / "0" / "properties").write_text(f"heap_type 1\nsize_in_bytes {mem}\n")


def test_kfd_topology_and_ladder(tmp_path, monkeypatch):
    for v in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"):
        monkeypatch.delenv(v, raising=False)
    cpu = tmp_path / "0"
    cpu.mkdir()
    (cpu / "properties").write_text("cpu_cores_count 16\ngfx_target_version 0\n")
    _node(tmp_path, 1, 100302, 512 * 2**20)  # an iGPU-sized carve-out
    _node(tmp_path, 2, 110000, 24 * 2**30)  # the real card
    gpus = gpucheck.kfd_gpus(tmp_path)
    assert [g["gfx"] for g in gpus] == ["gfx1032", "gfx1100"]
    tries = gpucheck.attempts_for(gpus)
    assert tries[0] == {"ROCR_VISIBLE_DEVICES": "1"}  # the 24 GB card first
    assert {"ROCR_VISIBLE_DEVICES": "0", "HSA_OVERRIDE_GFX_VERSION": "10.3.0"} in tries


def test_ladder_falls_back_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(gpucheck, "kfd_gpus", lambda root=None: [{"index": 0, "gfx": "gfx1031", "mem": 8 << 30}])
    for v in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"):
        monkeypatch.delenv(v, raising=False)
    seen = []

    def crash(env, timeout=240, log=None):
        seen.append(env)
        return {"ok": False, "error": "it crashed (SIGSEGV)", "env": env}

    monkeypatch.setattr(gpucheck, "run_probe", crash)
    dev, env, note, fresh = gpucheck.ensure("auto", log=lambda m: None)
    assert dev == "cpu" and fresh and "SIGSEGV" in note
    assert seen == [{}, {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}]
    assert gpucheck.ensure("auto")[3] is False  # saved, not tested again


def test_probe_child_runs():
    """The real child process, whatever PyTorch this machine has."""
    r = gpucheck.run_probe({}, timeout=120)
    assert "rc" in r and (r.get("ok") or r.get("error"))
