"""Test the GPU in a throwaway process before trusting it with real work.

A GPU build of PyTorch that can't drive a card doesn't always fail with an
error. ROCm in particular can segfault, which takes the whole process with
it. The usual causes on AMD:

- an integrated GPU (most Ryzen 7000 and newer desktop chips have one) that
  ROCm enumerates next to the real card and then crashes on
- a card whose exact chip isn't in the PyTorch build, which needs
  HSA_OVERRIDE_GFX_VERSION to run the kernels built for its sibling (an RX
  6700 XT is gfx1031 and runs the gfx1030 build)

So the first time a model needs the GPU, each GPU is tried on its own in a
child process with a small convolution and matrix test, checked against
the CPU result. The first that passes is used; if none does, the CPU is.
The verdict is saved and reused until PyTorch, the GPUs, the kernel or
EpochColor change. `epochcolor device --test` runs it again and prints
every attempt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def state_path() -> Path:
    from .models.weights import config_dir

    return config_dir() / "gpu-check.json"


# ----------------------------------------------------------------- sysfs


def gfx_name(v: int) -> str:
    """KFD gfx_target_version (110001) to an arch name (gfx1101)."""
    major, minor, step = v // 10000, (v // 100) % 100, v % 100
    return f"gfx{major}{minor:x}{step:x}"


def kfd_gpus(root: Path = KFD_NODES) -> list[dict]:
    """AMD GPUs as ROCm will number them: GPU nodes of the KFD topology in
    node order. Memory is the node's local memory, which for an integrated
    GPU is only its small carve-out."""
    out = []
    try:
        nodes = sorted(root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 1 << 30)
    except OSError:
        return out
    for node in nodes:
        try:
            props = dict(line.split()[:2] for line in (node / "properties").read_text().splitlines()
                         if len(line.split()) >= 2)
        except OSError:
            continue
        ver = int(props.get("gfx_target_version", "0") or 0)
        if not ver:
            continue  # a CPU node
        mem = 0
        for bank in (node / "mem_banks").glob("*/properties"):
            try:
                for line in bank.read_text().splitlines():
                    k, _, v = line.partition(" ")
                    if k == "size_in_bytes":
                        mem += int(v)
            except (OSError, ValueError):
                pass
        out.append({"index": len(out), "node": node.name, "gfx": gfx_name(ver), "version": ver,
                    "mem": mem, "cpu_cores": int(props.get("cpu_cores_count", "0") or 0)})
    return out


def override_for(gfx: str) -> str | None:
    """The HSA_OVERRIDE_GFX_VERSION worth trying when a chip fails as is."""
    if gfx.startswith("gfx103") and gfx != "gfx1030":
        return "10.3.0"
    if gfx.startswith("gfx110") and gfx not in ("gfx1100", "gfx1101", "gfx1102"):
        return "11.0.0"
    if gfx.startswith("gfx115"):
        return "11.0.0"
    return None


# ----------------------------------------------------------------- probe


def probe() -> dict:
    """Runs inside the child. Imports what the worker imports, in the same
    order, then a small test on cuda:0 compared with the CPU."""
    import numpy  # noqa: F401
    import cv2  # noqa: F401
    import av  # noqa: F401

    from . import models  # noqa: F401
    import torch

    t0 = time.time()
    res = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
           "cuda": getattr(torch.version, "cuda", None)}
    if not torch.cuda.is_available():
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            dev = torch.device("xpu")
            res.update(name=xpu.get_device_name(0), backend="xpu")
        else:
            res.update(ok=False, error="PyTorch sees no GPU",
                       cpu_build=not res["hip"] and not res["cuda"])
            return res
    else:
        dev = torch.device("cuda:0")
        p = torch.cuda.get_device_properties(0)
        res.update(name=p.name, mem=int(p.total_memory), arch=getattr(p, "gcnArchName", ""),
                   backend="rocm" if res["hip"] else "cuda")
    torch.manual_seed(0)
    x = torch.randn(1, 3, 256, 256)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=2, dilation=2), torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 2, 1)).eval()
    a = torch.randn(512, 512)
    with torch.no_grad():
        ref = net(x)
        ref_mm = torch.softmax(a @ a.T, dim=1)
        g = net.to(dev)(x.to(dev)).cpu()
        g_mm = torch.softmax(a.to(dev) @ a.to(dev).T, dim=1).cpu()
    err = float((g - ref).abs().max() / (ref.abs().max() + 1e-6))
    err_mm = float((g_mm - ref_mm).abs().max())
    res["rel_error"] = round(err, 6)
    res["seconds"] = round(time.time() - t0, 2)
    if not (err < 2e-2 and err_mm < 2e-2):  # TF32 and fast-math kernels sit far below this
        res.update(ok=False, error=f"GPU results don't match the CPU (error {err:.3g}, {err_mm:.3g})")
        return res
    res["ok"] = True
    return res


def _probe_cmd() -> tuple[list[str], dict]:
    cmd = [sys.executable] + (["-s"] if sys.flags.no_user_site else []) + ["-m", "epochcolor.gpucheck", "--probe"]
    env = dict(os.environ)
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return cmd, env


def run_probe(extra_env: dict, timeout: float = 240, log=None) -> dict:
    from .resources import describe_exit

    cmd, env = _probe_cmd()
    env.update(extra_env)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "env": extra_env, "error": f"no answer in {timeout:.0f}s (hung in the driver)"}
    res = None
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("{"):
            try:
                res = json.loads(line)
                break
            except ValueError:
                pass
    noise = (p.stderr or "").strip()
    if log and noise:
        log(noise[-4000:])
    if res is None:
        res = {"ok": False, "error": describe_exit(p.returncode) if p.returncode else "no result",
               "stderr": noise[-1500:]}
    res["env"] = extra_env
    res["rc"] = p.returncode
    res["wall"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------- ladder


def fingerprint() -> dict:
    from . import __version__
    from .torch_setup import active_dir, installed_variant

    d = active_dir()
    try:
        mt = int((d / "torch" / "version.py").stat().st_mtime) if d else 0
    except OSError:
        mt = 0
    keep = ("HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    return {"app": __version__, "torch": installed_variant() or "own", "torch_mtime": mt,
            "gpus": [g["gfx"] for g in kfd_gpus()], "kernel": os.uname().release if hasattr(os, "uname") else "",
            "python": sys.version.split()[0], "env": {k: os.environ[k] for k in keep if k in os.environ}}


def attempts_for(gpus: list[dict]) -> list[dict]:
    """Environments to try, best guess first. A user's own ROCm variables
    are left alone: they know their machine."""
    if any(k in os.environ for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION")):
        return [{}]
    if not gpus:
        return [{}]  # NVIDIA, Intel, or a system without KFD: one plain try
    tries = []
    for g in sorted(gpus, key=lambda g: (-g["mem"], g["index"])):  # the card with VRAM first
        sel = {"ROCR_VISIBLE_DEVICES": str(g["index"])} if len(gpus) > 1 else {}
        tries.append(dict(sel))
        ov = override_for(g["gfx"])
        if ov:
            tries.append({**sel, "HSA_OVERRIDE_GFX_VERSION": ov})
    return tries


def run_ladder(log=print, progress=None) -> dict:
    gpus = kfd_gpus()
    if gpus:
        log("AMD GPUs seen by the kernel: " + ", ".join(
            f"#{g['index']} {g['gfx']} ({g['mem'] / 2**30:.1f} GB)" for g in gpus))
    tries = attempts_for(gpus)
    attempts = []
    chosen = None
    for i, env in enumerate(tries):
        if progress:
            progress("testing the GPU (first run only)", i, len(tries))
        label = " ".join(f"{k}={v}" for k, v in env.items()) or "as is"
        log(f"trying the GPU {label} ...")
        r = run_probe(env, log=lambda s: log("  | " + s.replace("\n", "\n  | ")))
        attempts.append(r)
        if r.get("ok"):
            log(f"  works: {r.get('name')} ({r.get('backend')}, {r.get('seconds')}s)")
            chosen = r
            break
        log(f"  failed: {r.get('error')}")
    if progress:
        progress("testing the GPU (first run only)", len(tries), len(tries))
    result = {"fingerprint": fingerprint(), "time": time.time(), "attempts": attempts}
    if chosen:
        result.update(device="cuda" if chosen.get("backend") in ("rocm", "cuda") else "xpu",
                      env=chosen["env"], name=chosen.get("name", ""), backend=chosen.get("backend"))
        if chosen["env"].get("HSA_OVERRIDE_GFX_VERSION"):
            result["note"] = (f"{chosen.get('name')} needed HSA_OVERRIDE_GFX_VERSION="
                              f"{chosen['env']['HSA_OVERRIDE_GFX_VERSION']} to run.")
    elif attempts and all(a.get("cpu_build") for a in attempts):
        result.update(device="cpu", env={}, name="CPU")  # a CPU-only PyTorch: nothing to report
    else:
        why = "; ".join(str(a.get("error")) for a in attempts) or "no GPU"
        result.update(device="cpu", env={}, name="CPU",
                      note=f"The GPU didn't pass its test, so the CPU is used (much slower). "
                           f"What happened: {why}. `epochcolor device --test` shows the details.")
    try:
        state_path().parent.mkdir(parents=True, exist_ok=True)
        state_path().write_text(json.dumps(result, indent=1))
    except OSError:
        pass
    return result


def saved() -> dict | None:
    try:
        r = json.loads(state_path().read_text())
    except (OSError, ValueError):
        return None
    return r if r.get("fingerprint") == fingerprint() else None


def ensure(pref: str = "auto", log=None, progress=None) -> tuple[str, dict, str | None, bool]:
    """(device, env to set before torch starts the GPU, note, fresh).

    Only 'auto' and a bare 'cuda' get tested; an explicit choice (cpu,
    cuda:1, xpu) is used as given. fresh is True when the test just ran,
    so the caller can show the note once."""
    if pref not in ("auto", "cuda"):
        return pref, {}, None, False
    from .torch_setup import have_torch

    if not have_torch():
        return pref, {}, None, False  # loading the model reports the missing PyTorch
    r = saved()
    fresh = False
    if r is None:
        if log is None:
            from .resources import open_log

            f = open_log("gpu-check.log")
            f.write(f"--- {time.ctime()}\n")

            def log(msg, _f=f):
                _f.write(msg + "\n")
        r = run_ladder(log=log, progress=progress)
        fresh = True
    dev = r.get("device", "cpu")
    if pref == "cuda" and dev == "cpu":
        raise RuntimeError(r.get("note") or "the GPU didn't pass its test")
    return dev, dict(r.get("env") or {}), r.get("note"), fresh


def forget() -> None:
    try:
        state_path().unlink()
    except OSError:
        pass


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        try:
            out = probe()
        except Exception as e:  # noqa: BLE001 - reported to the parent
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
        os._exit(0)  # skip interpreter teardown, which some GPU stacks crash in
