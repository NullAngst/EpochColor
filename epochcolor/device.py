"""Pick a PyTorch device. ROCm builds of PyTorch report through torch.cuda,
so AMD and NVIDIA share the "cuda" name here."""

from __future__ import annotations


def pick_device(pref: str = "auto") -> tuple[str, str]:
    """Return (device string, human description)."""
    import torch

    if pref == "cpu":
        return "cpu", "CPU"
    if pref.startswith("cuda") or pref == "auto":
        if torch.cuda.is_available():
            dev = pref if pref.startswith("cuda") else "cuda"
            idx = torch.device(dev).index or 0
            backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            return dev, f"{torch.cuda.get_device_name(idx)} ({backend})"
        if pref != "auto":
            raise RuntimeError("asked for a GPU but PyTorch sees none (no CUDA or ROCm device)")
    if pref.startswith("xpu") or pref == "auto":
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            dev = pref if pref.startswith("xpu") else "xpu"
            return dev, f"{xpu.get_device_name(0)} (XPU)"
        if pref != "auto":
            raise RuntimeError("asked for an Intel GPU but PyTorch sees no XPU device")
    if pref not in ("auto",):
        raise RuntimeError(f"unknown device {pref!r}")
    return "cpu", "CPU (no GPU found by PyTorch)"
