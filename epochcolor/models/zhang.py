"""Zhang et al. colorization models (ECCV 2016 and SIGGRAPH 2017).

Architectures match richzhang/colorization (BSD-2-Clause) so its released
state dicts load as-is. SIGGRAPH17 is the interactive model from the
"Real-Time User-Guided Image Colorization" paper and takes colour hints;
ECCV16 is automatic only.
"""

from __future__ import annotations

import numpy as np

from ..hints import Hints
from .base import ColorModel, ModelInfo

L_CENT, L_NORM, AB_NORM = 50.0, 100.0, 110.0
# The interactive demo (ideepcolor) centres the hint mask on 0.5, so "no
# hint" reads as -0.5 and "hint" as +0.5. colorizers' siggraph17 forward
# passes the raw 0/1 mask but never feeds hints, so ideepcolor is the one
# to follow here.
MASK_CENT = 0.5


def _snap(n: float, mult: int = 8) -> int:
    return max(mult, int(round(n / mult)) * mult)


def _net_size(h: int, w: int, short: int) -> tuple[int, int]:
    s = short / min(h, w)
    return _snap(h * s), _snap(w * s)


def build_eccv16():
    import torch.nn as nn

    def block(cin, cout, n, stride_last=1, dilation=1, norm=True):
        layers = []
        for i in range(n):
            stride = stride_last if i == n - 1 else 1
            layers += [
                nn.Conv2d(cin if i == 0 else cout, cout, 3, stride=stride, padding=dilation,
                          dilation=dilation, bias=True),
                nn.ReLU(True),
            ]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        return nn.Sequential(*layers)

    class ECCVGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model1 = block(1, 64, 2, stride_last=2)
            self.model2 = block(64, 128, 2, stride_last=2)
            self.model3 = block(128, 256, 3, stride_last=2)
            self.model4 = block(256, 512, 3)
            self.model5 = block(512, 512, 3, dilation=2)
            self.model6 = block(512, 512, 3, dilation=2)
            self.model7 = block(512, 512, 3)
            self.model8 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1, bias=True), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, padding=1, bias=True), nn.ReLU(True),
                nn.Conv2d(256, 313, 1, bias=True),
            )
            self.softmax = nn.Softmax(dim=1)
            self.model_out = nn.Conv2d(313, 2, 1, bias=False)
            self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        def forward(self, l):
            x = self.model1((l - L_CENT) / L_NORM)
            for m in (self.model2, self.model3, self.model4, self.model5, self.model6,
                      self.model7, self.model8):
                x = m(x)
            return self.upsample4(self.model_out(self.softmax(x))) * AB_NORM

    return ECCVGenerator()


def build_siggraph17():
    import torch
    import torch.nn as nn

    def block(cin, cout, n, dilation=1, norm=True):
        layers = []
        for i in range(n):
            layers += [
                nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=dilation, dilation=dilation, bias=True),
                nn.ReLU(True),
            ]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        return nn.Sequential(*layers)

    def tail(c, n_conv, last_act):
        layers = [nn.ReLU(True)]
        for i in range(n_conv):
            layers.append(nn.Conv2d(c, c, 3, padding=1, bias=True))
            layers.append(last_act if i == n_conv - 1 and last_act is not None else nn.ReLU(True))
        return layers

    class SIGGRAPHGenerator(nn.Module):
        def __init__(self, classes=529):
            super().__init__()
            self.model1 = block(4, 64, 2)
            self.model2 = block(64, 128, 2)
            self.model3 = block(128, 256, 3)
            self.model4 = block(256, 512, 3)
            self.model5 = block(512, 512, 3, dilation=2)
            self.model6 = block(512, 512, 3, dilation=2)
            self.model7 = block(512, 512, 3)
            self.model8up = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True))
            self.model3short8 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=True))
            self.model8 = nn.Sequential(*tail(256, 2, None), nn.BatchNorm2d(256))
            self.model9up = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True))
            self.model2short9 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=True))
            self.model9 = nn.Sequential(*tail(128, 1, None), nn.BatchNorm2d(128))
            self.model10up = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=True))
            self.model1short10 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, bias=True))
            self.model10 = nn.Sequential(*tail(128, 1, nn.LeakyReLU(negative_slope=0.2)))
            # classification head, unused at inference but present in the weights
            self.model_class = nn.Sequential(nn.Conv2d(256, classes, 1, bias=True))
            self.model_out = nn.Sequential(nn.Conv2d(128, 2, 1, bias=True), nn.Tanh())

        def forward(self, l, ab, mask):
            x = torch.cat(((l - L_CENT) / L_NORM, ab / AB_NORM, mask - MASK_CENT), dim=1)
            c1 = self.model1(x)
            c2 = self.model2(c1[:, :, ::2, ::2])
            c3 = self.model3(c2[:, :, ::2, ::2])
            c4 = self.model4(c3[:, :, ::2, ::2])
            c7 = self.model7(self.model6(self.model5(c4)))
            c8 = self.model8(self.model8up(c7) + self.model3short8(c3))
            c9 = self.model9(self.model9up(c8) + self.model2short9(c2))
            c10 = self.model10(self.model10up(c9) + self.model1short10(c1))
            return self.model_out(c10) * AB_NORM

    return SIGGRAPHGenerator()


class _ZhangBase(ColorModel):
    builder = None
    infer_short = 256  # both were trained at 256x256

    def __init__(self, weights: str | None = None, info: ModelInfo | None = None, params=None):
        self.weights = weights
        if info is not None:
            self.info = info
        self.net = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        from pathlib import Path

        from .weights import load_state_dict

        if not self.weights:
            raise RuntimeError(f"{self.info.name} has no weights file")
        path = Path(self.weights)
        net = type(self).builder()
        sd = load_state_dict(path)
        try:
            missing, unexpected = net.load_state_dict(sd, strict=False)
        except RuntimeError as e:  # shape mismatch
            raise RuntimeError(f"{path.name} does not fit the {self.info.name} architecture: {e}") from None
        # older checkpoints predate BatchNorm's step counter, which inference ignores
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(
                f"{path.name} does not fit the {self.info.name} architecture "
                f"(missing {len(missing)} keys, unexpected {len(unexpected)})"
            )
        self.net = net.to(device).eval()
        self.device = device

    def _resize(self, arr: np.ndarray, h: int, w: int) -> np.ndarray:
        import cv2

        interp = cv2.INTER_AREA if h < arr.shape[0] else cv2.INTER_CUBIC
        return cv2.resize(arr, (w, h), interpolation=interp)


    def features(self, L: np.ndarray):
        """Mid-level features (conv4, 512 channels at 1/8 size) for matching
        saved colours. Returns a torch tensor (C, h, w), unit length per pixel."""
        import torch

        h, w = L.shape
        nh, nw = _net_size(h, w, self.infer_short)
        l = torch.from_numpy(self._resize(L, nh, nw)).to(self.device)[None, None]
        with torch.inference_mode():
            f = self._encode(l)
        f = f[0].float()
        return f / (f.norm(dim=0, keepdim=True) + 1e-6)


class ECCV16(_ZhangBase):
    info = ModelInfo(
        name="eccv16",
        mode="automatic",
        description="Zhang et al. 2016, automatic. Muted, conservative colour.",
        takes_hints=False,
        needs_weights=True,
        needs_torch=True,
        license="BSD-2-Clause",
    )
    builder = staticmethod(build_eccv16)

    def _encode(self, l):
        x = self.net.model1((l - L_CENT) / L_NORM)
        for m in (self.net.model2, self.net.model3, self.net.model4):
            x = m(x)
        return x

    def predict(self, L, hints=None):
        import torch

        h, w = L.shape
        nh, nw = _net_size(h, w, self.infer_short)
        l = torch.from_numpy(self._resize(L, nh, nw)).to(self.device)[None, None]
        with torch.inference_mode():
            ab = self.net(l)[0].permute(1, 2, 0).float().cpu().numpy()
        return self._resize(ab, h, w).astype(np.float32)


class SIGGRAPH17(_ZhangBase):
    info = ModelInfo(
        name="siggraph17",
        mode="hint",
        description="Zhang et al. 2017, automatic or guided by painted hints.",
        takes_hints=True,
        needs_weights=True,
        needs_torch=True,
        license="BSD-2-Clause",
    )
    builder = staticmethod(build_siggraph17)

    def _encode(self, l):
        import torch

        z = torch.zeros_like(l)
        x = torch.cat(((l - L_CENT) / L_NORM, z, z, z - MASK_CENT), dim=1)
        c1 = self.net.model1(x)
        c2 = self.net.model2(c1[:, :, ::2, ::2])
        c3 = self.net.model3(c2[:, :, ::2, ::2])
        return self.net.model4(c3[:, :, ::2, ::2])

    def predict(self, L, hints=None):
        import torch

        h, w = L.shape
        nh, nw = _net_size(h, w, self.infer_short)
        l = torch.from_numpy(self._resize(L, nh, nw)).to(self.device)[None, None]
        if hints is not None and hints.count:
            hs = hints.resized(nw, nh).binarized(0.3)
            ab_in = torch.from_numpy(hs.ab).permute(2, 0, 1)[None].to(self.device)
            m_in = torch.from_numpy(hs.mask)[None, None].to(self.device)
        else:
            ab_in = torch.zeros((1, 2, nh, nw), device=self.device)
            m_in = torch.zeros((1, 1, nh, nw), device=self.device)
        with torch.inference_mode():
            ab = self.net(l, ab_in, m_in)[0].permute(1, 2, 0).float().cpu().numpy()
        return self._resize(ab, h, w).astype(np.float32)
