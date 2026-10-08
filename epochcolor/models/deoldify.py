"""DeOldify (Jason Antic, 2018-2019) behind the adapter interface.

Three generators, all a U-Net on a ResNet body, trained with NoGAN:

- Video: ResNet-101 "wide" decoder, trained for frames that don't flicker.
  The calmest of the three; made for film.
- Stable: the same network, trained for portraits and landscapes with
  fewer odd colour splotches.
- Artistic: ResNet-34 "deep" decoder, bolder and more varied colour, and
  more likely to get something wrong.

The network takes a grey RGB square (render_size, DeOldify's render factor
x 16), ImageNet-normalized, and returns RGB in the same normalization.
DeOldify itself keeps the original luma and takes only the colour from
that output, which is how EpochColor works anyway: the output goes to Lab
and only a/b is kept.
"""

from __future__ import annotations

import numpy as np

from .base import ColorModel, ModelInfo

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class DeOldifyAdapter(ColorModel):
    info = ModelInfo(
        name="deoldify",
        mode="automatic",
        description="DeOldify, automatic.",
        takes_hints=False,
        needs_weights=True,
        needs_torch=True,
        license="MIT",
    )

    def __init__(self, weights: str | None = None, info: ModelInfo | None = None, params=None):
        self.weights = weights
        if info is not None:
            self.info = info
        p = dict(params or {})
        self.variant = p.get("variant", "wide")
        self.render_size = int(p.get("render_size", 336))
        if self.render_size % 16:
            raise ValueError("DeOldify render_size has to be a multiple of 16")
        self.net = None
        self.device = "cpu"

    def build(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch's weight_norm deprecation notice
            return self._build()

    def _build(self):
        from .deoldify_arch.resnet import body
        from .deoldify_arch.unet import DynamicUnetDeep, DynamicUnetWide
        from .deoldify_arch.utils import NormType

        common = dict(n_classes=3, blur=True, blur_final=True, self_attention=True, y_range=(-3.0, 3.0),
                      norm_type=NormType.Spectral, last_cross=True, bottle=False)
        if self.variant == "wide":  # Video and Stable
            return DynamicUnetWide(body(101), nf_factor=2, **common)
        if self.variant == "deep":  # Artistic
            return DynamicUnetDeep(body(34), nf_factor=1.5, **common)
        raise RuntimeError(f"DeOldify variant must be wide or deep, not {self.variant}")

    def load(self, device: str) -> None:
        from pathlib import Path

        from .weights import load_state_dict

        if not self.weights:
            raise RuntimeError(f"{self.info.name} has no weights file")
        path = Path(self.weights)
        net = self.build()
        sd = load_state_dict(path)
        try:
            missing, unexpected = net.load_state_dict(sd, strict=False)
        except RuntimeError as e:
            raise RuntimeError(f"{path.name} does not fit DeOldify-{self.variant}: {e}") from None
        # checkpoints from older PyTorch have no BatchNorm step counters; nothing reads them at inference
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            hint = " (variant: wide for Video and Stable, deep for Artistic)" if len(missing) > 20 else ""
            raise RuntimeError(f"{path.name} does not fit DeOldify-{self.variant}: missing {len(missing)} "
                               f"keys, unexpected {len(unexpected)}{hint}")
        self.net = net.to(device).eval()
        self.device = device

    def _input(self, L: np.ndarray):
        import cv2
        import torch

        from ..color import lab_to_srgb

        s = self.render_size
        h, w = L.shape
        Ls = cv2.resize(L, (s, s), interpolation=cv2.INTER_AREA if min(h, w) > s else cv2.INTER_LINEAR)
        rgb = lab_to_srgb(Ls, np.zeros(Ls.shape + (2,), np.float32))
        x = (np.clip(rgb, 0, 1) - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].float().to(self.device)

    def predict(self, L, hints=None):
        import cv2
        import torch

        from ..color import srgb_to_lab

        h, w = L.shape
        with torch.inference_mode():
            out = self.net(self._input(L))[0].permute(1, 2, 0).float().cpu().numpy()
        rgb = np.clip(out * STD + MEAN, 0, 1)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return srgb_to_lab(rgb)[..., 1:].astype(np.float32)

    def features(self, L: np.ndarray):
        """ResNet layer3 features (1/16 size) for matching saved colours."""
        import torch

        with torch.inference_mode():
            enc = self.net.layers[0]
            f = enc[:7](self._input(L))[0].float()
        return f / (f.norm(dim=0, keepdim=True) + 1e-6)
