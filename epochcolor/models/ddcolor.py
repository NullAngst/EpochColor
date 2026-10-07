"""DDColor (Kang et al., ICCV 2023) behind the adapter interface.

The network takes a grey RGB image at 512x512 (the luma, with a and b set to
zero, converted back to RGB) and returns a/b in Lab units. The architecture
code is vendored unchanged in ddcolor_arch/, Apache-2.0.
"""

from __future__ import annotations

import numpy as np

from .base import ColorModel, ModelInfo


class DDColorAdapter(ColorModel):
    info = ModelInfo(
        name="ddcolor",
        mode="automatic",
        description="DDColor, automatic.",
        takes_hints=False,
        needs_weights=True,
        needs_torch=True,
        license="see model card",
    )

    def __init__(self, weights: str | None = None, info: ModelInfo | None = None, params=None):
        self.weights = weights
        if info is not None:
            self.info = info
        p = dict(params or {})
        self.model_size = p.get("model_size", "large")
        self.input_size = int(p.get("input_size", 512))
        self.net = None
        self.device = "cpu"

    def build(self):
        from .ddcolor_arch.model import DDColor

        if self.model_size not in ("tiny", "large"):
            raise RuntimeError(f"DDColor model_size must be tiny or large, not {self.model_size}")
        return DDColor(
            encoder_name="convnext-t" if self.model_size == "tiny" else "convnext-l",
            decoder_name="MultiScaleColorDecoder",
            input_size=[self.input_size, self.input_size],
            num_output_channels=2, last_norm="Spectral", do_normalize=False,
            num_queries=100, num_scales=3, dec_layers=9,
        )

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
            raise RuntimeError(f"{path.name} does not fit DDColor-{self.model_size}: {e}") from None
        # mean/std are buffers set in the constructor; older checkpoints lack them
        missing = [k for k in missing if k not in ("mean", "std")]
        if missing or unexpected:
            hint = " (is model_size right? tiny and large differ)" if len(missing) > 20 else ""
            raise RuntimeError(f"{path.name} does not fit DDColor-{self.model_size}: missing "
                               f"{len(missing)} keys, unexpected {len(unexpected)}{hint}")
        self.net = net.to(device).eval()
        self.device = device

    def _input(self, L: np.ndarray):
        import cv2
        import torch

        from ..color import lab_to_srgb

        s = self.input_size
        Ls = cv2.resize(L, (s, s), interpolation=cv2.INTER_AREA if L.shape[0] > s else cv2.INTER_CUBIC)
        rgb = lab_to_srgb(Ls, np.zeros(Ls.shape + (2,), np.float32))
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None].float().to(self.device)

    def predict(self, L, hints=None):
        import cv2
        import torch

        h, w = L.shape
        with torch.inference_mode():
            ab = self.net(self._input(L))[0].permute(1, 2, 0).float().cpu().numpy()
        return cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    def features(self, L: np.ndarray):
        """ConvNeXt stage 3 features (1/16 size) for matching saved colours."""
        import torch

        with torch.inference_mode():
            x = self.net.normalize(self._input(L))
            self.net.encoder(x)
            f = self.net.encoder.hooks[2].feature[0].float()
        return f / (f.norm(dim=0, keepdim=True) + 1e-6)
