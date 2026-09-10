"""Trainable MoveNet-inspired multi-person pose network.

Google has not published the exact MultiPose Lightning training graph.  This
module keeps the documented MobileNetV2 + stride-4 FPN and four pose fields,
and adds center-offset and box-size fields required for multi-person decoding.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True)
        )


class Head(nn.Sequential):
    def __init__(self, cin: int, cout: int, bias: float | None = None):
        layers = [ConvBNReLU(cin, cin, 3), nn.Conv2d(cin, cout, 1)]
        super().__init__(*layers)
        if bias is not None:
            nn.init.constant_(self[-1].bias, bias)


class MoveNetMultiPose(nn.Module):
    """MobileNetV2/FPN network whose outputs are at input stride 4."""

    def __init__(self, fpn_channels: int = 96, pretrained: bool = False,
                 num_keypoints: int = 17, pretrained_path: str | Path | None = None):
        super().__init__()
        self.num_keypoints = num_keypoints
        weights = MobileNet_V2_Weights.DEFAULT if pretrained and pretrained_path is None else None
        self.backbone = mobilenet_v2(weights=weights).features
        if pretrained_path is not None:
            self._load_local_backbone(pretrained_path)
        # features 3, 6, 13 and 18 have strides 4, 8, 16 and 32.
        self.lateral = nn.ModuleList([
            nn.Conv2d(24, fpn_channels, 1), nn.Conv2d(32, fpn_channels, 1),
            nn.Conv2d(96, fpn_channels, 1), nn.Conv2d(1280, fpn_channels, 1),
        ])
        self.smooth = nn.ModuleList([ConvBNReLU(fpn_channels, fpn_channels) for _ in range(3)])
        self.heads = nn.ModuleDict(OrderedDict([
            ("center_heatmap", Head(fpn_channels, 1, -2.19)),
            ("keypoint_regression", Head(fpn_channels, num_keypoints * 2)),
            ("keypoint_heatmap", Head(fpn_channels, num_keypoints, -2.19)),
            ("keypoint_offset", Head(fpn_channels, num_keypoints * 2)),
            ("center_offset", Head(fpn_channels, 2)),
            ("box_size", Head(fpn_channels, 2)),
        ]))

    def _load_local_backbone(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            preferred = path / "mobilenet_v2-7ebf99e0.pth"
            candidates = [preferred] if preferred.is_file() else sorted(path.glob("mobilenet_v2*.pth"))
            if not candidates:
                raise FileNotFoundError(f"No mobilenet_v2*.pth found in {path}")
            path = candidates[0]
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        prefix = "features."
        state = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
        if not state:
            # Also accept a checkpoint containing the features module directly.
            state = checkpoint
        incompatible = self.backbone.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Incompatible MobileNetV2 backbone at {path}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
        self.pretrained_source = str(path.resolve())

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        result = []
        wanted = {3, 6, 13, 18}
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in wanted:
                result.append(x)
        return result

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self._features(x)
        p = self.lateral[-1](feats[-1])
        pyramid = [p]
        for level in range(2, -1, -1):
            p = self.lateral[level](feats[level]) + F.interpolate(
                p, size=feats[level].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[level](p)
            pyramid.append(p)
        stride4 = pyramid[-1]
        return {name: head(stride4) for name, head in self.heads.items()}
