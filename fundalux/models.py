"""Transfer-learning backbones with a small screening head."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import IMAGENET_MEAN, IMAGENET_STD
from .inference import ensemble_probabilities  # noqa: F401  (re-exported)


class FundusNet(nn.Module):
    """ImageNet backbone + one linear head.

    Output column 0 is a binary logit (referable DR, or "ungradable" for the
    quality gate). Columns 1.. are DR-grade logits when `num_grades` > 0.
    """

    def __init__(self, arch: str = "efficientnet_b3", num_grades: int = 5, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        import torchvision.models as tvm

        if not hasattr(tvm, arch):
            raise ValueError(f"Unknown architecture: {arch}")
        net = getattr(tvm, arch)(weights="DEFAULT" if pretrained else None)
        norm = nn.Identity()
        if arch.startswith("efficientnet"):
            self.features = net.features
            feat_dim = net.classifier[-1].in_features
        elif arch.startswith("convnext"):
            self.features = net.features
            norm = net.classifier[0]  # ConvNeXt normalises the pooled features before its classifier
            feat_dim = net.classifier[-1].in_features
        elif arch.startswith(("resnet", "resnext", "wide_resnet")):
            self.features = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4)
            feat_dim = net.fc.in_features
        else:
            raise ValueError(f"Unsupported architecture family: {arch} (use efficientnet_*, convnext_* or resnet*)")
        self.arch = arch
        self.num_grades = num_grades
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, 1 + num_grades)
        # Normalisation is part of the model, so every caller (training, the app,
        # the ONNX export) feeds plain 0-255 RGB and cannot get it wrong.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is a batch of RGB images with values 0-255 (uint8 or float), NCHW."""
        return self.features((x.float() / 255.0 - self.mean) / self.std)

    def forward_head(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.norm(self.pool(feats)).flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """RGB uint8 HxWx3 -> uint8 CHW tensor (FundusNet normalises internally)."""
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def save_weights(model: nn.Module, path) -> None:
    """Save a plain state_dict (float32, CPU) so it loads on any torch version."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, str(path))


def load_weights(model: nn.Module, path) -> nn.Module:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    return model.eval()
