"""
Perception module — TRAP formula: f(x)

Frozen backbone (DINOv2-small or ResNet-50) that maps an image to
(logits, feature_vector).  No gradient updates after load.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


@dataclass
class PerceptionOutput:
    logits: torch.Tensor        # (N, 100)
    probs: torch.Tensor         # (N, 100) softmax
    features: torch.Tensor      # (N, D)  penultimate-layer embedding
    preds: torch.Tensor         # (N,)    argmax predictions


class PerceptionModel:
    """Wrapper around a frozen image classifier."""

    def __init__(
        self,
        backbone: str = 'resnet50',
        num_classes: int = 100,
        checkpoint: Optional[Path] = None,
        device: str = 'cpu',
    ):
        self.device = device
        self.backbone_name = backbone
        self.model, self.feature_dim = self._build(backbone, num_classes)
        if checkpoint is not None:
            self.model.load_state_dict(
                torch.load(checkpoint, map_location=device)
            )
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    def _build(self, backbone: str, num_classes: int):
        if backbone == 'resnet50':
            m = tvm.resnet50(weights=None)
            m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            m.maxpool = nn.Identity()
            feat_dim = m.fc.in_features
            m.fc = nn.Linear(feat_dim, num_classes)
            return m, feat_dim
        elif backbone == 'dinov2':
            # DINOv2-small — requires internet or local cache
            m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            feat_dim = m.embed_dim
            head = nn.Linear(feat_dim, num_classes)
            # wrap: forward returns (logits, features)
            return _DINOv2Wrapper(m, head), feat_dim
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> PerceptionOutput:
        """
        Args:
            images: (N, C, H, W) tensor, already normalised

        Returns:
            PerceptionOutput with logits, probs, features, preds
        """
        images = images.to(self.device)

        if self.backbone_name == 'dinov2':
            features, logits = self.model(images)
        else:
            # Hook the penultimate layer to capture features
            features_list: list[torch.Tensor] = []

            def hook(module, inp, out):
                features_list.append(out.detach().flatten(1))

            h = self.model.avgpool.register_forward_hook(hook)
            logits = self.model(images)
            h.remove()
            features = features_list[0]

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        return PerceptionOutput(
            logits=logits.cpu(),
            probs=probs.cpu(),
            features=features.cpu(),
            preds=preds.cpu(),
        )


class _DINOv2Wrapper(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return features, logits
