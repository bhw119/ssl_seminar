# -*- coding: utf-8 -*-
"""이미지 인코더(torchvision ResNet)와 분류 헤드. sslexp.model.Net과 같은 인터페이스(features / logits / forward)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ImgNet(nn.Module):
    def __init__(self, n_classes: int, arch: str = "resnet50", pretrained: bool = True, heads=("main",)):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None       # 대회 규정: ImageNet 사전학습은 허용
        net = getattr(torchvision.models, arch)(weights=weights)
        self.embed_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.encoder = net
        self.heads = nn.ModuleDict({h: nn.Linear(self.embed_dim, n_classes) for h in heads})

    def features(self, x):
        return self.encoder(x).float()          # 혼합 정밀도에서도 손실은 float32로 계산한다

    def logits(self, z, head: str = "main", use_bias: bool = True):
        h = self.heads[head]
        z = z.float()
        return torch.nn.functional.linear(z, h.weight.float(), h.bias.float() if use_bias else None)

    def forward(self, x, head: str = "main"):
        return self.logits(self.features(x), head)
