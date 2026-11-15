# -*- coding: utf-8 -*-
"""시계열 인코더(InceptionTime 계열)와 분류 헤드.

모든 방법이 같은 인코더를 쓴다. 헤드는 방법에 따라 1개("main") 또는 3개(SC-SSL: "main", "bal", "exp").
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self, c_in: int, n_filters: int = 32, kernel_sizes=(9, 19, 39), bottleneck: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(c_in, bottleneck, 1, bias=False) if c_in > 1 else nn.Identity()
        cb = bottleneck if c_in > 1 else c_in
        self.convs = nn.ModuleList([nn.Conv1d(cb, n_filters, k, padding=k // 2, bias=False) for k in kernel_sizes])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(c_in, n_filters, 1, bias=False))
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.bottleneck(x)
        out = torch.cat([conv(z) for conv in self.convs] + [self.pool(x)], dim=1)
        return self.act(self.bn(out))


class Encoder(nn.Module):
    """(B, T, C) -> (B, embed_dim). Inception 블록 depth개, 3개마다 residual, 시간축 평균(GAP)."""

    def __init__(self, in_channels: int = 22, embed_dim: int = 128, depth: int = 6, n_filters: int = 32,
                 kernel_sizes=(9, 19, 39), bottleneck: int = 32):
        super().__init__()
        width = n_filters * (len(kernel_sizes) + 1)
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        c = in_channels
        for i in range(depth):
            self.blocks.append(InceptionBlock(c, n_filters, kernel_sizes, bottleneck))
            if i % 3 == 0:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(c, width, 1, bias=False), nn.BatchNorm1d(width)))
            c = width
        self.proj = nn.Linear(width, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        z = x.transpose(1, 2)
        res = None
        for i, blk in enumerate(self.blocks):
            if i % 3 == 0:
                res = self.shortcuts[i // 3](z)
            z = blk(z)
            if i % 3 == 2:
                z = torch.relu(z + res)
        return self.proj(z.mean(dim=2))


class Net(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 22, heads=("main",), **enc_kw):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, **enc_kw)
        self.heads = nn.ModuleDict({h: nn.Linear(self.encoder.embed_dim, n_classes) for h in heads})

    def features(self, x):
        return self.encoder(x)

    def logits(self, z, head: str = "main", use_bias: bool = True):
        h = self.heads[head]
        return h(z) if use_bias else z @ h.weight.t()

    def forward(self, x, head: str = "main"):
        return self.logits(self.features(x), head)


class EMA:
    """평가에 쓰는 지수이동평균 모델(FixMatch 계열의 관례)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))    # 학습 초반에는 평균 구간을 짧게 둔다
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, bm in zip(self.model.buffers(), model.buffers()):
            be.copy_(bm)
