# -*- coding: utf-8 -*-
"""이미지 증강(FixMatch의 표준 구성). 입력은 device에 올린 (B, 3, S, S) uint8, 출력은 정규화된 (B, 3, crop, crop) float.

weak  : 무작위 crop + 좌우 반전
strong: weak + RandAugment(연산 2개, 세기 무작위) + Cutout

속도를 위해 전부 GPU에서 한다. torchvision의 RandAugment는 배치 전체에 같은 연산을 적용하므로, 배치를 group장씩
나눠 묶음마다 연산을 새로 뽑는다(group=1이면 표본마다 따로 뽑지만 느리다). crop·반전·Cutout은 표본마다 따로다.
"""
from __future__ import annotations

import torch
import torchvision.transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MAGNITUDES = (5, 7, 9, 11, 13, 15)


class ImageAug:
    def __init__(self, crop: int = 160, group: int = 4, cutout: float = 0.5):
        self.crop, self.group, self.cutout = crop, max(1, group), cutout
        self._ra = {m: T.RandAugment(num_ops=2, magnitude=m) for m in MAGNITUDES}
        self._mean = self._std = None

    # ---- 부품
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._mean.device != x.device:
            self._mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1) * 255.0
            self._std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1) * 255.0
        return (x.float() - self._mean) / self._std

    def _crop_params(self, n: int, S: int, device):
        m = S - self.crop
        return dict(oy=torch.randint(0, m + 1, (n,), device=device), ox=torch.randint(0, m + 1, (n,), device=device),
                    flip=torch.rand(n, device=device) < 0.5)

    def _crop_flip(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        B, c = x.shape[0], self.crop
        ar = torch.arange(c, device=x.device)
        ys = p["oy"][:, None] + ar[None]
        xs = p["ox"][:, None] + torch.where(p["flip"][:, None], (c - 1 - ar)[None], ar[None])
        b = torch.arange(B, device=x.device)
        return x[b[:, None, None], :, ys[:, :, None], xs[:, None, :]].permute(0, 3, 1, 2)       # (B, 3, c, c)

    def _rand_augment(self, x: torch.Tensor) -> torch.Tensor:
        m = MAGNITUDES[int(torch.randint(len(MAGNITUDES), (1,)))]
        return self._ra[m](x)

    def _cutout_params(self, n: int, device):
        c = self.crop
        side = (torch.rand(n, device=device) * self.cutout * c).long().clamp_min(1)
        return dict(cy=torch.randint(0, c, (n,), device=device), cx=torch.randint(0, c, (n,), device=device), side=side)

    def _cutout(self, x: torch.Tensor, p: dict) -> torch.Tensor:
        c = self.crop
        ar = torch.arange(c, device=x.device)
        half = (p["side"] // 2)[:, None]
        my = (ar[None] >= p["cy"][:, None] - half) & (ar[None] <= p["cy"][:, None] + half)
        mx = (ar[None] >= p["cx"][:, None] - half) & (ar[None] <= p["cx"][:, None] + half)
        mask = my[:, :, None] & mx[:, None, :]
        return torch.where(mask[:, None], torch.full_like(x, 127), x)

    # ---- methods.py가 쓰는 인터페이스
    def weak(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(self._crop_flip(x, self._crop_params(x.shape[0], x.shape[-1], x.device)))

    def strong(self, x: torch.Tensor) -> torch.Tensor:
        z = self._crop_flip(x, self._crop_params(x.shape[0], x.shape[-1], x.device)).contiguous()
        z = torch.cat([self._rand_augment(z[i:i + self.group]) for i in range(0, z.shape[0], self.group)])
        return self._norm(self._cutout(z, self._cutout_params(z.shape[0], z.device)))

    def group_views(self, xl: torch.Tensor, xu: torch.Tensor, mu: int):
        """InterLUDE용. labeled i와 그 옆의 unlabeled mu장에 같은 약한 증강·같은 강한 증강(같은 crop 위치, 같은 RandAugment 연산, 같은 Cutout)을 준다."""
        B, S, dev = xl.shape[0], xl.shape[-1], xl.device
        rep = lambda p: {k: v.repeat_interleave(mu, dim=0) for k, v in p.items()}
        pw, pc, pk = self._crop_params(B, S, dev), self._crop_params(B, S, dev), self._cutout_params(B, dev)
        xlw, xuw = self._crop_flip(xl, pw), self._crop_flip(xu, rep(pw))
        sl, su = self._crop_flip(xl, pc).contiguous(), self._crop_flip(xu, rep(pc)).contiguous().view(B, mu, 3, self.crop, self.crop)
        out_l, out_u = [], []
        for i in range(B):
            g = self._rand_augment(torch.cat([sl[i:i + 1], su[i]]))       # 묶음 (1 + mu)장에 같은 연산
            out_l.append(g[:1]); out_u.append(g[1:])
        xls = self._cutout(torch.cat(out_l), pk); xus = self._cutout(torch.cat(out_u), rep(pk))
        return self._norm(xlw), self._norm(xuw), self._norm(xls), self._norm(xus)

    def center(self, x: torch.Tensor) -> torch.Tensor:
        """평가용: 가운데 crop + 정규화."""
        o = (x.shape[-1] - self.crop) // 2
        return self._norm(x[:, :, o:o + self.crop, o:o + self.crop])
