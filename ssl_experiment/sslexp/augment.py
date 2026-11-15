# -*- coding: utf-8 -*-
"""시계열 증강. 입력은 표준화된 (B, T, C) 텐서.

weak  : 채널별 크기 변화(±5%) + 작은 잡음
strong: weak보다 큰 크기 변화·잡음 + 시점 masking(구름 결측 모사) + 밴드 drop + 완만한 진폭 왜곡

시간축을 밀거나 늘리는 증강(shift, time warp)은 넣지 않았다. 식생의 계절 시기가 곧 클래스 정보라서
그런 증강은 클래스를 바꿀 수 있기 때문이다. masking은 값을 0(= 표준화 후 평균)으로 둔다.

증강 파라미터를 따로 뽑아 여러 표본에 같이 적용할 수 있게 했다(InterLUDE는 labeled 1개와 그 옆의
unlabeled μ개에 같은 증강을 쓴다). 잡음(jitter)만 표본마다 독립이다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

STRONG_OPS = ("mask", "banddrop", "warp")


def weak_params(n: int, C: int, device, gen: torch.Generator | None = None) -> dict:
    return dict(scale=1.0 + 0.05 * torch.randn(n, 1, C, device=device, generator=gen))


def strong_params(n: int, T: int, C: int, device, gen: torch.Generator | None = None, ops=STRONG_OPS,
                  n_seg: int = 3, seg_min: int = 4, seg_max: int = 12, p_band: float = 0.15) -> dict:
    p = dict(scale=1.0 + 0.10 * torch.randn(n, 1, C, device=device, generator=gen))
    if "mask" in ops:
        keep = torch.ones(n, T, 1, device=device)
        t = torch.arange(T, device=device)[None, :]
        for _ in range(n_seg):
            start = torch.randint(0, T, (n, 1), device=device, generator=gen)
            length = torch.randint(seg_min, seg_max + 1, (n, 1), device=device, generator=gen)
            keep[((t >= start) & (t < start + length)).unsqueeze(-1)] = 0.0
        p["keep_t"] = keep
    if "banddrop" in ops:
        half = C // 2 if C % 2 == 0 else C
        d = (torch.rand(n, 1, half, device=device, generator=gen) >= p_band).float()
        p["keep_c"] = torch.cat([d, d], dim=-1) if half * 2 == C else d   # 중심 픽셀 채널과 주변 평균 채널을 같이 끈다
    if "warp" in ops:
        knots = 1.0 + 0.10 * torch.randn(n, 1, 4, device=device, generator=gen)
        p["warp"] = F.interpolate(knots, size=T, mode="linear", align_corners=True).transpose(1, 2)   # (n, T, 1)
    return p


def expand(p: dict, rep: int) -> dict:
    """파라미터 한 벌을 rep개 표본에 되풀이한다(같은 증강을 여러 표본에 적용)."""
    return {k: v.repeat_interleave(rep, dim=0) for k, v in p.items()}


def apply_weak(x: torch.Tensor, p: dict) -> torch.Tensor:
    return x * p["scale"] + 0.03 * torch.randn_like(x)


def apply_strong(x: torch.Tensor, p: dict) -> torch.Tensor:
    out = x * p["scale"]
    if "warp" in p:
        out = out * p["warp"]
    out = out + 0.08 * torch.randn_like(x)
    if "keep_t" in p:
        out = out * p["keep_t"]
    if "keep_c" in p:
        out = out * p["keep_c"]
    return out


def weak(x: torch.Tensor) -> torch.Tensor:
    return apply_weak(x, weak_params(x.shape[0], x.shape[2], x.device))


def strong(x: torch.Tensor, ops=STRONG_OPS) -> torch.Tensor:
    return apply_strong(x, strong_params(x.shape[0], x.shape[1], x.shape[2], x.device, ops=ops))


class SeriesAug:
    """시계열 증강 묶음. methods.py의 학습 스텝은 이 인터페이스(weak / strong / group_views)만 쓴다.
    이미지 실험(sslimg)은 같은 인터페이스의 이미지 증강을 넣는다."""

    def __init__(self, ops=STRONG_OPS):
        self.ops = tuple(ops)

    def weak(self, x: torch.Tensor) -> torch.Tensor:
        return weak(x)

    def strong(self, x: torch.Tensor) -> torch.Tensor:
        return strong(x, self.ops)

    def group_views(self, xl: torch.Tensor, xu: torch.Tensor, mu: int):
        """InterLUDE용. labeled i와 그 옆의 unlabeled mu개(xu의 i*mu .. i*mu+mu-1)에 같은 약한 증강·같은 강한 증강을 준다.
        반환: (labeled 약, unlabeled 약, labeled 강, unlabeled 강)."""
        B, T, C = xl.shape
        pw, ps = weak_params(B, C, xl.device), strong_params(B, T, C, xl.device, ops=self.ops)
        pw_u, ps_u = expand(pw, mu), expand(ps, mu)
        return apply_weak(xl, pw), apply_weak(xu, pw_u), apply_strong(xl, ps), apply_strong(xu, ps_u)
