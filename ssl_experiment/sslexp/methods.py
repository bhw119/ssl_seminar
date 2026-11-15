# -*- coding: utf-8 -*-
"""비교 방법 6종의 학습 한 스텝.

supervised   labeled만 사용
pseudolabel  Self-training의 딥러닝 구현(Pseudo-Label). 약한 증강 한 벌, 고신뢰도 예측을 정답처럼 사용
fixmatch     약한 증강 예측을 pseudo-label로, 강한 증강 예측을 그에 맞춤 (고정 임계값)
cgmatch      FixMatch + Count-Gap 3분할(easy: CE / ambiguous: GCE / hard: 제외), 임계값은 EMA
interlude    FixMatch + embedding fusion(교차 배열, 이웃 혼합) + delta consistency loss
scssl        FixMatch + 균형 분류기·expansive 분류기(logit 보정, 클래스별 임계값) + 추론 시 bias 제거

세 최신 기법은 원 논문의 식을 따르되 시계열 입력과 짧은 학습 길이에 맞춘 부분이 있다. 차이는 README의
"원 논문과 다른 점"에 적었다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import augment as A

METHODS = ("supervised", "pseudolabel", "fixmatch", "cgmatch", "interlude", "scssl")
HEADS = {"scssl": ("main", "bal", "exp")}
SC_ANCHOR_C = {"consistent": 4, "uniform": 5, "inverse": 6, "gaussian": 4, "inv_gaussian": 6}


@dataclass
class Cfg:
    iters: int = 6000
    bs: int = 32                 # labeled 배치
    mu: int = 7                  # unlabeled 배치 = mu * bs
    lr: float = 1e-3
    wd: float = 1e-4
    grad_clip: float = 1.0
    ema: float = 0.999
    eval_every: int = 250
    tau: float = 0.95            # FixMatch 계열의 confidence 임계값
    lambda_u: float = 1.0
    strong_ops: tuple = A.STRONG_OPS
    depth: int = 6               # 인코더 Inception 블록 수
    n_filters: int = 32          # 블록의 가지별 필터 수(폭 = 4 * n_filters)
    embed_dim: int = 128
    pl_warmup: float = 0.4       # Pseudo-Label: unlabeled 손실 가중치를 학습의 앞 40% 동안 선형 증가
    cg_warmup: int = 500         # CGMatch: labeled만으로 학습하는 구간 t0
    cg_init: int = 250           # CGMatch: warm-up의 마지막 구간 동안 예측을 모아 Count-Gap 큐를 채운다
    cg_m: float = 0.999          # CGMatch: 임계값 EMA momentum
    cg_q: float = 0.7            # CGMatch: GCE의 q
    il_alpha: float = 0.1        # InterLUDE: 혼합 강도
    il_lambda_dc: float = 1.0    # InterLUDE: delta consistency 가중치
    sc_tau_b: float = 2.0        # SC-SSL: 균형 분류기 logit 보정 계수
    sc_tau_e: float = 4.0        # SC-SSL: expansive 분류기 logit 보정 계수
    sc_rho_max: float = 0.95
    sc_lambda_u: float = 2.0     # SC-SSL: 두 분류기의 unlabeled 손실 가중치
    sc_est_frac: float = 0.15    # SC-SSL: 이 시점에 unlabeled 분포를 추정하고 임계값을 정한다
    sc_alpha: float = 0.005      # SC-SSL: 임계값 감소 폭
    sc_nu: float = 1.0           # SC-SSL: bias가 이 값을 넘는 클래스의 임계값을 낮춘다
    sc_rho_min: float = 0.3


@dataclass
class State:
    K: int
    T: int
    n_u: int
    device: torch.device
    n_l_per_class: np.ndarray
    order: np.ndarray                       # order[k] = labeled가 k번째로 많은 클래스
    head: np.ndarray | None
    aug: object = None                      # 증강 묶음(weak / strong / group_views). 없으면 시계열 증강
    stats: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.aug is None:
            self.aug = A.SeriesAug()
        self.reset_stats()
        self.pl_hist = torch.zeros(self.K, device=self.device)
        # CGMatch
        self.cg_Q = torch.zeros(self.n_u, self.K, device=self.device)
        self.tau_e = 1.0 / self.K
        self.tau_a = 0.0
        # SC-SSL
        n = torch.tensor(self.n_l_per_class, dtype=torch.float32, device=self.device)
        self.dp = torch.log(n / n.sum())
        self.rho_b = torch.full((self.K,), 0.95, device=self.device)
        self.rho_e = torch.full((self.K,), 0.95, device=self.device)
        self.sc_estimated = False

    def reset_stats(self):
        self.stats = dict(loss=0.0, steps=0, n_u=0, sel=0, sel_ok=0, easy=0, easy_ok=0, amb=0, amb_ok=0, hard=0, hard_ok=0)

    def add_sel(self, mask, yhat, yu):
        self.stats["n_u"] += int(mask.numel()); self.stats["sel"] += int(mask.sum())
        self.stats["sel_ok"] += int((mask & (yhat == yu)).sum())
        self.pl_hist += torch.bincount(yhat[mask], minlength=self.K).float()


def _ce(logits, target):
    return F.cross_entropy(logits, target, reduction="none")


def _three_views(model, st, xl, xu):
    B, U = xl.shape[0], xu.shape[0]
    a = st.aug
    z = model.features(torch.cat([a.weak(xl), a.weak(xu), a.strong(xu)]))
    return z[:B], z[B:B + U], z[B + U:]


# --------------------------------------------------------------------------- 기준·고전·표준
def step_supervised(model, st, cfg, t, xl, yl, xu=None, yu=None, upos=None):
    return _ce(model(st.aug.weak(xl)), yl).mean()


def step_pseudolabel(model, st, cfg, t, xl, yl, xu, yu, upos):
    B = xl.shape[0]
    lg = model(torch.cat([st.aug.weak(xl), st.aug.weak(xu)]))
    lg_l, lg_u = lg[:B], lg[B:]
    conf, yhat = lg_u.detach().softmax(1).max(1)
    mask = conf >= cfg.tau
    st.add_sel(mask, yhat, yu)
    w = min(t / max(cfg.pl_warmup * st.T, 1.0), 1.0)
    return _ce(lg_l, yl).mean() + cfg.lambda_u * w * (mask.float() * _ce(lg_u, yhat)).mean()


def step_fixmatch(model, st, cfg, t, xl, yl, xu, yu, upos):
    zl, zw, zs = _three_views(model, st, xl, xu)
    conf, yhat = model.logits(zw).detach().softmax(1).max(1)
    mask = conf >= cfg.tau
    st.add_sel(mask, yhat, yu)
    return _ce(model.logits(zl), yl).mean() + cfg.lambda_u * (mask.float() * _ce(model.logits(zs), yhat)).mean()


# --------------------------------------------------------------------------- CGMatch
def step_cgmatch(model, st, cfg, t, xl, yl, xu, yu, upos):
    t0 = cfg.cg_warmup
    if t <= t0:
        if t > t0 - cfg.cg_init:     # warm-up 끝부분의 예측으로 큐를 채운다
            with torch.no_grad():
                yhat = model(st.aug.weak(xu)).argmax(1)
            st.cg_Q[upos, yhat] += 1.0
        return _ce(model(st.aug.weak(xl)), yl).mean()

    zl, zw, zs = _three_views(model, st, xl, xu)
    lg_w, lg_s = model.logits(zw), model.logits(zs)
    conf, yhat = lg_w.detach().softmax(1).max(1)
    st.cg_Q[upos, yhat] += 1.0
    top2 = st.cg_Q[upos].topk(2, dim=1).values
    cg = top2[:, 0] - top2[:, 1]
    m = cfg.cg_m
    st.tau_e = m * st.tau_e + (1 - m) * float(conf.mean())
    st.tau_a = m * st.tau_a + (1 - m) * float(cg.mean())
    easy = conf >= st.tau_e
    amb = (~easy) & (cg >= st.tau_a)
    hard = ~(easy | amb)

    loss_e = (easy.float() * _ce(lg_s, yhat)).sum() / easy.sum().clamp_min(1)
    q = cfg.cg_q
    pw = lg_w.softmax(1).gather(1, yhat[:, None]).squeeze(1).clamp_min(1e-7)
    ps = lg_s.softmax(1).gather(1, yhat[:, None]).squeeze(1).clamp_min(1e-7)
    gce = (1 - pw.pow(q)) / q + (1 - ps.pow(q)) / q
    loss_a = (amb.float() * gce).sum() / amb.sum().clamp_min(1)
    lam_a = ((t - t0) / max(st.T - t0, 1)) ** 2

    ok = yhat == yu
    st.add_sel(easy | amb, yhat, yu)
    for name, msk in (("easy", easy), ("amb", amb), ("hard", hard)):
        st.stats[name] += int(msk.sum()); st.stats[name + "_ok"] += int((msk & ok).sum())
    return _ce(model.logits(zl), yl).mean() + loss_e + lam_a * loss_a


# --------------------------------------------------------------------------- InterLUDE
def step_interlude(model, st, cfg, t, xl, yl, xu, yu, upos):
    B, mu = xl.shape[0], cfg.mu
    # labeled i와 그 옆의 unlabeled mu개는 같은 약한 증강·같은 강한 증강을 쓴다
    xlw, xuw, xls, xus = st.aug.group_views(xl, xu, mu)
    shp = xlw.shape[1:]
    Xw = torch.cat([xlw[:, None], xuw.reshape(B, mu, *shp)], dim=1)       # (B, 1+mu, ...)
    Xs = torch.cat([xls[:, None], xus.reshape(B, mu, *shp)], dim=1)
    # 교차 배열: i마다 [labeled 약, unlabeled 약 x mu, labeled 강, unlabeled 강 x mu]
    X = torch.stack([Xw, Xs], dim=1).reshape(B * 2 * (1 + mu), *shp)
    Z = model.features(X)
    Zf = (1 - cfg.il_alpha) * Z + cfg.il_alpha * torch.roll(Z, shifts=1, dims=0)     # z'_j = (1-a) z_j + a z_{j-1}, 원형
    lg = model.logits(Zf).view(B, 2, 1 + mu, -1)
    lw_l, ls_l, lw_u, ls_u = lg[:, 0, 0], lg[:, 1, 0], lg[:, 0, 1:], lg[:, 1, 1:]

    loss_l = _ce(lw_l, yl).mean()
    qw = lw_u.reshape(B * mu, -1); qs = ls_u.reshape(B * mu, -1)
    conf, yhat = qw.detach().softmax(1).max(1)
    mask = conf > cfg.tau
    loss_u = (mask.float() * _ce(qs, yhat)).mean()
    d_l = lw_l.softmax(1) - ls_l.softmax(1)
    d_u = (lw_u.softmax(-1) - ls_u.softmax(-1)).mean(dim=1)
    loss_dc = (d_l - d_u).pow(2).sum(1).mean()
    st.add_sel(mask, yhat, yu.view(B, mu).reshape(-1))
    return loss_l + cfg.lambda_u * loss_u + cfg.il_lambda_dc * loss_dc


# --------------------------------------------------------------------------- SC-SSL
def step_scssl(model, st, cfg, t, xl, yl, xu, yu, upos):
    zl, zw, zs = _three_views(model, st, xl, xu)
    # 기본 SSL 손실(FixMatch, main 헤드)
    conf, yhat = model.logits(zw, "main").detach().softmax(1).max(1)
    mask = conf >= cfg.tau
    loss = _ce(model.logits(zl, "main"), yl).mean() + cfg.lambda_u * (mask.float() * _ce(model.logits(zs, "main"), yhat)).mean()
    # 균형 분류기(bal)와 expansive 분류기(exp): logit 보정된 지도 손실 + 클래스별 임계값의 consistency 손실
    for head, tau_adj, rho in (("bal", cfg.sc_tau_b, st.rho_b), ("exp", cfg.sc_tau_e, st.rho_e)):
        loss = loss + _ce(model.logits(zl, head) + tau_adj * st.dp, yl).mean()
        c2, y2 = model.logits(zw, head).detach().softmax(1).max(1)
        m2 = c2 >= rho[y2]
        loss = loss + cfg.sc_lambda_u * (m2.float() * _ce(model.logits(zs, head), y2)).mean()
        if head == "bal":
            st.add_sel(m2, y2, yu)
    return loss


def sc_anchors(K: int, gamma_l: float) -> dict[str, np.ndarray]:
    """후보 분포(위치 k = labeled가 k번째로 많은 클래스). labeled가 균형이면 long-tail 후보는 uniform과 같아 뺀다."""
    k = np.arange(K)
    g = np.exp(-0.5 * (k - (K - 1) / 2) ** 2 / (K / 6))
    out = {"uniform": np.ones(K), "gaussian": g, "inv_gaussian": g.max() + g.min() - g}
    if gamma_l > 1.0 + 1e-6:
        lt = gamma_l ** (-k / max(K - 1, 1))
        out["consistent"] = lt; out["inverse"] = lt[::-1].copy()
    out = {n: np.maximum(v, 1e-12) for n, v in out.items()}      # 클래스가 많으면 inv_gaussian의 가운데가 0이 되어 KL이 inf가 된다
    return {n: v / v.sum() for n, v in out.items()}


@torch.no_grad()
def sc_estimate(model, st, cfg, xu_all: torch.Tensor, bs: int = 2048, prep=None) -> dict:
    """unlabeled 분포 추정: bias를 뺀 균형 분류기 예측으로 클래스별 개수를 세고, KL이 가장 작은 후보 분포를 고른다.
    그 결과로 expansion factor c와 non-head 클래스의 임계값 초깃값을 정한다(원 논문 식 (9), (10))."""
    was = model.training; model.eval()
    cnt = torch.zeros(st.K, device=st.device)
    for i in range(0, xu_all.shape[0], bs):
        xb = xu_all[i:i + bs]
        z = model.features(xb if prep is None else prep(xb))     # prep: 평가용 전처리(이미지의 중앙 crop·정규화 등)
        cnt += torch.bincount(model.logits(z, "bal", use_bias=False).argmax(1), minlength=st.K).float()
    model.train(was)
    p = (cnt.cpu().numpy() + 1.0); p = p / p.sum()
    n = st.n_l_per_class[st.order].astype(float)
    gamma_l = float(n.max() / max(n.min(), 1.0))
    anchors = sc_anchors(st.K, gamma_l)
    p_pos = p[st.order]
    kl = {name: float(np.sum(p_pos * np.log(p_pos / q))) for name, q in anchors.items()}
    best = min(kl, key=kl.get)
    c = SC_ANCHOR_C[best]
    q = anchors[best]; gamma_u = float(q.max() / q.min())
    rb = cfg.sc_rho_max - (c - 4) / 10.0 * min(gamma_u / 50.0, 1.0)
    re = cfg.sc_rho_max - (c - 3) / 5.0 * min(gamma_u / 20.0, 1.0)
    non_head = torch.ones(st.K, dtype=torch.bool, device=st.device)
    if st.head is not None:
        non_head = torch.from_numpy(~st.head).to(st.device)
    st.rho_b[non_head] = max(rb, cfg.sc_rho_min); st.rho_e[non_head] = max(re, cfg.sc_rho_min)
    st.sc_estimated = True
    st.extra["sc"] = dict(anchor=best, kl=kl, c=c, gamma_u=gamma_u, est_dist=p.tolist(), rho_b0=float(rb), rho_e0=float(re))
    return st.extra["sc"]


@torch.no_grad()
def sc_update_thresholds(model, st, cfg) -> None:
    """균형 분류기의 bias가 nu를 넘는 non-head 클래스의 임계값을 alpha만큼 낮춘다(원 논문 식 (8)). 평가 주기마다 한 번 부른다."""
    b = model.heads["bal"].bias
    sel = b > cfg.sc_nu
    if st.head is not None:
        sel = sel & torch.from_numpy(~st.head).to(st.device)
    st.rho_b[sel] = (st.rho_b[sel] - cfg.sc_alpha).clamp_min(cfg.sc_rho_min)
    st.rho_e[sel] = (st.rho_e[sel] - cfg.sc_alpha).clamp_min(cfg.sc_rho_min)


STEPS = dict(supervised=step_supervised, pseudolabel=step_pseudolabel, fixmatch=step_fixmatch,
             cgmatch=step_cgmatch, interlude=step_interlude, scssl=step_scssl)


def lr_at(t: int, T: int, lr: float) -> float:
    """FixMatch의 cosine 감쇠: lr * cos(7 pi t / 16 T)."""
    return lr * math.cos(7 * math.pi * t / (16 * T))
