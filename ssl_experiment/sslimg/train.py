# -*- coding: utf-8 -*-
"""이미지 실험의 학습 한 번과 평가. 방법별 손실은 sslexp.methods의 스텝 함수를 그대로 쓰고, 증강·모델·최적화만 이미지용이다."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from sslexp import methods as M
from sslexp.model import EMA
from sslexp.train import summarize

from .aug import ImageAug
from .model import ImgNet


@dataclass
class ImgCfg(M.Cfg):
    iters: int = 4000
    bs: int = 16                 # labeled 배치
    mu: int = 7                  # unlabeled 배치 = mu * bs (FixMatch의 값)
    lr: float = 0.003            # 사전학습된 인코더의 학습률
    head_lr_mult: float = 10.0   # 새로 만든 분류 헤드는 lr * 이 값
    momentum: float = 0.9
    wd: float = 5e-4
    eval_every: int = 250
    grad_clip: float = 5.0
    arch: str = "resnet50"
    pretrained: bool = True
    cache_size: int = 176        # 저장해 둔 정사각 이미지의 한 변
    crop: int = 160              # 학습·평가에 쓰는 crop 크기
    aug_group: int = 4           # RandAugment 연산을 몇 장마다 새로 뽑는가
    amp: bool = True             # 혼합 정밀도(cuda에서만)


@torch.no_grad()
def predict(net, arr: np.ndarray, aug: ImageAug, device, head: str = "main", use_bias: bool = True, bs: int = 256, amp: bool = False) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(arr), bs):
        xb = torch.from_numpy(np.array(arr[i:i + bs])).to(device).permute(0, 3, 1, 2)
        with torch.autocast(device_type=device.type, enabled=amp):
            z = net.features(aug.center(xb))
        out.append(net.logits(z, head, use_bias).argmax(1).cpu())
    return torch.cat(out).numpy()


def train_run(data, xl: torch.Tensor, yl: torch.Tensor, xu: torch.Tensor, cond: str, method: str, seed: int, cfg: ImgCfg,
              device: torch.device, verbose: bool = True, progress=None) -> dict:
    """xl, xu는 device에 올린 (N, 3, S, S) uint8. unlabeled의 레이블은 공개되지 않아 pseudo-label 정확도는 재지 못한다."""
    assert method in M.METHODS, method
    t_start = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    K, T = data.K, cfg.iters
    amp = bool(cfg.amp and device.type == "cuda")
    net = ImgNet(K, cfg.arch, cfg.pretrained, heads=M.HEADS.get(method, ("main",))).to(device)
    ema = EMA(net, cfg.ema)
    opt = torch.optim.SGD([dict(params=net.encoder.parameters(), lr=cfg.lr), dict(params=net.heads.parameters(), lr=cfg.lr * cfg.head_lr_mult)],
                          lr=cfg.lr, momentum=cfg.momentum, nesterov=True, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    aug = ImageAug(cfg.crop, cfg.aug_group)
    n_u = xu.shape[0]
    st = M.State(K=K, T=T, n_u=n_u, device=device, n_l_per_class=np.asarray(data.n_l), order=np.asarray(data.order), head=data.head, aug=aug)
    step = M.STEPS[method]
    eval_head, eval_bias = ("bal", False) if method == "scssl" else ("main", True)
    U = cfg.bs * cfg.mu
    perm = torch.randperm(n_u, device=device); ptr = 0
    yu_unknown = torch.full((U,), -1, dtype=torch.long, device=device)
    best = dict(acc=-1.0, it=0, sd=None)
    hist = []
    est_iter = int(round(cfg.sc_est_frac * T))

    net.train()
    for t in range(1, T + 1):
        for g, base in zip(opt.param_groups, (cfg.lr, cfg.lr * cfg.head_lr_mult)):
            g["lr"] = M.lr_at(t, T, base)
        li = torch.randint(xl.shape[0], (cfg.bs,), device=device)
        with torch.autocast(device_type=device.type, enabled=amp):
            if method == "supervised":
                loss = step(net, st, cfg, t, xl[li], yl[li])
            else:
                if ptr + U > n_u:
                    perm = torch.randperm(n_u, device=device); ptr = 0
                upos = perm[ptr:ptr + U] if n_u >= U else torch.randint(n_u, (U,), device=device)
                ptr += U
                loss = step(net, st, cfg, t, xl[li], yl[li], xu[upos], yu_unknown, upos)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        scaler.step(opt); scaler.update()
        ema.update(net)
        if progress is not None:
            progress(t, None)
        st.stats["loss"] += float(loss.detach()); st.stats["steps"] += 1

        if method == "scssl" and t == est_iter:
            with torch.autocast(device_type=device.type, enabled=amp):
                info = M.sc_estimate(net, st, cfg, xu, bs=256, prep=aug.center)
            if verbose:
                print(f"      [SC-SSL] 추정 분포: {info['anchor']} (c={info['c']}), non-head 임계값 bal {info['rho_b0']:.2f} / exp {info['rho_e0']:.2f}")

        if t % cfg.eval_every == 0 or t == T:
            if method == "scssl" and st.sc_estimated:
                M.sc_update_thresholds(net, st, cfg)
            sv = summarize(data.yv, predict(ema.model, data.xv, aug, device, eval_head, eval_bias, amp=amp), K)
            s = st.stats
            row = dict(it=t, loss=s["loss"] / max(s["steps"], 1), val_acc=sv["acc"], val_f1=sv["macro_f1"], util=s["sel"] / max(s["n_u"], 1))
            if method == "cgmatch":
                tot = max(s["easy"] + s["amb"] + s["hard"], 1)
                row.update(easy=s["easy"] / tot, amb=s["amb"] / tot, hard=s["hard"] / tot, tau_e=st.tau_e, tau_a=st.tau_a)
            hist.append(row)
            if progress is not None:
                progress(t, row)
            st.reset_stats()
            if sv["acc"] > best["acc"]:
                best = dict(acc=sv["acc"], it=t, sd={k: v.detach().clone() for k, v in ema.model.state_dict().items()})
            net.train()

    ema.model.load_state_dict(best["sd"])
    pt = predict(ema.model, data.xt, aug, device, eval_head, eval_bias, amp=amp)
    test = summarize(data.yt, pt, K, data.head)
    hit = (pt == data.yt)
    test["head_acc"] = float(hit[data.head[data.yt]].mean()); test["tail_acc"] = float(hit[~data.head[data.yt]].mean())
    ph = st.pl_hist.cpu().numpy()
    return dict(method=method, cond=cond, seed=seed, K=K, n_labeled=int(xl.shape[0]), n_unlabeled=int(n_u),
                best_iter=best["it"], val_acc=best["acc"], test=test,
                pl_head_frac=float(ph[data.head].sum() / max(ph.sum(), 1.0)), pl_hist=ph.tolist(),
                history=hist, extra=st.extra, cfg=asdict(cfg), seconds=time.time() - t_start)
