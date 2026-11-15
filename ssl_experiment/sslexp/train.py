# -*- coding: utf-8 -*-
"""한 번의 학습(조건 x 방법 x seed)과 평가."""
from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import torch

from . import augment as A
from . import methods as M
from .model import EMA, Net


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> np.ndarray:
    f = np.zeros(K)
    for c in range(K):
        tp = np.sum((y_pred == c) & (y_true == c)); fp = np.sum((y_pred == c) & (y_true != c)); fn = np.sum((y_pred != c) & (y_true == c))
        f[c] = 2 * tp / max(2 * tp + fp + fn, 1)
    return f


def summarize(y_true: np.ndarray, y_pred: np.ndarray, K: int, head: np.ndarray | None = None) -> dict:
    f = f1_scores(y_true, y_pred, K)
    out = dict(acc=float((y_true == y_pred).mean()), macro_f1=float(f.mean()), per_class_f1=f.tolist())
    if head is not None:
        out["head_f1"] = float(f[head].mean()); out["tail_f1"] = float(f[~head].mean())
    return out


@torch.no_grad()
def predict(net, x: torch.Tensor, idx: np.ndarray, head: str = "main", use_bias: bool = True, bs: int = 2048, aug=None) -> np.ndarray:
    net.eval()
    out = []
    ii = torch.as_tensor(idx, device=x.device)
    for i in range(0, len(ii), bs):
        xb = x[ii[i:i + bs]]
        if aug is not None:
            xb = aug(xb)
        out.append(net.logits(net.features(xb), head, use_bias).argmax(1).cpu())
    return torch.cat(out).numpy()


def train_run(bundle, x: torch.Tensor, y: torch.Tensor, cond: dict, method: str, seed: int, cfg: M.Cfg,
              device: torch.device, verbose: bool = True, progress=None) -> dict:
    """x, y는 device에 올린 전체 데이터. cond는 data.build_condition의 결과.
    progress(t, row): 스텝마다 (t, None), 평가 때마다 (t, 평가 기록)으로 불린다(진행 표시용)."""
    assert method in M.METHODS, method
    t_start = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    K, T = bundle.K, cfg.iters
    l_idx = torch.as_tensor(cond["l_idx"], device=device); u_idx = torch.as_tensor(cond["u_idx"], device=device)
    n_u = len(u_idx)
    net = Net(K, in_channels=x.shape[2], heads=M.HEADS.get(method, ("main",)), depth=cfg.depth, n_filters=cfg.n_filters, embed_dim=cfg.embed_dim).to(device)
    ema = EMA(net, cfg.ema)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    st = M.State(K=K, T=T, n_u=n_u, device=device, n_l_per_class=np.asarray(cond["n_l"]), order=np.asarray(cond["order"]), head=cond["head"], aug=A.SeriesAug(cfg.strong_ops))
    step = M.STEPS[method]
    eval_head, eval_bias = ("bal", False) if method == "scssl" else ("main", True)
    U = cfg.bs * cfg.mu
    perm = torch.randperm(n_u, device=device); ptr = 0
    y_val = y[torch.as_tensor(bundle.val, device=device)].cpu().numpy()
    best = dict(f1=-1.0, it=0, sd=None)
    hist = []
    est_iter = int(round(cfg.sc_est_frac * T))

    net.train()
    for t in range(1, T + 1):
        for g in opt.param_groups:
            g["lr"] = M.lr_at(t, T, cfg.lr)
        li = l_idx[torch.randint(len(l_idx), (cfg.bs,), device=device)]
        if method == "supervised":
            loss = step(net, st, cfg, t, x[li], y[li])
        else:
            if ptr + U > n_u:
                perm = torch.randperm(n_u, device=device); ptr = 0
            upos = perm[ptr:ptr + U] if n_u >= U else torch.randint(n_u, (U,), device=device)
            ptr += U
            ui = u_idx[upos]
            loss = step(net, st, cfg, t, x[li], y[li], x[ui], y[ui], upos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(net)
        if progress is not None:
            progress(t, None)
        st.stats["loss"] += float(loss.detach()); st.stats["steps"] += 1

        if method == "scssl" and t == est_iter:
            info = M.sc_estimate(net, st, cfg, x[u_idx])
            if verbose:
                print(f"      [SC-SSL] 추정 분포: {info['anchor']} (c={info['c']}), non-head 임계값 bal {info['rho_b0']:.2f} / exp {info['rho_e0']:.2f}")

        if t % cfg.eval_every == 0 or t == T:
            if method == "scssl" and st.sc_estimated:
                M.sc_update_thresholds(net, st, cfg)
            pv = predict(ema.model, x, bundle.val, eval_head, eval_bias)
            sv = summarize(y_val, pv, K)
            s = st.stats
            row = dict(it=t, loss=s["loss"] / max(s["steps"], 1), val_f1=sv["macro_f1"], val_acc=sv["acc"],
                       util=s["sel"] / max(s["n_u"], 1), pl_acc=s["sel_ok"] / max(s["sel"], 1))
            if method == "cgmatch":
                for nm in ("easy", "amb", "hard"):
                    row[nm] = s[nm]; row[nm + "_acc"] = s[nm + "_ok"] / max(s[nm], 1)
                row["tau_e"], row["tau_a"] = st.tau_e, st.tau_a
            hist.append(row)
            if progress is not None:
                progress(t, row)
            st.reset_stats()
            if sv["macro_f1"] > best["f1"]:
                best = dict(f1=sv["macro_f1"], it=t, sd={k: v.detach().clone() for k, v in ema.model.state_dict().items()})
            net.train()

    ema.model.load_state_dict(best["sd"])
    final = ema.model
    y_te = y[torch.as_tensor(bundle.test, device=device)].cpu().numpy()
    y_u = y[u_idx].cpu().numpy()
    test = summarize(y_te, predict(final, x, bundle.test, eval_head, eval_bias), K, cond["head"])
    u_pred = predict(final, x, cond["u_idx"], eval_head, eval_bias)
    res = dict(method=method, cond=cond["cond"], seed=seed, K=K, n_labeled=int(len(l_idx)), n_unlabeled=int(n_u),
               n_l=np.asarray(cond["n_l"]).tolist(), n_u=np.asarray(cond["n_u"]).tolist(),
               head=None if cond["head"] is None else cond["head"].tolist(),
               best_iter=best["it"], val_f1=best["f1"], test=test,
               u_acc=float((u_pred == y_u).mean()), u_pred_hist=np.bincount(u_pred, minlength=K).tolist(),
               pl_hist=st.pl_hist.cpu().tolist(), history=hist, extra=st.extra, cfg=asdict(cfg),
               seconds=time.time() - t_start)
    if method == "supervised":
        # 증강의 클래스 보존성: 같은 모델이 강한 증강 전후에 다른 클래스를 내는 비율(테스트 구역)
        clean = predict(final, x, bundle.test)
        flip = {}
        for name, ops in (("all", cfg.strong_ops), *[(o, (o,)) for o in cfg.strong_ops], ("weak", None)):
            aug = A.weak if ops is None else (lambda xb, ops=ops: A.strong(xb, ops))
            flip[name] = float((predict(final, x, bundle.test, aug=aug) != clean).mean())
        res["aug_flip_rate"] = flip
    return res
