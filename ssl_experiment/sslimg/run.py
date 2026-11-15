# -*- coding: utf-8 -*-
"""이미지 실험 실행기: 데이터 준비 -> (조건 x 방법 x seed) 학습 -> 결과 표.

조건
    in    unlabeled = 같은 200종(U_in 26,640장). 공식 설정
    open  unlabeled = U_in + 다른 종(U_out 일부). 닫힌 클래스 가정이 깨지는 조건(선택)

결과는 실행마다 JSON 하나({out}/runs/{조건}__{방법}__seed{S}.json)로 저장하고 이미 있는 실행은 건너뛴다.
이미지 데이터를 GPU에 통째로 올려 쓰므로 한 번에 하나씩 돈다(병렬 워커 없음).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from sslexp import methods as M
from sslexp.run import (COND_LABELS, LABELS, _gain, _jsonable, _md, _ms, _result_line, _save_result, fmt_time, hw_profile, run_path, step_cost)

from . import data as D
from .train import ImgCfg, train_run

DEFAULT_CONDS = ("in",)
DEFAULT_SEEDS = (0, 1, 2)


def prepare(root: str, cache_dir: str, size: int = 176, u_out_n: int = 0, do_download: bool = True, verbose: bool = True,
            archive_dir: str | None = None, source: str = "kaggle") -> D.ImgData:
    """이미지를 준비하고(do_download), 정사각 uint8 캐시를 만든 뒤 읽는다. 캐시가 이미 있으면 내려받기와 변환을 건너뛴다.
    source: 이미지를 받는 곳("kaggle" 또는 "gdrive"). archive_dir: 직접 받아 둔 폴더가 있으면 거기서 쓴다."""
    c = Path(cache_dir)
    need = [f"l_train_{size}", f"val_{size}", f"test_{size}", f"u_train_in_{size}"]
    meta = json.loads((c / f"meta_{size}.json").read_text()) if (c / f"meta_{size}.json").exists() else {}
    have = all(k in meta and (c / f"{k}.u8").exists() for k in need) and (u_out_n == 0 or any(k.startswith(f"u_train_out_{size}_n") for k in meta))
    if not have:
        if do_download:
            D.download(root, with_out=u_out_n > 0, archive_dir=archive_dir, source=source)
        D.build_cache(root, cache_dir, size=size, u_out_n=u_out_n, verbose=verbose)
    return D.load_data(cache_dir, size=size, u_out_n=u_out_n, verbose=verbose)


def benchmark(data: D.ImgData, cfg: ImgCfg, device: torch.device, steps: int = 8) -> float:
    """짧은 시험 학습으로 '이미지 한 장을 한 스텝 처리하는 시간'을 잰다(사전학습 가중치는 받지 않는다)."""
    from .aug import ImageAug
    from .model import ImgNet

    n = cfg.bs * (1 + 2 * cfg.mu)
    amp = bool(cfg.amp and device.type == "cuda")
    net = ImgNet(data.K, cfg.arch, pretrained=False).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    aug = ImageAug(cfg.crop, cfg.aug_group)
    xb = D.to_tensor(data.xu_in[:n], device); yb = torch.randint(data.K, (xb.shape[0],), device=device)
    t0 = None
    for i in range(steps + 3):
        if i == 3:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = torch.nn.functional.cross_entropy(net(aug.strong(xb)), yb)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (steps * xb.shape[0])


def _saved_rate(out: str, cfg: ImgCfg) -> float | None:
    v = []
    for p in (Path(out) / "runs").glob("*.json"):
        r = json.loads(p.read_text())
        if r.get("cfg", {}).get("iters") == cfg.iters and r.get("seconds"):
            v.append(r["seconds"] / (cfg.iters * step_cost(r["method"], cfg)))
    return float(np.median(v)) if v else None


def estimate_time(data: D.ImgData, out: str, methods=M.METHODS, conditions=DEFAULT_CONDS, seeds=DEFAULT_SEEDS,
                  cfg: ImgCfg | None = None, device: torch.device | None = None, verbose: bool = True) -> dict:
    cfg = cfg or ImgCfg()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = [(c, s, m) for c in conditions for s in seeds for m in methods]
    left = [k for k in todo if not run_path(out, k[0], k[2], k[1]).exists()]
    per = _saved_rate(out, cfg) if (Path(out) / "runs").exists() else None
    src = "끝난 실행의 기록"
    if per is None:
        per = benchmark(data, cfg, device); src = "시험 학습 8스텝"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    by_method = {m: per * cfg.iters * step_cost(m, cfg) for m in methods}
    total = sum(by_method[m] for _, _, m in left)
    if verbose:
        print(f"[예상 소요 시간] 남은 {len(left)}회 / 전체 {len(todo)}회 -> 약 {fmt_time(total)}  ({src} 기준, device {device})")
        print("                 1회당: " + ", ".join(f"{LABELS[m]} {fmt_time(v)}" for m, v in by_method.items()))
        print("                 평가 시간은 빠져 있다. 실행이 하나 끝날 때마다 다시 계산한다.")
    return dict(per_sample=per, total=total, by_method=by_method, n_left=len(left), n_total=len(todo))


def run_grid(data: D.ImgData, out: str, methods=M.METHODS, conditions=DEFAULT_CONDS, seeds=DEFAULT_SEEDS,
             cfg: ImgCfg | None = None, device: torch.device | None = None, verbose: bool = True) -> None:
    cfg = cfg or ImgCfg()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (Path(out) / "runs").mkdir(parents=True, exist_ok=True)
    todo = [(c, s, m) for c in conditions for s in seeds for m in methods]
    left = [k for k in todo if not run_path(out, k[0], k[2], k[1]).exists()]
    print(f"[실행] 전체 {len(todo)}회 중 남은 것 {len(left)}회 | device {device} | {cfg.arch} (사전학습 {'사용' if cfg.pretrained else '안 함'}) | iters {cfg.iters} | crop {cfg.crop}px")
    if not left:
        return
    tqdm = None
    if verbose:
        hw_profile()
        estimate_time(data, out, methods, conditions, seeds, cfg, device)
        try:
            from tqdm.auto import tqdm
        except ImportError:
            pass
    xl, yl = D.to_tensor(data.xl, device), torch.from_numpy(data.yl).to(device)
    xu_of = {}

    def unlabeled(cond):
        if cond not in xu_of:
            xu_of.clear()       # 조건이 바뀌면 앞 조건의 unlabeled를 GPU에서 내린다
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if cond == "open":
                if data.xu_out is None:
                    raise ValueError("조건 'open'에는 다른 종 unlabeled가 필요합니다. prepare(..., u_out_n=...)으로 준비하세요.")
                xu_of[cond] = torch.cat([D.to_tensor(data.xu_in, device), D.to_tensor(data.xu_out, device)])
            else:
                xu_of[cond] = D.to_tensor(data.xu_in, device)
        return xu_of[cond]

    units = [cfg.iters * step_cost(m, cfg) for _, _, m in left]
    outer = tqdm(total=sum(units), desc=f"전체 0/{len(left)}회", bar_format="{desc} {percentage:3.0f}%|{bar}| 경과 {elapsed} · 남은 시간 {remaining}") if tqdm else None
    t0 = time.time(); done_units = 0; done_sec = 0.0
    for i, (c, s, m) in enumerate(left, 1):
        cost = step_cost(m, cfg)
        inner = tqdm(total=cfg.iters, desc=f"  {c} · {LABELS[m]} · seed{s}", leave=False,
                     bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} 스텝 · 남은 시간 {remaining}{postfix}") if tqdm else None

        def prog(t, row, inner=inner, cost=cost):
            if inner is None:
                return
            if row is None:
                inner.update(1); outer.update(cost)
            else:
                inner.set_postfix_str(f"val acc {row['val_acc']:.3f}", refresh=False)

        res = train_run(data, xl, yl, unlabeled(c), c, m, s, cfg, device, verbose=verbose and tqdm is None, progress=prog if tqdm else None)
        if inner is not None:
            inner.close()
        _save_result(run_path(out, c, m, s), res)
        done_units += units[i - 1]; done_sec += res["seconds"]
        remain = (done_sec / done_units) * sum(units[i:])
        line = (_result_line(i, len(left), (c, s, m), dict(f1=res["test"]["macro_f1"], acc=res["test"]["acc"], best=res["best_iter"], sec=res["seconds"]))
                + f", 누적 {fmt_time(time.time() - t0)}, 남은 예상 {fmt_time(remain)}")
        if outer is not None:
            outer.set_description_str(f"전체 {i}/{len(left)}회"); outer.write(line)
        else:
            print(line)
    if outer is not None:
        outer.close()
    print(f"[완료] {len(left)}회, 총 {fmt_time(time.time() - t0)}")


def collect(out: str):
    import pandas as pd

    rows = []
    for p in sorted((Path(out) / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        last = [h for h in r["history"] if h["it"] <= r["best_iter"]][-1] if r["history"] else {}
        rows.append(dict(cond=r["cond"], method=r["method"], seed=r["seed"], acc=r["test"]["acc"], macro_f1=r["test"]["macro_f1"],
                         head_acc=r["test"]["head_acc"], tail_acc=r["test"]["tail_acc"], val_acc=r["val_acc"],
                         util=last.get("util", np.nan), pl_head_frac=r.get("pl_head_frac", np.nan), best_iter=r["best_iter"], minutes=r["seconds"] / 60))
    return pd.DataFrame(rows)


def tables(out: str, save: bool = True) -> dict:
    """test_acc(공식 지표인 top-1 정확도) / gain_vs_fixmatch / gain_vs_selftraining / tail_acc(labeled가 적은 절반의 종) / diagnosis."""
    df = collect(out)
    if df.empty:
        print("결과가 아직 없습니다."); return {}
    conds = [c for c in ("in", "open") if c in set(df.cond)]
    methods = [m for m in M.METHODS if m in set(df.method)]
    piv = df.pivot_table(index=["cond", "seed"], columns="method", values="acc")
    res = {"test_acc": _ms(df, "acc", conds, methods),
           "gain_vs_fixmatch": _gain(piv, "fixmatch", ("cgmatch", "interlude", "scssl"), conds),
           "gain_vs_selftraining": _gain(piv, "pseudolabel", ("fixmatch", "cgmatch", "interlude", "scssl"), conds),
           "tail_acc": _ms(df, "tail_acc", conds, methods), "head_acc": _ms(df, "head_acc", conds, methods)}
    diag = df.groupby(["cond", "method"])[["util", "pl_head_frac", "val_acc", "acc", "minutes"]].mean().round(3).reset_index()
    diag["method"] = diag["method"].map(LABELS)
    res["diagnosis"] = diag.rename(columns=dict(util="pseudo-label 활용률", pl_head_frac="pseudo-label 중 head 종의 비율", val_acc="val 정확도", acc="test 정확도", minutes="분/회"))
    if save:
        d = Path(out) / "tables"; d.mkdir(exist_ok=True)
        md = []
        for name, t in res.items():
            t.to_csv(d / f"{name}.csv")
            md.append(f"### {name}\n\n{_md(t)}\n")
        (d / "tables.md").write_text("\n".join(md))
        df.to_csv(d / "runs.csv", index=False)
    return res
