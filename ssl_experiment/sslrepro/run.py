from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sslexp.run import COND_LABELS, _md, _save_result, fmt_time, run_path

METHODS = ("supervised", "fixmatch", "cgmatch", "interlude", "scssl")
ABLATIONS = ("interlude_noDC", "interlude_noEF", "cgmatch_easyonly", "cgmatch_noCG", "scssl_uniform", "scssl_origopt", "scssl_origopt_u2")
LABELS = dict(supervised="Supervised only", fixmatch="FixMatch", cgmatch="CGMatch", interlude="InterLUDE", scssl="SC-SSL",
              interlude_noDC="InterLUDE, DC 제거 (EF만)", interlude_noEF="InterLUDE, EF 제거 (DC만)",
              cgmatch_easyonly="CGMatch, ambiguous 손실 제거", cgmatch_noCG="CGMatch, Count-Gap 대신 확신도로 분할",
              scssl_uniform="SC-SSL, unlabeled 분포를 균등으로 가정", scssl_origopt="SC-SSL, 원 코드 최적화 (상수 lr)",
              scssl_origopt_u2="SC-SSL, 원 코드 최적화 + unlabeled ×2")
BASE = dict(interlude_noDC="interlude", interlude_noEF="interlude", cgmatch_easyonly="cgmatch", cgmatch_noCG="cgmatch",
            scssl_uniform="scssl", scssl_origopt="scssl", scssl_origopt_u2="scssl")
DEFAULT_CONDS = ("in",)
DEFAULT_SEEDS = (0, 1, 2)
DATASET = {"in": "semi_inat2020", "open": "semi_inat2020_out"}
ROOT = Path(__file__).resolve().parent.parent

@dataclass
class ReproCfg:
    iters: int = 10000
    bs: int = 32
    uratio: int = 5
    lr: float = 0.01
    wd: float = 5e-4
    ema: float = 0.999
    tau: float = 0.95
    img_size: int = 224
    crop_ratio: float = 0.875
    eval_every: int | None = None
    num_classes: int = 200
    n_ulb: int = 26640
    pretrained: bool = True
    amp: bool = True
    workers: int = 10
    cg_warmup: int | None = None
    cg_queue: int = 32

def method_args(method: str, cfg: ReproCfg) -> dict:
    wu = cfg.cg_warmup if cfg.cg_warmup is not None else (2048 if cfg.iters >= 40000 else max(256, (cfg.iters // 20) // 32 * 32))
    cg = dict(algorithm="cgmatch", warm_up_iter=wu, warm_up_eval_every=max(1, wu // cfg.cg_queue), queue_size=cfg.cg_queue,
              lambda_h=1.0, ema_p=0.999, n_sigma=2, per_class=False, dist_align=True, dist_uniform=False, dist_target="lb")
    il = dict(algorithm="interlude", lambda_rel=1.0, fusion_strength=0.1, rel_T=1.0)
    sc = dict(algorithm="fixmatch", imb_algorithm="scssl", est_dist="from_lb", t_step=0.005, coe=1.0,
              lb_imb_ratio=1, ulb_imb_ratio=1, ulb_num_labels=cfg.n_ulb // cfg.num_classes)
    return dict(supervised=dict(algorithm="fullysupervised"), fixmatch=dict(algorithm="fixmatch"), cgmatch=cg, interlude=il, scssl=sc,
                interlude_noDC={**il, "lambda_rel": 0.0}, interlude_noEF={**il, "fusion_strength": 0.0},
                cgmatch_easyonly={**cg, "use_ambiguous": False}, cgmatch_noCG={**cg, "cg_criterion": "conf"},
                scssl_uniform={**sc, "est_dist": "uniform"}, scssl_origopt={**sc, "orig_optim": True},
                scssl_origopt_u2={**sc, "orig_optim": True, "uratio": 2})[method]

def usb_config(method: str, cond: str, seed: int, cfg: ReproCfg, data: str, out: str) -> dict:
    name = f"{cond}__{method}__seed{seed}"
    ck = Path(out).resolve() / "ckpt"
    ev = cfg.eval_every or (max(500, cfg.iters // 20) if cfg.iters < 40000 else 2000)
    c = dict(save_dir=str(ck), save_name=name, resume=True, load_path=str(ck / name / "latest_model.pth"), overwrite=True,
             use_tensorboard=True, use_wandb=False, use_aim=False,
             num_train_iter=cfg.iters, epoch=min(40, cfg.iters), num_eval_iter=ev, num_log_iter=min(200, ev),
             num_labels=3959, batch_size=cfg.bs, eval_batch_size=256, uratio=cfg.uratio,
             hard_label=True, T=0.5, p_cutoff=cfg.tau, ulb_loss_ratio=1.0, ema_m=cfg.ema,
             img_size=cfg.img_size, crop_ratio=cfg.crop_ratio,
             optim="SGD", lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd, layer_decay=1.0,
             amp=cfg.amp, clip=0.0, use_cat=True,
             net="resnet50_in", net_from_name=False, use_pretrain=cfg.pretrained, pretrain_path="",
             data_dir=str(Path(data).resolve()), dataset=DATASET[cond], num_classes=cfg.num_classes, train_sampler="RandomSampler",
             num_workers=cfg.workers, seed=seed, world_size=1, rank=0,
             multiprocessing_distributed=False, distributed=False,
             dist_url="tcp://127.0.0.1:10601", dist_backend="nccl", gpu=0, include_lb_to_ulb=False)
    c.update(method_args(method, cfg))
    return c

def run_one(method: str, cond: str, seed: int, cfg: ReproCfg, data: str, out: str) -> dict:
    import yaml

    from .evaluate import evaluate_run

    c = usb_config(method, cond, seed, cfg, data, out)
    d = Path(c["save_dir"]) / c["save_name"]
    d.mkdir(parents=True, exist_ok=True)
    cp = d / "config.yaml"
    yaml.safe_dump(c, open(cp, "w"), sort_keys=False)
    t0 = time.time()
    if not (d / "DONE").exists():
        with open(d / "train.log", "a") as f:
            rc = subprocess.run([sys.executable, "-m", "sslrepro.train", "--c", str(cp)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f"{c['save_name']} 학습 실패 (rc={rc}). 로그: {d / 'train.log'}")
        (d / "DONE").touch()
    sec = time.time() - t0
    res = evaluate_run(str(cp), method, cond, seed)
    res.update(cfg=asdict(cfg), seconds=sec)
    return res

def run_grid(data: str, out: str, methods=METHODS + ABLATIONS, conditions=DEFAULT_CONDS, seeds=DEFAULT_SEEDS,
             cfg: ReproCfg | None = None) -> None:
    cfg = cfg or ReproCfg()
    (Path(out) / "runs").mkdir(parents=True, exist_ok=True)
    todo = [(c, s, m) for c in conditions for s in seeds for m in methods]
    left = [k for k in todo if not run_path(out, k[0], k[2], k[1]).exists()]
    print(f"[실행] 전체 {len(todo)}회 중 남은 것 {len(left)}회 | iters {cfg.iters} | {cfg.img_size}px | 배치 {cfg.bs} / {cfg.bs * cfg.uratio}")
    t0 = time.time()
    for i, (c, s, m) in enumerate(left, 1):
        res = run_one(m, c, s, cfg, data, out)
        _save_result(run_path(out, c, m, s), res)
        print(f"[{i}/{len(left)}] {c} · {LABELS[m]} · seed{s}: test acc {res['test']['acc']:.3f}, best {res['best_iter']} 스텝, "
              f"{fmt_time(res['seconds'])}, 누적 {fmt_time(time.time() - t0)}")
    print(f"[완료] {len(left)}회, 총 {fmt_time(time.time() - t0)}")

def collect(out: str):
    import pandas as pd

    rows = []
    for p in sorted((Path(out) / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        rows.append(dict(cond=r["cond"], method=r["method"], seed=r["seed"], acc=r["test"]["acc"], macro_f1=r["test"]["macro_f1"],
                         head_acc=r["test"]["head_acc"], tail_acc=r["test"]["tail_acc"], top5_acc=r["test"]["top5_acc"],
                         val_acc=r["val_acc"], best_iter=r["best_iter"], last_acc=r["extra"]["last"]["test_acc"],
                         minutes=r["seconds"] / 60))
    return pd.DataFrame(rows)

def _ms(df, value: str, conds, methods):
    import pandas as pd

    g = df.groupby(["method", "cond"])[value].agg(["mean", "std", "count"])
    t = pd.DataFrame(index=[LABELS[m] for m in methods], columns=[COND_LABELS.get(c, c) for c in conds], dtype=object)
    for m in methods:
        for c in conds:
            if (m, c) in g.index:
                a = g.loc[(m, c)]
                t.loc[LABELS[m], COND_LABELS.get(c, c)] = f"{a['mean']:.3f} ± {0.0 if np.isnan(a['std']) else a['std']:.3f}" + ("" if a["count"] >= 3 else f" (n={int(a['count'])})")
    return t.fillna("")

def _gain(df, value: str, pairs, conds):
    import pandas as pd

    piv = df.pivot_table(index=["cond", "seed"], columns="method", values=value)
    t = pd.DataFrame(index=[LABELS[m] for m, _ in pairs if m in piv.columns], columns=[COND_LABELS.get(c, c) for c in conds], dtype=object)
    for m, b in pairs:
        if m not in piv.columns or b not in piv.columns:
            continue
        for c in conds:
            if c in piv.index.get_level_values(0):
                d = (piv.loc[c, m] - piv.loc[c, b]).dropna()
                if len(d):
                    t.loc[LABELS[m], COND_LABELS.get(c, c)] = f"{d.mean():+.3f} ± {0.0 if len(d) < 2 else d.std():.3f}"
    return t.fillna("")

def _extra(out: str, key: str, methods):
    import pandas as pd

    rows = {}
    for p in sorted((Path(out) / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        if r["method"] in methods and key in r["extra"]:
            rows.setdefault((r["method"], r["cond"]), []).append(r["extra"][key])
    t = {}
    for (m, c), v in rows.items():
        t[f"{LABELS[m]} · {COND_LABELS.get(c, c)}"] = {k: f"{np.mean([x[k] for x in v]):.3f}" + ("" if len(v) >= 3 else f" (n={len(v)})") for k in v[0]}
    return pd.DataFrame(t).T

def tables(out: str, save: bool = True) -> dict:
    df = collect(out)
    if df.empty:
        print("결과가 아직 없습니다.")
        return {}
    conds = [c for c in ("in", "open") if c in set(df.cond)]
    main = [m for m in METHODS if m in set(df.method)]
    abl = [m for m in ABLATIONS if m in set(df.method)]
    res = {"test_acc": _ms(df, "acc", conds, main),
           "gain_vs_fixmatch": _gain(df, "acc", [(m, "fixmatch") for m in ("cgmatch", "interlude", "scssl") if m in main], conds),
           "tail_acc": _ms(df, "tail_acc", conds, main), "head_acc": _ms(df, "head_acc", conds, main)}
    if abl:
        res["ablation_test_acc"] = _ms(df, "acc", conds, [m for m in METHODS if m in set(BASE[a] for a in abl)] + abl)
        res["ablation_gain_vs_full"] = _gain(df, "acc", [(a, BASE[a]) for a in abl], conds)
    if any(m.startswith("scssl") for m in set(df.method)):
        res["scssl_heads"] = _extra(out, "heads_test_acc", ("scssl", "scssl_uniform"))
    if "interlude" in set(df.method):
        res["interlude_ef_at_eval"] = _extra(out, "ef_at_eval_test_acc", ("interlude",))
    diag = df.groupby(["cond", "method"])[["val_acc", "acc", "last_acc", "top5_acc", "best_iter", "minutes"]].mean().round(3).reset_index()
    diag["method"] = diag["method"].map(LABELS)
    res["diagnosis"] = diag.rename(columns=dict(val_acc="val 정확도", acc="test 정확도", last_acc="마지막 스텝 test 정확도",
                                                top5_acc="test top-5", best_iter="선택된 스텝", minutes="분/회"))
    if save:
        d = Path(out) / "tables"
        d.mkdir(exist_ok=True)
        md = []
        for name, t in res.items():
            t.to_csv(d / f"{name}.csv")
            md.append(f"### {name}\n\n{_md(t)}\n")
        (d / "tables.md").write_text("\n".join(md))
        df.to_csv(d / "runs.csv", index=False)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/semi-inat-2020")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS + ABLATIONS))
    ap.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--iters", type=int, default=ReproCfg.iters)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    run_grid(a.data, a.out, a.methods, a.conditions, a.seeds, ReproCfg(iters=a.iters))
    for name, t in tables(a.out).items():
        print(f"\n== {name}\n{t.to_string()}")

if __name__ == "__main__":
    main()
