# -*- coding: utf-8 -*-
"""합성 데이터로 전체 경로를 짧게 돌려 보는 점검(실데이터·GPU 없이 CPU로 1~2분).

    python -m sslexp.smoke

실험 결과를 내는 용도가 아니다. 코드가 끝까지 도는지, 손실이 유한한지, 결과 파일과 표가 만들어지는지만 본다.
"""
from __future__ import annotations

import math
import tempfile

import numpy as np
import torch

from . import augment as A
from . import data as D
from . import methods as M
from .run import auto_workers, estimate_time, fmt_time, hw_profile, run_grid, step_cost, tables


def synthetic(n_per=420, K=6, T=40, C=22, seed=0) -> D.HLS:
    """클래스마다 계절 곡선의 위상·진폭이 다른 시계열. 지점은 클래스별로 공간에 뭉쳐 있다(블록 분할이 의미를 갖도록)."""
    g = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi, T)[None, :, None]
    xs, ys, xy = [], [], []
    for k in range(K):
        n = n_per + 60 * k
        phase = 0.5 * k; amp = 1.0 + 0.15 * k
        base = amp * np.sin(t + phase) + 0.3 * np.cos(2 * t + k)
        w = g.normal(1.0, 0.2, size=(1, 1, C))
        xs.append((base * w + g.normal(0, 0.6, size=(n, T, C))).astype(np.float32))
        ys.append(np.full(n, k))
        xy.append(g.uniform(0, 300_000, size=(n, 2)))
    x = np.concatenate(xs); y = np.concatenate(ys)
    return D.HLS(x=torch.from_numpy(x), y=torch.from_numpy(y).long(), sample_ids=np.arange(len(y)), xy=np.concatenate(xy))


def main():
    ok = []

    def check(name, cond):
        ok.append(bool(cond)); print(("  ok  " if cond else "  FAIL ") + name)

    hls = synthetic()
    sp = D.block_split(hls.xy, hls.y.numpy(), block_km=25.0, tries=50)
    check("블록 분할: 세 구역이 겹치지 않는다", len(set(sp["train"]) & set(sp["test"])) == 0 and len(set(sp["train"]) & set(sp["val"])) == 0)
    check("블록 분할: 한 블록은 한 구역에만", len(set(sp["block_id"][sp["train"]]) & set(sp["block_id"][sp["test"]])) == 0)
    b = D.make_bundle(hls, sp, min_train=160, verbose=False)
    check("클래스 유지", b.K == 6)

    c0, cA, cB, cBc = (D.build_condition(b, c, 0) for c in ("base", "A", "B", "Bc"))
    check("labeled와 unlabeled가 겹치지 않는다", all(len(set(c["l_idx"]) & set(c["u_idx"])) == 0 for c in (c0, cA, cB, cBc)))
    check("base와 A의 unlabeled 집합이 같다", np.array_equal(c0["u_idx"], cA["u_idx"]))
    check("A의 labeled는 base의 labeled에 포함", set(cA["l_idx"]) <= set(c0["l_idx"]))
    check("테스트 구역 지점은 학습에 쓰이지 않는다", all(len(set(c["l_idx"]) & set(b.test)) + len(set(c["u_idx"]) & set(b.test)) == 0 for c in (c0, cB)))
    o = cB["order"]
    check("B: labeled는 head에서 tail로 줄고 unlabeled는 늘어난다", cB["n_l"][o[0]] > cB["n_l"][o[-1]] and cB["n_u"][o[0]] < cB["n_u"][o[-1]])
    check("Bc: labeled와 unlabeled가 같은 방향", cBc["n_u"][o[0]] > cBc["n_u"][o[-1]])
    check("seed가 다르면 labeled가 다르다", not np.array_equal(D.build_condition(b, "A", 1)["l_idx"], cA["l_idx"]))

    x = b.x[:8]
    xs = A.strong(x)
    check("증강: 모양 유지, 유한", xs.shape == x.shape and torch.isfinite(xs).all() and torch.isfinite(A.weak(x)).all())
    p = A.expand(A.strong_params(2, x.shape[1], x.shape[2], x.device), 4)
    check("증강: 같은 파라미터 공유(시점 mask가 묶음 안에서 같다)", torch.equal(p["keep_t"][0], p["keep_t"][3]) and p["keep_t"].shape[0] == 8)

    an = M.sc_anchors(6, 25.0)
    check("SC-SSL 후보 분포 5종, 합 1", len(an) == 5 and all(abs(v.sum() - 1) < 1e-9 for v in an.values()))
    check("SC-SSL: labeled가 균형이면 후보 3종", len(M.sc_anchors(6, 1.0)) == 3)

    cfg = M.Cfg(iters=240, bs=16, mu=3, eval_every=60, cg_warmup=60, cg_init=30, depth=3, n_filters=8, embed_dim=32, ema=0.98, lr=3e-3)
    with tempfile.TemporaryDirectory() as out:
        run_grid(b, out, conditions=("A", "B"), seeds=(0,), cfg=cfg, device=torch.device("cpu"), verbose=False)
        t = tables(out)
        check("표 생성", {"test_macro_f1", "gain_vs_fixmatch", "gain_vs_selftraining", "tail_f1", "diagnosis"} <= set(t))
        import pandas as pd

        runs = pd.read_csv(__import__("pathlib").Path(out) / "tables" / "runs.csv")
        chance = 1.0 / b.K
        worst = runs.groupby("method").acc.max()
        check(f"모든 방법이 학습된다(조건별 최고 test 정확도 > 우연 {chance:.2f}의 2배): " + ", ".join(f"{m} {v:.2f}" for m, v in worst.items()), (worst > 2 * chance).all())
        check("12회 실행 결과", len(list((__import__("pathlib").Path(out) / "runs").glob("*.json"))) == 12)
        import json, pathlib

        rs = [json.loads(p.read_text()) for p in (pathlib.Path(out) / "runs").glob("*.json")]
        check("손실·점수가 유한", all(np.isfinite(h["loss"]) for r in rs for h in r["history"]) and all(0 <= r["test"]["macro_f1"] <= 1 for r in rs))
        sc = [r for r in rs if r["method"] == "scssl" and r["cond"] == "B"][0]
        check("SC-SSL: 분포 추정 기록", "sc" in sc["extra"] and sc["extra"]["sc"]["anchor"] in M.SC_ANCHOR_C)
        sup = [r for r in rs if r["method"] == "supervised"][0]
        check("Supervised: 증강 전후 예측 변화율 기록", "aug_flip_rate" in sup and 0 <= sup["aug_flip_rate"]["all"] <= 1)
        cg = [r for r in rs if r["method"] == "cgmatch"][0]
        check("CGMatch: 3분할 통계 기록", all(k in cg["history"][-1] for k in ("easy", "amb", "hard")))
        n0 = len(rs)
        with tempfile.TemporaryDirectory() as out2:
            small = M.Cfg(iters=20, bs=8, mu=3, eval_every=10, cg_warmup=6, cg_init=3, depth=3, n_filters=8, embed_dim=32)
            run_grid(b, out2, methods=("supervised", "fixmatch"), conditions=("A",), seeds=(0,), cfg=small, device=torch.device("cpu"), verbose=True)
            check("진행 표시를 켠 실행(verbose=True)", len(list((pathlib.Path(out2) / "runs").glob("*.json"))) == 2)
        run_grid(b, out, conditions=("A", "B"), seeds=(0,), cfg=cfg, device=torch.device("cpu"), verbose=False)
        est = estimate_time(b, out, conditions=("A", "B", "Bc"), seeds=(0,), cfg=cfg, device=torch.device("cpu"), verbose=False)
        check("예상 소요 시간: 끝난 실행의 기록으로 계산, 남은 6회", est["n_left"] == 6 and est["total"] > 0)
        check("스텝당 표본 수: supervised < pseudolabel < fixmatch < interlude", step_cost("supervised", cfg) < step_cost("pseudolabel", cfg) < step_cost("fixmatch", cfg) < step_cost("interlude", cfg))
        check("시간 표기", fmt_time(3725) == "1시간 02분" and fmt_time(125) == "2분" and fmt_time(42) == "42초")
        hw = hw_profile(verbose=False)
        check("런타임 확인: vCPU·RAM", hw["vcpu"] >= 1 and hw["ram_gb"] > 0)
        check("워커 수: GPU 없으면 1", auto_workers(dict(cuda=False, vcpu=8, gpu_free_gb=0), 54) == 1)
        check("워커 수: vCPU 2, GPU 여유 15 GB -> 2", auto_workers(dict(cuda=True, vcpu=2, gpu_free_gb=15.0), 54) == 2)
        check("워커 수: vCPU 12, GPU 여유 15 GB -> 8(상한)", auto_workers(dict(cuda=True, vcpu=12, gpu_free_gb=15.0), 54) == 8)
        check("워커 수: GPU 여유 3 GB -> 1", auto_workers(dict(cuda=True, vcpu=12, gpu_free_gb=3.0), 54) == 1)
        check("워커 수: 남은 실행보다 많지 않다", auto_workers(dict(cuda=True, vcpu=12, gpu_free_gb=40.0), 3) == 3)
        with tempfile.TemporaryDirectory() as out3:
            small = M.Cfg(iters=40, bs=8, mu=3, eval_every=20, cg_warmup=10, cg_init=5, depth=3, n_filters=8, embed_dim=32)
            run_grid(b, out3, methods=("supervised", "pseudolabel", "fixmatch", "cgmatch"), conditions=("A",), seeds=(0,), cfg=small,
                     device=torch.device("cpu"), verbose=True, workers=2)
            par = {p.name: json.loads(p.read_text()) for p in (pathlib.Path(out3) / "runs").glob("*.json")}
            check("병렬 실행(워커 2개): 4회 모두 저장", len(par) == 4)
            with tempfile.TemporaryDirectory() as out4:
                run_grid(b, out4, methods=("fixmatch",), conditions=("A",), seeds=(0,), cfg=small, device=torch.device("cpu"), verbose=False, workers=1)
                seq = json.loads(next((pathlib.Path(out4) / "runs").glob("*.json")).read_text())
                check("병렬과 순차의 결과가 같다(같은 seed)", abs(seq["test"]["macro_f1"] - par["A__fixmatch__seed0.json"]["test"]["macro_f1"]) < 1e-6)
        check("다시 실행하면 건너뛴다", len(list((pathlib.Path(out) / "runs").glob("*.json"))) == n0)
        print("\n" + t["test_macro_f1"].to_string() + "\n\n" + t["gain_vs_fixmatch"].to_string())
    print(f"\n{sum(ok)}/{len(ok)} 통과")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
