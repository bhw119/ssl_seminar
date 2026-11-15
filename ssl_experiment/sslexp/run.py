# -*- coding: utf-8 -*-
"""실험 실행기: 데이터 준비 -> (조건 x 방법 x seed) 학습 -> 결과 표.

결과는 실행마다 JSON 하나({out}/runs/{조건}__{방법}__seed{S}.json)로 저장하고, 이미 있는 실행은 건너뛴다.
Colab이 끊겨도 같은 셀을 다시 실행하면 이어서 돈다. 설정(iters 등)을 바꿀 때는 out 폴더를 새로 잡는다.

CLI 예:
    python -m sslexp.run --csv /path/HLS_TimeSeries_2022.csv --out ./results
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import methods as M
from .train import train_run

DEFAULT_CONDS = ("base", "A", "B")
DEFAULT_SEEDS = (0, 1, 2)
LABELS = dict(supervised="Supervised only", pseudolabel="Self-training", fixmatch="FixMatch",
              cgmatch="CGMatch", interlude="InterLUDE", scssl="SC-SSL")
COND_LABELS = dict(base="기준 조건", A="조건 A (극소량 레이블)", B="조건 B (분포 불일치)", Bc="조건 Bc (long-tail, 일치)",
                   **{"in": "같은 종 unlabeled", "open": "다른 종 포함 (open-set)"})


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"JSON으로 저장할 수 없는 값: {type(o)}")


def prepare(csv: str, dense_cache: str | None = None, xy_cache: str | None = None, work_dir: str = ".",
            split: str = "block", block_km: float = 25.0, min_train: int = 300, split_seed: int = 0, verbose: bool = True) -> D.Bundle:
    """데이터를 읽고 분할한 뒤 표본이 부족한 클래스를 빼고 표준화한다. 분할은 split_seed로 고정되어 모든 실행이 같은 테스트 구역을 쓴다."""
    hls = D.load_hls(csv, dense_cache, xy_cache, work_dir)
    y = hls.y.numpy()
    sp = D.block_split(hls.xy, y, block_km=block_km, seed=split_seed) if split == "block" else D.random_split(y, seed=split_seed)
    return D.make_bundle(hls, sp, min_train=min_train, verbose=verbose)


def run_path(out: str, cond: str, method: str, seed: int) -> Path:
    return Path(out) / "runs" / f"{cond}__{method}__seed{seed}.json"


def step_cost(method: str, cfg: M.Cfg) -> int:
    """한 스텝에 인코더를 통과하는 표본 수. 실행 시간이 대체로 이 값에 비례한다."""
    b, mu = cfg.bs, cfg.mu
    return {"supervised": b, "pseudolabel": b * (1 + mu), "interlude": 2 * b * (1 + mu)}.get(method, b * (1 + 2 * mu))


def fmt_time(sec: float) -> str:
    sec = max(float(sec), 0.0)
    h, m = int(sec // 3600), int(sec % 3600 // 60)
    return f"{h}시간 {m:02d}분" if h else (f"{m}분" if m else f"{int(sec)}초")


def _sec_per_sample_saved(out: str, cfg: M.Cfg) -> float | None:
    """이미 끝난 실행의 기록에서 '표본 하나를 한 스텝 처리하는 시간'을 구한다(같은 iters의 실행만)."""
    v = []
    for p in (Path(out) / "runs").glob("*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if r.get("cfg", {}).get("iters") == cfg.iters and r.get("seconds"):
            v.append(r["seconds"] / (cfg.iters * step_cost(r["method"], M.Cfg(**{k: r["cfg"][k] for k in ("bs", "mu")}))))
    return float(np.median(v)) if v else None


def benchmark(bundle: D.Bundle, cfg: M.Cfg, device: torch.device, steps: int = 20) -> float:
    """짧은 시험 학습으로 '표본 하나를 한 스텝 처리하는 시간'을 잰다(실제 실행 전 예상 시간 계산용)."""
    from .model import Net

    n = cfg.bs * (1 + 2 * cfg.mu)
    net = Net(bundle.K, in_channels=bundle.x.shape[2], depth=cfg.depth, n_filters=cfg.n_filters, embed_dim=cfg.embed_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr)
    xb = bundle.x[:n].to(device); yb = torch.randint(bundle.K, (xb.shape[0],), device=device)
    t0 = None
    for i in range(steps + 5):
        if i == 5:      # 앞의 몇 스텝은 준비 시간이 섞여 있어 뺀다
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        loss = torch.nn.functional.cross_entropy(net(xb), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / (steps * xb.shape[0])


def estimate_time(bundle: D.Bundle, out: str, methods=M.METHODS, conditions=DEFAULT_CONDS, seeds=DEFAULT_SEEDS,
                  cfg: M.Cfg | None = None, device: torch.device | None = None, verbose: bool = True) -> dict:
    """남은 실행의 예상 소요 시간. 끝난 실행이 있으면 그 기록으로, 없으면 짧은 시험 학습으로 계산한다."""
    cfg = cfg or M.Cfg()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    todo = [(c, s, m) for c in conditions for s in seeds for m in methods]
    left = [k for k in todo if not run_path(out, k[0], k[2], k[1]).exists()]
    per = _sec_per_sample_saved(out, cfg) if (Path(out) / "runs").exists() else None
    src = "끝난 실행의 기록"
    if per is None:
        per = benchmark(bundle, cfg, device); src = "시험 학습 20스텝"
    by_method = {m: per * cfg.iters * step_cost(m, cfg) for m in methods}
    total = sum(by_method[m] for _, _, m in left)
    if verbose:
        print(f"[예상 소요 시간] 남은 {len(left)}회 / 전체 {len(todo)}회 -> 약 {fmt_time(total)}  ({src} 기준, device {device})")
        print("                 1회당: " + ", ".join(f"{LABELS[m]} {fmt_time(v)}" for m, v in by_method.items()))
        print("                 평가·저장 시간은 빠져 있어 실제로는 조금 더 걸린다. 실행이 하나 끝날 때마다 다시 계산한다.")
    return dict(per_sample=per, total=total, by_method=by_method, n_left=len(left), n_total=len(todo))


def hw_profile(verbose: bool = True) -> dict:
    """런타임 확인: GPU 종류와 여유 메모리, vCPU 수, RAM."""
    hw = dict(cuda=torch.cuda.is_available(), gpu="없음", gpu_total_gb=0.0, gpu_free_gb=0.0, vcpu=os.cpu_count() or 1, ram_gb=0.0)
    if hw["cuda"]:
        free, total = torch.cuda.mem_get_info()
        hw.update(gpu=torch.cuda.get_device_name(0), gpu_total_gb=total / 2 ** 30, gpu_free_gb=free / 2 ** 30)
    try:
        import psutil

        hw["ram_gb"] = psutil.virtual_memory().total / 2 ** 30
    except ImportError:
        try:
            hw["ram_gb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        except (ValueError, OSError):
            pass
    if verbose:
        g = f"{hw['gpu']} (전체 {hw['gpu_total_gb']:.0f} GB, 여유 {hw['gpu_free_gb']:.1f} GB)" if hw["cuda"] else "없음"
        print(f"[런타임] GPU {g} | vCPU {hw['vcpu']} | RAM {hw['ram_gb']:.0f} GB")
    return hw


def auto_workers(hw: dict, n_left: int, mem_per_worker_gb: float = 1.5, max_workers: int = 8) -> int:
    """동시에 돌릴 학습 수. 워커 하나가 CPU 코어 하나와 GPU 메모리 mem_per_worker_gb를 쓴다고 보고 둘 중 작은 쪽에 맞춘다.
    GPU가 없으면 1(CPU에서는 한 학습이 이미 코어를 다 쓴다)."""
    if not hw["cuda"]:
        return 1
    by_cpu = max(1, hw["vcpu"])
    by_mem = int(max(hw["gpu_free_gb"] - 1.0, 0.0) // mem_per_worker_gb)
    return max(1, min(by_cpu, by_mem, n_left, max_workers))


def _save_result(p: Path, res: dict) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, default=_jsonable, ensure_ascii=False))
    os.replace(tmp, p)


def _result_line(i: int, n: int, key, res: dict) -> str:
    c, s, m = key
    return (f"  [{i}/{n}] {c:>4} {LABELS[m]:<15} seed{s} | test macro-F1 {res['f1']:.3f} acc {res['acc']:.3f} "
            f"(best it {res['best']}) | {fmt_time(res['sec'])}")


def _worker(wid: int, bundle: D.Bundle, out: str, cfg: M.Cfg, device_str: str, u_cap, tasks, msgs, threads: int) -> None:
    """병렬 워커. 작업 큐에서 (조건, seed, 방법)을 하나씩 꺼내 학습하고 결과를 저장한다. 진행 상황은 msgs로 주 프로세스에 보낸다
    (워커의 print는 노트북 셀에 보이지 않는다)."""
    import queue
    import traceback

    try:
        torch.set_num_threads(max(1, threads))
        device = torch.device(device_str)
        x, y = bundle.x.to(device), bundle.y.to(device)
        while True:
            try:
                key = tasks.get(timeout=3)
            except queue.Empty:
                break
            c, s, m = key
            p = run_path(out, c, m, s)
            if p.exists():
                msgs.put(("skip", wid, key)); continue
            msgs.put(("start", wid, key))

            def prog(t, row, key=key):
                if row is None:
                    if t % 20 == 0 or t == cfg.iters:
                        msgs.put(("step", wid, key, t))
                else:
                    msgs.put(("eval", wid, key, float(row["val_f1"])))

            try:
                res = train_run(bundle, x, y, D.build_condition(bundle, c, s, u_cap=u_cap), m, s, cfg, device, verbose=False, progress=prog)
                _save_result(p, res)
                msgs.put(("done", wid, key, dict(f1=res["test"]["macro_f1"], acc=res["test"]["acc"], best=res["best_iter"], sec=res["seconds"])))
            except Exception:
                msgs.put(("error", wid, key, traceback.format_exc()))
    except Exception:
        import traceback as tb

        msgs.put(("error", wid, None, tb.format_exc()))
    finally:
        msgs.put(("exit", wid))


def _run_parallel(bundle, out, cfg, device, u_cap, left, n_workers, tqdm) -> None:
    import queue

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")          # CUDA는 fork된 프로세스에서 쓸 수 없다
    bundle.x.share_memory_(); bundle.y.share_memory_()      # 워커들이 입력 데이터를 복사하지 않고 같이 본다
    tasks, msgs = ctx.Queue(), ctx.Queue()
    for k in left:
        tasks.put(k)
    threads = max(1, (os.cpu_count() or 1) // n_workers)
    procs = [ctx.Process(target=_worker, args=(w, bundle, out, cfg, str(device), u_cap, tasks, msgs, threads), daemon=True) for w in range(n_workers)]
    for p in procs:
        p.start()
    units = {k: cfg.iters * step_cost(k[2], cfg) for k in left}
    total = sum(units.values())
    outer = tqdm(total=total, desc=f"전체 0/{len(left)}회", position=0,
                 bar_format="{desc} {percentage:3.0f}%|{bar}| 경과 {elapsed} · 남은 시간 {remaining}") if tqdm else None
    bars = [tqdm(total=cfg.iters, desc=f"  워커 {w + 1}: 준비 중", position=w + 1, leave=False,
                 bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} 스텝{postfix}") for w in range(n_workers)] if tqdm else None
    last_t = {}; n_done = 0; done_units = 0.0; errors = []; alive = n_workers; t0 = time.time(); quiet = 0

    def say(line):
        (outer.write(line) if outer is not None else print(line))

    while alive > 0:
        try:
            msg = msgs.get(timeout=30)
        except queue.Empty:
            quiet += 30
            if not any(p.is_alive() for p in procs):
                say("  [경고] 워커가 모두 종료 신호 없이 끝났습니다. 같은 셀을 다시 실행하면 남은 것부터 이어서 돕니다."); break
            if quiet % 600 == 0:
                say(f"  [대기] {quiet // 60}분째 새 진행이 없습니다(살아 있는 워커 {sum(p.is_alive() for p in procs)}개).")
            continue
        quiet = 0
        kind, wid = msg[0], msg[1]
        if kind == "start":
            c, s, m = msg[2]; last_t[wid] = 0
            if bars:
                bars[wid].reset(total=cfg.iters); bars[wid].set_description_str(f"  워커 {wid + 1}: {c} · {LABELS[m]} · seed{s}"); bars[wid].set_postfix_str("")
        elif kind == "step":
            key, t = msg[2], msg[3]
            d = t - last_t.get(wid, 0); last_t[wid] = t
            done_units += d * step_cost(key[2], cfg)
            if bars:
                bars[wid].update(d); outer.update(d * step_cost(key[2], cfg))
        elif kind == "eval":
            if bars:
                bars[wid].set_postfix_str(f"val F1 {msg[3]:.3f}", refresh=False)
        elif kind == "done":
            n_done += 1
            el = time.time() - t0
            remain = (total - done_units) * el / max(done_units, 1.0)
            if outer is not None:
                outer.set_description_str(f"전체 {n_done}/{len(left)}회")
            say(_result_line(n_done, len(left), msg[2], msg[3]) + f", 누적 {fmt_time(el)}, 남은 예상 {fmt_time(remain)}")
        elif kind == "error":
            errors.append((msg[2], msg[3]))
            say(f"  [오류] 워커 {wid + 1} {msg[2]}\n{msg[3]}")
        elif kind == "exit":
            alive -= 1
            if bars:
                bars[wid].set_description_str(f"  워커 {wid + 1}: 끝")
    for p in procs:
        p.join(timeout=10)
    if bars:
        for b in bars:
            b.close()
        outer.close()
    still = [k for k in left if not run_path(out, k[0], k[2], k[1]).exists()]
    print(f"[완료] {len(left) - len(still)}회 끝남, 총 {fmt_time(time.time() - t0)} (워커 {n_workers}개)" +
          (f" | 끝나지 않은 실행 {len(still)}회: 같은 셀을 다시 실행하면 이어서 돕니다" if still else ""))


def run_grid(bundle: D.Bundle, out: str, methods=M.METHODS, conditions=DEFAULT_CONDS, seeds=DEFAULT_SEEDS,
             cfg: M.Cfg | None = None, device: torch.device | None = None, u_cap: int | None = None, verbose: bool = True,
             workers: int | str = "auto", mem_per_worker_gb: float = 1.5) -> None:
    """남은 실행을 돌린다. workers="auto"면 런타임(GPU 여유 메모리, vCPU 수)을 보고 동시에 돌릴 학습 수를 정한다. 1이면 차례로 돈다.
    verbose=True면 시작할 때 예상 소요 시간을, 도는 동안 진행 막대(전체 + 실행별)를 보여 준다."""
    cfg = cfg or M.Cfg()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (Path(out) / "runs").mkdir(parents=True, exist_ok=True)
    meta = dict(K=bundle.K, class_ids=bundle.class_ids, dropped=bundle.dropped, counts=bundle.counts, split=bundle.split_info)
    (Path(out) / "data_meta.json").write_text(json.dumps(meta, default=_jsonable, ensure_ascii=False, indent=1))
    todo = [(c, s, m) for c in conditions for s in seeds for m in methods]
    left = [k for k in todo if not run_path(out, k[0], k[2], k[1]).exists()]
    print(f"[실행] 전체 {len(todo)}회 중 남은 것 {len(left)}회 | device {device} | iters {cfg.iters}")
    if not left:
        return
    tqdm = None
    seq_total = None
    if verbose:
        seq_total = estimate_time(bundle, out, methods, conditions, seeds, cfg, device)["total"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            from tqdm.auto import tqdm
        except ImportError:
            print("(tqdm이 없어 진행 막대 대신 실행별 요약만 출력합니다)")
    if workers == "auto":
        hw = hw_profile(verbose=verbose)
        n_workers = auto_workers(hw, len(left), mem_per_worker_gb) if device.type == "cuda" else 1
    else:
        n_workers = max(1, min(int(workers), len(left)))
    if n_workers > 1:
        if verbose:
            print(f"[병렬] 워커 {n_workers}개가 동시에 돈다(GPU 하나를 나눠 씀). 위 예상 시간" + (f" {fmt_time(seq_total)}" if seq_total else "") +
                  f"은 차례로 돌 때의 값이다. 병렬에서는 그보다 짧지만 1/{n_workers}까지 줄지는 않는다. 실제 남은 시간은 진행 막대를 본다.")
        _run_parallel(bundle, out, cfg, device, u_cap, sorted(left, key=lambda k: -step_cost(k[2], cfg)), n_workers, tqdm)
        return

    x, y = bundle.x.to(device), bundle.y.to(device)
    units = [cfg.iters * step_cost(m, cfg) for _, _, m in left]
    outer = tqdm(total=sum(units), desc=f"전체 0/{len(left)}회", bar_format="{desc} {percentage:3.0f}%|{bar}| 경과 {elapsed} · 남은 시간 {remaining}") if tqdm else None
    t0 = time.time(); done_units = 0; done_sec = 0.0
    for i, (c, s, m) in enumerate(left, 1):
        cond = D.build_condition(bundle, c, s, u_cap=u_cap)
        cost = step_cost(m, cfg)
        inner = tqdm(total=cfg.iters, desc=f"  {c} · {LABELS[m]} · seed{s}", leave=False,
                     bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} 스텝 · 남은 시간 {remaining}{postfix}") if tqdm else None

        def prog(t, row, inner=inner, cost=cost):
            if inner is None:
                return
            if row is None:         # 스텝마다 한 번
                inner.update(1); outer.update(cost)
            else:                   # 평가 때: 점수만 갱신(스텝은 위에서 이미 셌다)
                inner.set_postfix_str(f"val F1 {row['val_f1']:.3f}", refresh=False)

        res = train_run(bundle, x, y, cond, m, s, cfg, device, verbose=verbose and tqdm is None, progress=prog if tqdm else None)
        if inner is not None:
            inner.close()
        _save_result(run_path(out, c, m, s), res)
        done_units += units[i - 1]; done_sec += res["seconds"]
        remain = (done_sec / done_units) * sum(units[i:])       # 실제로 걸린 시간으로 다시 계산한 남은 시간
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
        rows.append(dict(cond=r["cond"], method=r["method"], seed=r["seed"], macro_f1=r["test"]["macro_f1"], acc=r["test"]["acc"],
                         tail_f1=r["test"].get("tail_f1", np.nan), head_f1=r["test"].get("head_f1", np.nan),
                         val_f1=r["val_f1"], u_acc=r["u_acc"], util=last.get("util", np.nan), pl_acc=last.get("pl_acc", np.nan),
                         best_iter=r["best_iter"], minutes=r["seconds"] / 60, n_labeled=r["n_labeled"], n_unlabeled=r["n_unlabeled"],
                         flip_all=(r.get("aug_flip_rate") or {}).get("all", np.nan)))
    return pd.DataFrame(rows)


def _ms(df, value: str, conds, methods):
    import pandas as pd

    g = df.groupby(["method", "cond"])[value].agg(["mean", "std", "count"])
    out = pd.DataFrame(index=[LABELS[m] for m in methods], columns=[COND_LABELS.get(c, c) for c in conds], dtype=object)
    for m in methods:
        for c in conds:
            if (m, c) in g.index:
                a = g.loc[(m, c)]
                out.loc[LABELS[m], COND_LABELS.get(c, c)] = f"{a['mean']:.3f} ± {0.0 if np.isnan(a['std']) else a['std']:.3f}" + ("" if a["count"] >= 3 else f" (n={int(a['count'])})")
    return out.fillna("")


def _md(t) -> str:
    """DataFrame -> markdown 표(tabulate 없이)."""
    t = t.reset_index() if t.index.name or not isinstance(t.index[0], (int, np.integer)) else t
    cols = [str(c) if str(c) != "index" else "방법" for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def _gain(piv, baseline: str, rows, conds):
    """같은 (조건, seed)의 baseline과 짝지어 뺀 이득의 평균 ± 표준편차."""
    import pandas as pd

    t = pd.DataFrame(index=[LABELS[m] for m in rows if m in piv.columns], columns=[COND_LABELS.get(c, c) for c in conds], dtype=object)
    if baseline not in piv.columns:
        return t.fillna("")
    for m in rows:
        if m not in piv.columns:
            continue
        for c in conds:
            if c in piv.index.get_level_values(0):
                d = (piv.loc[c, m] - piv.loc[c, baseline]).dropna()
                if len(d):
                    t.loc[LABELS[m], COND_LABELS.get(c, c)] = f"{d.mean():+.3f} ± {0.0 if len(d) < 2 else d.std():.3f}"
    return t.fillna("")


def tables(out: str, save: bool = True) -> dict:
    """슬라이드의 표와 같은 구성으로 결과를 모은다.
    test_macro_f1(Table 15) / gain_vs_fixmatch(Table 16, 모듈 고유의 효과) / gain_vs_selftraining(Table 17, 고전 기법 대비 총 효과)
    / tail_f1(조건 B의 tail 클래스 F1) / diagnosis(원인 진단용)."""
    df = collect(out)
    if df.empty:
        print("결과가 아직 없습니다."); return {}
    conds = [c for c in ("base", "A", "B", "Bc") if c in set(df.cond)]
    methods = [m for m in M.METHODS if m in set(df.method)]
    piv = df.pivot_table(index=["cond", "seed"], columns="method", values="macro_f1")
    res = {"test_macro_f1": _ms(df, "macro_f1", conds, methods),
           "gain_vs_fixmatch": _gain(piv, "fixmatch", ("cgmatch", "interlude", "scssl"), conds),
           "gain_vs_selftraining": _gain(piv, "pseudolabel", ("fixmatch", "cgmatch", "interlude", "scssl"), conds)}
    lt = [c for c in conds if c in ("B", "Bc")]
    if lt:
        res["tail_f1"] = _ms(df[df.cond.isin(lt)], "tail_f1", lt, methods)
    diag = df.groupby(["cond", "method"])[["util", "pl_acc", "u_acc", "acc", "flip_all", "minutes"]].mean().round(3).reset_index()
    diag["method"] = diag["method"].map(LABELS)
    res["diagnosis"] = diag.rename(columns=dict(util="pseudo-label 활용률", pl_acc="pseudo-label 정확도", u_acc="unlabeled 정확도(학습 구역)",
                                                acc="test 정확도", flip_all="강한 증강 전후 예측 변화율", minutes="분/회"))
    if save:
        d = Path(out) / "tables"; d.mkdir(exist_ok=True)
        md = []
        for name, t in res.items():
            t.to_csv(d / f"{name}.csv")
            md.append(f"### {name}\n\n{_md(t)}\n")
        (d / "tables.md").write_text("\n".join(md))
        df.to_csv(d / "runs.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--dense-cache"); ap.add_argument("--xy-cache")
    ap.add_argument("--out", required=True); ap.add_argument("--split", default="block", choices=["block", "random"])
    ap.add_argument("--block-km", type=float, default=25.0); ap.add_argument("--min-train", type=int, default=300)
    ap.add_argument("--methods", nargs="+", default=list(M.METHODS)); ap.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS)); ap.add_argument("--iters", type=int, default=M.Cfg.iters)
    a = ap.parse_args()
    b = prepare(a.csv, a.dense_cache, a.xy_cache, work_dir=a.out, split=a.split, block_km=a.block_km, min_train=a.min_train)
    D.describe_conditions(b, a.conditions)
    run_grid(b, a.out, a.methods, a.conditions, a.seeds, M.Cfg(iters=a.iters))
    for name, t in tables(a.out).items():
        print(f"\n== {name}\n{t.to_string()}")


if __name__ == "__main__":
    main()
