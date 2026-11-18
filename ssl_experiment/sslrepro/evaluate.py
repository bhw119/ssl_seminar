from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from sslexp.train import summarize

def load_algorithm(cfg_path: str, ckpt: str = "model_best.pth"):
    argv = sys.argv
    sys.argv = [argv[0], "--c", str(cfg_path)]
    try:
        from . import train as T
        args = T.get_config()
    finally:
        sys.argv = argv
    from semilearn.algorithms import get_algorithm
    from semilearn.imb_algorithms import get_imb_algorithm
    from semilearn.core.utils import get_net_builder, send_model_cuda

    args.gpu = 0
    args.use_tensorboard = False
    args.fusion_at_eval = False
    build = get_imb_algorithm if getattr(args, "imb_algorithm", None) else get_algorithm
    alg = build(args, get_net_builder(args.net, args.net_from_name), None, None)
    alg.model = send_model_cuda(args, alg.model)
    alg.ema_model = send_model_cuda(args, alg.ema_model, clip_batch=False)
    alg.load_model(os.path.join(args.save_dir, args.save_name, ckpt))
    alg.call_hook("before_run", "EMAHook")
    return alg, args

def split_loader(alg, args, split: str, shuffle: bool = False, workers: int = 8):
    from semilearn.datasets.cv_datasets import get_semi_inat_split

    ds = get_semi_inat_split(args, args.algorithm, args.data_dir, split)
    g = torch.Generator().manual_seed(0)
    return torch.utils.data.DataLoader(ds, batch_size=args.eval_batch_size, shuffle=shuffle, generator=g,
                                       num_workers=workers, drop_last=False)

@torch.no_grad()
def predict(alg, loader, scssl: bool = False) -> tuple[dict, np.ndarray]:
    model = alg.model
    model.eval()
    alg.ema.apply_shadow()
    if scssl:
        w = torch.ones((1, 3, alg.args.img_size, alg.args.img_size)).cuda(alg.gpu)
        o = model(w, use_zero=False)
        b1, b2 = o["logits"][0].float().cpu(), o["logits_aux"][0].float().cpu()
    out, ys = {}, []
    for d in loader:
        o = model(d["x_lb"].cuda(alg.gpu))
        if scssl:
            aux = o["logits_aux"].float().cpu()
            parts = dict(h2=aux - alg.coe * b2, h1=aux - alg.coe * b1, h1_raw=o["logits"].float().cpu(), h2_raw=aux)
        else:
            parts = dict(main=o["logits"].float().cpu())
        for k, v in parts.items():
            out.setdefault(k, []).append(v)
        ys.append(d["y_lb"])
    alg.ema.restore()
    model.train()
    return {k: torch.cat(v).numpy() for k, v in out.items()}, torch.cat(ys).numpy()

def top5(logits: np.ndarray, y: np.ndarray) -> float:
    return float((np.argsort(-logits, 1)[:, :5] == y[:, None]).any(1).mean())

def read_history(ckpt_dir: Path) -> list[dict]:
    p = ckpt_dir / "eval_log.jsonl"
    if not p.exists():
        return []
    hist = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = "h2/top-1-acc" if "h2/top-1-acc" in r else next((k for k in r if k.startswith("eval") and k.endswith("/top-1-acc")), None)
        if key is None:
            continue
        row = dict(it=r["it"], val_acc=r[key])
        for k in ("h1/top-1-acc", "h1/raw-acc", "h2/raw-acc"):
            if k in r:
                row[k.replace("/top-1-acc", "").replace("/raw-acc", "_raw").replace("/", "_")] = r[k]
        hist.append(row)
    return hist

def checkpoint_iter(ckpt_dir: Path) -> int:
    best, it = -1.0, 0
    for line in (ckpt_dir / "eval_log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for k, v in r.items():
            if "top-1-acc" in k and v > best:
                best, it = v, r["it"]
    return it

def evaluate_checkpoint(cfg_path: str, ckpt: str, method: str) -> dict:
    alg, args = load_algorithm(cfg_path, ckpt)
    scssl = method.startswith("scssl")
    main = "h2" if scssl else "main"
    K = args.num_classes
    n_l = np.bincount(np.asarray(alg.dataset_dict["train_lb"].targets), minlength=K)
    order = np.argsort(-n_l, kind="stable")
    head = np.zeros(K, dtype=bool)
    head[order[: K // 2]] = True

    lv, yv = predict(alg, split_loader(alg, args, "val"), scssl)
    lt, yt = predict(alg, split_loader(alg, args, "test"), scssl)
    pt = lt[main].argmax(1)
    test = summarize(yt, pt, K, head)
    hit = pt == yt
    test["head_acc"] = float(hit[head[yt]].mean())
    test["tail_acc"] = float(hit[~head[yt]].mean())
    test["top5_acc"] = top5(lt[main], yt)
    res = dict(K=K, n_labeled=int(n_l.sum()), n_unlabeled=len(alg.dataset_dict["train_ulb"]),
               val_acc=float((lv[main].argmax(1) == yv).mean()), test=test, extra={})
    if scssl:
        res["extra"]["heads_test_acc"] = {k: float((v.argmax(1) == yt).mean()) for k, v in lt.items()}
        res["extra"]["heads_val_acc"] = {k: float((v.argmax(1) == yv).mean()) for k, v in lv.items()}
    if args.algorithm == "interlude":
        ef = dict(no_fusion=test["acc"])
        alg.fusion_at_eval = True
        ef["fusion_file_order"] = float((predict(alg, split_loader(alg, args, "test"))[0]["main"].argmax(1) == yt).mean())
        lo, ys = predict(alg, split_loader(alg, args, "test", shuffle=True))
        ef["fusion_shuffled"] = float((lo["main"].argmax(1) == ys).mean())
        alg.fusion_at_eval = False
        res["extra"]["ef_at_eval_test_acc"] = ef
    del alg
    torch.cuda.empty_cache()
    return res

def evaluate_run(cfg_path: str, method: str, cond: str, seed: int) -> dict:
    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    ckpt_dir = Path(cfg["save_dir"]) / cfg["save_name"]
    hist = read_history(ckpt_dir)
    best = evaluate_checkpoint(cfg_path, "model_best.pth", method)
    last = evaluate_checkpoint(cfg_path, "latest_model.pth", method)
    res = dict(method=method, cond=cond, seed=seed, K=best["K"], n_labeled=best["n_labeled"], n_unlabeled=best["n_unlabeled"],
               best_iter=checkpoint_iter(ckpt_dir), val_acc=best["val_acc"], test=best["test"], history=hist,
               extra=dict(best["extra"], last=dict(it=hist[-1]["it"] if hist else None, val_acc=last["val_acc"],
                                                   test_acc=last["test"]["acc"], extra=last["extra"])))
    return res
