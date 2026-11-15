# -*- coding: utf-8 -*-
"""이미지 실험 점검: 공식 데이터와 같은 폴더·주석 형식의 작은 가짜 데이터를 만들어 전체 경로를 돌린다(CPU, 수 분).

    python -m sslimg.smoke
"""
from __future__ import annotations

import csv
import json
import pathlib
import tempfile

import numpy as np
import torch

from sslexp import methods as M

from . import data as D
from .aug import ImageAug
from .run import estimate_time, prepare, run_grid, tables
from .train import ImgCfg


def fake_dataset(root: pathlib.Path, K: int = 6, seed: int = 0) -> None:
    """클래스마다 색과 줄무늬 방향이 다른 그림. 공식 형식: trainval_images/{종}/{번호}.jpg, u_train_in/{번호}.jpg, test/{번호}.jpg, annotation/anno_*.json, solution.csv"""
    from PIL import Image

    g = np.random.default_rng(seed)

    def img(k):
        h, w = int(g.integers(40, 60)), int(g.integers(48, 72))
        yy, xx = np.mgrid[0:h, 0:w]
        base = np.zeros((h, w, 3), np.float32)
        base[..., k % 3] = 150 + 20 * (k // 3)
        base[..., (k + 1) % 3] = 60 * np.sin((xx if k % 2 else yy) / (2.0 + k)) + 80
        return Image.fromarray(np.clip(base + g.normal(0, 25, base.shape), 0, 255).astype(np.uint8))

    ann = root / "annotation"; ann.mkdir(parents=True)

    def write(split, items):     # items: (file_name, 클래스)
        ims, an = [], []
        for i, (fn, k) in enumerate(items):
            p = root / fn; p.parent.mkdir(parents=True, exist_ok=True)
            im = img(k if k >= 0 else int(g.integers(0, K))); im.save(p, quality=90)
            ims.append(dict(file_name=fn, width=im.size[0], height=im.size[1], id=i))
            if split != "test":
                an.append(dict(image_id=i, id=i, category_id=(k if split in ("l_train", "val") else -1)))
        (ann / f"anno_{split}.json").write_text(json.dumps(dict(info={}, annotations=an, images=ims)))

    l_items, v_items, n = [], [], {}
    for k in range(K):
        n[k] = 0
        for _ in range(6 + 3 * (K - 1 - k)):       # long-tail: 종 0이 가장 많다
            l_items.append((f"trainval_images/{k}/{n[k]}.jpg", k)); n[k] += 1
        for _ in range(4):
            v_items.append((f"trainval_images/{k}/{n[k]}.jpg", k)); n[k] += 1
    write("l_train", l_items); write("val", v_items)
    uk = g.integers(0, K, 150)
    write("u_train_in", [(f"u_train_in/{i}.jpg", int(k)) for i, k in enumerate(uk)])
    write("u_train_out", [(f"u_train_out/{i}.jpg", -1) for i in range(80)])
    tk = np.repeat(np.arange(K), 6)
    write("test", [(f"test/{i}.jpg", int(k)) for i, k in enumerate(tk)])
    with open(root / "solution.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Id", "Category"])
        for i, k in enumerate(tk):
            w.writerow([i, int(k)])


def main():
    ok = []

    def check(name, cond):
        ok.append(bool(cond)); print(("  ok  " if cond else "  FAIL ") + name)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp); root = tmp / "raw"; cache = tmp / "cache"; out = tmp / "out"
        fake_dataset(root)
        lists = D.load_lists(str(root))
        check("목록: 분할 5개, test 레이블은 solution.csv에서", set(lists) == {"l_train", "val", "test", "u_train_in", "u_train_out"} and (lists["test"]["y"] >= 0).all())
        check("목록: unlabeled 레이블은 -1", (lists["u_train_in"]["y"] == -1).all() and (lists["u_train_out"]["y"] == -1).all())
        data = prepare(str(root), str(cache), size=40, u_out_n=50, do_download=False, verbose=False)
        check("캐시: 모양과 개수", data.xl.shape == (len(lists["l_train"]["y"]), 40, 40, 3) and len(data.xu_in) == 150 and len(data.xu_out) == 50 and data.K == 6)
        check("캐시: labeled 레이블 보존", np.array_equal(data.yl, lists["l_train"]["y"]) and np.array_equal(data.yt, lists["test"]["y"]))
        check("head = labeled가 많은 절반", data.n_l[data.head].min() >= data.n_l[~data.head].max())
        mt = (cache / "l_train_40.u8").stat().st_mtime
        prepare(str(root), str(cache), size=40, u_out_n=50, do_download=False, verbose=False)
        check("캐시가 있으면 다시 만들지 않는다", (cache / "l_train_40.u8").stat().st_mtime == mt)

        # 이미지를 받는 경로 세 가지: (1) 압축 파일이 든 폴더 (2) 이미 풀린 폴더(Kaggle이 이렇게 줄 수 있다) (3) kagglehub
        import shutil, sys, tarfile, types

        def fresh(name):
            r2 = tmp / name; r2.mkdir()
            shutil.copytree(root / "annotation", r2 / "annotation"); shutil.copy(root / "solution.csv", r2 / "solution.csv")
            return r2

        arch = tmp / "archives"; (arch / "sub").mkdir(parents=True)
        with tarfile.open(arch / "trainval_images.tar.gz", "w:gz") as t:
            t.add(root / "trainval_images", arcname="trainval_images")
        with tarfile.open(arch / "sub" / "u_train_in_copy", "w") as t:      # 확장자가 없고 한 단계 안쪽에 있어도 알아본다
            t.add(root / "u_train_in", arcname="u_train_in")
        with tarfile.open(arch / "test.tar", "w") as t:
            t.add(root / "test", arcname="nested/test")                      # 한 단계 안쪽에 풀려도 찾는다
        with tarfile.open(arch / "u_train_out.tar", "w") as t:
            t.add(root / "u_train_out", arcname="u_train_out")
        r2 = fresh("raw2")
        D.download(str(r2), archive_dir=str(arch), source="none")
        d2 = prepare(str(r2), str(tmp / "cache2"), size=40, do_download=False, verbose=False)
        check("압축 파일 폴더에서 준비", np.array_equal(np.asarray(d2.xl), np.asarray(data.xl)) and len(d2.xu_in) == 150)
        check("다른 종 묶음은 필요할 때만 푼다", D.find_base(r2, D.FIRST["u_train_out"]) is None)

        r3 = fresh("raw3")
        D.download(str(r3), archive_dir=str(root), source="none")            # 풀린 폴더를 그대로 쓴다(바로가기)
        check("이미 풀린 폴더에서 준비", (r3 / "trainval_images").is_symlink() and D.find_base(r3, D.FIRST["test"]) is not None)

        real = sys.modules.get("kagglehub"); fake = types.ModuleType("kagglehub")
        fake.competition_download = lambda name: str(arch)
        sys.modules["kagglehub"] = fake
        try:
            r4 = fresh("raw4"); D.download(str(r4), source="kaggle")
            check("kagglehub가 준 폴더에서 준비", all(D.find_base(r4, D.FIRST[k]) is not None for k in ("trainval", "u_train_in", "test")))
            fake.competition_download = lambda name: (_ for _ in ()).throw(RuntimeError("403 규칙 동의 필요"))
            r5 = fresh("raw5")
            try:
                D.download(str(r5), source="kaggle"); msg = ""
            except RuntimeError as e:
                msg = str(e)
            check("받지 못하면 빠진 것과 받는 방법을 알려 준다", "trainval" in msg and "kaggle.com/c/semi-inat-2020" in msg and "drive.google.com/file/d/" in msg)
        finally:
            if real is not None:
                sys.modules["kagglehub"] = real
            else:
                del sys.modules["kagglehub"]

        aug = ImageAug(crop=32, group=4)
        x = D.to_tensor(data.xl[:12], torch.device("cpu"))
        w, s, c = aug.weak(x), aug.strong(x), aug.center(x)
        check("증강: 모양·유한", w.shape == s.shape == c.shape == (12, 3, 32, 32) and torch.isfinite(w).all() and torch.isfinite(s).all())
        check("증강: 약한 증강과 강한 증강이 다르다", (w - s).abs().mean() > 0.05)
        a, b2, c2, d2 = aug.group_views(x[:3], D.to_tensor(data.xu_in[:6], torch.device("cpu")), 2)
        check("묶음 증강: 모양", a.shape == (3, 3, 32, 32) and b2.shape == (6, 3, 32, 32) and c2.shape == a.shape and d2.shape == b2.shape)

        cfg = ImgCfg(iters=30, bs=4, mu=3, eval_every=15, cg_warmup=8, cg_init=4, arch="resnet18", pretrained=False, cache_size=40, crop=32, amp=False, lr=0.01)
        dev = torch.device("cpu")
        est = estimate_time(data, str(out), conditions=("in",), seeds=(0,), cfg=cfg, device=dev, verbose=False)
        check("예상 소요 시간", est["n_left"] == 6 and est["total"] > 0)
        run_grid(data, str(out), conditions=("in",), seeds=(0,), cfg=cfg, device=dev, verbose=False)
        run_grid(data, str(out), methods=("fixmatch", "cgmatch"), conditions=("open",), seeds=(0,), cfg=cfg, device=dev, verbose=False)
        rs = [json.loads(p.read_text()) for p in (out / "runs").glob("*.json")]
        check("실행 8회 저장(같은 종 6 + open-set 2)", len(rs) == 8)
        check("손실·정확도가 유한", all(np.isfinite(h["loss"]) for r in rs for h in r["history"]) and all(0 <= r["test"]["acc"] <= 1 for r in rs))
        op = [r for r in rs if r["cond"] == "open"][0]; inn = [r for r in rs if r["cond"] == "in" and r["method"] == "fixmatch"][0]
        check("open-set 조건은 unlabeled가 더 많다", op["n_unlabeled"] == inn["n_unlabeled"] + 50)
        sc = [r for r in rs if r["method"] == "scssl"][0]
        check("SC-SSL: 분포 추정 기록", "sc" in sc["extra"])
        t = tables(str(out))
        check("표 생성", {"test_acc", "gain_vs_fixmatch", "gain_vs_selftraining", "tail_acc", "diagnosis"} <= set(t))
        n0 = len(rs)
        run_grid(data, str(out), conditions=("in",), seeds=(0,), cfg=cfg, device=dev, verbose=False)
        check("다시 실행하면 건너뛴다", len(list((out / "runs").glob("*.json"))) == n0)
        print("\n" + t["test_acc"].to_string())
    print(f"\n{sum(ok)}/{len(ok)} 통과")
    raise SystemExit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
