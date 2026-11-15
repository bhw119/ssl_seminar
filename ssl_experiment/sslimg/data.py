# -*- coding: utf-8 -*-
"""Semi-Aves(Semi-Supervised iNaturalist-Aves, FGVC7 @ CVPR 2020) 데이터 준비.

공식 분할을 그대로 쓴다(https://github.com/cvl-umass/semi-inat-2020).
    l_train      labeled 3,959장, 200종 (종마다 5~43장, long-tail)
    val          2,000장 (종마다 10장)      -> 모델 선택
    test         8,000장 (종마다 40장)      -> 평가. 레이블은 대회 뒤 공개된 solution.csv
    u_train_in   unlabeled 26,640장, 같은 200종 (레이블 비공개)
    u_train_out  unlabeled 122,208장, 다른 종 (open-set 조건에서만 일부를 쓴다)

주석은 COCO 형식 JSON(anno_{split}.json)이고 file_name은 'trainval_images/0/0.jpg', 'u_train_in/0.jpg', 'test/0.jpg' 꼴이다.

학습 속도를 위해 한 번만 JPEG를 풀어 정사각 uint8 배열(.u8 memmap)로 저장해 두고, 학습 때는 그 배열을 GPU에 올려 쓴다.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

GDRIVE = {      # 공식 README의 Google Drive 파일
    "trainval": "1xsgOcEWKG9CszNNT_EXN3YB1OLPYNbf8",     # labeled train + val, 533 MB
    "u_train_in": "1BiEkIp8yuqB5Vau_ZAFAhwp8CNoqVjRk",   # 2.3 GB
    "test": "1OVEA2lNJnYM5zxh3W_o_Q6K5lsNmJ9Hy",         # 722 MB
    "u_train_out": "18gZhCEKJkxKkUbG49syAuWPdfL0Lm9tp",  # 11.0 GB (open-set 조건에서만)
}
ANN_URL = "https://github.com/cvl-umass/semi-inat-2020/raw/master/annotation.tar.gz"
SOL_URL = "https://github.com/cvl-umass/semi-inat-2020/raw/master/solution.csv"
SPLIT_DIR = {"l_train": "trainval_images", "val": "trainval_images", "test": "test", "u_train_in": "u_train_in", "u_train_out": "u_train_out"}


# --------------------------------------------------------------------------- 내려받기
KAGGLE_COMP = "semi-inat-2020"
KEY_DIR = {"trainval": "trainval_images", "u_train_in": "u_train_in", "test": "test", "u_train_out": "u_train_out"}
FIRST = {"trainval": "trainval_images/0/0.jpg", "u_train_in": "u_train_in/0.jpg", "test": "test/0.jpg", "u_train_out": "u_train_out/0.jpg"}   # 주석 파일의 첫 이미지
HELP = """이미지를 준비하지 못했습니다. 빠진 것: {missing}
- Kaggle로 받을 때: https://www.kaggle.com/c/semi-inat-2020 에서 대회 규칙에 동의(Join/Late Submission)했는지, Kaggle API 토큰이 설정됐는지 확인하세요.
- 직접 받을 때: 아래 주소에서 받아 한 폴더에 넣고 그 경로를 archive_dir(노트북의 ARCHIVES)로 주세요. 압축을 풀지 않아도 됩니다.
{links}
찾아본 폴더의 내용:
{tree}"""


def find_base(root, rel_file: str, depth: int = 4) -> Path | None:
    """root 아래(깊이 depth까지)에서 (폴더 / rel_file)이 있는 폴더를 찾는다. 압축이 한두 단계 안쪽에 풀려도 찾기 위해서다."""
    root = Path(root)
    if not root.is_dir():
        return None
    if (root / rel_file).exists():
        return root
    if depth > 0:
        for p in sorted(root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                f = find_base(p, rel_file, depth - 1)
                if f is not None:
                    return f
    return None


def find_dir(root, name: str, depth: int = 3) -> Path | None:
    """root 아래에서 이름이 name인 폴더를 찾는다."""
    root = Path(root)
    if (root / name).is_dir():
        return root / name
    if depth > 0 and root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                f = find_dir(p, name, depth - 1)
                if f is not None:
                    return f
    return None


def _tree(root, depth: int = 2, limit: int = 40) -> str:
    out = []

    def walk(p, d):
        if len(out) >= limit or not p.is_dir():
            return
        kids = sorted(p.iterdir())
        for k in kids[:12]:
            if len(out) >= limit:
                return
            out.append("  " * (depth - d + 1) + k.name + ("/" if k.is_dir() else f"  ({k.stat().st_size / 2 ** 20:.0f} MB)"))
            if k.is_dir() and d > 1:
                walk(k, d - 1)
        if len(kids) > 12:
            out.append("  " * (depth - d + 1) + f"... 외 {len(kids) - 12}개")

    walk(Path(root), depth)
    return "\n".join(out) if out else "  (비어 있음)"


def _is_archive(p: Path) -> bool:
    if not p.is_file() or p.suffix in (".extracted", ".json", ".csv", ".u8", ".npy", ".jpg", ".txt") or p.stat().st_size < 1024:
        return False
    try:
        return zipfile.is_zipfile(p) or tarfile.is_tarfile(p)
    except OSError:
        return False


def _extract_from(src: Path, root: Path, with_out: bool, depth: int = 2) -> None:
    """src(와 그 아래 depth단계)에 있는 압축 파일(tar, tar.gz, zip. 확장자가 아니라 내용으로 판별)을 root에 푼다.
    주석 묶음은 건드리지 않고(GitHub의 것을 쓴다), 다른 종 unlabeled 묶음은 with_out일 때만 푼다."""
    if not src.is_dir():
        return
    for p in sorted(src.iterdir()):
        if p.is_dir():
            if depth > 0 and not p.name.startswith("."):
                _extract_from(p, root, with_out, depth - 1)
            continue
        low = p.name.lower()
        mark = root / (p.name + ".extracted")
        if mark.exists() or "anno" in low or ("out" in low and not with_out) or not _is_archive(p):
            continue
        print(f"  압축 풀기: {p.name} ({p.stat().st_size / 2 ** 30:.2f} GB)")
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                z.extractall(root)
        else:
            with tarfile.open(p) as t:
                try:
                    t.extractall(root, filter="data")      # 묶음 밖 경로로 풀리는 것을 막는다(Python 3.12+)
                except TypeError:
                    t.extractall(root)
        mark.write_text("ok")


def _link_from(src: Path, root: Path, keys) -> None:
    """src에 이미 풀려 있는 이미지 폴더가 있으면 root에 바로가기(symlink)를 만든다."""
    for k in keys:
        if find_base(root, FIRST[k]) is None:
            b = find_base(src, FIRST[k])
            if b is not None and not (root / KEY_DIR[k]).exists():
                os.symlink(b / KEY_DIR[k], root / KEY_DIR[k])


def kaggle_path() -> str:
    """Kaggle 대회 파일을 받는다(이미 받았으면 저장해 둔 위치만 돌려준다). 대회 규칙 동의와 API 토큰이 필요하다."""
    import kagglehub

    return kagglehub.competition_download(KAGGLE_COMP)


def download(root: str, with_out: bool = False, archive_dir: str | None = None, source: str = "kaggle") -> None:
    """주석과 이미지를 root에 준비한다. 이미 있는 것은 건너뛴다.
    주석(JSON)과 test 레이블(solution.csv)은 GitHub에서 받는다.
    이미지는 archive_dir(직접 받아 둔 폴더)가 있으면 거기서, 없으면 source에 따라 받는다.
        source="kaggle" : kagglehub로 대회 파일 전체를 받는다(약 15 GB, 다른 종 unlabeled 포함)
        source="gdrive" : 공식 README의 Google Drive 파일을 gdown으로 받는다(자동 내려받기가 막히는 경우가 많다)
    받은 곳에 폴더가 이미 풀려 있으면 그대로 쓰고, 압축 파일이면 root에 푼다."""
    r = Path(root); r.mkdir(parents=True, exist_ok=True)
    if find_dir(r, "annotation", depth=0) is None:
        subprocess.run(["curl", "-sL", "-o", str(r / "annotation.tar.gz"), ANN_URL], check=True)
        with tarfile.open(r / "annotation.tar.gz") as t:
            try:
                t.extractall(r, filter="data")
            except TypeError:
                t.extractall(r)
    if not (r / "solution.csv").exists():
        subprocess.run(["curl", "-sL", "-o", str(r / "solution.csv"), SOL_URL], check=True)
    need = ["trainval", "u_train_in", "test"] + (["u_train_out"] if with_out else [])
    missing = lambda: [k for k in need if find_base(r, FIRST[k]) is None]
    looked = [r]
    if missing():
        _extract_from(r, r, with_out, depth=0)
    if missing() and archive_dir:
        if Path(archive_dir).is_dir():
            looked.append(Path(archive_dir))
            _link_from(Path(archive_dir), r, missing()); 
            if missing():
                _extract_from(Path(archive_dir), r, with_out)
        else:
            print(f"  (archive_dir {archive_dir} 폴더가 없습니다.)")
    if missing() and source == "kaggle":
        try:
            kp = Path(kaggle_path())
            print(f"  Kaggle 파일 위치: {kp}")
            looked.append(kp)
            _link_from(kp, r, missing())
            if missing():
                _extract_from(kp, r, with_out)
        except Exception as e:
            import re

            print(f"  Kaggle 내려받기 실패: {type(e).__name__}: " + re.sub(r"KGAT_[A-Za-z0-9]+", "KGAT_***", str(e))[:300])
    if missing() and source == "gdrive":
        for key in missing():
            try:
                import gdown

                print(f"  내려받기: {key}")
                if gdown.download(id=GDRIVE[key], output=str(r) + "/", quiet=False) is None:
                    raise RuntimeError("gdown이 파일을 받지 못했습니다.")
            except Exception as e:
                print(f"  내려받기 실패({key}): {type(e).__name__}")
                break
        _extract_from(r, r, with_out, depth=0)
    if missing():
        raise RuntimeError(HELP.format(missing=", ".join(missing()), links="\n".join(f"  {k}: https://drive.google.com/file/d/{GDRIVE[k]}/view" for k in missing()),
                                       tree="\n".join(f"{p}\n{_tree(p)}" for p in looked)))


# --------------------------------------------------------------------------- 목록
def load_lists(root: str) -> dict:
    """분할별 (파일 경로, 레이블). unlabeled는 레이블 -1. test 레이블은 solution.csv(Id = 이미지 id)에서 읽는다."""
    r = Path(root)
    ann = find_dir(r, "annotation")
    if ann is None:
        raise FileNotFoundError(f"{r} 아래에 annotation 폴더가 없습니다(annotation.tar.gz를 풀어야 합니다).")
    sol_path = r / "solution.csv"
    if not sol_path.exists():
        raise FileNotFoundError(f"{sol_path}가 없습니다(test 레이블).")
    with open(sol_path) as f:
        sol = {int(row["Id"]): int(row["Category"].split()[0]) for row in csv.DictReader(f)}
    out = {}
    for split in ("l_train", "val", "test", "u_train_in", "u_train_out"):
        p = ann / f"anno_{split}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ims = sorted(d["images"], key=lambda i: i["id"])
        lab = {a["image_id"]: a["category_id"] for a in d.get("annotations", [])}
        y = np.array([sol[i["id"]] if split == "test" else lab.get(i["id"], -1) for i in ims], dtype=np.int64)
        out[split] = dict(files=[i["file_name"] for i in ims], y=y)
    return out


# --------------------------------------------------------------------------- 캐시
def _load_one(args):
    path, size = args
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2
        return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)


def build_cache(root: str, cache_dir: str, size: int = 176, splits=("l_train", "val", "test", "u_train_in"),
                u_out_n: int = 0, workers: int | None = None, verbose: bool = True) -> dict:
    """분할마다 (N, size, size, 3) uint8 배열을 cache_dir에 만든다. 짧은 변을 size로 맞추고 가운데를 자른다.
    이미 만든 분할은 건너뛴다. u_out_n > 0이면 u_train_out에서 그만큼만 뽑아(고정 seed) 넣는다."""
    lists = load_lists(root)
    cdir = Path(cache_dir); cdir.mkdir(parents=True, exist_ok=True)
    meta_p = cdir / f"meta_{size}.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    splits = list(splits) + (["u_train_out"] if u_out_n > 0 else [])
    workers = workers or os.cpu_count() or 1
    for split in splits:
        files, y = lists[split]["files"], lists[split]["y"]
        if split == "u_train_out":
            pick = np.sort(np.random.default_rng(0).choice(len(files), size=min(u_out_n, len(files)), replace=False))
            files, y = [files[i] for i in pick], y[pick]
        name = f"{split}_{size}" + (f"_n{len(files)}" if split == "u_train_out" else "")
        arr_p = cdir / f"{name}.u8"
        if meta.get(name, {}).get("n") == len(files) and arr_p.exists() and arr_p.stat().st_size == len(files) * size * size * 3:
            continue
        base = find_base(Path(root), files[0])
        if base is None:
            raise FileNotFoundError(f"{root} 아래에서 {files[0]}을 찾지 못했습니다. 이미지를 받았는지 확인하세요.\n{_tree(root)}")
        paths = [str(base / f) for f in files]
        miss = [p for p in paths[:50] if not os.path.exists(p)]
        if miss:
            raise FileNotFoundError(f"이미지 파일이 없습니다(예: {miss[0]}). 압축을 푼 위치를 확인하세요.")
        tmp = arr_p.with_suffix(".tmp")
        mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(len(files), size, size, 3))
        it = ((p, size) for p in paths)
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None
        bar = tqdm(total=len(files), desc=f"  캐시 {split}", leave=False) if (tqdm and verbose) else None
        if workers > 1:
            from multiprocessing import Pool

            with Pool(workers) as pool:
                for i, a in enumerate(pool.imap(_load_one, it, chunksize=32)):
                    mm[i] = a
                    if bar:
                        bar.update(1)
        else:
            for i, a in enumerate(map(_load_one, it)):
                mm[i] = a
                if bar:
                    bar.update(1)
        if bar:
            bar.close()
        mm.flush(); del mm
        os.replace(tmp, arr_p)
        np.save(cdir / f"{name}_y.npy", y)
        meta[name] = dict(n=len(files), size=size)
        meta_p.write_text(json.dumps(meta))
        if verbose:
            print(f"  캐시 완료: {name} ({len(files):,}장, {arr_p.stat().st_size / 2 ** 30:.2f} GB)")
    return meta


def copy_cache(src: str, dst: str) -> None:
    """캐시를 다른 곳(예: Drive <-> 런타임 로컬 디스크)으로 복사한다. 이미 같은 크기의 파일이 있으면 건너뛴다."""
    s, d = Path(src), Path(dst); d.mkdir(parents=True, exist_ok=True)
    for p in s.iterdir():
        q = d / p.name
        if p.is_file() and not (q.exists() and q.stat().st_size == p.stat().st_size):
            shutil.copy2(p, q)


@dataclass
class ImgData:
    K: int
    size: int
    xl: np.ndarray; yl: np.ndarray            # labeled (N, S, S, 3) uint8
    xv: np.ndarray; yv: np.ndarray            # val
    xt: np.ndarray; yt: np.ndarray            # test
    xu_in: np.ndarray                         # unlabeled, 같은 종
    xu_out: np.ndarray | None                 # unlabeled, 다른 종(일부)
    n_l: np.ndarray                           # 종별 labeled 수
    order: np.ndarray                         # order[k] = labeled가 k번째로 많은 종
    head: np.ndarray                          # labeled가 많은 쪽 절반


def load_data(cache_dir: str, size: int = 176, u_out_n: int = 0, verbose: bool = True) -> ImgData:
    c = Path(cache_dir)
    meta = json.loads((c / f"meta_{size}.json").read_text())

    def arr(name):
        return np.memmap(c / f"{name}.u8", dtype=np.uint8, mode="r", shape=(meta[name]["n"], size, size, 3))

    yl = np.load(c / f"l_train_{size}_y.npy"); yv = np.load(c / f"val_{size}_y.npy"); yt = np.load(c / f"test_{size}_y.npy")
    K = int(max(yl.max(), yv.max(), yt.max())) + 1
    n_l = np.bincount(yl, minlength=K)
    order = np.argsort(-n_l, kind="stable")
    head = np.zeros(K, dtype=bool); head[order[: K // 2]] = True
    xu_out = None
    if u_out_n > 0:
        name = [k for k in meta if k.startswith(f"u_train_out_{size}_n")]
        if not name:
            raise FileNotFoundError("u_train_out 캐시가 없습니다. build_cache(..., u_out_n=...)을 먼저 실행하세요.")
        xu_out = arr(name[0])
    d = ImgData(K=K, size=size, xl=arr(f"l_train_{size}"), yl=yl, xv=arr(f"val_{size}"), yv=yv, xt=arr(f"test_{size}"), yt=yt,
                xu_in=arr(f"u_train_in_{size}"), xu_out=xu_out, n_l=n_l, order=order, head=head)
    if verbose:
        print(f"[데이터] 종 {K} | labeled {len(yl):,} (종마다 {n_l.min()}~{n_l.max()}, 중앙값 {int(np.median(n_l))}) | val {len(yv):,} | test {len(yt):,}")
        print(f"         unlabeled 같은 종 {len(d.xu_in):,}" + (f" | 다른 종 {len(xu_out):,}" if xu_out is not None else "") + f" | 저장 크기 {size}px")
    return d


def to_tensor(a: np.ndarray, device) -> torch.Tensor:
    """(N, S, S, 3) uint8 배열 -> device의 (N, 3, S, S) uint8 텐서."""
    return torch.from_numpy(np.array(a)).to(device).permute(0, 3, 1, 2).contiguous()
