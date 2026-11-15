# -*- coding: utf-8 -*-
"""데이터: HLS 시계열 로드, 공간 블록 분할, 실험 조건별 labeled/unlabeled 구성.

별도 연구 저장소의 코드는 가져오지 않는다. 같은 CSV(HLS_TimeSeries_2022.csv)와 그 저장소가 만들어 둔
dense 캐시(hls_dense_cache.pt), 좌표 캐시(site_xy.npy)를 **읽기만** 한다. 캐시가 없으면 CSV를 직접 읽고
이 실험의 작업 폴더(work_dir)에 따로 저장한다(연구 저장소의 캐시 경로에는 쓰지 않는다).

CSV 스키마(long format): sample_id, timestep(1..122), 22개 시계열 채널, class, latitude, longitude.
latitude/longitude 열은 투영 좌표(미터)다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_BASE = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "NDVI", "NDBI", "MNDWI", "UI")
TIME_VARYING_COLS: list[str] = list(_BASE) + [f"nearby_{c}" for c in _BASE]

# 실험 조건. l: labeled 구성, u: unlabeled 구성.
#   balanced  : 클래스당 n개
#   longtail  : n_k = n_max * gamma_l^(-k/(K-1)), k = 0(head) .. K-1(tail)
#   uniform   : 모든 클래스 같은 개수
#   inverse   : labeled와 반대 방향의 long-tail (labeled tail 클래스에 unlabeled가 가장 많다)
#   consistent: labeled와 같은 방향의 long-tail
CONDITIONS: dict[str, dict] = {
    "base": dict(l="balanced", n=100, u="uniform", desc="기준 조건(가정 성립)"),
    "A": dict(l="balanced", n=4, u="uniform", desc="극소량 레이블"),
    "B": dict(l="longtail", n_max=100, gamma_l=25.0, u="inverse", gamma_u=25.0, desc="클래스 분포 불일치"),
    # 선택 조건: B에서 '불일치'만 뺀 대조군(long-tail labeled + 같은 방향 unlabeled). 기본 실행에는 포함하지 않는다.
    "Bc": dict(l="longtail", n_max=100, gamma_l=25.0, u="consistent", gamma_u=25.0, desc="long-tail, 분포 일치(B의 대조군)"),
}
N_RESERVED = 100  # 클래스마다 앞 100개 자리는 labeled 후보로 비워 둔다(조건 간 unlabeled 집합을 같게 하려고)


@dataclass
class HLS:
    x: torch.Tensor            # (N, T, C) float32, 표준화 전
    y: torch.Tensor            # (N,) 0..23
    sample_ids: np.ndarray
    xy: np.ndarray | None      # (N, 2) 투영 좌표(미터) 또는 None


def _read_csv_dense(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    need = set(TIME_VARYING_COLS + ["class", "sample_id", "timestep"])
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"CSV에 다음 열이 없습니다: {sorted(miss)}")
    ids = np.sort(df["sample_id"].unique())
    ts = np.sort(df["timestep"].unique())
    r = df["sample_id"].map({s: i for i, s in enumerate(ids)}).to_numpy()
    c = df["timestep"].map({t: i for i, t in enumerate(ts)}).to_numpy()
    x = np.zeros((len(ids), len(ts), len(TIME_VARYING_COLS)), dtype=np.float32)
    for ch, name in enumerate(TIME_VARYING_COLS):
        x[r, c, ch] = df[name].to_numpy(dtype=np.float32)
    cls = df.drop_duplicates("sample_id").set_index("sample_id").loc[ids, "class"].to_numpy()
    raw = sorted(int(v) for v in np.unique(cls))
    lm = {v: i for i, v in enumerate(raw)}
    y = np.array([lm[int(v)] for v in cls], dtype=np.int64)
    return ids.astype(np.int64), x, y


def load_hls(csv_path: str, dense_cache: str | None = None, xy_cache: str | None = None, work_dir: str = ".") -> HLS:
    """HLS 데이터와 지점 좌표를 읽는다. 파일이 없으면 즉시 에러(합성 데이터로 대체하지 않는다)."""
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    own_dense, own_xy = work / "hls_dense.pt", work / "site_xy.npy"
    blob = None
    for p in (dense_cache, str(own_dense)):
        if p and os.path.exists(p):
            blob = torch.load(p, weights_only=False, map_location="cpu"); break
    if blob is not None:
        x, y = blob["x_time"].float(), blob["y"].long()
        ids = np.asarray(blob["sample_ids"].tolist() if hasattr(blob["sample_ids"], "tolist") else blob["sample_ids"], dtype=np.int64)
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"HLS CSV가 없습니다: {csv_path}")
        ids, xn, yn = _read_csv_dense(csv_path)
        x, y = torch.from_numpy(xn), torch.from_numpy(yn)
        torch.save(dict(x_time=x, y=y, sample_ids=torch.from_numpy(ids)), own_dense)

    xy = None
    for p in (xy_cache, str(own_xy)):
        if p and os.path.exists(p):
            a = np.load(p)
            if a.shape == (len(ids), 2):
                xy = a; break
    if xy is None and os.path.exists(csv_path):
        import pandas as pd

        d = pd.read_csv(csv_path, usecols=["sample_id", "latitude", "longitude"]).drop_duplicates("sample_id").set_index("sample_id")
        xy = d.loc[ids, ["longitude", "latitude"]].to_numpy(dtype=np.float64)
        np.save(own_xy, xy)
    return HLS(x=x, y=y, sample_ids=ids, xy=xy)


# --------------------------------------------------------------------------- 분할
def block_split(xy: np.ndarray, y: np.ndarray, block_km: float = 25.0, frac=(0.6, 0.2, 0.2), tries: int = 300, seed: int = 0) -> dict:
    """공간 블록 단위 train/val/test 분할. 블록 배정을 여러 번 뽑아, 모든 클래스가 세 구역에 고르게 들어가는 배정을 고른다.
    같은 블록의 지점은 한 구역에만 들어가므로 인접 지점이 학습과 테스트에 함께 들어가지 않는다."""
    if xy is None:
        raise ValueError("공간 블록 분할에는 지점 좌표가 필요합니다(CSV의 latitude/longitude 또는 site_xy.npy).")
    size = block_km * 1000.0
    b = np.floor(xy / size).astype(np.int64)
    _, bid = np.unique(b, axis=0, return_inverse=True)
    bid = bid.reshape(-1)
    nb = int(bid.max()) + 1
    bsz = np.bincount(bid, minlength=nb)
    K = int(y.max()) + 1
    cnt = np.zeros((nb, K), dtype=np.int64)
    np.add.at(cnt, (bid, y), 1)
    tot = cnt.sum(0).astype(np.float64)
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    for _ in range(tries):
        perm = rng.permutation(nb)
        cum = np.cumsum(bsz[perm]) / bsz.sum()
        part = np.zeros(nb, dtype=np.int64)
        part[perm[cum > frac[0]]] = 1
        part[perm[cum > frac[0] + frac[1]]] = 2
        ratio = np.stack([cnt[part == p].sum(0) / np.maximum(tot, 1) / frac[p] for p in range(3)])
        score = float(ratio[:, tot > 0].min())
        if score > best_score:
            best, best_score = part.copy(), score
    part_of = best[bid]
    return dict(train=np.where(part_of == 0)[0], val=np.where(part_of == 1)[0], test=np.where(part_of == 2)[0],
                block_id=bid, n_blocks=nb, score=best_score, block_km=block_km)


def random_split(y: np.ndarray, frac=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    """클래스별 무작위 분할. 공간 자기상관 때문에 성능이 높게 나오므로 블록 분할과의 비교용으로만 쓴다."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0]); n = len(idx)
        a, b2 = int(n * frac[0]), int(n * (frac[0] + frac[1]))
        out["train"].append(idx[:a]); out["val"].append(idx[a:b2]); out["test"].append(idx[b2:])
    return {k: np.sort(np.concatenate(v)) for k, v in out.items()}


# --------------------------------------------------------------------------- 묶음
@dataclass
class Bundle:
    x: torch.Tensor                 # (N, T, C) 표준화된 입력
    y: torch.Tensor                 # (N,) 0..K-1, 제외된 클래스는 -1
    K: int
    class_ids: list[int]            # 새 라벨 -> 원래 클래스 번호
    dropped: list[int]              # 표본이 부족해 뺀 원래 클래스 번호
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    counts: dict = field(default_factory=dict)   # 구역별 클래스 개수(새 라벨 기준)
    split_info: dict = field(default_factory=dict)


def class_table(y: np.ndarray, split: dict, K: int | None = None) -> np.ndarray:
    K = int(y.max()) + 1 if K is None else K
    return np.stack([np.bincount(y[split[k]][y[split[k]] >= 0], minlength=K) for k in ("train", "val", "test")], 1)


def make_bundle(hls: HLS, split: dict, min_train: int = 300, min_val: int = 10, min_test: int = 20, verbose: bool = True) -> Bundle:
    """표본이 부족한 클래스를 빼고 라벨을 다시 매긴 뒤, 학습 구역 통계로 채널별 표준화를 한다.
    min_train: 학습 구역의 클래스당 최소 지점 수(앞 100개는 labeled 후보, 나머지가 unlabeled 후보)."""
    y0 = hls.y.numpy()
    tab = class_table(y0, split)
    keep = [c for c in range(tab.shape[0]) if tab[c, 0] >= min_train and tab[c, 1] >= min_val and tab[c, 2] >= min_test]
    drop = [c for c in range(tab.shape[0]) if c not in keep]
    if len(keep) < 2:
        raise ValueError(f"조건을 만족하는 클래스가 {len(keep)}개뿐입니다. min_train({min_train})을 낮추거나 분할을 확인하세요.\n{tab}")
    remap = -np.ones(tab.shape[0], dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    y = remap[y0]
    tr = split["train"][y[split["train"]] >= 0]; va = split["val"][y[split["val"]] >= 0]; te = split["test"][y[split["test"]] >= 0]
    mu = hls.x[tr].mean(dim=(0, 1), keepdim=True); sd = hls.x[tr].std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    x = (hls.x - mu) / sd
    new_tab = class_table(y, dict(train=tr, val=va, test=te), K=len(keep))
    if verbose:
        print(f"[데이터] 지점 {len(y0):,} | 클래스 {tab.shape[0]} -> 사용 {len(keep)} | 제외 {drop if drop else '없음'}")
        print(f"[분할]   train {len(tr):,} / val {len(va):,} / test {len(te):,}" + (f" | 블록 {split.get('n_blocks')}개({split.get('block_km')} km)" if "n_blocks" in split else " | 무작위"))
        print("         원래 클래스: (train, val, test) 지점 수")
        for c in range(tab.shape[0]):
            print(f"         {c:>2}: {tuple(int(v) for v in tab[c])}" + ("" if c in keep else "   <- 제외"))
    return Bundle(x=x, y=torch.from_numpy(y), K=len(keep), class_ids=keep, dropped=drop, train=tr, val=va, test=te,
                  counts=dict(train=new_tab[:, 0].tolist(), val=new_tab[:, 1].tolist(), test=new_tab[:, 2].tolist()),
                  split_info={k: v for k, v in split.items() if k in ("n_blocks", "score", "block_km")})


# --------------------------------------------------------------------------- 조건
def _lt(n_max: float, gamma: float, K: int) -> np.ndarray:
    return n_max * gamma ** (-np.arange(K) / max(K - 1, 1))


def build_condition(b: Bundle, cond: str, seed: int, u_cap: int | None = None) -> dict:
    """조건(cond)과 seed에 맞는 labeled/unlabeled 지점 번호를 만든다.
    클래스마다 학습 구역 지점을 seed로 섞은 뒤, 앞에서부터 labeled를, N_RESERVED번째부터 unlabeled를 자른다.
    그래서 base와 A는 같은 seed에서 unlabeled 집합이 같고, A의 labeled는 base의 labeled에 포함된다.
    long-tail 조건의 클래스 순서: 학습 구역 지점이 적은 클래스가 labeled head(레이블 많음)이고 많은 클래스가 tail이다.
    inverse에서 tail 클래스에 unlabeled가 가장 많이 필요하기 때문에 이렇게 둔다."""
    spec = CONDITIONS[cond]
    y = b.y.numpy(); K = b.K
    per = []
    for c in range(K):
        idx = b.train[y[b.train] == c]
        per.append(np.random.default_rng(100_003 * seed + c).permutation(idx))
    avail = np.array([len(p) for p in per])
    room = avail - N_RESERVED
    if spec["l"] == "balanced":
        order = np.arange(K)
        n_l = np.full(K, int(spec["n"]))
        u = int(room.min()) if u_cap is None else min(int(room.min()), u_cap)
        n_u = np.full(K, u)
    else:
        order = np.argsort(avail, kind="stable")          # order[k] = k번째(head부터) 클래스
        n_pos = np.maximum(np.round(_lt(spec["n_max"], spec["gamma_l"], K)), 1).astype(int)
        r = _lt(1.0, spec["gamma_u"], K)
        r = r[::-1] if spec["u"] == "inverse" else r      # inverse: tail일수록 unlabeled가 많다
        u_max = float(np.min(room[order] / r))
        if u_cap is not None:
            u_max = min(u_max, float(u_cap))
        u_pos = np.maximum(np.floor(u_max * r), 1).astype(int)
        n_l = np.zeros(K, dtype=int); n_u = np.zeros(K, dtype=int)
        n_l[order] = n_pos; n_u[order] = u_pos
    if (n_l > N_RESERVED).any() or (n_u > room).any() or (n_u < 1).any():
        raise ValueError(f"조건 {cond}을 만들 지점이 부족합니다. 클래스별 학습 지점 {avail.tolist()}, 필요 labeled {n_l.tolist()}, unlabeled {n_u.tolist()}")
    l_idx = np.concatenate([per[c][: n_l[c]] for c in range(K)])
    u_idx = np.concatenate([per[c][N_RESERVED: N_RESERVED + n_u[c]] for c in range(K)])
    head = None
    if spec["l"] == "longtail":
        head = np.zeros(K, dtype=bool); head[order[: K // 2]] = True
    return dict(cond=cond, seed=seed, l_idx=l_idx, u_idx=u_idx, n_l=n_l, n_u=n_u, order=order, head=head, spec=dict(spec))


def describe_conditions(b: Bundle, conds=("base", "A", "B"), seed: int = 0) -> None:
    """실행 전에 조건별 labeled/unlabeled 규모를 확인한다."""
    for c in conds:
        d = build_condition(b, c, seed)
        print(f"[조건 {c}] {CONDITIONS[c]['desc']} | labeled {len(d['l_idx']):,} (클래스당 {d['n_l'].min()}~{d['n_l'].max()}) | "
              f"unlabeled {len(d['u_idx']):,} (클래스당 {d['n_u'].min()}~{d['n_u'].max()})")
        if d["head"] is not None:
            o = d["order"]
            print("          head->tail 순서의 (labeled, unlabeled): " + " ".join(f"({d['n_l'][k]},{d['n_u'][k]})" for k in o))
