# 공식 구현 재현 (`sslrepro`, `semilearn`)

`sslexp`, `sslimg`는 논문의 식을 옮긴 구현이다. `sslrepro`는 세 논문(InterLUDE, CGMatch, SC-SSL)의 공개 코드를 써서 같은 Semi-Aves에서 다시 돌린 비교와 ablation이다. 224px, 10,000 스텝, 시드 1회.

- 코드: SC-SSL 공개 코드(USB 라이브러리, `../semilearn/`)에 CGMatch 공개 코드를 옮겨 합쳤다. InterLUDE 공개 코드는 USB 기반이 아니어서 `semilearn/algorithms/interlude/`로 다시 썼다.
- 데이터: 원본 이미지와 `annotation/`의 json을 그대로 읽는다(`semilearn/datasets/cv_datasets/semi_inat.py`). test 레이블은 `solution.csv`에서 읽는다.
- 실행: `run.py`가 방법마다 설정 파일(`ckpt/{조건}__{방법}__seed{S}/config.yaml`)을 만들어 `train.py`로 학습하고, `evaluate.py`로 val 최고 체크포인트와 마지막 체크포인트를 val · test에서 평가해 `runs/{조건}__{방법}__seed{S}.json`에 저장한다.

## 실행 방법 (로컬 GPU)

추가 패키지: `pyyaml`, `ruamel.yaml==0.17.21`, `scikit-learn`, `tensorboard`, `timm==0.6.12`, `transformers==4.40.2`, `tqdm`, `progress` (`torch 2.4`, `numpy<2`에서 확인)

Kaggle `semi-inat-2020`의 원본 폴더(`annotation/`, `trainval_images/`, `u_train_in/`, `test/`)와 [`solution.csv`](https://github.com/cvl-umass/semi-inat-2020)를 `ssl_experiment/data/semi-inat-2020/`에 둔다.

```bash
cd ssl_experiment
python3 -m sslrepro.smoke    # 합성 데이터로 전체 경로 점검 (GPU, 1분)
python3 -m sslrepro.run --out ../../results_repro --seeds 0
python3 -m sslrepro.run --out ../../results_repro_40k --seeds 0 --iters 40000 \
    --methods supervised fixmatch interlude
```

- 한 번에 하나씩 돈다. 처음에는 `--methods fixmatch --iters 1000`처럼 하나만 짧게 돌려 시간을 확인한다.
- RTX A6000 한 장에서 10k 실행 10회가 약 14시간 걸린다. 끊기면 같은 명령을 다시 실행한다. 끝난 실행은 건너뛰고, 중단된 실행은 마지막 체크포인트에서 이어 간다.
- 결과는 `--out`의 `tables/`와 `runs/`에, 체크포인트(`.pth`)는 `ckpt/`에 저장된다. 체크포인트가 커서 저장소 밖을 `--out`으로 준다.

## 기본 설정 (모든 방법 공통)

| 항목 | 값 |
| --- | --- |
| 인코더 | ResNet-50, ImageNet 사전학습 |
| 입력 | 256으로 줄여 224×224 crop, 평가는 가운데 crop |
| 증강 | 약: crop + 좌우 반전 / 강: RandomResizedCrop + RandAugment(3, 10) |
| 배치 | labeled 32, unlabeled 160 |
| 최적화 | SGD(momentum 0.9), lr 0.01, cosine 감쇠, weight decay 5e-4, 10,000 스텝, 혼합 정밀도 |
| 모델 선택 | EMA(0.999) 모델, 500 스텝마다 val top-1 정확도 |
| 평가 | test 8,000장(종마다 40장). head 종은 labeled가 많은 절반 |
| 조건 | `in`(같은 200종 26,640장). `open`도 지원하지만 돌리지 않았다 |

- 방법별 하이퍼파라미터는 논문 값 그대로다(τ = 0.95, InterLUDE δ = 0.1 · λ_DC = 1, CGMatch queue 32).
- SC-SSL은 FixMatch 위에 얹고, 사후 보정한 보조 head(h2)로 보고한다. 보정 전후 4가지 head의 정확도는 `extra.heads_test_acc`에 남긴다.
- InterLUDE는 embedding fusion을 끄고 평가한다. 공식 코드처럼 켠 결과(test 파일 순서 / 섞음)는 `extra.ef_at_eval_test_acc`에 남긴다.

## 원 논문과 다른 점

| 항목 | 원 논문 | `sslrepro` |
| --- | --- | --- |
| 데이터 · 학습 길이 | CIFAR · STL 등, WRN을 처음부터 2^18 ~ 2^20 스텝 | Semi-Aves, 사전학습 인코더로 10,000 스텝 (1 ~ 4%) |
| 배치 | 64 / 448 (SC-SSL 64 / 128) | 32 / 160 |
| CGMatch warm-up | 2,048 스텝 동안 매 스텝 unlabeled 전체를 평가 | 480 스텝, 15 스텝마다 32번만 평가 |
| CGMatch · SC-SSL의 unlabeled 분포 | 균등 또는 설정으로 주어짐 | labeled 분포와 같다고 가정 |
| InterLUDE 평가 | 평가 때도 embedding fusion | 끄고 평가 |
| SC-SSL 학습률 | 상수 | 다른 방법과 같은 cosine |

## Ablation (`--methods`에 이름을 준다)

| 이름 | 바꾼 것 |
| --- | --- |
| `interlude_noDC` | delta-consistency 손실 제거 (embedding fusion만) |
| `interlude_noEF` | embedding fusion 제거 (delta-consistency만) |
| `cgmatch_easyonly` | ambiguous 표본의 GCE 손실 제거 |
| `cgmatch_noCG` | Count-Gap 대신 확신도로 easy / ambiguous 분할 |
| `scssl_uniform` | unlabeled 분포를 균등으로 가정 |
| `scssl_origopt` | SC-SSL 원 코드의 최적화 (상수 lr, BN · bias에도 weight decay) |
| `scssl_origopt_u2` | `scssl_origopt`에 unlabeled 배수 2 (원 논문의 64 / 128 비율) |

결과 표는 test_acc, gain_vs_fixmatch, tail_acc, head_acc에 더해 `ablation_test_acc`, `ablation_gain_vs_full`(같은 시드의 전체 버전과 뺀 값), `scssl_heads`, `interlude_ef_at_eval`, `diagnosis`(val · test · 마지막 스텝 test 정확도, 선택된 스텝)가 있다.

## 결과 (Semi-Aves In-class, Test top-1 정확도, 시드 1회)

| 방법 | 10k 스텝 | Tail 종 | FixMatch 대비 | 40k 스텝 |
| --- | --- | --- | --- | --- |
| Supervised only | 0.416 | 0.385 | | 0.410 |
| FixMatch | 0.484 | 0.444 | | 0.478 |
| CGMatch | **0.514** | **0.487** | **+0.029** | |
| InterLUDE | 0.476 | 0.428 | −0.009 | **0.491** |
| SC-SSL | 0.439 | 0.441 | −0.046 | |

| Ablation (10k) | 설정 | Test | 전체 대비 |
| --- | --- | --- | --- |
| InterLUDE | DC 제거 (EF만) | 0.473 | −0.002 |
| InterLUDE | EF 제거 (DC만) | 0.479 | +0.003 |
| CGMatch | ambiguous 손실 제거 | 0.498 | −0.015 |
| CGMatch | Count-Gap 대신 확신도로 분할 | 0.515 | +0.001 |
| SC-SSL | unlabeled 분포를 균등으로 가정 | 0.431 | −0.008 |
| SC-SSL | 원 코드 최적화 (상수 lr, weight decay 전체 적용) | 0.412 | −0.027 |
| SC-SSL | 원 코드 최적화 + unlabeled 배수 2 (원 논문 비율) | 0.320 | −0.119 |

- 직접 구현(`sslimg`)과 같이 CGMatch가 In-class에서 FixMatch보다 높다(+0.029). ambiguous 손실을 빼면 0.015 내려가지만, Count-Gap을 확신도로 바꿔도 그대로다.
- InterLUDE는 10k에서 FixMatch보다 낮고 40k에서 높다(+0.013). EF와 DC를 하나씩 빼도 차이가 0.003 이내다.
- InterLUDE 공식 코드처럼 평가 때도 embedding fusion을 켜도 test는 오르지 않는다(0.476 → 0.474). fusion은 배치 안의 이웃 표본을 섞으므로, 종별로 정렬된 val에서만 오른다.
- SC-SSL은 전체 정확도가 FixMatch보다 0.046 낮지만, 5개 방법 중 유일하게 tail 종이 head 종(0.437)보다 높다. 직접 구현(+0.012)과 어긋난다. 원 코드의 최적화 설정(상수 lr, unlabeled 배수 2)으로 다시 돌려도 더 낮았다(0.412, 0.320). 두 실행 모두 val 정확도가 5,000 스텝 무렵까지 0.15 아래에 머물다 10k 스텝 끝까지 오르고 있어, 짧은 학습에서는 원 설정이 불리하다. 이 데이터에서 SC-SSL이 FixMatch보다 낮은 원인은 찾지 못했다.
- 전체 표(head 종, SC-SSL head별, fusion, 마지막 스텝 체크포인트)는 `results/repro/tables/tables.md`, `results/repro_40k/tables/tables.md`에 있다.

**한계**

- 시드 1회다. 원 논문의 2^18 ~ 2^20 스텝 대비 1 ~ 4%만 학습했다.
- 40k 실행 중 FixMatch와 InterLUDE 일부는 네트워크 드라이브에서 읽어 소요 시간이 다른 실행과 비교되지 않는다.

## 파일

```
sslrepro/   run.py (방법 설정, 실행기, 표, CLI)  train.py (학습)  evaluate.py (val · test 평가)  smoke.py
semilearn/  USB 라이브러리. 추가한 것: algorithms/interlude, algorithms/cgmatch, imb_algorithms/scssl,
            datasets/cv_datasets/semi_inat.py, nets/resnet/resnet_in.py
results/repro/      tables/, runs/ (실행별 json, 10k 스텝)
results/repro_40k/  tables/, runs/ (실행별 json, 40k 스텝)
```
