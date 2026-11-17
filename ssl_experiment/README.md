# 실험 코드: 실행 상세

결과와 실행 방법의 요약은 저장소 최상위 [README](../README.md)에 있다. 이 문서는 노트북 설정과 구현의 상세다.

## 시계열 실험 (`sslexp`, `SSL_experiment_colab.ipynb`)

노트북 첫 셀의 경로 상수. 위치가 다르면 여기서 고친다.

| 상수 | 기본값 | 내용 |
| --- | --- | --- |
| `DATA` | `/content/drive/MyDrive/AI4Biodynamics/data` | `HLS_TimeSeries_2022.csv`, `hls_dense_cache.pt`, `site_xy.npy` (읽기만 함) |
| `CODE_DIR` | `/content/drive/MyDrive/ssl_seminar/ssl_experiment` | 이 폴더 |
| `OUT` | `/content/drive/MyDrive/ssl_seminar/results_v1` | 결과 저장 |

- 처음에는 `SEEDS = [0]`, `CONDITIONS = ["A"]`로 한 묶음만 돌려 시간과 출력을 확인한다.
- `WORKERS = "auto"`는 vCPU 수와 GPU 여유 메모리에 맞춰 동시에 돌릴 학습 수를 정한다(상한 8). GPU 메모리 오류가 나면 줄인다. 병렬로 돌려도 결과는 같다.
- 설정(`iters` 등)을 바꿀 때는 `OUT`을 새 폴더로 잡는다.
- `MIN_TRAIN`(기본 300): 학습 구역 지점이 이보다 적은 클래스는 뺀다. 노트북 2번 셀의 출력에서 빠진 클래스와 조건별 규모를 확인한다.

기본 설정

- 분할: 지점 좌표를 25 km 블록으로 나눠 블록 단위로 train / val / test(60 / 20 / 20)에 배정. 테스트 구역은 labeled로도 unlabeled로도 쓰지 않는다.
- 조건: `base`(클래스당 100), `A`(클래스당 4), `B`(labeled long-tail 100 → 4, unlabeled는 반대 방향). 선택 조건 `Bc`(같은 방향의 long-tail)는 노트북 마지막 셀에 있고 돌리지 않았다.
- 학습: 6,000 스텝, labeled 배치 32, unlabeled 배치 224, AdamW(lr 1e-3), EMA 0.999, 250 스텝마다 val macro-F1로 모델 선택.
- 방법 이름: `supervised`, `pseudolabel`(Self-training), `fixmatch`, `cgmatch`, `interlude`, `scssl`

## 이미지 실험 (`sslimg`, `SSL_image_colab.ipynb`)

데이터 준비 (처음 한 번, 약 18분)

1. https://www.kaggle.com/c/semi-inat-2020 에서 **Join the competition**을 눌러 규칙에 동의한다.
2. Kaggle API 토큰을 Colab 보안 비밀에 `KAGGLE_API_TOKEN`이라는 이름으로 넣고 "노트북 액세스"를 켠다. 토큰을 노트북 셀에 직접 적지 않는다.
3. 노트북이 대회 파일(15.35 GB)을 받아 정사각 uint8 캐시(약 3.8 GB)를 만들고 Drive에 복사한다. 다음 세션부터는 캐시만 복사한다.
4. 직접 받아 둔 폴더가 있으면 노트북의 `ARCHIVES`에 경로를 적는다.

기본 설정

| 항목 | 값 |
| --- | --- |
| 인코더 | ResNet-50, ImageNet 사전학습 |
| 입력 | 176×176으로 저장, 학습은 160×160 무작위 crop, 평가는 가운데 crop |
| 증강 | 약: crop + 좌우 반전 / 강: RandAugment + Cutout |
| 배치 | labeled 16, unlabeled 112 |
| 최적화 | SGD(Nesterov), 인코더 lr 0.003, 분류 헤드 lr 0.03, cosine 감쇠, 4,000 스텝, 혼합 정밀도 |
| 조건 | `in`(같은 200종 26,640장), `open`(`in` + 다른 종 26,640장, 노트북의 선택 셀) |

- 한 번에 하나씩 돈다. 이미지 데이터를 GPU에 통째로 올리기 때문에 병렬 워커는 쓰지 않는다.
- unlabeled의 레이블이 공개되지 않아 pseudo-label 정확도는 잴 수 없다. 활용률과 head 종 비율을 남긴다.

## 결과 파일

| 파일 | 내용 |
| --- | --- |
| `tables/test_macro_f1.csv` (시계열), `tables/test_acc.csv` (이미지) | 방법 × 조건의 test 성능 |
| `tables/gain_vs_fixmatch.csv`, `gain_vs_selftraining.csv` | 같은 시드끼리 뺀 이득 |
| `tables/tail_f1.csv` (시계열), `head_acc.csv` · `tail_acc.csv` (이미지) | head / tail 클래스 성능 |
| `tables/diagnosis.csv` | pseudo-label 활용률 · 정확도, 강한 증강 전후 예측 변화율 |
| `tables/tables.md` | 위 표를 모은 것 |
| `runs/*.json` | 실행별 원자료 (학습 곡선, 클래스별 성능, pseudo-label 클래스 분포, 설정) |

## 원 논문과 다른 점

세 기법은 논문의 식을 따랐지만 재현이 아니라 옮긴 구현이다. 방법별 하이퍼파라미터는 논문 값 그대로다(τ = 0.95, α = 0.1, τ_b = 2, τ_e = 4 등).

| 항목 | 원 논문 | 이 코드 |
| --- | --- | --- |
| 시계열 증강 | 이미지 weak / RandAugment | 약: 크기 변화 + 잡음 / 강: 시점 masking + 밴드 drop + 진폭 왜곡 |
| 시계열 인코더 · 최적화 | WRN-28-2 등, SGD, 약 100만 스텝 | InceptionTime 계열, AdamW, 6,000 스텝 |
| 이미지 해상도 · 학습 길이 | 224×224, 수만 ~ 백만 스텝 | 160×160, 4,000 스텝 |
| 이미지 RandAugment | 표본마다 연산을 뽑음 | 4장 묶음마다 뽑음 (`ImgCfg(aug_group=1)`이면 표본마다) |
| CGMatch warm-up | 2,048 스텝 | 500 스텝 |
| CGMatch 임계값 초깃값 | 명시 없음 | 확신 임계값 1/K, Count-Gap 임계값 0에서 시작해 EMA로 갱신 |
| InterLUDE | InterLUDE와 InterLUDE+ | InterLUDE만 |
| SC-SSL 분포 추정 시점 | "several epochs" | 학습의 15% 시점에 한 번 |
| SC-SSL 임계값 감소 | 식 (8), 주기 명시 없음 | 250 스텝마다 한 번, 하한 0.3 |
| SC-SSL 후보 분포의 불균형비 | 실험 설정으로 주어짐 | labeled의 불균형비. labeled가 균형이면 후보 3종만 사용 |
| SC-SSL 손실 재가중 | unlabeled 개수 사전정보로 재가중 | 구현하지 않음 |
| SC-SSL 배치 (이미지) | 64 / 128 | 다른 방법과 같은 16 / 112 |

## 파일

```
sslexp/   data.py (분할, 조건 구성)  augment.py  model.py  methods.py (방법 6종)  train.py  run.py (실행기, 표, CLI)  smoke.py
sslimg/   data.py (내려받기, 캐시)   aug.py      model.py  train.py               run.py   smoke.py
```
