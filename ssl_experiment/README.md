# 위성 시계열 데이터 기반 Semi-Supervised Learning 실험

기계지능특론 세미나용 실험 코드다. HLS 2022 위성 시계열(24,009개 지점, 24개 클래스, 122시점 × 22채널)을 inductive SSL 설정으로 돌려, 세 논문(CGMatch, InterLUDE, SC-SSL)이 각자 완화한 가정의 위배 조건에서 고전 기법보다 실제로 더 기여하는지 본다.

별도 연구 저장소의 코드는 쓰지 않는다. 같은 데이터 파일만 읽기 전용으로 쓴다. 데이터 파일은 이 저장소에 들어 있지 않다.

> 실험 결과와 해석은 저장소 최상위 [README](../README.md)에, 결과 원자료는 [`results/`](../results)에 있다. 이 문서는 구현과 실행 방법의 상세다.

## 실행 (Colab)

1. 이 폴더(`ssl_experiment`)를 Google Drive의 `MyDrive/ssl_seminar/` 아래에 올린다.
2. `SSL_experiment_colab.ipynb`를 Colab에서 열고 GPU 런타임으로 위에서부터 실행한다.
3. 결과 표는 `MyDrive/ssl_seminar/results_v1/tables/`에 저장된다.

노트북의 경로 상수는 다음과 같다. 위치가 다르면 첫 셀에서 고친다.

| 상수 | 기본값 | 내용 |
| --- | --- | --- |
| `DATA` | `/content/drive/MyDrive/AI4Biodynamics/data` | `HLS_TimeSeries_2022.csv`, `hls_dense_cache.pt`, `site_xy.npy` (읽기만 함) |
| `CODE_DIR` | `/content/drive/MyDrive/ssl_seminar/ssl_experiment` | 이 폴더 |
| `OUT` | `/content/drive/MyDrive/ssl_seminar/results_v1` | 결과 저장 |

주 실험 셀은 시작할 때 예상 소요 시간을 출력하고, 도는 동안 진행 막대 두 줄(전체 진행과 남은 시간 / 지금 도는 실행의 스텝과 val F1)을 보여 준다. 실행이 하나 끝날 때마다 실제 걸린 시간으로 남은 시간을 다시 계산한다. 예상 시간은 스텝당 처리하는 표본 수에 비례한다고 보고 계산한 값이라 평가·저장 시간만큼 실제가 조금 더 길다.

**병렬 실행**: 주 실험 셀의 `WORKERS = "auto"`는 런타임을 읽어 동시에 돌릴 학습 수를 정한다. 워커 하나가 CPU 코어 하나와 GPU 메모리 1.5 GB를 쓴다고 보고, vCPU 수와 GPU 여유 메모리 중 작은 쪽에 맞춘다(상한 8). vCPU가 2개인 런타임은 2개, 12개인 런타임은 최대 8개가 된다. 워커들은 GPU 하나를 나눠 쓰므로 워커 수만큼 빨라지지는 않는다. GPU 메모리가 부족하다는 오류가 나면 `WORKERS`를 줄이고, 차례로 돌리려면 1로 둔다. 병렬로 돌려도 결과는 차례로 돌린 것과 같다(실행마다 seed가 고정).

끊기면 같은 셀을 다시 실행한다. 끝난 실행은 건너뛴다. 설정(`iters` 등)을 바꿀 때는 `OUT`을 새 폴더로 잡는다.

로컬에서 코드만 확인하려면 `python3 -m sslexp.smoke`를 실행한다(합성 데이터, CPU 약 4분, 34개 검사).

## 실험 구성

**분할**: 지점 좌표를 25 km 블록으로 나눠 블록 단위로 train / val / test(60 / 20 / 20)에 배정한다. 테스트 구역의 지점은 labeled로도 unlabeled로도 쓰이지 않는다. 분할은 고정이고 seed는 labeled 추출과 모델 초기화만 바꾼다.

**방법 6종**

| 이름 | 표의 이름 | 내용 |
| --- | --- | --- |
| `supervised` | Supervised only | labeled만 사용 |
| `pseudolabel` | Self-training | 고전 기법. 고신뢰도 예측을 정답처럼 사용 |
| `fixmatch` | FixMatch | 세 논문의 공통 기반 |
| `cgmatch` | CGMatch | Count-Gap 3분할 |
| `interlude` | InterLUDE | Embedding fusion + delta consistency |
| `scssl` | SC-SSL | 분포 추정, 두 분류기, 추론 시 bias 제거 |

**조건 3종**

| 이름 | labeled | unlabeled | 의미 |
| --- | --- | --- | --- |
| `base` | 클래스당 100개 | 클래스당 같은 개수 | 가정 성립(대조군) |
| `A` | 클래스당 4개 | `base`와 같은 집합 | 극소량 레이블 |
| `B` | long-tail (최대 100개, 최소 4개, 불균형비 25) | 반대 방향의 long-tail | 클래스 분포 불일치 |

- 조건 B의 labeled는 클래스당 25개가 아니라 **long-tail**이다. labeled가 균형이면 "반대 방향의 분포"를 만들 수 없기 때문이다. 평균은 클래스당 약 30개다.
- 조건 B의 클래스 순서는 학습 구역 지점이 적은 클래스가 labeled head, 많은 클래스가 tail이다. tail 클래스에 unlabeled가 가장 많이 필요해서 이렇게 뒀다.
- 선택 조건 `Bc`(long-tail labeled + 같은 방향의 unlabeled)는 B에서 "불일치"만 뺀 대조군이다. 기본 실행에는 없고 노트북 마지막 셀에서 돌릴 수 있다.

**기본 설정**: 6,000 스텝, labeled 배치 32, unlabeled 배치 224, AdamW(lr 1e-3), EMA 0.999, 250 스텝마다 val macro-F1로 모델 선택, seed 3개. 총 54회 학습.

## 결과물

| 파일 | 내용 |
| --- | --- |
| `tables/test_macro_f1.csv` | 방법 × 조건의 Test macro-F1 (평균 ± 표준편차) |
| `tables/gain_vs_fixmatch.csv` | FixMatch 대비 이득 (같은 seed끼리 뺀 값). 모듈 고유의 효과 |
| `tables/gain_vs_selftraining.csv` | Self-training 대비 이득. 고전 기법 대비 총 효과 |
| `tables/tail_f1.csv` | 조건 B의 tail 클래스 F1 |
| `tables/diagnosis.csv` | pseudo-label 활용률·정확도, 학습 구역 unlabeled 정확도, 강한 증강 전후 예측 변화율 |
| `tables/tables.md` | 위 표를 markdown으로 모은 것 |
| `runs/*.json` | 실행별 원자료 (학습 곡선, 클래스별 F1, pseudo-label 클래스 분포 등) |

세 최신 기법은 모두 FixMatch 위에 모듈을 얹은 것이라, 논문 고유의 효과는 FixMatch 대비 이득으로 본다. Self-training 대비 이득에는 FixMatch 자체의 효과가 섞여 있다.

원인 진단 세 항목은 따로 돌릴 필요 없이 주 실험에서 함께 기록된다.
- 증강의 클래스 보존성: `supervised` 실행이 끝날 때 테스트 구역에서 강한 증강 전후에 예측이 바뀌는 비율을 잰다(`aug_flip_rate`).
- Pseudo-label 품질: 모든 SSL 실행이 평가 주기마다 활용률과 정확도를 남긴다(`history`).
- 분포 차이: 학습 구역의 unlabeled 정확도(`u_acc`)와 테스트 구역 정확도의 차이로 본다.

## 실행 전에 확인할 것

노트북 2번 셀이 클래스별 (train, val, test) 지점 수와 조건별 labeled / unlabeled 규모를 출력한다. 학습을 시작하기 전에 이 출력을 본다.

- `MIN_TRAIN`(기본 300): 학습 구역 지점이 이보다 적은 클래스는 실험에서 뺀다. 빠진 클래스가 출력에 표시된다. 너무 많이 빠지면 값을 낮춘다. 단, 100보다는 커야 한다(클래스당 앞 100개가 labeled 후보).
- unlabeled 규모: `base`와 `A`의 unlabeled는 "가장 지점이 적은 클래스의 여유분 × 클래스 수"다. 너무 작으면 `MIN_TRAIN`을 올려 희소 클래스를 뺀다.
- 인코더는 이 실험에서 처음부터 학습하므로 연구 저장소 체크포인트의 레이블 누출 문제는 없다.

## 원 논문과 다른 점

세 최신 기법은 논문의 식을 따랐지만, 이미지가 아닌 시계열 입력과 짧은 학습 길이에 맞춘 부분이 있다. 발표에서 "재현"이 아니라 "시계열 데이터로 옮긴 구현"이라고 말해야 한다.

| 항목 | 원 논문 | 이 코드 |
| --- | --- | --- |
| 증강 | 이미지 weak / RandAugment | weak: 크기 변화 + 잡음 / strong: 시점 masking + 밴드 drop + 진폭 왜곡 |
| 인코더 · 최적화 | WRN-28-2 등, SGD lr 0.03, 약 100만 스텝 | InceptionTime 계열, AdamW lr 1e-3, 6,000 스텝 |
| CGMatch warm-up | 2,048 스텝, 마지막 1,000 스텝의 예측으로 큐 초기화 | 500 스텝, 마지막 250 스텝 |
| CGMatch 임계값 초깃값 | 논문에 명시 없음 | 확신 임계값 1/K, Count-Gap 임계값 0에서 시작해 EMA(0.999)로 갱신 |
| InterLUDE | InterLUDE와 InterLUDE+ | InterLUDE만 구현 |
| SC-SSL 분포 추정 시점 | "several epochs" | 학습의 15% 시점에 한 번 |
| SC-SSL 임계값 감소 | 식 (8), 적용 주기는 논문에 명시 없음 | 평가 주기(250 스텝)마다 한 번, 하한 0.3 |
| SC-SSL 후보 분포의 불균형비 | 실험 설정으로 주어짐 | labeled의 불균형비를 사용. labeled가 균형이면 long-tail 후보를 빼고 3종만 사용 |
| SC-SSL 손실 재가중 | unlabeled 개수 사전정보로 재가중(Lai et al., 2022) | 구현하지 않음 |
| 하이퍼파라미터 | 이미지 벤치마크 기준 | 논문 값 그대로(τ = 0.95, α = 0.1, τ_b = 2, τ_e = 4 등), 재조정하지 않음 |

## 확인된 것과 확인되지 않은 것

- 확인됨: 합성 데이터에서 여섯 방법 모두 끝까지 돌고 학습된다. 분할·조건 구성의 누수 검사(테스트 구역 미사용, labeled와 unlabeled 분리 등)를 통과한다. CSV 파싱과 연구 저장소 캐시 형식 읽기는 같은 스키마의 가짜 CSV로 확인했다.
- 실제 HLS 데이터로 실행함(2026-11-13, Colab T4, 54회, 6시간 46분). 24개 클래스가 모두 조건 구성에 쓰였다(빠진 클래스 없음). 출력이 담긴 노트북은 `notebooks_with_outputs/SSL_experiment_colab_output.ipynb`다.
- 처음 실행할 때는 `SEEDS = [0]`, `CONDITIONS = ["A"]`로 한 묶음만 돌려 시간과 출력을 확인한 뒤 전체를 돌린다.

## 파일

```
sslexp/
  data.py      데이터 로드, 공간 블록 분할, 조건별 labeled/unlabeled 구성
  augment.py   시계열 증강 (weak / strong)
  model.py     인코더, 분류 헤드, EMA
  methods.py   방법 6종의 학습 스텝, 설정(Cfg)
  train.py     학습 한 번과 평가
  run.py       실행기(run_grid), 표(tables), CLI
  smoke.py     합성 데이터 점검
SSL_experiment_colab.ipynb
```


---

# 이미지 실험: Semi-Aves

세 논문이 원래 대상으로 한 이미지 데이터에서 같은 여섯 방법을 비교한다. 시계열 실험과 달리 증강을 바꾸지 않고 표준 구성을 쓴다.

## 데이터

Semi-Supervised iNaturalist-Aves (FGVC7 워크숍, CVPR 2020). 공식 분할을 그대로 쓴다.

| 분할 | 장수 | 내용 |
| --- | --- | --- |
| labeled | 3,959 | 조류 200종, 종마다 5~43장(중앙값 20). 원래 long-tail이고 레이블이 적다 |
| unlabeled (같은 종) | 26,640 | 레이블 비공개 |
| unlabeled (다른 종) | 122,208 | open-set 조건에서만 일부 사용 |
| val | 2,000 | 종마다 10장. 모델 선택에 사용 |
| test | 8,000 | 종마다 40장. 레이블은 대회 뒤 공개된 `solution.csv` |

- 출처와 이용 조건: https://github.com/cvl-umass/semi-inat-2020 . 비상업적 연구·교육 목적으로만 쓰고 이미지를 재배포하지 않는다.
- 주석 파일의 실제 형식과 장수는 이 코드로 읽어 확인했다(위 표와 일치).

## 실행 (Colab)

1. `ssl_experiment` 폴더(`sslexp`, `sslimg` 포함)를 Drive의 `MyDrive/ssl_seminar/`에 올린다.
2. `SSL_image_colab.ipynb`를 GPU 런타임에서 위부터 실행한다.
3. 처음에는 Kaggle에서 대회 파일(15.35 GB)을 받아 정사각 uint8 캐시(약 3.8 GB)를 만들고 Drive에 복사해 둔다. 다음 세션부터는 캐시만 복사한다.
   - 받기 전에 https://www.kaggle.com/c/semi-inat-2020 에서 **Join the competition**을 눌러 규칙에 동의해야 한다.
   - Kaggle API 토큰이 필요하다. Colab 보안 비밀에 `KAGGLE_API_TOKEN`이라는 이름으로 넣고 "노트북 액세스"를 켠다(예전 방식의 `KAGGLE_USERNAME`, `KAGGLE_KEY`도 읽는다). 토큰을 노트북 셀에 직접 적지 않는다.
   - 주석(JSON)과 test 레이블(`solution.csv`)은 GitHub에서 받는다.
   - 직접 받아 둔 폴더가 있으면 노트북의 `ARCHIVES`에 경로를 적는다. 압축 파일이든 풀린 폴더든 그대로 쓴다.
   - 공식 README의 Google Drive 주소는 자동 내려받기가 막혀 있었다(2026-11-13 확인). `source="gdrive"`로 시도는 할 수 있다.
4. 결과 표는 `MyDrive/ssl_seminar/results_img_v1/tables/`에 저장된다. 공식 지표는 test top-1 정확도다.

## 기본 설정

| 항목 | 값 |
| --- | --- |
| 인코더 | ResNet-50, ImageNet 사전학습(대회 규정상 허용) |
| 입력 | 짧은 변 176으로 줄여 가운데 176×176을 저장, 학습은 160×160 무작위 crop, 평가는 가운데 crop |
| 증강 | 약: crop + 좌우 반전 / 강: RandAugment(연산 2개, 세기 무작위) + Cutout |
| 배치 | labeled 16, unlabeled 112 (μ = 7) |
| 최적화 | SGD(momentum 0.9, Nesterov), 인코더 lr 0.003, 분류 헤드 lr 0.03, cosine 감쇠, 4,000 스텝, 혼합 정밀도 |
| 방법별 하이퍼파라미터 | 논문 값 그대로 (τ = 0.95, α = 0.1, τ_b = 2, τ_e = 4 등) |

## 조건

| 이름 | unlabeled | 의미 |
| --- | --- | --- |
| `in` | 같은 200종 26,640장 | 공식 설정. 기본 실행 |
| `open` | 같은 종 26,640장 + 다른 종 26,640장 | 닫힌 클래스 가정이 깨지는 조건. 노트북의 선택 셀 |

이 데이터는 labeled가 원래 long-tail(5~43장)이고 종당 평균 20장이라, 시계열 실험의 조건 A·B처럼 인위적으로 만들지 않아도 "레이블이 적다"와 "불균형하다"가 함께 들어 있다. 표의 `tail_acc`는 labeled가 적은 절반의 종에 대한 정확도다.

## 원 논문·공식 벤치마크와 다른 점

| 항목 | 원래 | 이 코드 |
| --- | --- | --- |
| 해상도 | 224×224 | 160×160 (시간 때문에. `ImgCfg(cache_size=256, crop=224)`와 `SIZE = 256`으로 바꿀 수 있으나 훨씬 느리다) |
| 학습 길이 | 수만~백만 스텝 | 4,000 스텝. unlabeled 한 장이 학습 중 약 15번 보인다 |
| RandAugment | 표본마다 연산을 뽑음 | 4장 묶음마다 연산을 뽑음(GPU에서 처리하려고). `ImgCfg(aug_group=1)`이면 표본마다 |
| CGMatch | warm-up 2,048 스텝 | 500 스텝. 학습이 짧아 Count-Gap이 약 15번의 예측으로만 계산된다. 이 방법에는 불리한 설정이다 |
| SC-SSL | 배치 64 / 128 | 다른 방법과 같은 배치(16 / 112) |

- unlabeled의 레이블이 공개되지 않아 pseudo-label 정확도는 잴 수 없다. 대신 활용률과, pseudo-label 중 head 종이 차지하는 비율을 남긴다.

## 확인된 것과 확인되지 않은 것

- 확인됨: 공식 주석 파일(JSON, solution.csv)을 읽어 분할별 장수와 종 수가 맞는 것. 공식과 같은 폴더·주석 형식의 가짜 데이터로 캐시 만들기부터 여섯 방법의 학습, 표 생성까지 도는 것(`python3 -m sslimg.smoke`, 21개 검사).
- 실제 데이터로 실행함(Colab T4). Kaggle 내려받기와 캐시 만들기 18분, `in` 조건 18회 8시간 51분(2026-11-14), `open` 조건 18회 8시간 54분(2026-11-15). 출력이 담긴 노트북은 `notebooks_with_outputs/SSL_image_colab_output.ipynb`다.
- 처음에는 노트북 주석대로 줄여서 한 번 돌려 본다.
- 한 번에 하나씩 돈다. 이미지 데이터(약 2.8 GB)를 GPU에 통째로 올리기 때문에 병렬 워커는 쓰지 않는다.

## 파일

```
sslimg/
  data.py    내려받기, 주석 읽기, uint8 캐시 만들기
  aug.py     이미지 증강 (약한 증강 / RandAugment + Cutout / InterLUDE용 묶음 증강)
  model.py   ResNet 인코더와 분류 헤드
  train.py   학습 한 번과 평가, 설정(ImgCfg)
  run.py     실행기(prepare, run_grid), 표(tables)
  smoke.py   가짜 데이터 점검
SSL_image_colab.ipynb
```
