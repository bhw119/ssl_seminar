# Semi-Supervised Learning 세미나: 가정이 깨지는 조건에서의 SSL 기법 비교 실험

대학원 수업 「기계지능특론」의 Semi-Supervised Learning(SSL) 세미나(2026년 11월)의 Experiment 섹션에 쓴 코드와 실행 결과다.

발표는 교안의 고전 SSL 기법(Self-training, Generative model, S3VM, Graph-based)이 어떤 가정 위에 서 있는지 정리하고, 최신 논문 세 편이 그 가정을 어떻게 완화했는지 살펴본 뒤, 실험으로 확인하는 순서다. 이 저장소의 중심은 마지막의 실험이다.

| 논문 | 이어받은 고전 기법 | 완화한 가정 |
| --- | --- | --- |
| CGMatch (CVPR 2025) | Self-training | confidence가 높은 예측은 맞다 |
| InterLUDE (arXiv 2024) | Graph-based | labeled 손실과 unlabeled 손실은 따로 계산해 더하면 된다 |
| SC-SSL (AAAI 2026) | Generative model | labeled와 unlabeled의 클래스 분포가 같다 |

## 결과 요약

- SSL의 이득은 조건부였다. 레이블이 충분한 조건에서는 없었고, 레이블이 적거나 불균형한 조건에서만 나타났다.
- 이득은 고전 Self-training이 아니라 FixMatch 이후의 consistency 기반 기법에서 나왔다. Self-training은 5개 조건 모두 Supervised보다 낮았다.
- 가정이 깨지는 조건과 기법의 대응이 가장 분명한 것은 SC-SSL이다. 두 데이터 모두에서 레이블이 적은 tail 클래스의 성능을 올렸다.
- CGMatch는 이미지의 in-class 조건에서만, InterLUDE는 이미지의 open-set 조건에서만 FixMatch보다 높았다.
- 시드 3회에 차이가 0.01–0.05 수준인 작은 실험이다. [한계](#한계)를 함께 읽어야 한다.

## 연구 질문

**각 논문이 완화한 가정이 실제로 깨지는 조건에서, 그 기법이 정말로 더 기여하는가.**

| 논문 | 완화한 가정 | 가정이 깨지는 조건 | 기대 |
| --- | --- | --- | --- |
| CGMatch | confidence = 정확성 | 극소량 레이블 | 조건 A에서 이득이 가장 크다 |
| SC-SSL | labeled · unlabeled의 클래스 분포가 같다 | 클래스 분포 불일치 | 조건 B에서 이득이 가장 크다 |
| InterLUDE | labeled · unlabeled를 따로 계산해 더한다 | (데이터 쪽 조건 없음) | 레이블이 적은 조건 A에서 이득 |

판정 기준은 "FixMatch 대비 이득이 기준 조건보다 해당 위배 조건에서 더 큰가"다. 세 기법은 모두 FixMatch 위에 모듈을 얹은 방법이라, 모듈 고유의 효과는 FixMatch와 비교해야 보인다.

## 실험 설계

설정은 두 데이터 모두 inductive SSL이다. labeled와 레이블을 숨긴 unlabeled로 학습하고, 학습에 쓰지 않은 test로 평가한다.

**데이터** (레이블을 얻기 어려운 두 도메인)

| 항목 | 위성 시계열 (HLS, 2022) | 이미지 (Semi-Aves) |
| --- | --- | --- |
| 분류 대상 | 생태계 유형 24종 (대한민국) | 조류 200종 |
| 레이블 비용 | 현장조사 필요 | 전문가의 종 판별 필요 |
| 표본 수 | 24,009 지점 (학습 14,359 / 검증 4,799 / 테스트 4,851) | labeled 3,959 / unlabeled 26,640 / 검증 2,000 / 테스트 8,000 |
| 입력 | 122시점 × 22채널 | RGB 이미지 |
| 분할 | 25 km 공간 블록 단위 (테스트 구역은 학습에 쓰지 않음) | 챌린지 공식 분할 |
| 역할 | 가정 위배 조건을 직접 만들어 통제한 실험 | 세 기법의 원래 도메인에서 재확인 |

세 기법은 이미지용으로 설계됐다. 시계열에서 결과가 나오지 않으면 기법의 한계인지 시계열로 옮긴 구현의 한계인지 구분할 수 없어서, 같은 손실 코드로 이미지에서도 확인했다.

**비교 방법 6종**: Supervised only(기준), Self-training(교안의 고전 기법), FixMatch(세 논문의 공통 기반), CGMatch, InterLUDE, SC-SSL

**조건 5종**

| 데이터 | 조건 | labeled | unlabeled | 의미 |
| --- | --- | --- | --- | --- |
| 시계열 | 기준 조건 (`base`) | 클래스당 100 | 클래스별 균등 | 가정이 성립하는 대조군 |
| 시계열 | 조건 A (`A`) | 클래스당 4 | 클래스별 균등 | 극소량 레이블 |
| 시계열 | 조건 B (`B`) | long-tail (100 → 4) | 반대 방향 long-tail | 클래스 분포 불일치 |
| 이미지 | In-class (`in`) | 종당 5–43 (공식) | 같은 200종 26,640장 | 자연 발생 소량 · long-tail |
| 이미지 | Open-set (`open`) | 위와 같음 | 위 + 다른 종 26,640장 | 레이블에 없는 클래스가 섞임 |

**공통 설정**

- 방법별 손실 코드(`sslexp/methods.py`)는 두 데이터에서 같고, 인코더와 증강만 바꿨다.
- 시계열: InceptionTime 계열 인코더를 처음부터 학습, 6,000 스텝. 약한 증강은 크기 변화와 잡음, 강한 증강은 시점 masking, 밴드 drop, 진폭 왜곡.
- 이미지: ImageNet 사전학습 ResNet-50, 4,000 스텝, 160px. 약한 증강은 crop과 좌우 반전, 강한 증강은 RandAugment와 Cutout.
- 하이퍼파라미터는 원 논문 값을 그대로 쓰고 재조정하지 않았다. 조건마다 시드 3회, 모두 90회 학습(시계열 54회, 이미지 36회)이다.
- 지표는 시계열이 test macro-F1, 이미지가 test top-1 정확도(챌린지 공식 지표)다. Head와 tail은 레이블 수가 많은 절반과 적은 절반의 클래스다.

## 결과

값은 시드 3회의 평균 ± 표준편차다. 원자료는 [`results/`](results)에 있다.

### 위성 시계열 (Test macro-F1)

| 방법 | 기준 조건 | 조건 A (극소량 레이블) | 조건 B (분포 불일치) |
| --- | --- | --- | --- |
| Supervised only | **0.659 ± 0.007** | 0.407 ± 0.013 | 0.506 ± 0.017 |
| Self-training | 0.652 ± 0.010 | 0.402 ± 0.013 | 0.498 ± 0.026 |
| FixMatch | 0.654 ± 0.004 | **0.441 ± 0.011** | 0.507 ± 0.028 |
| CGMatch | 0.654 ± 0.007 | 0.433 ± 0.028 | 0.500 ± 0.023 |
| InterLUDE | 0.658 ± 0.007 | 0.428 ± 0.038 | 0.493 ± 0.005 |
| SC-SSL | 0.644 ± 0.004 | 0.435 ± 0.036 | **0.555 ± 0.029** |

FixMatch 대비 이득 (같은 시드끼리 뺀 값)

| 방법 | 기준 조건 | 조건 A | 조건 B |
| --- | --- | --- | --- |
| CGMatch | +0.000 ± 0.010 | −0.008 ± 0.017 | −0.006 ± 0.005 |
| InterLUDE | +0.004 ± 0.007 | −0.014 ± 0.030 | −0.014 ± 0.032 |
| SC-SSL | −0.009 ± 0.005 | −0.007 ± 0.025 | **+0.049 ± 0.021** |

- **기준 조건**: 여섯 방법이 0.64–0.66으로 같다. 클래스당 레이블 100개면 unlabeled가 더해 줄 것이 없었다.
- **조건 A**: FixMatch가 0.407에서 0.441로 오른다. 세 최신 기법은 FixMatch를 넘지 못했다. CGMatch에 대한 기대는 확인되지 않았다.
- **조건 B**: SC-SSL만 오른다(세 시드 모두 FixMatch보다 높음). Tail 클래스 F1이 0.429에서 0.512로 올랐다. 판정 기준을 만족한 것은 SC-SSL이다.
- **원인 진단**: FixMatch가 채택한 pseudo-label의 정확도는 기준 조건 0.85, 조건 A 0.55, 조건 B 0.52였다. 강한 증강을 주면 예측의 56–62%가 바뀌었다. 세 논문이 공통으로 전제하는 "의미를 보존하는 강한 증강"이 시계열에서는 약하게만 성립한다.

### 이미지: Semi-Aves (Test top-1 정확도)

| 방법 | In-class 전체 | Head | Tail | Open-set 전체 | Head | Tail |
| --- | --- | --- | --- | --- | --- | --- |
| Supervised only | 0.392 ± 0.004 | 0.423 | 0.361 | 0.387 ± 0.007 | 0.419 | 0.355 |
| Self-training | 0.384 ± 0.006 | 0.419 | 0.348 | 0.381 ± 0.003 | 0.415 | 0.347 |
| FixMatch | 0.403 ± 0.005 | 0.460 | 0.346 | 0.387 ± 0.003 | 0.438 | 0.336 |
| CGMatch | **0.418 ± 0.003** | **0.479** | 0.357 | 0.389 ± 0.006 | **0.448** | 0.330 |
| InterLUDE | 0.402 ± 0.003 | 0.442 | 0.362 | 0.404 ± 0.006 | 0.438 | 0.370 |
| SC-SSL | 0.415 ± 0.004 | 0.420 | **0.409** | **0.408 ± 0.005** | 0.387 | **0.429** |

FixMatch 대비 이득과 pseudo-label 활용률

| 방법 | 이득 (In-class) | 이득 (Open-set) | 활용률 (In-class → Open-set) |
| --- | --- | --- | --- |
| FixMatch | | | 0.27 → 0.17 |
| CGMatch | **+0.015 ± 0.005** | +0.002 ± 0.008 | 0.82 → 0.85 |
| InterLUDE | −0.001 ± 0.005 | **+0.017 ± 0.004** | 0.15 → 0.09 |
| SC-SSL | +0.012 ± 0.009 | **+0.021 ± 0.003** | 0.21 → 0.13 |

- **In-class**: CGMatch와 SC-SSL이 가장 높다. CGMatch는 unlabeled의 82%를 학습에 쓴다(FixMatch는 27%). confidence는 낮지만 예측이 일관된 표본을 살린다는 설계가 그대로 나타났다.
- **이득이 간 곳**: FixMatch와 CGMatch의 이득은 head 종에 몰려 있고, tail 종은 Supervised보다 조금 낮다. SC-SSL은 반대로 tail을 0.361에서 0.409로 올리고 head는 Supervised 수준에 둔다.
- **Open-set**: 다른 종을 섞으면 FixMatch와 CGMatch의 이득이 사라져 Supervised와 같아진다. InterLUDE와 SC-SSL은 유지된다. SC-SSL은 tail이 더 오르는 대신 head가 Supervised보다 낮아진다(0.387).
- **원 논문과의 대조**: CGMatch 논문은 open-set 열위를 스스로 한계로 적었고, InterLUDE 논문은 open-set 의료 영상에서 좋은 결과를 보고했다. 이 데이터의 벤치마크 논문(Su et al., 2021)도 다른 종이 섞이면 FixMatch가 57.4%에서 49.7%로 떨어져 Supervised(52.7%)보다 낮아진다고 보고했다. 세 가지 모두 이 결과와 방향이 같다.
- 절대 정확도가 벤치마크보다 낮은 것은 해상도를 160px로 줄이고 4,000 스텝만 학습했기 때문이다. 방법 사이의 비교가 목적이다.

### 연구 질문에 대한 판정

수치는 FixMatch 대비 이득이다(Self-training은 Supervised 대비).

| 기법 | 기대 | 위성 시계열 | 이미지 | 판정 |
| --- | --- | --- | --- | --- |
| CGMatch | 극소량 레이블에서 이득 | 기준 +0.000 / 조건 A −0.008 | In-class +0.015 / Open-set +0.002 | 이미지 in-class에서만 이득 |
| InterLUDE | 레이블이 적을 때 이득 | 기준 +0.004 / 조건 A −0.014 | In-class −0.001 / Open-set +0.017 | 이미지 open-set에서만 이득 |
| SC-SSL | 분포 불일치에서 이득 | 기준 −0.009 / 조건 B +0.049 | In-class +0.012 / Open-set +0.021 | **두 데이터에서 확인 (tail 개선)** |
| Self-training | (고전 기법) | −0.005 ~ −0.008 | −0.008 / −0.006 | 5개 조건 모두 Supervised 이하 |

기법이 다루지 않은 위배 조건에서는 이득이 사라졌다(CGMatch의 open-set). "가정과 데이터가 맞아야 unlabeled가 도움이 된다"는 교안의 결론과 같다.

## 한계

- 시드가 3회이고 차이가 0.01–0.05로 작다. 통계 검정은 하지 않았다. Supervised는 unlabeled를 쓰지 않아 in-class와 open-set이 같은 설정인데도 0.392와 0.387로 0.005 달랐다. 이보다 작은 차이는 해석하지 않았다.
- 이미지 데이터는 분할이 고정이라 시드는 초기값과 배치 순서만 바꾼다. 표준편차가 실제 불확실성보다 작게 나온다.
- 학습이 짧다(6,000 / 4,000 스텝). 절대 성능은 원 벤치마크보다 낮다.
- 시계열은 이미지용 기법을 옮긴 구현이다. 증강과 인코더가 다르고 하이퍼파라미터를 조정하지 않았다. 재현이 아니라 이식이다. 원 논문과 다른 점은 [`ssl_experiment/README.md`](ssl_experiment/README.md)에 표로 정리했다.
- 조건 B에서는 "분포 불일치"의 효과와 "레이블이 불균형하다"는 것 자체의 효과를 분리하지 못했다. 대조 조건 `Bc`는 코드에 있지만 돌리지 않았다.
- Open-set은 세 논문이 대상으로 삼은 조건이 아니다. 추가 관찰이다.
- SC-SSL의 분포 추정에 결함이 있었다. 클래스가 200개인 이미지 실험에서 후보 분포 5종 중 inverse-Gaussian의 KL이 무한대가 되어 후보에서 빠졌다(`results/image/runs/*scssl*.json`의 `extra.sc.kl`). 실제로 선택된 후보는 uniform 또는 inverse였다. 이 저장소의 코드는 고친 버전이고, 표의 결과는 고치기 전에 얻은 것이다. 고친 코드로 다시 돌리지는 않았다.

## 저장소 구성

```
README.md
ssl_experiment/               실험 코드
  README.md                   구현 상세, 실행 방법, 원 논문과 다른 점
  sslexp/                     시계열 실험 + 방법 6종의 손실 (methods.py)
  sslimg/                     이미지 실험 (데이터 준비, 증강, 인코더, 실행기)
  SSL_experiment_colab.ipynb  시계열 실험 Colab 노트북 (출력 없음)
  SSL_image_colab.ipynb       이미지 실험 Colab 노트북 (출력 없음)
notebooks_with_outputs/       실제 실행 출력이 담긴 노트북 (Colab T4)
  SSL_experiment_colab_output.ipynb   시계열 54회, 6시간 46분
  SSL_image_colab_output.ipynb        이미지 in-class 18회 8시간 51분 + open-set 18회 8시간 54분
results/
  timeseries/                 tables/ (요약 표 csv · md), runs/ (실행별 json 54개), data_meta.json
  image/                      tables/, runs/ (실행별 json 36개)
```

결과 파일의 내용은 다음과 같다.

| 파일 | 내용 |
| --- | --- |
| `results/timeseries/tables/test_macro_f1.csv` | 방법 × 조건의 test macro-F1 |
| `results/timeseries/tables/gain_vs_fixmatch.csv` | FixMatch 대비 이득 |
| `results/timeseries/tables/diagnosis.csv` | pseudo-label 활용률 · 정확도, 강한 증강 전후 예측 변화율 |
| `results/image/tables/test_acc.csv`, `head_acc.csv`, `tail_acc.csv` | test top-1 (전체 / head / tail) |
| `results/image/tables/gain_vs_fixmatch.csv`, `diagnosis.csv` | FixMatch 대비 이득, 활용률 |
| `results/*/runs/*.json` | 실행별 원자료 (학습 곡선, 클래스별 성능, pseudo-label 클래스 분포, 설정) |

## 실행 방법

필요한 패키지는 `torch`, `torchvision`, `numpy`, `pandas`, `psutil`이고, 이미지 데이터를 받을 때 `kagglehub`를 쓴다.

**데이터 없이 코드만 확인** (합성 데이터, CPU에서 몇 분)

```bash
cd ssl_experiment
python3 -m sslexp.smoke    # 시계열 파이프라인과 방법 6종, 34개 검사
python3 -m sslimg.smoke    # 이미지 파이프라인, 21개 검사
```

**실험 실행** (Colab, GPU 런타임)

1. `ssl_experiment` 폴더를 Google Drive의 `MyDrive/ssl_seminar/` 아래에 올린다.
2. `SSL_experiment_colab.ipynb`(시계열) 또는 `SSL_image_colab.ipynb`(이미지)를 위에서부터 실행한다.
3. 끊기면 같은 셀을 다시 실행한다. 끝난 실행은 건너뛴다.

경로 상수, 병렬 실행, Kaggle 내려받기 절차는 [`ssl_experiment/README.md`](ssl_experiment/README.md)에 있다.

## 데이터

데이터는 이 저장소에 없다.

- **위성 시계열 (HLS 2022)**: 별도 연구에서 쓰는 데이터라 공개하지 않는다. 그래서 시계열 실험은 이 저장소만으로 다시 돌릴 수 없고, 코드 점검(`sslexp.smoke`)과 결과 확인만 할 수 있다.
- **Semi-Aves**: Semi-Supervised iNaturalist-Aves (FGVC7 워크숍, CVPR 2020). Kaggle 대회 `semi-inat-2020`에서 받는다. 비상업적 연구 · 교육 목적으로만 쓸 수 있고 이미지를 재배포할 수 없다. 출처와 이용 조건은 https://github.com/cvl-umass/semi-inat-2020 에 있다.

## 참고문헌

1. X. Zhu, "Semi-Supervised Learning Tutorial," ICML, 2007.
2. K. Sohn et al., "FixMatch: Simplifying semi-supervised learning with consistency and confidence," NeurIPS, 2020.
3. B. Cheng et al., "CGMatch: A different perspective of semi-supervised learning," CVPR, 2025.
4. Z. Huang, X. Yu, D. Zhu, and M. C. Hughes, "InterLUDE: Interactions between labeled and unlabeled data to enhance semi-supervised learning," arXiv:2403.10658, 2024.
5. S. Tian, X. Wei, and S. Zhang, "Sampling control for imbalanced calibration in semi-supervised learning," AAAI, 2026.
6. C. Wei, K. Shen, Y. Chen, and T. Ma, "Theoretical analysis of self-training with deep networks on unlabeled data," ICLR, 2021.
7. J.-C. Su and S. Maji, "The Semi-Supervised iNaturalist-Aves Challenge at FGVC7 Workshop," arXiv:2103.06937, 2021.
8. J.-C. Su, Z. Cheng, and S. Maji, "A realistic evaluation of semi-supervised learning for fine-grained classification," CVPR, 2021.
