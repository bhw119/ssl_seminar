# Semi-Supervised Learning 세미나: 가정이 깨지는 조건에서의 SSL 기법 비교 실험

대학원 수업 「기계지능특론」의 Semi-Supervised Learning(SSL) 세미나의 Experiment 섹션에 쓴 코드와 실행 결과.

질문은 하나다. **각 논문이 완화한 가정이 실제로 깨지는 조건에서, 그 기법이 FixMatch보다 더 기여하는가.**

- 방법 6종: Supervised only, Self-training, FixMatch, CGMatch(CVPR 2025), InterLUDE(arXiv 2024), SC-SSL(AAAI 2026)
- 데이터 2종: 위성 시계열(HLS 2022, 생태계 유형 24종), 이미지(Semi-Aves, 조류 200종)
- 설정: inductive SSL. 방법별 손실 코드는 두 데이터에서 같고 인코더와 증강만 다르다. 조건마다 시드 3회, 모두 90회 학습.

## 실행 방법

필요한 패키지: `torch`, `torchvision`, `numpy`, `pandas`, `psutil`, 이미지 데이터를 받을 때 `kagglehub`

**1. 코드 점검** (데이터 불필요, 합성 데이터, CPU에서 몇 분)

```bash
cd ssl_experiment
python3 -m sslexp.smoke    # 시계열 파이프라인과 방법 6종, 34개 검사
python3 -m sslimg.smoke    # 이미지 파이프라인, 21개 검사
```

**2. 실험 실행** (Colab, GPU 런타임)

1. `ssl_experiment` 폴더를 Google Drive의 `MyDrive/ssl_seminar/` 아래에 올린다.
2. 노트북을 위에서부터 실행한다.
   - 시계열: `SSL_experiment_colab.ipynb` (T4에서 54회, 6시간 46분)
   - 이미지: `SSL_image_colab.ipynb` (T4에서 조건당 18회, 약 9시간)
3. 끊기면 같은 셀을 다시 실행한다. 끝난 실행은 건너뛴다.
4. 결과는 `MyDrive/ssl_seminar/results_v1/`(시계열)과 `results_img_v1/`(이미지)의 `tables/`와 `runs/`에 저장된다.

시계열은 명령줄로도 돌릴 수 있다. 방법, 조건, 시드를 주지 않으면 전부 돈다.

```bash
cd ssl_experiment
python3 -m sslexp.run --csv HLS_TimeSeries_2022.csv --out results_v1
python3 -m sslexp.run --csv HLS_TimeSeries_2022.csv --out results_v1 \
    --methods fixmatch scssl --conditions B --seeds 0    # 일부만
```

**3. 데이터** (저장소에 없음)

- 위성 시계열: 별도 연구의 데이터라 공개하지 않는다. 시계열 실험은 코드 점검과 결과 확인만 할 수 있다.
- Semi-Aves: 노트북이 Kaggle 대회 `semi-inat-2020`에서 받는다. 대회 참가 동의와 Kaggle API 토큰(Colab 보안 비밀 `KAGGLE_API_TOKEN`)이 필요하다. 비상업적 연구 · 교육 목적으로만 쓸 수 있고 이미지를 재배포할 수 없다([이용 조건](https://github.com/cvl-umass/semi-inat-2020)).

경로 상수, 병렬 실행, 기본 설정, 원 논문과 다른 점은 [`ssl_experiment/README.md`](ssl_experiment/README.md)에 있다.

## 결과

값은 시드 3회의 평균 ± 표준편차다.

**위성 시계열 (Test macro-F1)**

| 방법 | 기준 조건 (클래스당 100) | 조건 A (클래스당 4) | 조건 B (분포 불일치) |
| --- | --- | --- | --- |
| Supervised only | **0.659 ± 0.007** | 0.407 ± 0.013 | 0.506 ± 0.017 |
| Self-training | 0.652 ± 0.010 | 0.402 ± 0.013 | 0.498 ± 0.026 |
| FixMatch | 0.654 ± 0.004 | **0.441 ± 0.011** | 0.507 ± 0.028 |
| CGMatch | 0.654 ± 0.007 | 0.433 ± 0.028 | 0.500 ± 0.023 |
| InterLUDE | 0.658 ± 0.007 | 0.428 ± 0.038 | 0.493 ± 0.005 |
| SC-SSL | 0.644 ± 0.004 | 0.435 ± 0.036 | **0.555 ± 0.029** |

조건 B는 labeled가 long-tail(100 → 4)이고 unlabeled가 반대 방향의 long-tail이다.

**이미지: Semi-Aves (Test top-1 정확도)**

| 방법 | In-class | Tail 종 | Open-set | Tail 종 |
| --- | --- | --- | --- | --- |
| Supervised only | 0.392 ± 0.004 | 0.361 | 0.387 ± 0.007 | 0.355 |
| Self-training | 0.384 ± 0.006 | 0.348 | 0.381 ± 0.003 | 0.347 |
| FixMatch | 0.403 ± 0.005 | 0.346 | 0.387 ± 0.003 | 0.336 |
| CGMatch | **0.418 ± 0.003** | 0.357 | 0.389 ± 0.006 | 0.330 |
| InterLUDE | 0.402 ± 0.003 | 0.362 | 0.404 ± 0.006 | 0.370 |
| SC-SSL | 0.415 ± 0.004 | **0.409** | **0.408 ± 0.005** | **0.429** |

In-class는 unlabeled가 같은 200종 26,640장이고, Open-set은 여기에 다른 종 26,640장을 더한 것이다. Tail 종은 레이블이 적은 절반의 종이다.

**FixMatch 대비 이득** (같은 시드끼리 뺀 값의 평균)

| 방법 | 시계열 기준 | 시계열 A | 시계열 B | 이미지 In-class | 이미지 Open-set |
| --- | --- | --- | --- | --- | --- |
| CGMatch | +0.000 | −0.008 | −0.006 | **+0.015** | +0.002 |
| InterLUDE | +0.004 | −0.014 | −0.014 | −0.001 | **+0.017** |
| SC-SSL | −0.009 | −0.007 | **+0.049** | +0.012 | **+0.021** |

- SC-SSL만 기대대로다. 분포가 어긋난 조건 B에서만 오르고, 두 데이터 모두에서 tail 클래스를 올린다.
- CGMatch는 이미지 In-class에서만, InterLUDE는 이미지 Open-set에서만 FixMatch보다 높다.
- Self-training은 5개 조건 모두 Supervised보다 낮다.
- 레이블이 충분한 기준 조건에서는 SSL의 이득이 없다.

**한계**

- 시드 3회에 차이가 0.01 ~ 0.05인 작은 실험이다. 통계 검정은 하지 않았다.
- 학습이 짧다(시계열 6,000 스텝, 이미지 4,000 스텝 · 160px). 절대 성능은 원 벤치마크보다 낮다(해상도 및 학습량 차이).
- 시계열은 이미지용 기법을 옮긴 구현이고 하이퍼파라미터를 조정하지 않았다.
- 표의 SC-SSL 결과는 분포 추정의 결함을 고치기 전에 얻은 것이다. 이미지 실험에서 후보 분포 5종 중 inverse-Gaussian이 KL 무한대로 빠졌다. 저장소의 코드는 고친 버전이고 다시 돌리지는 않았다.

## 저장소 구성

```
ssl_experiment/           실험 코드
  sslexp/                 시계열 실험, 방법 6종의 손실 (methods.py)
  sslimg/                 이미지 실험 (데이터 준비, 증강, 인코더, 실행기)
  *.ipynb                 Colab 노트북 (출력 없음)
notebooks_with_outputs/   실행 출력이 담긴 노트북
results/
  timeseries/             tables/ (요약 표), runs/ (실행별 json 54개)
  image/                  tables/, runs/ (실행별 json 36개)
```

요약 표는 `results/*/tables/tables.md`에 모여 있다. 실행별 json에는 학습 곡선, 클래스별 성능, pseudo-label 통계가 들어 있다.

## 참고문헌

1. K. Sohn et al., "FixMatch: Simplifying semi-supervised learning with consistency and confidence," NeurIPS, 2020.
2. B. Cheng et al., "CGMatch: A different perspective of semi-supervised learning," CVPR, 2025.
3. Z. Huang et al., "InterLUDE: Interactions between labeled and unlabeled data to enhance semi-supervised learning," arXiv:2403.10658, 2024.
4. S. Tian, X. Wei, and S. Zhang, "Sampling control for imbalanced calibration in semi-supervised learning," AAAI, 2026.
5. J.-C. Su, Z. Cheng, and S. Maji, "A realistic evaluation of semi-supervised learning for fine-grained classification," CVPR, 2021.
