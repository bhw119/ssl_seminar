### test_acc

| 방법 | 같은 종 unlabeled |
| --- | --- |
| Supervised only | 0.416 ± 0.000 (n=1) |
| FixMatch | 0.484 ± 0.000 (n=1) |
| CGMatch | 0.514 ± 0.000 (n=1) |
| InterLUDE | 0.476 ± 0.000 (n=1) |
| SC-SSL | 0.439 ± 0.000 (n=1) |

### gain_vs_fixmatch

| 방법 | 같은 종 unlabeled |
| --- | --- |
| CGMatch | +0.029 ± 0.000 |
| InterLUDE | -0.009 ± 0.000 |
| SC-SSL | -0.046 ± 0.000 |

### tail_acc

| 방법 | 같은 종 unlabeled |
| --- | --- |
| Supervised only | 0.385 ± 0.000 (n=1) |
| FixMatch | 0.444 ± 0.000 (n=1) |
| CGMatch | 0.487 ± 0.000 (n=1) |
| InterLUDE | 0.428 ± 0.000 (n=1) |
| SC-SSL | 0.441 ± 0.000 (n=1) |

### head_acc

| 방법 | 같은 종 unlabeled |
| --- | --- |
| Supervised only | 0.448 ± 0.000 (n=1) |
| FixMatch | 0.525 ± 0.000 (n=1) |
| CGMatch | 0.541 ± 0.000 (n=1) |
| InterLUDE | 0.523 ± 0.000 (n=1) |
| SC-SSL | 0.437 ± 0.000 (n=1) |

### ablation_test_acc

| 방법 | 같은 종 unlabeled |
| --- | --- |
| CGMatch | 0.514 ± 0.000 (n=1) |
| InterLUDE | 0.476 ± 0.000 (n=1) |
| SC-SSL | 0.439 ± 0.000 (n=1) |
| InterLUDE, DC 제거 (EF만) | 0.473 ± 0.000 (n=1) |
| InterLUDE, EF 제거 (DC만) | 0.479 ± 0.000 (n=1) |
| CGMatch, ambiguous 손실 제거 | 0.498 ± 0.000 (n=1) |
| CGMatch, Count-Gap 대신 확신도로 분할 | 0.515 ± 0.000 (n=1) |
| SC-SSL, unlabeled 분포를 균등으로 가정 | 0.431 ± 0.000 (n=1) |
| SC-SSL, 원 코드 최적화 (상수 lr) | 0.412 ± 0.000 (n=1) |
| SC-SSL, 원 코드 최적화 + unlabeled ×2 | 0.320 ± 0.000 (n=1) |

### ablation_gain_vs_full

| 방법 | 같은 종 unlabeled |
| --- | --- |
| InterLUDE, DC 제거 (EF만) | -0.002 ± 0.000 |
| InterLUDE, EF 제거 (DC만) | +0.003 ± 0.000 |
| CGMatch, ambiguous 손실 제거 | -0.015 ± 0.000 |
| CGMatch, Count-Gap 대신 확신도로 분할 | +0.001 ± 0.000 |
| SC-SSL, unlabeled 분포를 균등으로 가정 | -0.008 ± 0.000 |
| SC-SSL, 원 코드 최적화 (상수 lr) | -0.027 ± 0.000 |
| SC-SSL, 원 코드 최적화 + unlabeled ×2 | -0.119 ± 0.000 |

### scssl_heads

| 방법 | h2 | h1 | h1_raw | h2_raw |
| --- | --- | --- | --- | --- |
| SC-SSL · 같은 종 unlabeled | 0.439 (n=1) | 0.439 (n=1) | 0.429 (n=1) | 0.442 (n=1) |
| SC-SSL, unlabeled 분포를 균등으로 가정 · 같은 종 unlabeled | 0.431 (n=1) | 0.430 (n=1) | 0.426 (n=1) | 0.439 (n=1) |

### interlude_ef_at_eval

| 방법 | no_fusion | fusion_file_order | fusion_shuffled |
| --- | --- | --- | --- |
| InterLUDE · 같은 종 unlabeled | 0.476 (n=1) | 0.474 (n=1) | 0.474 (n=1) |

### diagnosis

| cond | method | val 정확도 | test 정확도 | 마지막 스텝 test 정확도 | test top-5 | 선택된 스텝 | 분/회 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in | CGMatch | 0.517 | 0.514 | 0.517 | 0.749 | 8000.0 | 85.9 |
| in | CGMatch, ambiguous 손실 제거 | 0.507 | 0.498 | 0.505 | 0.74 | 9500.0 | 85.783 |
| in | CGMatch, Count-Gap 대신 확신도로 분할 | 0.524 | 0.515 | 0.51 | 0.759 | 8000.0 | 85.767 |
| in | FixMatch | 0.486 | 0.484 | 0.484 | 0.725 | 10000.0 | 74.517 |
| in | InterLUDE | 0.482 | 0.476 | 0.472 | 0.718 | 9500.0 | 81.483 |
| in | InterLUDE, DC 제거 (EF만) | 0.494 | 0.474 | 0.472 | 0.716 | 9000.0 | 81.067 |
| in | InterLUDE, EF 제거 (DC만) | 0.502 | 0.479 | 0.476 | 0.715 | 9000.0 | 80.45 |
| in | SC-SSL | 0.442 | 0.439 | 0.443 | 0.684 | 8500.0 | 107.75 |
| in | SC-SSL, 원 코드 최적화 (상수 lr) | 0.426 | 0.412 | 0.412 | 0.649 | 10000.0 | 152.155 |
| in | SC-SSL, 원 코드 최적화 + unlabeled ×2 | 0.327 | 0.32 | 0.32 | 0.575 | 10000.0 | 93.371 |
| in | SC-SSL, unlabeled 분포를 균등으로 가정 | 0.44 | 0.431 | 0.431 | 0.681 | 10000.0 | 107.833 |
| in | Supervised only | 0.421 | 0.416 | 0.416 | 0.67 | 10000.0 | 10.3 |
