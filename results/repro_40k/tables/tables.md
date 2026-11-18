### test_acc

| 방법 | 같은 종 unlabeled |
| --- | --- |
| Supervised only | 0.410 ± 0.000 (n=1) |
| FixMatch | 0.478 ± 0.000 (n=1) |
| InterLUDE | 0.491 ± 0.000 (n=1) |

### gain_vs_fixmatch

| 방법 | 같은 종 unlabeled |
| --- | --- |
| InterLUDE | +0.013 ± 0.000 |

### tail_acc

| 방법 | 같은 종 unlabeled |
| --- | --- |
| Supervised only | 0.371 ± 0.000 (n=1) |
| FixMatch | 0.426 ± 0.000 (n=1) |
| InterLUDE | 0.448 ± 0.000 (n=1) |

### head_acc

| 방법 | 같은 종 unlabeled |
| --- | --- |
| Supervised only | 0.450 ± 0.000 (n=1) |
| FixMatch | 0.530 ± 0.000 (n=1) |
| InterLUDE | 0.535 ± 0.000 (n=1) |

### interlude_ef_at_eval

| 방법 | no_fusion | fusion_file_order | fusion_shuffled |
| --- | --- | --- | --- |
| InterLUDE · 같은 종 unlabeled | 0.491 (n=1) | 0.490 (n=1) | 0.487 (n=1) |

### diagnosis

| cond | method | val 정확도 | test 정확도 | 마지막 스텝 test 정확도 | test top-5 | 선택된 스텝 | 분/회 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in | FixMatch | 0.488 | 0.478 | 0.477 | 0.716 | 26000.0 | 681.3 |
| in | InterLUDE | 0.5 | 0.492 | 0.486 | 0.715 | 22000.0 | 669.917 |
| in | Supervised only | 0.421 | 0.41 | 0.417 | 0.665 | 36000.0 | 38.65 |
