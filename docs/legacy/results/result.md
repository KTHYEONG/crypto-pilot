## L1 breadth-collapse fix — production replay reveals a new regression

### 1. 실행 조건

| 항목 | 값 |
|---|---|
| 실행 디렉터리 | `logs/futures/compound/20260730_095358/` |
| reference date / seed | `2026-07-15 / 42` |
| data manifest | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` |
| 입력 | 5,442 bars × 51 symbols |
| 실행 | `L2_DRY_RUN=1`, local sync, full phase, entrypoint `src/execution/opt_main_futures.py` |
| integrity / dry-run | `true / true` |
| L2 / L3 | `no_evidence / reject` |

`docs/specs/l1_admission_bootstrap_consistency_and_growth_lever_survey.md` (F1 bootstrap
통일, F2 concept-layer 해체 2→11 family별 leg, F3 `max_leg_weight` 0.70→0.25)를 적용한
재실행이다. `/check` PASS(mypy/pytest/coverage 전체 그린, Cov 98%) 이후 실행했다.

### 2. 무엇이 바뀌었나 — 그리고 무엇이 더 나빠졌나

**F2 breadth 진단 자체는 재확인됐다.** 등가중(스크래치) 측정과 무관하게, family별
개별 t-stat이 이번 실행에서도 다수 유의했다(`trend_ema t=1.49~1.73`,
`volume_zscore t=2.11`, `bollinger_bandwidth t=2.16~2.19` 등, 11개 중 5개가
`evidence_weight=1.0` 스크린 통과). 2-concept 구조가 breadth를 grid-average로 뭉갰다는
진단은 유효하다.

**그러나 실제 프로덕션 결과는 개선이 아니라 악화됐다.**

| Metric | 수정 전 (K=2) | 수정 후 (K=11, 프로덕션) |
|---|---:|---:|
| posterior | 0.795 | **0.049** |
| net_ann | +2.73% | **−4.67%** |
| positive_folds | — | 2/5 (<0.5) |
| stressed_net_ann | — | −6.25% (음수) |
| admitted | False | False (3개 사유 동시 발생) |

원인을 admission 창의 fold별 실제 배치로 추적했다:

```
fold0 net_ann=-22.24%  top_legs=[aroon_oscillator:xs 0.75, volume_zscore:ts 0.25]
fold1 net_ann=+1.58%   전량 미배치
fold2 net_ann=-1.31%
fold3 net_ann=+6.69%
fold4 net_ann=-8.08%
```

`aroon_oscillator`는 이전 spec의 전체 기간 측정에서 `alpha_ann=-16.2%`로 이미 **음의
엣지가 확인된 family**다. `fold0` 시점에는 누적 prequential 증거가
`config.l1_leg.warmup_folds=2`개 fold뿐이라 추정이 극도로 불안정했고, 그 상태에서
우연히 좋아 보인 `aroon_oscillator`가 스크리닝을 통과해 자본의 75%를 배정받았다.

**근본 원인: K를 2→11로 늘리면서 다중검정 문제가 새로 생겼다.** 후보가 많아질수록
"초기에 우연히 좋아 보이는" 나쁜 family가 하나라도 걸릴 확률이 커지는데,
`compute_evidence_weight`의 스크리닝에는 K에 비례한 보정(Šidák/Bonferroni 등)이나
최소 증거 fold 수 스케일링이 전혀 없다. 등가중 진단(Sharpe 1.729)은 breadth가
"존재한다"는 것만 증명했지, 기존 sizing 메커니즘이 그 breadth를 "안전하게 자본화할
수 있다"는 것은 증명하지 못했다 — 이번 프로덕션 실행이 그 차이를 실측으로 드러냈다.

### 3. L2 / L3 결과

| Metric | 값 |
|---|---|
| verdict | `no_evidence` |
| L1 admitted | `False` — `posterior_0.049_below_0.9`, `positive_folds_2/5_below_0.5`, `stressed_net_ann_-0.0625_not_positive` |
| active_days_ratio | 0.0000 (<0.10) |
| rebalances | 0 (<30) |
| target_weights | 전 구간 0 (cash-only) |
| L3 verdict | `reject` — `low_growth_probability`, `l2_not_pass` |

결과적으로 verdict는 이전과 동일하게 `no_evidence`/cash-only이지만, 내부 판정 사유가
"증거 부족(0.795, 표본 150일)"에서 "적극적으로 나쁜 배치(0.049, 3개 사유 동시 발생)"로
바뀌었다 — 겉보기엔 같은 cash-only이나 근본 원인이 달라졌다.

### 4. 결론

1. **F2(breadth 해체) 진단은 유효하나, F1~F3만으로는 배포 가능한 개선이 아니다.**
   구조적 breadth는 실재하지만, 이를 안전하게 자본화하려면 sizing/screening 레이어에
   K-스케일 다중검정 보정이 별도로 필요하다.
2. **다음 단일 개선 후보**: `compute_evidence_weight` 또는 그 상위 게이트에 (a) K에
   비례한 최소 누적 증거 fold 수 요구, (b) family-wise 다중검정 보정(Šidák) 중 하나를
   추가하는 후속 spec. 이는 사용자 확인 없이 임의로 끼워넣지 않고 별도 spec 사이클로
   분리해야 한다(설계 결정 포함).
3. 임계값(`min_growth_posterior_probability=0.90`, `L2GateConfig` 전체) 완화는
   여전히 금지 — 이번에도 적용하지 않았다.
4. F1(부트스트랩 방법론 통일)은 그대로 유효한 정합성 수정이다. F2/F3는 "breadth가
   존재한다"는 진단으로서는 유효하지만, 코드가 그 breadth를 안전하게 쓰도록 만드는
   후속 수정 없이는 프로덕션에 그대로 둘 수 없다.

### 5. Artifact

- [result.json](../../logs/futures/compound/20260730_095358/result.json)
- [spec](../specs/l1_admission_bootstrap_consistency_and_growth_lever_survey.md)
- [contract](../specs/l1_admission_bootstrap_consistency_and_growth_lever_survey_contract.json)
- [L1 leg evidence](../../logs/l1_admission.jsonl)

## 6. 2026-07-30 최신 재실행 및 net-evidence 비교

`2026-07-30` 기준으로 최신 구현을 `full/local` 모드에서 재실행했다.

| Metric | 이전 기준 실행 (`20260730_041009`) | 최신 실행 (`20260730_113351`) |
|---|---:|---:|
| reference date | 2026-07-15 | 2026-07-30 |
| 입력 | 5,442 bars × 51 symbols | 5,442 bars × 51 symbols |
| L2 verdict | `fail` | `no_evidence` |
| CAGR | +27.29% | 0.00% |
| Sharpe | 2.399 | 0.000 |
| annual turnover | 54.87 | 0.00 |
| active days / rebalances | 충족 시도 | 0 / 0 |
| L3 verdict | `reject` | `reject` |

최신 결과는 수익성 기준으로 개선되지 않았다. 개별 leg evidence에는 양의 alpha가 남아
있지만(`volume_zscore`, `bollinger_bandwidth`, `momentum_ts` 등), net 비용·fold
안정성·registry multiplicity 조건을 통과해 실제 자본 배치로 이어진 leg가 없어
cash-only로 종료됐다. 따라서 현재 병목은 signal 생성량보다
`signal → net evidence → admission → sizing` 경로에 있다.

두 실행 모두 5,442 bars × 51 symbols를 사용했으므로 최신 cash-only 결과는 데이터 기간
축소로 설명되지 않는다. 이전 실행의 +27.29% CAGR 역시 L2 excess-growth probability
`0.329`로 최종 게이트를 통과하지 못한 비배포 결과였다.

### 최신 artifact

- [latest result.json](../../logs/futures/compound/20260730_113351/result.json)
- [latest target weights](../../logs/futures/compound/20260730_113351/target_weights.npy)

## 7. 2026-07-30 최신 구현 재실행 — L1 attribution

최신 `L1 Signal Deficit Attribution & Shadow Incubation` 구현을 2026-07-30 기준으로
`full/local` replay했다.

| 항목 | 결과 |
|---|---:|
| artifact | `20260730_122624` |
| 입력 | 5,442 bars × 51 symbols |
| L1 bottleneck | `signal_generalization_failed` |
| economic / capital candidates | 2 / 0 |
| L2 / L3 | `no_evidence` / `reject` |
| CAGR / Sharpe | 0.00% / 0.000 |
| active days / rebalances | 0 / 0 |
| target weights | 전부 0 (cash-only) |

신호가 완전히 부족한 것은 아니다. `bollinger_bandwidth`(net alpha +91.64%, posterior
0.982), `volume_zscore`(+41.78%, 0.878), `keltner_breakout`(+12.86%, 0.685) 등
양의 net alpha leg가 확인됐다. 그러나 production admission은 posterior 0.000,
positive folds 0/3, stressed net alpha 0.0000 조건으로 실패했다. 동일 신호의
prequential shadow도 net alpha -0.83%, stressed -2.16%, posterior 0.405,
positive folds 1/3으로 자본화에 실패했다.

결론적으로 현 병목은 signal 생성량보다 signal의 시간적 일반화·비용 후 안정성 및
`signal → evidence → admission → sizing` 평가 경로다. 데이터 기간 축소는 없었고,
이전 실행과 동일한 5,442 × 51 입력을 사용했다.

### 최신 artifact

- [result.json](../../logs/futures/compound/20260730_122624/result.json)
- [manifest.json](../../logs/futures/compound/20260730_122624/manifest.json)
- [target weights](../../logs/futures/compound/20260730_122624/target_weights.npy)

## 8. 2026-07-30 L1 Capital Formation Redesign — 실배포 후 재실행

`docs/specs/l1_capital_formation_redesign.md` (C1 연속 shrinkage 레그 가중,
C2 fold0부터 prior 배포, C4 posterior 연속 handoff `φ`, C5 오버레이 북-구동
전환)를 구현하고 `/check` PASS(mypy 0 issue, 대상 테스트 전체 그린, lean_check
spec-compliance PASS) 후 `full/local` 재실행했다. 임계값
(`min_growth_posterior_probability=0.90`, `min_positive_fold_ratio=0.50`,
`L2GateConfig`, `L3ValidationConfig`)은 모두 무변경.

| Metric | 이전 (`20260730_122624`, 구조 수정 전) | 최신 (`20260730_134512`) |
|---|---:|---:|
| L1 verdict | `signal_generalization_failed` | `partial_evidence_sized` |
| L1 posterior | 0.405 (shadow) / n/a (production=0) | **0.7435** |
| handoff scale `φ` | 0 (배포 자체가 없음) | **0.609** |
| 배포된 book | 전 구간(5,442 bars) 0 | **4,020 / 5,442 bars 비영(非零)** |
| L2 verdict | `no_evidence` (평가 대상 book 자체가 없음) | `fail` (실체 book을 평가했으나 초과성장 기준 미달) |
| CAGR | 0.00% | **+22.30%** |
| Sharpe | 0.000 | **1.429** |
| annual turnover | 0.00 | 124.37 |
| max drawdown | 0.00% | 8.04% |
| L3 verdict | `reject` (`l2_not_pass`) | `reject` (`l2_not_pass`, posterior 0.456) |

**핵심 확인**: 이전 실행의 `target_weights.npy`는 5,442개 bar 전 구간에서
문자 그대로 전부 0이었다(`np.sum(np.abs(w).sum(axis=1) > 1e-9) == 0`). L1이
단 한 번도 실제 book을 만든 적이 없었다는 뜻이다. 재설계 후 처음으로 L1이
`φ=0.609`로 스케일된 실체 book을 만들었고, L2가 그 book을 실제로 평가해
CAGR +22.3%/Sharpe 1.43을 관측했다. 다만 `excess_growth_probability=0.737<0.9`,
`stressed_excess_growth_lcb90=-0.121`로 **초과성장(벤치마크 대비) 기준은
여전히 미달**이라 배포는 `fail`로 정직하게 차단됐다 — 임계값을 건드리지
않고도 진단이 "신호 없음"에서 "신호는 있으나 벤치마크 초과분 검증 미달"로
정확해졌다.

**부수 발견 (스펙 범위 밖 버그)**: 재실행 중 `project_terminal_portfolio_caps`
(`src/domain/futures/compound/allocator.py`)가 `max_iterations=64`로 수렴
실패했다. 실측 추적 결과 발산이 아니라 반복당 비율 ≈0.877의 정상 기하수렴이며
`1e-12` 허용오차 도달에 약 192회가 필요했다. 이 함수는 L1이 항상 0-book만
생성해 실제로 한 번도 0이 아닌 입력을 받아본 적이 없어 잠재해 있던 결함이다.
`max_iterations`를 64→256으로만 상향(캡·레버리지 임계값은 무변경)해 해결.

### artifact

- [result.json](../../logs/futures/compound/20260730_134512/result.json)
- [target weights](../../logs/futures/compound/20260730_134512/target_weights.npy)
