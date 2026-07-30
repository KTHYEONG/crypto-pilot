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
