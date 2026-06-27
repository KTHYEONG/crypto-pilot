# L2 Regime DEBUG 실행 결과

> 실행: `LOG_LEVEL=DEBUG uv run python src/execution/opt_main_futures.py --phase l2 --sync skip --timeframe 4h --trials 200`  
> 로그: `/tmp/regime_debug_run_latest_escalated.log`  
> 기준: 권한 상승 실행 결과. sandbox 실행은 Redis preflight 권한 문제로 Optuna fallback이 발생해 분석 기준에서 제외.

---

## 1. 최종 판정

| 항목 | 결과 |
|---|---:|
| L2 status | ❌ BLOCKED |
| blocker | `cagr` |
| 기간 | 2024-12-22 ~ 2025-09-30 |
| deployed CAGR | +3.9% |
| PnL | +3.3% |
| MDD | 9.0% |
| CVaR95 | 0.9% |
| RiskUtil | 30.1% |
| Sharpe | 0.364 |
| Sortino | 0.506 |
| Calmar | 0.427 |
| Fold pass | 66.7% |
| Trades | 121 |
| Sharpe uplift | +0.11 |
| DSR | 0.675 |
| PSR | 0.674 |
| L* | 2.0000 |

핵심은 risk는 안정화됐지만 growth와 efficiency가 부족하다는 점이다. MDD, CVaR, fold ratio, trades, DSR은 통과권이지만 CAGR 30%, Sharpe, Sortino, Sharpe uplift 기준에는 도달하지 못했다.

---

## 2. Regime 정책 동작 확인

| 지표 | 값 | 해석 |
|---|---:|---|
| regime states | 3 | production routing은 `bull/bear/crisis` 사용 |
| distribution | bull 34.7%, bear 28.1%, crisis 37.3% | 세 state 모두 충분히 활성 |
| routing proof | true | regime-conditioned path 사용 |
| mean lift | +54.42 bps | regime 조건부 lift 자체는 존재 |
| t-stat | 15.24 | 통계 신호는 강함 |
| fold pass | 1.00 | 3개 fold 기준 proof는 통과 |
| policy mode | `soft` | hard block 없이 downweight 중심 |
| allow / downweight / block / pooled | 5 / 10 / 0 / 243 | block은 실제로 제거됨 |
| unstable cells | 15 | fit/cal 방향 불일치 셀이 많음 |
| hard block eligible | 0 | 현재 hard block 후보 없음 |
| sign consistency | 0.50 | 방향 일관성은 낮음 |
| mean cal lift | -23.04 bps | calibration lift는 음수 |

적용한 regime 정책은 의도대로 작동한다. `soft` 모드는 hard block을 하지 않고, 불안정 cell을 downweight로만 처리한다. 이 덕분에 이전 hard filtering 계열보다 tail risk는 줄었고, 최종 MDD 9.0%, CVaR95 0.9%로 risk profile은 양호하다.

문제는 regime proof가 성과로 충분히 전환되지 않는다는 점이다. `effective_3`는 lift와 t-stat이 강하지만, DEBUG cell에서는 fit/OOS gap이 매우 크고 sign consistency가 낮다. 즉, regime은 “구분력”은 있지만 “배팅 가능한 방향 안정성”은 아직 부족하다.

---

## 3. 성과 분해

| 항목 | 값 | 의미 |
|---|---:|---|
| hybrid annual mean | 0.0228 |
| hybrid annual std | 0.0627 |
| hybrid Sharpe HAC | 0.3885 |
| baseline EW Sharpe HAC | 0.2806 |
| delta Sharpe | +0.1079 |
| mean ratio | 0.54 |
| std ratio | 0.38 |

Regime + soft policy는 baseline 대비 변동성을 크게 줄였고 Sharpe도 개선했다. 그러나 수익 평균도 같이 줄었다. 그래서 Sharpe uplift는 +0.11로 개선됐지만, 기준 +0.20에는 못 미쳤다.

Fold별 결과:

| Fold | Sharpe | CAGR | MDD | 판정 |
|---|---:|---:|---:|---|
| #1 | -0.376 | -4.9% | 8.4% | ❌ FAIL |
| #2 | 1.268 | +9.9% | 3.9% | ✅ PASS |
| #3 | 0.929 | +7.1% | 8.8% | ✅ PASS |

Fold #1의 손실이 전체 CAGR을 크게 낮춘다. Fold #2/#3은 통과 가능성이 있으나, 절대 CAGR과 Sharpe 기준으로는 아직 약하다.

---

## 4. Regime이 실패한 지점

### 4.1 fit/cal policy는 보수적으로 잘 작동

- `block=0`, `hard_block_eligible=0`은 의도한 결과다.
- `sign_consistency=0.50`이므로 hard block을 켜면 오히려 잘못된 배제 위험이 크다.
- 현재 구조에서는 `soft`가 맞다. `hybrid hard block`은 아직 켜면 안 된다.

### 4.2 regime cell의 OOS 안정성이 약함

DEBUG에서 worst cells는 fit과 OOS gap이 매우 크다.

| 예시 | fit | OOS | gap | selected_hit |
|---|---:|---:|---:|---:|
| bull / funding zscore | -503.0 bps | +882.0 bps | +1385.0 bps | 0.00 |
| crisis / trend pullback | +354.6 bps | -508.9 bps | -863.5 bps | 1.00 |
| bull / trend MA | +318.8 bps | -402.6 bps | -721.3 bps | 1.00 |

이 패턴은 regime 자체보다 `regime x family x TF` cell edge 추정이 OOS에서 흔들린다는 뜻이다. 특히 selected_hit=1.00인 cell에서도 OOS gap이 크게 음수인 경우가 있어, 현재 policy가 선택된 sleeve의 손실 cell을 충분히 피하지 못한다.

### 4.3 block comparison은 거의 0

| fold | hybrid log growth | baseline log growth | delta |
|---|---:|---:|---:|
| 0 | -0.0114 | -0.0114 | -0.0000 |
| 1 | +0.0252 | +0.0252 | +0.0000 |
| 2 | +0.0185 | +0.0185 | -0.0000 |

Regime policy가 최종 portfolio block return을 거의 바꾸지 못한다. 이는 regime이 routing layer에서는 작동하지만, 최종 weight 또는 rebalance 구조에서 성과 차이로 충분히 전달되지 않는다는 뜻이다.

---

## 5. 아키텍처 피드백

### 결론

현재 regime 모듈은 “risk reducer”로는 유효하지만 “growth selector”로는 부족하다. L2에서 regime의 역할은 hard filter가 아니라 다음 3개로 한정하는 편이 맞다.

1. **Exposure governor**: state별 gross cap으로 crisis/bear 구간 노출을 줄인다.
2. **Confidence modifier**: cell edge를 직접 차단하지 말고 downweight/conviction 조정에 사용한다.
3. **Diagnostics provider**: raw 6-state와 cell OOS gap은 production routing이 아니라 성능 진단에 사용한다.

### 유지할 것

- `policy_mode=soft` 기본값 유지.
- `hard_block_enabled=False` 유지.
- `sign_consistency` gate 유지.
- `risk_cap` 유지.
- 3-state production routing 유지.

### 수정 검토 대상

| 대상 | 이유 | 제안 |
|---|---|---|
| `mean_cal_lift_bps < 0`인 정책 | calibration에서 이미 음수 | downweight 강도를 더 키우거나 pooled fallback으로 전환 |
| `selected_hit=1` + OOS gap 음수 cell | 실제 선택된 손실 cell | per-cell `selected_oos_gap` 진단을 policy confidence에 반영 |
| block delta 0 | routing 효과가 portfolio return에 미전달 | regime policy 적용 전후 weight delta/gross delta 로그 추가 |
| Fold #1 손실 | 전체 blocker의 핵심 | fold-local policy fallback 또는 fold #1 regime state별 PnL 분해 필요 |

---

## 6. 다음 설계 방향

가장 우선순위가 높은 개선은 hard block 강화가 아니다. 현재 sign consistency가 0.50이고 hard-block 후보가 0이므로 hard block을 켜도 자산증식 개선 근거가 없다.

우선 적용할 구조:

1. `soft downweight`를 calibration lift 기반으로 연속화한다.
2. `mean_cal_lift_bps < 0`이고 `sign_consistency=False`인 cell은 pooled fallback으로 보낸다.
3. regime risk cap은 `crisis=0.55`, `bear=0.75`, `bull=1.0`을 유지하되, cap 적용 빈도와 cap 전후 PnL을 별도 로그로 추적한다.
4. policy 효과를 `sleeve_count`, `edge_pass`, `gross_before/after`, `return_before/after`로 분해해 block delta 0 원인을 제거한다.

현재 결과 기준으로 regime 모듈은 L2에서 손실 제한에는 기여하지만, 수익 창출을 직접 담당하기에는 edge 안정성이 부족하다. 자산증식 관점에서는 regime을 alpha selector로 쓰기보다, L1/L2 신호가 만든 alpha에 대해 state-aware exposure와 confidence만 조정하는 보조 계층으로 두는 것이 더 합리적이다.
