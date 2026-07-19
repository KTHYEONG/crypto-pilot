# L2 Phase 성과 개선 세션 결과 — 2026-07-19 (배포단계 격리 bull boost 반영)

## 세션 요약

L2가 이번 세션 최초로 **정상장·위기장 성과를 트레이드오프 없이 동시에 개선**하는 champion을 산출했다. 근본 원인은 "정상장 상방을 결정하는 유일한 레버(L*, 전역 정적 레버리지)가 위기 생존 요건에 의해 하한이 정해져 정상장이 항상 위기 방어의 인질로 잡히는" 비대칭 아키텍처였다 — bull 국면 노출을 늘리는 배포단계(deployment-only) boost를 **L* 캘리브레이션과 수학적으로 완전히 격리**해 도입함으로써 해소했다.

## 시행착오 및 근본 원인

- **1차 시도(실패, 되돌림)**: `apply_regime_risk_cap`의 `bull_gross_cap` 상한(1.0)을 weight 생성 단계에서 2.0으로 완화 — 실측 결과 120/300 trial 모두 `joint_feasible=0`, champion 산출 실패(직전보다 악화).
- **실패 원인 재추적**: fit-leg 수익률(`fit_rets_hybrid`)은 `apply_regime_risk_cap`을 거치지 않는 별도 가중치 파이프라인(`_fit_w`)이라 오염되지 않았으나, `calibrate_deployment_leverage`의 **Stage 2(OOS Adaptive)**에 전달되는 `oos_rets` 인자가 boost 반영된 `rets_hybrid`를 그대로 사용(`workflow.py:2037`)해 **boost로 커진 변동성을 보고 L*를 도로 깎는 피드백 루프**가 실제 원인이었다.
- **2차 시도(성공)**: boost를 weight 생성 단계가 아니라 **L* 확정 이후, `apply_deployment` 직전에만** 적용하도록 재설계. `calibrate_deployment_leverage`에 전달되는 `fit_rets`/`oos_rets`/`crisis_rets`는 boost 존재를 전혀 모르는 unit 시리즈 그대로 유지(핵심 불변식, 회귀테스트로 직접 검증) — 탐색범위도 실패 데이터를 반영해 `[1.0, 2.0]`→`[1.0, 1.3]`으로 보수화.

## 프로덕션 실측 (2026-07-19 기준일, 120 trials)

| 항목 | 이전(crisis-TF-fix만 적용) | 이번(배포단계 격리 boost) |
| :--- | ---: | ---: |
| gate-pass 후보 수 | 1 | **7** |
| champion 선정 경로 | 정상 gate-pass | 정상 gate-pass |
| 정상장 CAGR | +30.9% | **+42.5%** |
| Sharpe | 2.088 | **2.404** |
| Sortino | 3.379 | **4.247** |
| 정상장 MDD | 13.1% | **10.4%**(개선) |
| Fold 통과율 | 75.0%(3/4) | **100.0%(4/4)** |
| Sharpe Uplift | +0.38 | **+0.66** |
| L* 바인딩 사유 | `crisis_window`(위기 요건에 발목) | **`mdd`(순수 정상장 리스크로 전환)** |
| Crisis MDD | 19.11%(budget 21%) | **17.35%**(개선) |
| Crisis CAGR | -4.98%(간신히 통과, 마진 0.02%p) | **+3.51%**(양수 전환) |
| `[CRISIS-RELIABILITY]` | `stress_tested_pass` | ✅ **`stress_tested_pass verified=True`(유지, 마진 대폭 개선)** |

```
STATUS  : ✅ PASS

✅ [Growth    ] CAGR: +42.5% (>=30.0%) | PnL: +71.8% | Equity x1.72
✅ [Efficiency] Sharpe: 2.404 (>=1.000) | Sortino: 4.247 (>=1.500) | Calmar: 4.073 (>=1.000)
✅ [Risk      ] MDD: 10.4% (<=30.0%) | CVaR95: 0.9% (<=6.0%) | RiskUtil: 34.7%
✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 418 (>=30) | Friction: 100.0%
✅ [Uplift    ] Sharpe Uplift: +0.66 (>=+0.05)
✅ [Integrity ] PSR: 0.999 (>=0.90) | DSR: 0.991 (diag)
```

Champion: Trial #101, `growth_lcb=0.2910`, `L*=1.0000(binding=mdd)`, regime levers `hard_block=False asymmetry=True severity_gating=True crisis_gross_cap=0.23`.

원본 로그: `/tmp/l2_decoupled_boost_verify.log`.

## Verdict

- **L2 정상장**: ✅ PASS — 전 게이트 통과, 직전 대비 CAGR/Sharpe/Sortino/MDD/Fold/Uplift **전 지표 개선**.
- **L2 crisis 방어**: ✅ PASS — MDD·CAGR 둘 다 직전보다 개선(트레이드오프 없음).
- **핵심 검증**: L*의 바인딩 사유가 `crisis_window`에서 `mdd`로 전환 — boost가 crisis 방어를 훼손하지 않으면서 정상장 상방을 실제로 넓혔음을 확인.
- **`/check`**: PASS (Cov 39%, 캘리브레이션 무오염 회귀테스트 포함).

## 잔여 이슈

1. **joint_feasible 여전히 0/120**: 개별 게이트(정상장/위기)는 다수 trial이 통과하나, 13개 제약을 **동시에** 만족하는 엄밀한 joint-feasibility 기준으로는 아직 0건 — champion은 gate-pass 경로(promotion 게이트 기준)로 정상 선정되지만, 이 구조적 gap 자체는 다음 세션 재검토 대상.
2. **`[CRISIS-WINDOW-DETAIL]` 개별 라벨 불일치(경미)**: 상위 집계 `stress_tested_pass`와 달리 개별 윈도우 라벨이 `stress_tested_fail`로 표기되는 기존 관측 이슈 — 이번 실측에서도 재확인, 라벨링 로직 자체 점검 필요(수치상 실패는 아님).
3. **단일 seed 실측**: 오늘도 seed=42 단일 실행 기준 — 다중 seed 검증으로 이번 개선(gate-pass 7건, crisis CAGR 양전환)의 안정성 확인 필요.
4. **fold-level 게이트(`fold_pass_ratio`/`recent_fold`)는 boost 미반영**: 이번 spec은 의도적으로 fold-level 진단을 boost-blind 상태로 남겼다(안전한 보수적 편향) — boost 효과가 fold 단위에서도 온전히 인정되지 않아 실제 여력은 이번 실측보다 더 클 수 있음, 후속 spec 대상.
