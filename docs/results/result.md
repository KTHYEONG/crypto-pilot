# L2 Phase 성과 개선 세션 결과 — 2026-07-19 (6개 spec 누적: 정상장 첫 PASS 달성, crisis 방어로 병목 이동)

## 세션 개요

`docs/results/result.md`(2026-07-18)의 `❌ BLOCKED(cagr, +20.9%)` 상태를 출발점으로, 코드+데이터 기반 진단을 반복해 6개 spec을 순차 구현·실측 검증했다. 최종적으로 **오늘 세션 최초로 L2 정상장 게이트 전체 PASS**를 달성했으며, 잔여 병목이 "CAGR 미달"에서 "crisis stress test(LUNA/FTX) 방어 실패"로 명확히 이동했다.

| # | Spec / ADR | 대상 | 실측 효과 |
| :--- | :--- | :--- | :--- |
| 1 | `ADR_20260718_L2_DEPLOYMENT_MARGIN_CAGR_GATE` | leverage 캘리브레이션 안전마진 decouple + searchable화 | MDD 예산 활용 헤드룸 확보(margin 0.05~0.30 탐색) |
| 2 | `ADR_20260718_L2_FOLD_GRANULARITY_ROBUSTNESS` | L2 전용 walk-forward fold 개수 분리(L1/live 비영향) | fold_pass_ratio 50%→75%(4→8-fold 실험), 국소 손실 구간 격리 실증 |
| 3 | `ADR_20260718_L2_REGIME_CELL_ADMISSION_SEARCHABILITY` | regime-cell hard-block/passthrough 탐색공간 편입 | Sharpe/Sortino/Calmar/PSR 동시 통과 champion 최초 발견 |
| 4 | `ADR_20260718_L2_OPTUNA_CONSTRAINT_CAGR_UPLIFT_ALIGNMENT` | Optuna `constraints_func`에 cagr·sharpe_uplift 승격(10→12-tuple) | CAGR +17.5%→+30.4%, Sharpe Uplift -0.63→+0.65 (16개 blocker 전원 통과, worst_fold_cagr만 잔존) |
| 5 | `ADR_20260719_L2_OOS_WORST_FOLD_LEVERAGE_FLOOR_CLAMP` | OOS worst-fold CAGR 플로어 기반 레버리지 하향 클램프(사이징, 탐색 아님) | 메커니즘 자체는 이번 seed에서 미발동(champion 변동성으로 미검증) |
| 6 | `ADR_20260719_L2_ACTIVE_BLOCK_COUNT_LIGHTWEIGHT_FIX` | **버그 수정**: `lightweight=True`(전 탐색 trial 기본값) 시 `active_block_count`가 항상 0으로 계산되던 결함 제거 | **오늘 세션 최초 `STATUS: PASS`** — `feasible_trials`가 처음으로 비어있지 않게 됨 |

## 최종 실행 결과 (seed=42, n_trials=120, spec 1~6 전부 반영)

### 정상장 스코어카드: ✅ PASS

```
STATUS  : ✅ PASS

✅ [Growth    ] CAGR: +38.9% (>=30.0%) | PnL: +57.2% | Equity x1.57
✅ [Efficiency] Sharpe: 2.217 (>=1.000) | Sortino: 3.738 (>=1.500) | Calmar: 3.244 (>=1.000)
✅ [Risk      ] MDD: 12.0% (<=30.0%) | CVaR95: 0.9% (<=6.0%) | RiskUtil: 40.0%
✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 466 (>=30) | Friction: 99.8%
✅ [Uplift    ] Sharpe Uplift: +0.37 (>=+0.05)
✅ [Integrity ] PSR: 0.998 (>=0.90) | DSR: 0.994 (diag)
[Diag     ] RelMDD: 1.12x | Turnover: 0.057
```

| Metric | Value | Gate |
| :--- | ---: | :---: |
| Leverage (L*) | 1.1410 (binding: champion) | |
| CAGR | **+38.9%** | ✅ (≥30%) |
| MDD | 12.0% | ✅ (≤30%) |
| CVaR95 | 0.9% | ✅ (≤6%) |
| Sharpe / Sortino / Calmar | 2.217 / 3.738 / 3.244 | ✅ / ✅ / ✅ |
| Fold Pass Ratio | 100.0% (4/4) | ✅ (≥60%) |
| Sharpe Uplift | +0.37 | ✅ (≥+0.05) |
| PSR | 0.998 | ✅ (≥0.90) |

### Fold 상세 — 4/4 전원 PASS (오늘 세션 최초)

| Fold | Period | CAGR | MDD | Sharpe | Status | Symbols |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| 1 | 2025-03-20 ~ 2025-05-30 | +38.2% | 4.9% | 2.352 | ✅ PASS | 22 |
| 2 | 2025-05-30 ~ 2025-08-09 | +16.3% | 12.0% | 1.156 | ✅ PASS | 24 |
| 3 | 2025-08-09 ~ 2025-10-20 | +6.7% | 7.0% | 0.514 | ✅ PASS | 22 |
| 4 | 2025-10-20 ~ 2025-12-30 | +116.7% | 3.5% | 4.327 | ✅ PASS | 18 |

**참고**: 2026-07-18 result.md 및 오늘 세션 초반(margin/fold 세분화 spec 검증 단계)에서 Fold #2(2025-05-30~08-09)가 4개의 서로 다른 champion 전부에서 반복 실패(CAGR -0.0%~-20.2%)했던 것과 대조적으로, 이번 champion은 해당 구간도 +16.3%로 통과 — 국소 손실 구간이 champion의 파라미터/심볼 선택에 따라 회피 가능함을 재확인.

### Optuna Selection 진단

```
[EVAL] event=replay_flip trial=71 stored_pass=True replay_pass=False stored_cagr=0.3344 replay_cagr=0.3677 stored_mdd=0.1763 replay_mdd=0.1910
[EVAL] event=replay_flip trial=46 stored_pass=True replay_pass=False stored_cagr=0.3088 replay_cagr=0.3505 stored_mdd=0.1470 replay_mdd=0.1641
  ● [CHAMPION STORE] 신규 챔피언 갱신 (tf=8h, growth_lcb=0.2272)
```

- Study: `l2_study_8h_7d885f03f343` | InMemory | Trials 120 | Events 4,699 | Symbols 42
- `[REGIME-L2] proof_failed path=pooled_fallback effective_states=3` — bucket edge routing의 aggregate proof는 여전히 실패(정상 작동: nw_tstat=-8.01 등 강한 음의 신호를 정확히 감지해 pooled로 안전 폴백, 별도 조사에서 dead-code 아님을 확인 완료).
- `replay_flip` 진단 로그(spec 1에서 추가)가 처음으로 실제 값을 내며 정상 작동 — trial 71/46은 저장된 지표와 재검증 지표가 근소하게 달라(예: CAGR 33.4%→36.8%) gate-pass에서 탈락, 이후 다른 trial이 champion으로 확정됨. **이는 버그가 아니라 replay 검증이 의도대로 안전장치 역할을 수행한 것.**

### ⚠️ Crisis Stress Test: ❌ FAIL — 새로운(그러나 예견된) 병목

```
[CRISIS-RELIABILITY] status=stress_tested_fail verified=False detail=luna_ftx_2022_collapse:mdd_abs; luna_ftx_2022_collapse:cagr
[CRISIS-WINDOW-DETAIL] label=luna_ftx_2022_collapse status=stress_tested_fail mdd=0.2786 cagr=-0.2453 cvar95=0.0211 trades=466 symbols=45
```

| Metric | Value | Budget |
| :--- | ---: | :---: |
| MDD | 27.86% | ❌ 초과 |
| CAGR | -24.53% | ❌ 하한 미달(-5%) |
| CVaR95 | 2.11% | — |

**의미**: 정상장 게이트를 처음으로 완전히 통과한 champion이, 독립적인 LUNA/FTX 2022 붕괴 재현 윈도우에서는 붕괴한다 — 세션 초반 사용자가 지적했던 "CAGR 90%급과 L2 gate 모두 통과했지만 스트레스 테스트에서만 무너지는" 역사적 패턴이 버그 수정 이후 처음으로 재현·관측됐다. 이전에는 정상장 게이트 자체를 통과한 적이 없어 이 실패 모드를 직접 관측할 기회조차 없었다.

## 근본 원인 진단: `active_block_count` lightweight 버그 (spec 6)

- Optuna 탐색 루프의 **전 trial**은 `TieredContext.lightweight_eval=True`(기본값)로 평가됨.
- `evaluate_l2_trial(lightweight=True)`는 진단용 `block_metrics` 리스트를 비워둔 채 반환 → `active_block_count = len([m for m in block_metrics if ...]) = 0` (입력과 무관하게 항상 0).
- `l2_min_active_blocks=3`(기본값) 대비 `active_block_count=0`은 `optuna_constraint_values`의 6번째 슬롯을 **100% trial에서 위반**시켜, `select_layer2_champion`의 `feasible_trials`가 구조적으로 항상 공집합이었음.
- **오늘 실행된 8번의 이전 파이프라인 전부**에서 `[L2-SELECTION] feasible trials 없음 → fallback`이 예외 없이 재현된 진짜 원인 — spec 1~5가 전부 논리적으로 타당했음에도, TPESampler의 constrained 탐색 메커니즘은 이 버그 때문에 하루 종일 한 번도 제대로 작동한 적이 없었다.
- 수정: `active_block_count`를 `sim.all_turnovers` 슬라이싱만으로 `lightweight` 여부와 무관하게 항상 계산하도록 분리(진단용 `Layer2BlockMetric` 객체 생성은 계속 lightweight로 skip, 성능 예산 영향 없음).

## Verdict

- **L1**: 이번 세션 스코프 밖 — 변경 없음, 7/7 TF 정상 가정.
- **L2 정상장**: ✅ **PASS** (오늘 세션 최초, 16개 promotion blocker + worst_fold_cagr 전부 통과).
- **L2 crisis stress test**: ❌ **FAIL** (MDD 27.86%/CAGR -24.53%, LUNA/FTX 2022) — **다음 세션의 최우선 타겟**.
- **`/check`**: 6개 spec 전부 PASS(Cov 37~100%, spec별 상이).

## 잔여 이슈 (우선순위순)

1. **[최우선] Crisis stress test 방어 실패** — 정상장 PASS champion이 LUNA/FTX 재현에서 MDD 27.86%(예산 초과)·CAGR -24.53%(하한 미달)로 붕괴. spec 3(regime-cell hard-block searchability)이 이 특정 champion에서 실제로 `hybrid` 모드를 선택했는지, crisis-specific leverage margin(`l2_deploy_crisis_mdd_margin`, spec 1에서 분리됨)이 이 champion에 대해 올바르게 작동했는지 직접 검증 필요.
2. **`[REGIME-L2] proof_failed path=pooled_fallback`** — bucket edge routing의 aggregate proof가 지속 실패. 원인은 진단 완료(nw_tstat=-8.01, 강한 음의 신호로 정확히 pooled 폴백 — 버그 아님), 다만 `l2_routing_mode="bucket"`의 이 부분이 사실상 상시 비활성 상태라는 점은 아키텍처 재검토 대상.
3. **spec 5(worst-fold leverage clamp) 미검증** — 메커니즘 자체가 발동하는 champion을 이번 seed에서 만나지 못함. 고정 파라미터 기반 격리 스크립트로 clamp 전/후 직접 비교 필요.
4. **다중 seed/study 검증 부재** — 오늘 모든 실측이 단일 seed=42 기준. Optuna champion-to-champion 변동성이 크므로, 이번 PASS가 재현 가능한지 여러 seed로 확인 필요.
