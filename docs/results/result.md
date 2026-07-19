# L2 Phase 성과 개선 세션 결과 — 2026-07-19 (7개 spec 누적: 정상장 PASS + crisis MDD 방어 확보, 잔여 병목은 crisis 방향성)

## 세션 개요

`BLOCKED(cagr, +20.9%)`(2026-07-18)를 출발점으로 코드+데이터 기반 진단을 반복해 7개 spec을 순차 구현·실측 검증했다. **정상장 게이트 전체 PASS**를 달성했고, crisis stress test(LUNA/FTX)에서도 **MDD 예산을 처음으로 통과**시켰다. 잔여 병목은 crisis CAGR(방향성 손실) 단일 항목으로 좁혀졌다.

| # | Spec / ADR | 대상 | 실측 효과 |
| :--- | :--- | :--- | :--- |
| 1 | `ADR_20260718_L2_DEPLOYMENT_MARGIN_CAGR_GATE` | leverage 캘리브레이션 안전마진 decouple + searchable화 | MDD 예산 활용 헤드룸 확보 |
| 2 | `ADR_20260718_L2_FOLD_GRANULARITY_ROBUSTNESS` | L2 전용 walk-forward fold 개수 분리(L1/live 비영향) | fold_pass_ratio 50%→75%(4→8-fold 실험), 국소 손실 구간 격리 실증 |
| 3 | `ADR_20260718_L2_REGIME_CELL_ADMISSION_SEARCHABILITY` | regime-cell hard-block/passthrough 탐색공간 편입 | Sharpe/Sortino/Calmar/PSR 동시 통과 champion 최초 발견 |
| 4 | `ADR_20260718_L2_OPTUNA_CONSTRAINT_CAGR_UPLIFT_ALIGNMENT` | Optuna `constraints_func`에 cagr·sharpe_uplift 승격(10→12-tuple) | CAGR +17.5%→+30.4%, Sharpe Uplift -0.63→+0.65 |
| 5 | `ADR_20260719_L2_OOS_WORST_FOLD_LEVERAGE_FLOOR_CLAMP` | OOS worst-fold CAGR 플로어 기반 레버리지 하향 클램프 | 메커니즘 미발동(champion 변동성으로 미검증, 후속 필요) |
| 6 | `ADR_20260719_L2_ACTIVE_BLOCK_COUNT_LIGHTWEIGHT_FIX` | **버그 수정**: `lightweight=True` 시 `active_block_count` 항상 0 → `optuna_constraint_values` 6번째 슬롯 100% 위반 | **세션 최초 정상장 `STATUS: PASS`**(`feasible_trials` 최초로 비어있지 않게 됨) |
| 7 | `ADR_20260719_L2_CHAMPION_SELECTION_CRISIS_BLINDNESS_FIX` | **버그 수정**: `select_layer2_champion`이 `crisis_mdd_hybrid=None`으로 호출돼 항상 자동 feasible 처리(선정이 crisis를 전혀 못 봄) | **crisis MDD 27.86%→17.50%, 예산(21%) 최초 통과** |

## 최종 실행 결과 (seed=42, n_trials=120, spec 1~7 전부 반영)

### 정상장 스코어카드: ✅ PASS

```
STATUS  : ✅ PASS

✅ [Growth    ] CAGR: +33.7% (>=30.0%) | PnL: +57.4% | Equity x1.57
✅ [Efficiency] Sharpe: 2.098 (>=1.000) | Sortino: 3.795 (>=1.500) | Calmar: 3.378 (>=1.000)
✅ [Risk      ] MDD: 10.0% (<=30.0%) | CVaR95: 0.8% (<=6.0%) | RiskUtil: 33.3%
✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 792 (>=30) | Friction: 100.0%
✅ [Uplift    ] Sharpe Uplift: +0.29 (>=+0.05)
✅ [Integrity ] PSR: 0.998 (>=0.90) | DSR: 0.994 (diag)
[Diag     ] RelMDD: 1.01x | Turnover: 0.059
```

| Metric | Value | Gate |
| :--- | ---: | :---: |
| Leverage (L*) | 1.0000 (binding: mdd, 바닥값) | |
| CAGR | **+33.7%** | ✅ (≥30%) |
| MDD | 10.0% | ✅ (≤30%) |
| CVaR95 | 0.8% | ✅ (≤6%) |
| Sharpe / Sortino / Calmar | 2.098 / 3.795 / 3.378 | ✅ / ✅ / ✅ |
| Fold Pass Ratio | 100.0% (4/4) | ✅ (≥60%) |
| Sharpe Uplift | +0.29 | ✅ (≥+0.05) |
| PSR | 0.998 | ✅ (≥0.90) |

### Fold 상세 — 4/4 전원 PASS

| Fold | Period | CAGR | MDD | Sharpe | Status | Symbols |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| 1 | 2025-03-20 ~ 2025-05-30 | +1.8% | 10.0% | 0.209 | ✅ PASS | 33 |
| 2 | 2025-05-30 ~ 2025-08-09 | +21.9% | 5.5% | 2.024 | ✅ PASS | 28 |
| 3 | 2025-08-09 ~ 2025-10-20 | +15.5% | 4.7% | 1.466 | ✅ PASS | 26 |
| 4 | 2025-10-20 ~ 2025-12-30 | +122.0% | 5.1% | 3.831 | ✅ PASS | 23 |

### ✅ Crisis Stress Test: MDD 방어 최초 통과, CAGR만 잔존 실패

```
[CRISIS-RELIABILITY] status=stress_tested_fail verified=False detail=luna_ftx_2022_collapse:cagr
[CRISIS-WINDOW-DETAIL] label=luna_ftx_2022_collapse status=stress_tested_fail mdd=0.1750 cagr=-0.1458 cvar95=0.0137 trades=867 symbols=45
```

| Metric | spec 6까지(이전) | spec 7 적용(최신) | Budget | 판정 |
| :--- | ---: | ---: | :---: | :---: |
| MDD | 27.86% | **17.50%** | ≤21% | ✅ **최초 통과** |
| CAGR | -24.53% | **-14.58%** | ≥-5% | ❌ 미달(개선폭 +9.95%p) |
| CVaR95 | 2.11% | 1.37% | — | — |

**의미**: champion 선정이 crisis 데이터를 처음으로 참조하게 되면서, Optuna가 레버리지를 바닥값(L*=1.0)까지 낮춘 더 안전한 champion을 선택 — crisis MDD가 예산 안으로 들어왔다(10.4%p 개선). 정상장 CAGR은 38.9%→33.7%로 소폭 하락했으나 여전히 게이트를 여유 있게 통과 — 정상장 성과를 크게 희생하지 않고 crisis MDD 방어력을 확보한 좋은 트레이드오프.

**남은 문제**: crisis CAGR(-14.58%)은 이미 L*=1.0(무레버리지)에서도 미달 — **사이징(레버리지)으로는 더 개선 불가**. LUNA/FTX 붕괴 기간 동안 전략의 방향성(포지션의 롱/숏 선택) 자체가 손실을 내고 있다는 뜻. 다음 타겟은 regime cap·비대칭 롱/숏·hard-block 등 "무엇을 살지"를 결정하는 메커니즘.

## 근본 원인 진단 (spec 6·7 — 대칭적인 두 버그)

- **spec 6 (`active_block_count`)**: Optuna 탐색 루프의 전 trial이 `lightweight=True`로 평가되며 `active_block_count`가 입력과 무관하게 항상 0 → `l2_min_active_blocks=3` 제약을 100% trial에서 위반 → `feasible_trials`가 구조적으로 항상 공집합("존재하지 않는 신호로 인한 오판정=FAIL").
- **spec 7 (`crisis_mdd_hybrid`)**: `select_layer2_champion`이 `crisis_rets`/`crisis_replay_ctx`를 아예 받지 않아 replay 단계의 crisis 제약이 항상 `None`→자동 feasible(-1.0) 처리 → champion 선정이 crisis 데이터를 전혀 반영 못함("존재하지 않는 신호로 인한 오판정=PASS", spec 6과 정반대 방향의 동일 계열 결함).
- 두 버그 모두 `evaluate_l2_trial(lightweight=True)`가 진단/비용 절감을 위해 스킵하는 계산에, 하류의 제약 판정이 암묵적으로 의존하고 있었다는 공통 패턴 — `compute_crisis_mdd_budget`처럼 로직을 공유 함수로 추출해 재사용하는 방식으로 해결.

## Verdict

- **L1**: 이번 세션 스코프 밖 — 변경 없음, 7/7 TF 정상 가정.
- **L2 정상장**: ✅ **PASS** (CAGR +33.7%, 16개 promotion blocker + worst_fold_cagr 전부 통과).
- **L2 crisis MDD**: ✅ **최초 통과** (17.50% ≤ 21% 예산).
- **L2 crisis CAGR**: ❌ **잔존 실패** (-14.58% < -5% 하한) — **다음 세션 최우선 타겟**, 사이징이 아닌 방향성 문제로 명확히 좁혀짐.
- **`/check`**: 7개 spec 전부 PASS(Cov 30~100%, spec별 상이).

## 잔여 이슈 (우선순위순)

1. **[최우선] Crisis CAGR 방향성 손실** — L*=1.0(무레버리지)에서도 -14.58%. spec 3(regime-cell hard-block searchability)이 이 champion에서 실제로 `hybrid` 모드·비대칭 롱/숏·cooldown을 선택했는지 직접 확인 필요. LUNA/FTX 붕괴 구간에서 어느 포지션(롱/숏, 어느 심볼)이 손실을 주도했는지 attribution 진단 선행 권장.
2. **`[REGIME-L2] proof_failed path=pooled_fallback`** — bucket edge routing의 aggregate proof가 지속 실패. 원인 진단 완료(nw_tstat=-8.01, 강한 음의 신호로 정확히 pooled 폴백 — 버그 아님), `l2_routing_mode="bucket"`의 이 부분이 사실상 상시 비활성 상태라는 점은 아키텍처 재검토 대상.
3. **spec 5(worst-fold leverage clamp) 미검증** — 메커니즘 자체가 발동하는 champion을 아직 만나지 못함. 고정 파라미터 기반 격리 스크립트로 clamp 전/후 직접 비교 필요.
4. **다중 seed/study 검증 부재** — 오늘 모든 실측이 단일 seed=42 기준. champion-to-champion 변동성이 크므로, 이번 PASS + crisis MDD 통과가 재현 가능한지 여러 seed로 확인 필요.
