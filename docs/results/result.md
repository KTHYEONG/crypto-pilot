# L2 Phase 성과 개선 세션 결과 — 2026-07-19 (8개 spec 누적: 정상장 PASS 유지, crisis는 champion-selection fallback 취약점으로 재악화)

## 세션 개요

전일(spec 1~7) `crisis MDD 최초 통과(17.50%) + crisis CAGR 잔존 실패(-14.58%)`를 출발점으로, spec6·7과 동일 계열의 대칭 버그(champion 선정이 crisis **CAGR**을 못 보는 문제)를 spec8로 진단·수정했다. 단위 테스트 레벨에서는 정상 작동을 확인했으나, **동일 seed 프로덕션 실측에서 champion 선정이 non-deterministic-replay fallback 경로로 빠지며 crisis MDD가 오히려 30.92%로 재악화**되는 더 근본적인 결함이 새로 드러났다. 이번 세션은 spec8 구현 + fallback 분기 가시성 확보로 마무리하고, fallback 근본 원인 진단은 다음 세션으로 명시적으로 이월한다.

| # | Spec / ADR | 대상 | 실측 효과 |
| :--- | :--- | :--- | :--- |
| 1 | `ADR_20260718_L2_DEPLOYMENT_MARGIN_CAGR_GATE` | leverage 캘리브레이션 안전마진 decouple + searchable화 | MDD 예산 활용 헤드룸 확보 |
| 2 | `ADR_20260718_L2_FOLD_GRANULARITY_ROBUSTNESS` | L2 전용 walk-forward fold 개수 분리(L1/live 비영향) | fold_pass_ratio 50%→75%(4→8-fold 실험), 국소 손실 구간 격리 실증 |
| 3 | `ADR_20260718_L2_REGIME_CELL_ADMISSION_SEARCHABILITY` | regime-cell hard-block/passthrough 탐색공간 편입 | Sharpe/Sortino/Calmar/PSR 동시 통과 champion 최초 발견 |
| 4 | `ADR_20260718_L2_OPTUNA_CONSTRAINT_CAGR_UPLIFT_ALIGNMENT` | Optuna `constraints_func`에 cagr·sharpe_uplift 승격(10→12-tuple) | CAGR +17.5%→+30.4%, Sharpe Uplift -0.63→+0.65 |
| 5 | `ADR_20260719_L2_OOS_WORST_FOLD_LEVERAGE_FLOOR_CLAMP` | OOS worst-fold CAGR 플로어 기반 레버리지 하향 클램프 | 메커니즘 미발동(champion 변동성으로 미검증, 후속 필요) |
| 6 | `ADR_20260719_L2_ACTIVE_BLOCK_COUNT_LIGHTWEIGHT_FIX` | **버그 수정**: `lightweight=True` 시 `active_block_count` 항상 0 | 세션 최초 정상장 `STATUS: PASS` |
| 7 | `ADR_20260719_L2_CHAMPION_SELECTION_CRISIS_BLINDNESS_FIX` | **버그 수정**: `select_layer2_champion`이 `crisis_mdd_hybrid=None`으로 호출돼 항상 자동 feasible 처리 | crisis MDD 27.86%→17.50%, 예산(21%) 최초 통과 |
| 8 | `ADR_20260719_L2_CRISIS_CAGR_CHAMPION_SELECTION_BLINDNESS_FIX` | **버그 수정(spec7과 대칭)**: `compute_crisis_mdd_budget`이 이미 계산된 crisis CAGR을 버리고 MDD만 반환 → 선정 루프가 crisis CAGR을 전혀 못 봄 | 메커니즘 자체는 정상 작동 확인(단위테스트) — 그러나 실측에서 **fallback 취약점**을 새로 노출(아래 참조) |

## spec 8 근본 원인 (spec6·7과 동일 계열의 세 번째 사례)

- `compute_crisis_mdd_budget()`(`optimization/workflow.py`)은 crisis-window 재시뮬레이션(`_run_awf_simulation` + `apply_deployment`)을 이미 수행하면서 `DeploymentResult.mdd`만 읽고, 같은 객체에 이미 계산되어 있는 `DeploymentResult.cagr`은 버리고 있었다.
- crisis CAGR이 실제로 평가되는 유일한 지점은 `crisis_policy.py::evaluate_crisis_survival()`인데, 이는 **champion이 이미 확정된 뒤 실행되는 사후 리포트**이며 Optuna `constraints_func`나 `select_layer2_champion`의 입력이 아니었다.
- **해결**: `compute_crisis_mdd_budget` → `compute_crisis_replay_budget`(`CrisisReplayBudget` dataclass 반환: `mdd_hybrid`/`mdd_budget`/`cagr_hybrid`/`cagr_floor`)로 확장. `evaluate_layer2_gate`의 `optuna_constraint_values`를 12→13-tuple로 확장(13번째=crisis CAGR 제약). `Layer2AllocationConfig.l2_min_crisis_cagr`(-0.05, fixed/non-searchable) 신설 — `l2_min_worst_fold_cagr`와 값은 우연히 같으나 spec1의 crisis-margin decoupling 전례(`l2_deploy_crisis_mdd_margin` vs `l2_deploy_mdd_margin`)를 따라 독립 필드로 분리, `pipeline.py`의 `evaluate_crisis_survival` 호출도 동일 필드로 SSOT 정합.
- 단위 테스트로 13-tuple 구성/None-passthrough/champion 거부 로직 검증 완료, `/check` PASS(Cov 43~62%).

## 프로덕션 실측 (seed=42, n_trials=120, spec 1~8 전부 반영) — ⚠️ crisis 결과 재악화

### 정상장 스코어카드: ✅ PASS (spec7 대비 변화 없음)

```
STATUS  : ✅ PASS

✅ [Growth    ] CAGR: +59.1% (>=30.0%) | PnL: +51.5% | Equity x1.52
✅ [Efficiency] Sharpe: 2.178 (>=1.000) | Sortino: 3.724 (>=1.500) | Calmar: 3.381 (>=1.000)
✅ [Risk      ] MDD: 17.5% (<=30.0%) | CVaR95: 1.4% (<=6.0%) | RiskUtil: 58.3%
✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 477 (>=30) | Friction: 99.8%
✅ [Uplift    ] Sharpe Uplift: +0.37 (>=+0.05)
✅ [Integrity ] PSR: 0.998 (>=0.90) | DSR: 0.993 (diag)
```

| Fold | Period | CAGR | MDD | Sharpe | Status |
| :--- | :--- | ---: | ---: | ---: | :---: |
| 1 | 2025-03-20 ~ 2025-05-30 | +50.3% | 9.5% | 2.039 | ✅ PASS |
| 2 | 2025-05-30 ~ 2025-08-09 | +19.8% | 14.9% | 1.062 | ✅ PASS |
| 3 | 2025-08-09 ~ 2025-10-20 | +16.4% | 9.8% | 0.804 | ✅ PASS |
| 4 | 2025-10-20 ~ 2025-12-30 | +204.3% | 6.1% | 4.197 | ✅ PASS |

### ❌ Crisis Stress Test: MDD/CAGR 모두 재악화 — **champion-selection fallback 취약점**

```
[EVAL] event=replay_flip trial=71 stored_pass=True replay_pass=False stored_cagr=0.3344 replay_cagr=0.3677 stored_mdd=0.1763 replay_mdd=0.1910
[EVAL] event=replay_flip trial=46 stored_pass=True replay_pass=False stored_cagr=0.3088 replay_cagr=0.3505 stored_mdd=0.1470 replay_mdd=0.1641
[L2-SELECTION] No feasible candidate found within fallback window (reason=non_deterministic_replay)
[CRISIS-RELIABILITY] status=stress_tested_fail verified=False detail=luna_ftx_2022_collapse:mdd_abs; luna_ftx_2022_collapse:cagr
[CRISIS-WINDOW-DETAIL] label=luna_ftx_2022_collapse status=stress_tested_fail mdd=0.3092 cagr=-0.1145 cvar95=0.0275 trades=466 symbols=45
```

| Metric | spec 7(직전) | spec 8 적용 실측(오늘) | Budget | 판정 |
| :--- | ---: | ---: | :---: | :---: |
| MDD | 17.50% | **30.92%** | ≤21% | ❌ **재악화, 예산 재초과** |
| CAGR | -14.58% | -11.45% | ≥-5% | ❌ 여전히 미달(소폭 개선) |
| L* | 1.00(바닥값) | **1.79** | — | — |

**원인**: 이번 실행은 champion이 정상 선정되지 않고 **fallback 경로**로 빠졌다. gate-pass로 기록됐던 후보 2개(trial #71, #46)가 replay 재검증 시점에 flip(stored_pass=True → replay_pass=False)해 최종 feasible 후보가 0개가 됐고, 이때 `select_layer2_champion`은 **모든 gate 제약(crisis MDD·CAGR 포함)을 무시하고 objective 최댓값만 보는 "best diagnostic" fallback**을 champion으로 채택한다. 그 결과 crisis 방어 레버가 전혀 걸러지지 않은 채 L*가 1.79까지 상승했고 crisis MDD가 예산을 재초과했다.

**핵심 판단**: spec8의 crisis-CAGR 가시성 메커니즘 자체는 정상 작동한다(13-tuple 구조·replay flip 로그 정확히 노출, 단위테스트 통과). 그러나 **"non_deterministic_replay fallback이 발동하면 spec7·spec8의 crisis 방어 제약이 통째로 무력화된다"**는, 오늘 발견보다 더 근본적인 구조적 결함이 이번 실측으로 새로 드러났다. crisis CAGR/MDD 제약을 아무리 정교화해도 fallback 경로로 빠지는 한 무의미하다.

### 후속 조치 (이번 세션 내 완료)
- `select_layer2_champion`의 champion 확정 분기뿐 아니라 **fallback(non_deterministic_replay) 분기에도** `[ALGO] event=champion_regime_levers fallback=True ...` 로그를 추가 — 이번 실측처럼 fallback이 발동한 실행에서도 실제 선택된 regime 레버 값(policy_mode/hard_block/asymmetry/severity_gating/crisis_gross_cap)을 즉시 확인 가능하도록 배선. `/check` PASS(Cov 62%).
- fallback 근본 원인(non-deterministic replay 자체의 진단/수정) 및 fallback 경로에서도 crisis 제약을 최소한 지키게 하는 하드닝은 **다음 세션으로 명시 이월**(아래 잔여 이슈 #1).

## Verdict

- **L1**: 이번 세션 스코프 밖 — 변경 없음.
- **L2 정상장**: ✅ **PASS 유지** (CAGR +59.1%, 16개 promotion blocker + worst_fold_cagr 전부 통과).
- **L2 crisis MDD**: ❌ **재악화** (17.50%→30.92%, 예산 21% 재초과) — spec7의 성과가 fallback 발동으로 이번 실행에서 재현되지 않음.
- **L2 crisis CAGR**: ❌ **잔존 실패** (-14.58%→-11.45%, 소폭 개선이나 하한 -5% 미달 지속).
- **`/check`**: spec8 관련 전부 PASS(Cov 43~62%). `test_layer2_gate_fixes.py`의 `L2_ALLOC_SPACE`류 import 실패 4건은 base 브랜치에서도 동일 재현되는 **스코프 밖 기존 결함**(오늘 변경과 무관, 확인 완료).

## 잔여 이슈 (우선순위순)

1. **[신규 최우선] non_deterministic_replay fallback 자체가 crisis 방어를 통째로 무력화** — gate-pass로 기록된 trial이 replay 재검증에서 flip하는 근본 원인(비결정성 소스: 부동소수 non-associativity, ThreadPoolExecutor 평가 순서, 캐시 fingerprint 불일치 등 후보) 진단 필요. 최소한의 완화책으로 fallback 분기에서도 crisis MDD/CAGR 제약을 2차 필터로 적용하는 하드닝 검토.
2. **Crisis CAGR 방향성 손실** — spec8 가시성 확보는 완료됐으나, fallback 미발동 시(정상 champion 선정 시) 실제로 hard-block/비대칭 롱숏/severity-gating이 crisis CAGR을 개선하는지는 이번 실행에서 검증되지 못함(fallback으로 우회됨). #1 해결 후 재검증 필요.
3. **`[REGIME-L2] proof_failed path=pooled_fallback`** — bucket edge routing의 aggregate proof 지속 실패(nw_tstat=-8.01, 버그 아님으로 진단 완료). `l2_routing_mode="bucket"` 상시 비활성 상태, 아키텍처 재검토 대상.
4. **spec 5(worst-fold leverage clamp) 미검증** — 메커니즘 발동 champion 미조우, 고정 파라미터 격리 스크립트로 전/후 비교 필요.
5. **다중 seed/study 검증 부재** — 오늘도 단일 seed=42 기준. champion-to-champion 변동성 + fallback 발동 빈도 자체가 seed에 민감할 가능성 — #1 진단 시 다중 seed 스윕 병행 권장.
