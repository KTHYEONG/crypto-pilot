# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-06 (평가 기준 강화 → BLOCKED. fold 3 +6.8bps 정직하게 실패 처리)  
**이전 기준:** 2026-06-06 (Ranking 버그 수정 + Production Default 전환, pass_ratio 0.75)  
**현재 기본 모드:** `expected_edge_direct` (production default, 검증 완료)  
**평가 기준:** `min_fold_realized_edge_bps=15.0` (이전 0.0), `min_cagr_for_promotion=0.15` (이전 0.02)

---

## 실행 요약 (최신)

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견
[PIPELINE] raw=272819 labeled=6350 promoted=6350 fit=9869 cal=9165 oos=2355 n_folds=4 wf_scheme=anchored
[BRIDGE][WF] fold_cost_survival=[False, True, False, True] pass_ratio=0.50 min_required=0.60
[BRIDGE SUMMARY] Active Signals 0 (sel=0) | Status BLOCKED | Execution Time 34.10s
[BRIDGE SUMMARY][DIAG] selected_total=0 eligible_total=0 zero_reason=wf_fold_pass_ratio_fail gate_p50=nan mu_p50=nan utility_p50=nan breakeven_floor=nan
[BRIDGE SUMMARY][WF_DIAG] wf_selected=117 wf_eligible=1144 shadow=expected_edge_direct:off:0.00:-50.0:0.00 shadow_selected=24 shadow_realized=136.894 eu_p90=53.800 downside_p90=228.095
```

---

## 백테스트 성과 (OOS)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Equal Size | -17.6% | 47.0% | -0.37 | 559,693 | 8 | 0.66 | N |
| Rule Promo NL | -14.1% | 16.0% | -0.88 | 913,690 | 402 | 0.64 | N |
| Rule Promo Oracle | 2.6% | 3.6% | 0.71 | 1,015,037 | 182 | 0.17 | N |
| Kelly (No ML) | -0.1% | 0.4% | 0.00 | 997,350 | 2149 | 0.66 | N |
| ML Gate | 0.0% | 0.7% | 0.00 | 1,000,388 | 101 | 0.14 | N |
| ML Gate+Edge | -1.1% | 9.1% | -0.12 | 967,273 | 87 | 0.01 | N |
| ML Full (Capped) | 0.0% | 0.6% | 0.00 | 1,000,434 | 100 | 0.14 | N |
| Cand. ML | 0.0% | 0.7% | 0.00 | 1,000,178 | 100 | 0.70 | N |
| Direct Edge | 0.2% | 0.4% | 0.00 | 1,001,297 | 119 | 0.34 | N |
| Variant Prior | 1.6% | 0.6% | 0.00 | 1,009,208 | 154 | 0.31 | N |
| Promo Filter | -0.6% | 0.8% | 0.00 | 996,231 | 614 | 0.99 | N |
| Val. Selection | -0.5% | 0.4% | 0.00 | 997,245 | 3 | 0.03 | N |
| Identity Feat | -0.3% | 0.6% | 0.00 | 998,388 | 111 | 0.71 | N |
| Market Feat | 0.1% | 0.5% | 0.00 | 1,000,338 | 108 | 0.73 | N |

---

## 2026-06-06: 평가 기준 강화 (경제적 최소선 적용)

### 기준 변경
| 항목 | 이전 | 변경 후 | 이유 |
|---|---|---|---|
| `min_fold_realized_edge_bps` | 0.0 bps | **15.0 bps** | RT cost(7.5bps) × 2배 최소선 |
| `min_cagr_for_promotion` | 2% | **15%** | crypto 위험 프리미엄 최소선 |

### fold별 결과 (新기준 적용)

| Fold | OOS 기간 | 선택 수 | realized edge | hit rate | 결과 |
|---|---|---:|---:|---:|---|
| 1 | Oct-Nov 2025 | 23 | **-144.9 bps** | 8.7% | ❌ (수익 음수) |
| 2 | Nov-Dec 2025 | 24 | **+136.9 bps** | 50.0% | ✅ |
| 3 | Jan-Feb 2026 | 30 | **+6.8 bps** | 33.3% | ❌ (**新기준**으로 탈락: 6.8 < 15bps) |
| 4 | Feb-Mar 2026 | 40 | **+54.2 bps** | 35.0% | ✅ |

**pass_ratio = 2/4 = 0.50 < 0.60 → BLOCKED**

### 핵심 해석
- Fold 3의 +6.8bps는 거래비용(7.5bps)도 회수하지 못하는 수준 — 기존 0.0 기준이 잘못 통과시킨 것
- 현재 `gate p_pass max=0.4958` — 모델이 0.5 이상 확신을 한 번도 못 내는 상태
- **BLOCKED = 정직한 현재 상태.** 개선 없이 Optuna 진입은 의미 없음

### 다음 목표
3개 이상의 fold에서 realized edge ≥ 15bps 달성이 선행 조건:
1. Gate 모델이 p_pass > 0.55 신호를 실제로 골라낼 수 있어야 함
2. Fold 3 구간(Jan-Feb 2026) 신호 품질 개선 — 33% hit rate에서 +6.8bps는 손실 트레이드의 규모가 너무 큰 것

---

## ML Layer 결과

### 1. Realized fold survival로 전환
* `fold_cost_survival`은 예측 `mu` t-stat이 아니라 `selected fold`의 realized edge/log-growth 기준으로 평가되었습니다.
* 이번 실행에서는 4개 fold 모두 `pass=False`였고, 최종 `pass_ratio=0.00`으로 `min_wf_fold_pass_ratio=0.60`을 충족하지 못해 `fail-closed`로 종료되었습니다.
* 주요 원인은 `utility_topk`에서 `eligible=0`이었고, `zero_reason=no_eligible_after_breakeven_floor`가 반복된 점입니다.

### 2. Waterfall diagnostics로 막히는 지점이 명확해짐
* fold별 `waterfall_expected_utility_adj_p90_bps`는 약 `-66 ~ -71bps` 범위였고, `waterfall_downside_drag_p90_bps`는 약 `187 ~ 264bps` 수준이었습니다.
* 즉 현재 문제는 단순 gate 경직만이 아니라, `p_pass * mu`가 downside drag를 전혀 상쇄하지 못해 상위 10% 후보조차 음의 expected utility에 머무는 구조입니다.
* `breakeven_floor` 자체는 `3.8bps`로 낮은 편이라, 이번 실패를 cost floor 단독 문제로 보기는 어렵습니다.

### 3. Gate / utility 계약이 더 보수적으로 동작
* 새 `selection_gate_mode="soft_floor"`와 expected utility 계약이 반영된 뒤, gate가 단순 승률 문턱이 아니라 confidence input으로 해석되도록 바뀌었습니다.
* 그 결과 `ML Gate`, `ML Gate+Edge`, `ML Full (Capped)`, `Cand. ML`, `Direct Edge`, `Variant Prior`, `Identity Feat`, `Market Feat`는 모두 0 trades로 수렴했습니다.
* `Promo Filter`만 1 trade를 남겼지만 `pass_deploy=False`였고, `Val. Selection`은 76 trades로 늘었으나 `real_p50=-68.0bps`, `capture=-2.172`로 실수익성과는 맞지 않았습니다.

### 4. Shadow profile도 개선 신호를 주지 못함
* fold-level shadow diagnostics를 추가했지만, 이번 실행에서는 best shadow profile조차 `shadow_selected=0`이었습니다.
* 따라서 당장 `selection_gate_mode`, `selection_min_expected_utility_bps` 같은 threshold 완화만으로 문제를 해결할 가능성은 낮아졌습니다.

### 5. Calibration 상태
* `gate_calibration_method: isotonic`은 유지되었지만 일부 fold에서는 `calibration_probability_collapse`로 `used=False`가 발생했고, 이후 fold는 `calibration_accepted`로 전환되었습니다.
* 다만 gate가 최종 selection에 기여한 결과는 `eligible=0`이어서, calibration 회복이 실제 선택으로 이어지지 않았습니다.

---

## 2026-06-06: Ranking 버그 수정 + Production Default 전환 → PROMOTED

### 수정 내용 (3개 버그 + 2개 개선)
1. **P0-A (Ranking Bug)**: `select_candidate_events_for_portfolio` topk 정렬이 additive `utility_score` 사용 → `expected_utility_bps` (= mu_net) 기준으로 수정. Shadow와 Production 정렬 기준 일치.
2. **P0-B (Scoring Bug)**: `CandidateModelOutput.utility_score`가 항상 additive → `expected_edge_direct` 모드 시 `mu_net_decision_bps`로 수정.
3. **Config Default 전환**: `selection_utility_mode = "additive_drag"` → `"expected_edge_direct"` (A/B 검증 완료 → 승격).
4. **Model Capacity**: `max_depth 2→3`, `reg_lambda 100→30` (depth-2 = 사실상 선형 모델이었음).
5. **Early Stopping**: LightGBM gate/edge 모델에 `stopping_rounds=30` 추가.

### E2E 결과 (2026-06-06, 버그 수정 후)

| 지표 | 이전 (2026-06-06 A/B) | 최신 (버그 수정 + default 전환) | 변화 |
|---|---:|---:|---:|
| fold별 eligible | 219, 234, 291, 393 | 224, 234, 291, 395 | ≈ 동일 |
| fold별 selected | 21, 24, 26, 38 | 23, 24, 30, 40 | +14 |
| `fold_cost_survival` | [F,T,T,F] | **[F,T,T,T]** | **+1 pass** |
| `pass_ratio` | 0.50 | **0.75** | **+0.25** (min 0.60 ✅) |
| EU p90 (OOS 전체) | +54.1 bps | **+53.8 bps** | ≈ 동일 |
| Active Signals | 0 (BLOCKED) | **3050 (PROMOTED)** | **구조적 해소** |
| Selected events | 0 | **115** | +115 |
| Shadow realized (fold2) | +83.2 bps | **+136.9 bps** | +53.7 bps |

### 핵심 해석
- **PROMOTED 달성**: `pass_ratio 0.75 >= 0.60`. fold 1만 실패 (Oct-Nov 2025, realized=-144.9bps, hit=8.7% — 신호가 실제로 맞지 않는 기간).
- **Fold 1 실패는 정책/시장 문제**: EU p90 62.5bps로 selection은 정상 작동. 해당 기간 시장이 신호를 역전시킨 것.
- **Ranking 버그 수정 효과**: fold4 선택 품질 향상 (realized 54.2bps, hit=35%), fold3도 log_growth 양전.
- **Variant Prior 최우수**: 1.6% CAGR, MaxDD 0.6%. ML 기반 모델 중 유일하게 의미 있는 수익.

---

## 2026-06-06: Selection Utility Mode 도입 및 검증

### 문제 진단 (근본원인)
2026-06-05 결과의 `eligible=0`, `eu_p90=-69.5bps` 교착의 근본원인 규명:
- **`mu_net_decision_bps`**: `y_edge_bps = edge_after_hurdle_bps`의 **무조건부 평균** (Huber regression, 수십 bps 스케일). 이미 패자를 포함한 기댓값.
- **`q10_net_bps`**: `y_q10_bps = clip(mae_bps, -sl_thr_bps)`의 10분위 (quantile 0.10, 수백 bps 스케일). **return outcome이 아니라 경로 MAE proxy**.
- **현재 EU 공식**: `p_pass·mu − (1−p_pass)·|q10| − turnover`
  - **결함 1 (이중계상):** `mu_net`은 이미 기댓값인데 `p_pass`로 다시 축소 → 하방을 두 번 차감
  - **결함 2 (스케일 불일치):** 수십 bps 기댓값에서 수백 bps MAE proxy를 가산 차감 → EU 깊은 음수

### 개선: Selection Utility Mode (2 Pillar)

#### Pillar 1: EU Decomposition 정정
- **`expected_edge_direct` (신규)**: selection 점수 = `mu_net` 단독. `p_pass`/`q10`는 sizing(confidence discount, variance floor) + catastrophic veto로만 사용.
- **`additive_drag` (기본값 보존)**: 기존 공식 유지 (회귀 0, A/B 비교용).

#### Pillar 2: 동적(fold-adaptive) Breakeven Floor
- **`fold_adaptive` (신규)**: fold-local `cost_floor_bps` 분위수 기반 → 약한 fold 자동 강화, 강한 fold 개방.
- **`static` (기본값 보존)**: 고정 bps (회귀 0).

### E2E A/B 검증 결과 (2026-06-06)

| 지표 | Baseline (additive_drag/static) | 신규 (expected_edge_direct/fold_adaptive) | 개선량 |
|---|---:|---:|---:|
| fold별 eligible | 0, 0, 0, 0 | 219, 234, 291, 393 | **+1137 total** |
| fold별 selected | 0, 0, 0, 0 | 21, 24, 26, 38 | **+109 total** |
| `fold_cost_survival` | [F,F,F,F] | [F,T,T,F] | **+2 pass** |
| `pass_ratio` | 0.00 | 0.50 | **+0.50** (min 0.60) |
| EU p90 | -69.5 bps | +54.1 bps | **+123.6 bps** |
| Shadow realized edge | N/A | +83.2 bps (fold2) | **실수익성 양전** |

### 핵심 해석
- **구조적 교착 해소:** `eligible 0 → 1137`, 격자 변환이 아닌 **계약 자체 재설계**로 인한 도약.
- **여전히 BLOCKED:** `pass_ratio 0.50 < 0.60`이나 원인이 **정책/품질**로 전환 (데드락 아님).
- **Shadow A/B 우월성 자동 포착:** baseline 실행에서도 best shadow = `expected_edge_direct` + realized +83bps → production 변경 없이도 next step 가시화.
- **구현은 config-gated:** 기본값 보존, shadow 평가로 risk-free A/B 진행 중. 승격은 별도 결정.

---

## 다음 단계 (현재 BLOCKED)

PROMOTED 재달성 조건 (3/4 fold에서 realized edge ≥ 15bps):

1. **Gate 모델 개선 (최우선):** p_pass가 0.5를 한 번도 넘지 못하는 근본 원인 진단. label 전략 재검토 또는 피처 품질 개선.
2. **Fold 1 역전 구간 진단:** Oct-Nov 2025 hit_rate 8.7% → 어떤 signal family가 역방향 편향인지 `[DIAG][SELECT_VARIANT]` 로그 분석 후 비활성화.
3. **Fold 3 edge 개선:** Jan-Feb 2026 realized +6.8bps → hit_rate 33%에서 손실 트레이드 규모 축소 필요. payoff ratio 개선이 핵심.
4. **Optuna는 선행 조건 충족 후:** Gate가 실제로 좋은 신호를 구별할 때 Optuna가 의미를 가짐. 현재는 시기 상조.
