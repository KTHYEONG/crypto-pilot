# Mode Full (ML) — 최신 검증 결과

**실행 일시:** 2026-06-05 (새 `selection_gate_mode` + realized fold survival 계약 적용 후)  
**실행 명령어:** `UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase ml --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 0 (sel=0)`, `Status: BLOCKED (realized fold survival fail)`

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견
[PIPELINE] raw=272819 labeled=6350 promoted=6350 fit=9869 cal=9165 oos=2355 n_folds=4 wf_scheme=anchored
[BRIDGE][WF] fold_cost_survival=[False, False, False, False] pass_ratio=0.00 min_required=0.60
[BRIDGE SUMMARY] Active Signals 0 (sel=0) | Status BLOCKED | Execution Time 32.81s
[BRIDGE SUMMARY][DIAG] selected_total=0 eligible_total=0 selected_pre_group=0 policy=utility_topk zero_reason=wf_fold_pass_ratio_fail gate_p50=nan gate_p90=nan mu_p50=nan mu_p90=nan q10_p10=nan utility_p50=nan utility_p90=nan breakeven_floor=nan
[BRIDGE SUMMARY][WF_DIAG] wf_selected=0 wf_eligible=0 shadow= shadow_selected=0 shadow_realized=nan eu_p90=-69.525 downside_p90=228.049
```

---

## 백테스트 성과 (OOS)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Equal Size | -17.6% | 47.0% | -0.37 | 559,693 | 8 | 0.66 | N |
| Rule Promo NL | -14.1% | 16.0% | -0.88 | 913,690 | 402 | 0.64 | N |
| Rule Promo Oracle | 2.6% | 3.6% | 0.71 | 1,015,037 | 182 | 0.17 | N |
| Kelly (No ML) | -0.1% | 0.4% | 0.00 | 997,350 | 2149 | 0.66 | N |
| ML Gate | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| ML Gate+Edge | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| ML Full (Capped) | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| Cand. ML | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| Direct Edge | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| Variant Prior | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| Promo Filter | 0.0% | 0.0% | 0.00 | 1,000,148 | 1 | 0.00 | N |
| Val. Selection | 0.2% | 0.9% | 0.00 | 1,001,419 | 76 | 0.26 | N |
| Identity Feat | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| Market Feat | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |

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

## 다음 단계
1. threshold 완화보다 먼저 `q10` target/label 정의와 `edge_after_hurdle_bps`의 scale alignment를 재검토해야 합니다.
2. 현재 feature set이 payoff-tail을 설명하지 못하는지 확인하기 위해, `q10` 전용 diagnostics와 regime/holding-period conditional error 분석을 추가하는 편이 합리적입니다.
