# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-06 (4-Layer Eval Criteria Hardening 적용 후)
**현재 상태:** `BLOCKED` — `pass_ratio=0.00`
**평가 기준:** `min_fold_realized_edge_bps=15.0`, `min_cagr_for_promotion=0.15`, `min_edge_rank_ic=0.02`, `signal_prequalify_min_obs=30`

---

## 최신 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견
[PIPELINE] raw=272819 labeled=6350 fit=9869 cal=9165 oos=2355 n_folds=4 wf_scheme=anchored
[SIGNAL_PREQUALIFY] fold별 26~29% 이벤트 제거 (382~608개/fold)
[EDGE_GATE] fold1: rank_ic=-0.018 → rejected (prior-only)
[EDGE_GATE] fold2: rank_ic=-0.007 → rejected (prior-only)
[EDGE_GATE] fold3: rank_ic=-0.042 → rejected (prior-only)
[EDGE_GATE] fold4: rank_ic=-0.0003 → rejected (prior-only)
[BRIDGE][WF] fold_cost_survival=[False, False, False, False] pass_ratio=0.00 min_required=0.60
[BRIDGE SUMMARY] Active Signals 0 (sel=0) | Status blocked
[BRIDGE SUMMARY][WF_DIAG] wf_selected=22 wf_eligible=33 eu_p90=0.769 downside_p90=14.163
  shadow=expected_edge_direct:off shadow_selected=31 shadow_realized=114.316
```

---

## fold별 상세 결과

| Fold | OOS 기간 | selected | realized_mean | lift_bps | ML lift | 결과 |
|---|---|---:|---:|---:|---:|---|
| 1 | Oct-Nov 2025 | 0 | nan | nan | False | ❌ |
| 2 | Nov-Dec 2025 | 17 | -1.3 bps | -30.9 | False | ❌ |
| 3 | Jan-Feb 2026 | 3 | -571.1 bps | -567.3 | False | ❌ |
| 4 | Feb-Mar 2026 | 2 | -366.0 bps | -368.9 | False | ❌ |

---

## 백테스트 성과 (OOS Ablation — 6 Causal Variants)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| rule_stop_risk | -29.7% | 23.9% | -1.24 | 811,759 | 637 | 1.00 | N |
| prior_rank_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| prior_residual_rank_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| edge_plus_validated_gate_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| edge_plus_gate_event_kelly | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| full_portfolio_caps | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |

---

## 근본 진단

| 문제 | 내용 |
|---|---|
| Residual model IC < 0.02 | 현재 feature set(identity/market/symbol)으로 cross-sectional 순위 예측 불가. 4개 fold 전부 IC < 0.02 → prior-only 강제 전환 |
| Prior-only 붕괴 | eu_p90=0.769bps (이전 PROMOTED 시 53.8bps). prior 값만으로는 selection 구동 불가 (breakeven 3.8bps 미달) |
| Shadow signal 존재 | shadow_realized=114bps — signal 자체는 살아있으나 ML feature가 이를 포착하지 못함 |
| 이전 PROMOTED 허위 양성 | IC=-0.62 역방향 모델이 gate 없이 배포됐던 것. 새 기준이 이를 차단 |

---

## 다음 단계

1. **[최우선] Feature Engineering**: 진입 시점 momentum strength, variant별 rolling IC, regime-specific feature 등 cross-sectional 예측력 있는 feature 추가 → rank_ic ≥ 0.02 달성 목표
2. **Prior-only fallback 경로**: edge gate 미통과 시 variant prior ranking만으로 top-k 선택하는 임시 경로. shadow_realized=114bps 활용 가능성 존재
3. **Shadow 분석**: fold1 shadow_selected=31 이벤트 프로파일 → feature 개선 방향 도출
