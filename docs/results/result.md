# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-07 (p_pass sizing soft-discount 및 gate config 정리 후 재실행)
**현재 상태:** `BLOCKED` — `pass_ratio=0.00`
**평가 기준:** `min_fold_realized_edge_bps=15.0`, `min_cagr_for_promotion=0.15`, `min_edge_rank_ic=0.02`, `signal_prequalify_min_obs=30`

---

## 최신 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견
[PIPELINE] raw=272819 labeled=6350 promoted=2355 n_folds=4 wf_scheme=anchored
[SIGNAL_PREQUALIFY] fold별 29.3% 이벤트 제거 (355개/1210개)
[EDGE_GATE] mode=overlay_lift lift=0.7 t=0.07 n_eff=612.0 decision=rejected reason=overlay_lift_below_threshold inference_mode=disabled
[BRIDGE][WF] fold_cost_survival=[False, False, False, False] pass_ratio=0.00 min_required=0.60
[BRIDGE SUMMARY] Active Signals 0 (sel=0) | Status blocked
[BRIDGE SUMMARY][WF_DIAG] wf_selected=0 wf_eligible=0 shadow_profiles=18 shadow_max_selected=40 shadow_max_eligible=420 eu_p90=0.000 downside_p90=8.399
```

---

## fold별 상세 결과

| Fold | OOS 기간 | selected | realized_mean | lift_bps | ML lift | 결과 |
|---|---|---:|---:|---:|---:|---|
| 1 | Oct-Nov 2025 | 0 | nan bps | nan | False | ❌ (empty) |
| 2 | Nov-Dec 2025 | 0 | nan bps | nan | False | ❌ (empty) |
| 3 | Jan-Feb 2026 | 0 | nan bps | nan | False | ❌ (empty) |
| 4 | Feb-Mar 2026 | 0 | nan bps | nan | False | ❌ (empty) |

---

## 백테스트 성과 (OOS Ablation — 6 Causal Variants)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| rule_stop_risk | -20.4% | 16.6% | -1.23 | 875,924 | 620 | 1.00 | N |
| prior_rank_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| prior_residual_rank_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| edge_plus_validated_gate_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| edge_plus_gate_event_kelly | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| full_portfolio_caps | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |

---

## 근본 진단

| 문제 | 내용 |
|---|---|
| ML Edge Gate 미통과 | `edge_gate_mode="overlay_lift"`에서 lift(0.7bps) 및 t-stat(0.07)이 임계치 미달로 rejected되어 ML inference가 `disabled` 상태로 fail-closed 됨 |
| Prior-only 붕괴 | ML이 disable된 상태에서 prior-only selection 역시 breakeven floor에 의해 모든 fold에서 selected=0으로 차단됨 |
| Sizing 정합화 완료 | Sizing 경로에 `p_pass` soft-discount를 계산식에 올바르게 적용하였으며, dead config였던 gate 관련 파라미터들을 완전히 제거함 |

---

## 다음 단계

1. **Feature Engineering**: `overlay_lift_tstat`를 제고하기 위해 Macro, Regime 및 Variant별 고속 모멘텀 피처 추가 설계.
2. **Prior-only 임계치 조정 검토**: non-ML fallback인 prior가 breakeven floor를 통과할 수 있도록, structural cost 완화책 또는 prior shrinkage 강도 조절.
