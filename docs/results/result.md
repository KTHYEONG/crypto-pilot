# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-06 (integrity fix 반영 후 재실행)
**현재 상태:** `BLOCKED` — `pass_ratio=0.25`
**평가 기준:** `min_fold_realized_edge_bps=15.0`, `min_cagr_for_promotion=0.15`, `min_edge_rank_ic=0.02`, `signal_prequalify_min_obs=30`

---

## 최신 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견
[PIPELINE] raw=272819 labeled=6350 promoted=2355 n_folds=4 wf_scheme=anchored
[SIGNAL_PREQUALIFY] fold별 26~29% 이벤트 제거 (382~608개/fold)
[EDGE_GATE] fold1: rank_ic=-0.0192 → rejected inference_mode=prior_only
[EDGE_GATE] fold2: rank_ic=-0.0064 → rejected inference_mode=prior_only
[EDGE_GATE] fold3: rank_ic=-0.0294 → rejected inference_mode=prior_only
[EDGE_GATE] fold4: rank_ic=+0.0005 → rejected inference_mode=prior_only
[BRIDGE][WF] fold_cost_survival=[True, False, False, False] pass_ratio=0.25 min_required=0.60
[BRIDGE SUMMARY] Active Signals 0 (sel=0) | Status blocked
[BRIDGE SUMMARY][WF_DIAG] wf_selected=32 wf_eligible=46 eu_p90=2.436 downside_p90=14.163
  shadow_profiles=20 shadow_max_selected=70 shadow_max_eligible=751
```

---

## fold별 상세 결과

| Fold | OOS 기간 | selected | realized_mean | lift_bps | ML lift | 결과 |
|---|---|---:|---:|---:|---:|---|
| 1 | Oct-Nov 2025 | 20 | 265.4 bps | 238.6 | True | ✅ |
| 2 | Nov-Dec 2025 | 5 | 232.4 bps | 202.8 | True | ❌ (`selected_count < 20`) |
| 3 | Jan-Feb 2026 | 2 | -458.0 bps | -454.2 | False | ❌ |
| 4 | Feb-Mar 2026 | 5 | -524.6 bps | -527.5 | False | ❌ |

---

## 백테스트 성과 (OOS Ablation — 6 Causal Variants)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| rule_stop_risk | -29.7% | 23.9% | -1.24 | 811,759 | 637 | 1.00 | N |
| prior_rank_stop_risk | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | N |
| prior_residual_rank_stop_risk | 0.2% | 2.0% | 0.08 | 1,000,890 | 54 | 0.15 | N |
| edge_plus_validated_gate_stop_risk | 0.2% | 2.0% | 0.08 | 1,000,890 | 54 | 0.15 | N |
| edge_plus_gate_event_kelly | -1.4% | 4.9% | -0.28 | 991,936 | 56 | 0.15 | N |
| full_portfolio_caps | 0.2% | 1.0% | 0.18 | 1,001,147 | 55 | 0.15 | N |

---

## 근본 진단

| 문제 | 내용 |
|---|---|
| 기존 해석 무효화 | reject된 residual 모델이 실제로는 계속 사용되던 버그를 수정함. 이제 `prior_only` fallback이 정확히 적용됨 |
| prior-only는 완전 붕괴가 아님 | fold1은 `+265bps`, fold2는 `+232bps`로 baseline 대비 양의 lift를 보임 |
| 병목 1: 샘플 수 | fold2는 수익성은 충분하지만 `selected_count=5 < 20`으로 생존 실패 |
| 병목 2: 후반 fold 품질 | fold3~4는 prior-only selection이 음수 edge로 전환됨 |
| shadow 해석 제한 | shadow는 OOS profile search 결과이므로 진단 전용. deployable edge 증거로 사용 불가 |

---

## 다음 단계

1. **Feature Engineering**: residual model의 `rank_ic >= 0.02` 달성 시도. 특히 entry momentum strength, variant rolling IC, regime-specific feature 우선.
2. **Prior-only 운영 재검토**: fold1~2 양의 lift는 확인됐지만 fold3~4가 음수이므로, threshold 완화 대신 추가 causal filter가 필요.
3. **Fold sample 문제 분해**: `selected_count`를 늘리기 위한 top-k / breakeven floor / variant concentration 조합을 별도 실험으로 분리.
