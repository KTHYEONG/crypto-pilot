# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-07 (Fix1: prior_only fallback 활성화 / Fix2: edge_gate_mode=rank_ic 전환 후 재실행)
**현재 상태:** `BLOCKED` — `pass_ratio=0.00`
**평가 기준:** `min_fold_realized_edge_bps=15.0`, `min_cagr_for_promotion=0.15`, `min_edge_rank_ic=0.02`, `signal_prequalify_min_obs=30`

---

## 최신 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 (63 ready)
[PIPELINE] 4 folds / wf_scheme=anchored
[SIGNAL_PREQUALIFY] fold별 평균 ~29% 이벤트 제거
[EDGE_GATE] mode=rank_ic
  fold1: rejected (insufficient_obs, n=725)  → inference_mode=prior_only
  fold2: rejected (insufficient_obs, n=869)  → inference_mode=prior_only
  fold3: rejected (rank_ic=-0.005, t=-0.14)  → inference_mode=prior_only
  fold4: ACCEPTED (rank_ic=0.071, t=1.98)    → inference_mode=direct
[BRIDGE][WF] fold_cost_survival=[False, False, False, False] pass_ratio=0.00 min_required=0.60
[BRIDGE SUMMARY] Active Signals 0 (sel=0) | Status blocked
[BRIDGE SUMMARY][WF_DIAG] wf_selected=0 wf_eligible=0 shadow_profiles=18 shadow_max_selected=37 shadow_max_eligible=420 eu_p90=1.437 downside_p90=8.399
```

---

## fold별 상세 결과

| Fold | OOS 기간 | inference_mode | rank_ic | selected | eu_p90 | breakeven | 결과 |
|---|---|---|---:|---:|---:|---:|---|
| 1 | Oct-Nov 2025 | prior_only | n/a (insuf. obs) | 0 | 1.89 bps | 3.8 bps | ❌ (eu_p90 < breakeven) |
| 2 | Nov-Dec 2025 | prior_only | n/a (insuf. obs) | 0 | 1.50 bps | 3.8 bps | ❌ (eu_p90 < breakeven) |
| 3 | Jan-Feb 2026 | prior_only | -0.005 | 0 | 1.27 bps | 3.8 bps | ❌ (eu_p90 < breakeven) |
| 4 | Feb-Mar 2026 | **direct** | +0.071 | 0 | 1.08 bps | 3.8 bps | ❌ (eu_p90 < breakeven) |

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

## 근본 진단 (갱신)

| 항목 | 내용 |
|---|---|
| Fix 1 적용 확인 | `prior_only` fallback 경로가 실제로 활성화됨. fold 1-3에서 `disabled` 대신 `prior_only`로 동작. 코드 결함 수정 완료. |
| Fix 2 적용 확인 | `edge_gate_mode=rank_ic` 전환 완료. fold 4에서 `rank_ic=0.071, t=1.98`로 accept. 평가-제어 미스매치 제거됨. |
| 잔여 블로커 (C5) | prior mu 자체가 0.8~1.2 bps로 `breakeven_floor=3.8 bps`에 미달. eu_p90이 **모든 fold에서 breakeven 미만**이므로 eligible=0. "ML이 거래를 회피"가 아니라 "prior가 말하는 기대 수익 자체가 비용 이하"인 상태. |
| 룰 기반 엣지 부재 | `rule_stop_risk` = -20.4% CAGR. 순수 룰 자체가 비용 차감 후 손실. alpha 원천 문제. |

---

## 다음 단계

1. **Feature Engineering 우선 (C5 해결):** 현재 prior mu(0.8-1.2 bps) < breakeven(3.8 bps). 기대 수익을 올리는 유일한 경로는 피처 예측력 개선. 상위 신호 후보: `trend_pullback_continuation:tpc_50_200`(IS profit 74bps), `funding_zscore_carry:fzs_96`.
2. **breakeven_floor 재검토:** `min_net_floor_cost_fraction=0.5 × cost_floor_bps=7.5 → 3.8 bps`. eu_p90=1.4 bps 수준의 prior에 대해 floor가 과도하게 높을 수 있음. 단, 이는 완화가 아니라 cost 가정 검증이 먼저.
3. **fold 1-2 insufficient_obs 원인 분석:** `n=725/869`임에도 `insufficient_obs` → `min_n_eff=60` 체크이므로 실제 n_eff 계산값 확인 필요.
