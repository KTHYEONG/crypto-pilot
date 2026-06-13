## 2026-06-13 Layer1 Regime Decoupling & IC Mode Switch
- **Delta:** `l1_qualify_by_regime=False` (default) — qualification key 에서 `activation_context(entry_regime)` 제거, 전략단위 표본 풀링. `l1_opp_ic_mode="time_series"` (default) — per-symbol 이벤트 시계열 IC로 전환. `l1_pair_min_folds 3→2`, `l1_min_cross_section 3→2`.
- **Rationale:** regime을 alpha 게이트 차원에 포함 시 표본 파편화(RC-1) + fold 구조 불가능(RC-2) + cross-section 동시성 불가(RC-3)로 OUTER FOLDS 전멸(0/4). regime 조건화는 Fold IC 악화(-0.102)로 반증됨(see `project_regime_alpha_conditioning_disproved`). regime → risk overlay 강등, 비정상성 방어는 `positive_fold_ratio` 유지.
- **Edge Cases:** `l1_qualify_by_regime=True` 로 기존 regime-cell 모드 복원 가능(하위호환). time_series IC는 단일심볼 과적합 위험 → `positive_fold_ratio` + probe gate 로 방어.

# ADR: Layer1 Signal Readiness Workflow Refactor

- **Delta**: Layer1 검증 파이프라인을 Nested Anchored Walk-Forward 구조로 개편하고, target contract를 net에서 gross로 단일화.
- **Rationale**: Inner selection과 Outer evaluation을 격리하여 selection bias를 완전 제거하고, 비용(fee, slippage 등)을 Layer2로 이관하여 도메인 간의 책임 한계를 확실히 분리하기 위함.
