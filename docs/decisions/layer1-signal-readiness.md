## 2026-06-13 Layer1 Pipeline D2 Override Position Fix
- **Delta:** `run_l1_nested_swf` 파이프라인 상단에서 `l1_cfg = replace(cfg, ensemble_conditioning="archetype_only", ensemble_score_calibration_enabled=False)`를 치환하여 inner/outer `_fit_and_predict_single_fold` 호출 시 매개변수로 전달하고, MagicMock 호환성을 위한 `is_dataclass` 검사 분기 로직 추가.
- **Rationale:** 기존 D2 override 코드가 gate pass 통과 후(`fit_layer1_inference_artifact`) 시점에만 위치해 있어, Layer 1이 차단(BLOCKED) 상태일 때 dead override가 되던 치명적 위치 오류를 해소함. 이를 통해 게이트 검증 시에도 regime $\mu$ 조건화가 배제된 Pooled(Arch-Only) 모드가 안정적으로 작동하게 함.

## 2026-06-13 Layer1 Regime Decoupling & IC Mode Switch
- **Delta:** `l1_qualify_by_regime=False` (default) — qualification key 에서 `activation_context(entry_regime)` 제거, 전략단위 표본 풀링. `l1_opp_ic_mode="time_series"` (default) — per-symbol 이벤트 시계열 IC로 전환. `l1_pair_min_folds 3→2`, `l1_min_cross_section 3→2`.
- **Rationale:** regime을 alpha 게이트 차원에 포함 시 표본 파편화(RC-1) + fold 구조 불가능(RC-2) + cross-section 동시성 불가(RC-3)로 OUTER FOLDS 전멸(0/4). regime 조건화는 Fold IC 악화(-0.102)로 반증됨(see `project_regime_alpha_conditioning_disproved`). regime → risk overlay 강등, 비정상성 방어는 `positive_fold_ratio` 유지.
- **Edge Cases:** `l1_qualify_by_regime=True` 로 기존 regime-cell 모드 복원 가능(하위호환). time_series IC는 단일심볼 과적합 위험 → `positive_fold_ratio` + probe gate 로 방어.

# ADR: Layer1 Signal Readiness Workflow Refactor

- **Delta**: Layer1 검증 파이프라인을 Nested Anchored Walk-Forward 구조로 개편하고, target contract를 net에서 gross로 단일화.
- **Rationale**: Inner selection과 Outer evaluation을 격리하여 selection bias를 완전 제거하고, 비용(fee, slippage 등)을 Layer2로 이관하여 도메인 간의 책임 한계를 확실히 분리하기 위함.
