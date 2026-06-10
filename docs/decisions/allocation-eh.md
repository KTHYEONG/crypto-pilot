---
title: Allocation Ensemble — Decision Records
domain: futures/strategy
type: adr
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/candidate_ensemble.py
  - src/domain/futures/strategy/config.py
last_verified: 2026-06-10
---

## [2026-06-10] Regime-Cell Admission: Bayesian Posterior Probability 대체

- **Delta:** `min_obs=60 + min_tstat=1.0 + k0=30` 3개 하드코딩 임계값 → 단일 기준 `P(μ > δ | data) ≥ 0.70` (N-N conjugate posterior + Newey-West Ω_nw). `min_regime_cell_oos_obs: 60→10` (NW stability floor만), `regime_cell_admission_enabled: False→True`. `min_obs` OR-path 추가 + `_build_recommendations` cell_admitted bypass.
- **Rationale:** `min_obs=60`은 implicit t≥2.0 역산이나 `min_tstat=1.0`과 내적 모순. 70bps/39obs(t=14.6) REJECT·8bps/80obs(t=2.1) ADMIT이라는 통계적 역전 발생. Bayesian 단일 기준은 effect-size와 uncertainty를 동시 반영 → 강엣지·희소 신호 구출(RECOMMENDED 7→15), Fold4 최초 PASS.
- **Edge Cases:** τ² < 2 cell → fallback `admission_tau_prior_bps²=225`. bcr_48은 IS sparsity(35 IS events ÷ 6 regimes ≈ 6/cell < floor=10) 구조 문제 — 알고리즘 한계 아닌 데이터 가용성 한계.

## [2026-06-10] Ensemble Conditioning Default → "auto" + Fail-SAFE

- **Delta:** `ensemble_conditioning` 기본값 `"archetype_regime"` → `"auto"`. `oos_proof_events=None`일 때 `archetype_regime` 유지(fail-OPEN) → `archetype_only` 강등(fail-SAFE, `path="no_oos_evidence_failsafe"`). `regime_oos_stability_rho` 진단 필드(비-게이팅) 추가.
- **Rationale:** 기본값이 `archetype_regime`으로 고정되어 IS 내부검증용 `auto` 분기가 한 번도 발동 안 됨. proof 없는 fold에서 OOS 증거 없이 복잡한 조건화 축을 선택하는 fail-OPEN 구조가 전 fold IC 음수(-0.016~-0.120)의 구조적 기여 요인. 임계값 추가(curve-fitting)가 아닌 기존 메커니즘의 기본값·안전경로 수정.
- **Edge Cases:** `"archetype_regime"` 명시 시 기존 lift_proof 2차 게이트 유지(하위호환). `auto`가 `archetype_only`를 선택하면 fail-SAFE 미발동. 수치 불변(WF IC 동일)은 예상된 결과 — 근본 블로커는 admission OFF로 인한 풀 다양성 부재.
