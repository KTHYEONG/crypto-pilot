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

## [2026-06-10] In-Fold Validation IC Alignment & OOS Rank IC logging

- **Delta:** `_internal_validation_rank_ic`에서 `v_mu` 정적 평균 대신 `cell_val + v_offset` 동적 예측 적용. `candidate_workflow.py`에서 `oos_rank_ic`를 `ml_out` 진단에 저장하고 `bridge.py`에서 이를 가져와 `rank_ic` 리포트 작성.
- **Rationale:** 검증 단계와 OOS 예측 단계의 예측 수식 불일치(bug)로 인해 검증 Rank IC가 구조적 음수로 측정되어 shrinkage 및 conditioning 선택에 왜곡 발생. 이를 해결하고 실제 앙상블 OOS IC를 브릿지 요약 테이블에 표시하여 진단 편의성 증대.
- **Edge Cases:** 변이 관측치 미달 시 offset = 0.0으로 fallback 동작하여 하위 호환성 유지.

## [2026-06-10] calibrated_event_kelly 비중 분모 오류 수정 및 별칭 개편

- **Delta:** `select_candidate_events_for_portfolio`에 `q90_net_bps` 칼럼 추가. `build_candidate_target_weights`에서 켈리 분모 계산 시 $q_{10}$ 극단값 제곱을 정규화 변동성 $\sigma_R = \frac{q_{90} - q_{10}}{2.563}$ 제곱으로 대체. Ablation Row 이름 간소화.
- **Rationale:** 분모에 극단적 꼬리 위험값($q_{10}$)의 제곱을 다이렉트로 대입해 분모가 100~400배 왜곡되고 가중치가 0에 수렴하는 금융공학적 버그 해결. 변동성 스케일을 정상 복구하여 켈리 비중의 실제 영향도 검증을 가능하게 함.
- **Edge Cases:** `q90` 정보 누락 방지 및 0 분모 클리핑 가드 처리.

## [2026-06-10] 동적 포트폴리오 Cap 및 이중 볼 스케일링 방지 가드 도입

- **Delta:** `build_candidate_target_weights`에 `regime_code_1d` 배선 및 동적 `PortfolioCaps` 생성. `double_scaling_guard` 하이퍼파라미터화 및 `project_all_caps` 연산부 `target_ann_vol` 0/None 분기 핫픽스 적용.
- **Rationale:** 고정 Cap 제약과 이중 볼 스케일링으로 인한 비중 고사 문제를 해결하고, 켈리 비중의 성과(CAGR +2.6%)를 보존하면서 시장 국면에 유연하게 대응하는 자산 배분 구조 확립.
- **Edge Cases:** 볼 타겟팅이 생략되는 가드 동작 시에도 국면 multipliers 제약(crash=0.1 net_cap 등)을 정상 적용하여 위험 통제력 유지.
