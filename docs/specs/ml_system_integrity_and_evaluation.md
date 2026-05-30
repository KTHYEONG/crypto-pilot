---
title: ML 시스템 무결성 및 평가지표 견고화 (통합 SSOT)
domain: futures-alpha
type: spec
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/integrity.py
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/strategy/diagnostics.py
  - src/domain/futures/strategy/features.py
  - src/domain/futures/strategy/alpha_evaluation.py
last_verified: 2026-05-30
---

# ML 시스템 무결성 및 평가지표 견고화 통합 사양서

> **본 문서는 다음 4개 문서를 통합·대체하는 단일 SSOT다:**
> 1. `ml_data_feature_integrity.md` (데이터/피처 무결성 게이트)
> 2. `ml_alpha_evaluation_hardening.md` (평가지표 타당성 감사 및 개선)
> 3. `ml_feature_oos_coverage_refinement.md` (OOS 커버리지 정정 및 피처 효율화)
> 4. `ml_integrity_vectorized_optimization.md` (벡터화 및 JIT 최적화)

---

## 0. 개요 (TL;DR)

1.  **데이터 무결성 보장**: 학습 전 가격 오염(Zero-price), OHLC 정합성, OOS NaN 근원 분해를 통해 "데이터 천장" 여부를 판정한다.
2.  **피처 효율화 및 선택**: 중복(Alias) 제거, Constant/Drift 피처 동적 제거, 누수 없는 피처 선택(Construction) 로직을 적용한다.
3.  **평가 견고화**: Beta-residualized IC, Effective Breadth($N_{eff}$) 보정 Breakeven, Regime-based 게이트를 통해 "거래 가능한 엣지"를 판정한다.
4.  **기술적 최적화**: Scipy 의존성을 제거하고 Numba JIT 및 벡터화된 계산 엔진을 도입하여 진단 속도를 100배 이상 개선한다.

---

## 1. 데이터 및 피처 무결성 검증 (Integrity Gate)

### 1.1 데이터 무결성 (D-INT, D-MASK)
-   **Zero-price 오염**: 활성 마스크 내 가격이 0 이하인 경우 `HARD FAIL`.
-   **OOS NaN 근원 분해 (D-MASK)**: OOS target NaN을 4가지 범주(`universe_inactive`, `price_missing`, `warmup`, `kill`)로 분해하여 데이터 결함 여부를 판정한다.
-   **조건부 커버리지**: `coverage_within_eligible` ($|finite(target) \cap eligible| / |eligible|$)이 0.9 이상이어야 OOS가 건강한 것으로 간주한다.

### 1.2 피처 무결성 및 선택 (F-INT, F-EFF)
-   **피처 Alias 제거**: 정의상 상관관계가 $\pm 1.0$인 피처(예: `cs_rank_dollar_volume`, `carry_prior_6`, `xs_reversal_prior_6`)를 정적 제거한다.
-   **동적 선택 (Leak-free)**: Train fold 한정으로 Constant(std < $\epsilon$), Drift(PSI > 0.25), Redundant(|corr| > 0.95) 피처를 제거한다.
-   **효능 진단**: Beta-residualized rank IC와 부호 안정성을 기준으로 피처 효능을 평가한다.

---

## 2. ML Alpha 평가지표 견고화 (Evaluation Hardening)

### 2.1 구조적 결함 보정
1.  **C1: Beta/Size 팩터 제거**: Raw IC가 아닌 `beta-residualized rank IC`를 주요 게이팅 지표로 사용한다.
2.  **C2: Effective Breadth 보정**: 코인 간 고상관(BTC 팩터)을 반영하여 $N_{eff} = N / (1+(N-1)\rho)$ 기반의 Breakeven IC를 산출한다.
3.  **C3: 신호 일관성**: Clipped alpha(`max(ev,0)`)가 아닌 Pre-clip dense score로 랭킹 스킬을 판정한다.

### 2.2 단계별 개선안 (Phase 0-3)
-   **Phase 0 (결정적 측정)**: Beta-resid IC + $N_{eff}$ Breakeven 측정으로 KILL 확정/번복.
-   **Phase 1 (지표 견고화)**: `evaluate_alpha` PASS 판정 로직 재구성 (Resid IC, $N_{eff}$ BE, Regime 게이트).
-   **Phase 2 (신호 개선)**: Beta-neutral 잔존 alpha 피처 발굴.
-   **Phase 3 (거래성 보존)**: Soft-hurdle 또는 Rank-based sizing으로 스킬 파괴 방지.

---

## 3. 기술적 최적화 (Vectorized & JIT Engine)

### 3.1 Numba JIT 가속
-   Scipy `spearmanr`의 오버헤드를 제거하기 위해 Numba `njit` 기반의 커스텀 엔진 도입.
-   **주요 함수**: `_fast_rank1d`, `_fast_pearson_core`, `_numba_rolling_ic_spearman`.
-   **성능 목표**: Scipy 대비 100~250배 속도 향상 ($O(F \times T \times N \log N)$).

### 3.2 수치 정합성
-   JIT 구현체는 Scipy 결과와 부동 소수점 오차 범위($10^{-7}$) 내에서 일치해야 한다.

---

## 4. 의사결정 및 PASS 게이트

### 4.1 데이터 무결성 판정
-   `price_missing > 5%` → 데이터 결함 → KILL 번복 및 수집 수정.
-   `coverage_within_eligible ≈ 1.0` + `IC ≤ Breakeven` → KILL 확정 (신호 결함).

### 4.2 Alpha PASS 기준
-   **Primary**: `beta_resid_ic > N_eff_breakeven`
-   **Stability**: Fold별 부호 안정성 및 Regime(Bear) IC $\ge$ 0.
-   **Decision**: `_oos_diag`가 "sufficient_cofinite_check_ic"인 상태에서 IC Gap이 양수여야 함.

---

## 5. 구현 계약 (Contracts)

### `integrity.py`
```python
def verify_data_integrity(...) -> DataIntegrityReport
def verify_feature_integrity(...) -> FeatureIntegrityReport
def select_features(...) -> tuple[str, ...]
```

### `diagnostics.py`
```python
def rolling_ic(..., method="spearman") -> np.ndarray  # Numba-accelerated
```

### `alpha_evaluation.py`
```python
def effective_breadth_corr(...) -> float  # N_eff estimation
def diagnose_alpha_ic_decomposition(...) -> dict[str, float]
```

---

## 6. 완료 기준 및 검증

1.  **단위 테스트**: `test_integrity.py`, `test_alpha_evaluation.py`, `test_diagnostics_speed` 통과.
2.  **실측 검증**: `opt_main_futures.py` 실행 시 로그에 `[DATA-INT]`, `[FEAT-INT]`, `[RESID-IC]`, `[BE-EFF]` 노출 확인.
3.  **문서 정리**: 기존 4개 개별 사양서 삭제 및 본 통합 사양서로 참조 단일화.
