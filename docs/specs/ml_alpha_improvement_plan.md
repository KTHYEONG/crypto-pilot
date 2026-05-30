# Blueprint: ML Alpha 성능 개선 및 무결성 검증 실무 구현서

## 1. 개요 및 설계 방향
본 문서는 `docs/specs/ml_system_integrity_and_evaluation.md` 사양에 기반하여, 선물 ML Alpha 모델의 무결성을 확보하고 OOS 성과를 획기적으로 견고화하기 위한 실무 구현용 Blueprint이다. 

핵심 목적은 다음과 같다:
1. **데이터/피처 무결성 게이트(Integrity Gate)의 엄격화**: 데이터 천장 판정 및 누수 없는 피처 동적 필터링.
2. **평가 지표의 정밀화 (Evaluation Hardening)**: raw IC 대신 시장 베타와 크기 효과를 통제한 `beta-residualized rank IC`와 고상관 시장 보정 `Effective Breadth` 기반의 `Breakeven IC`를 도입하여 시장을 이기는 엣지만을 생존시킴.
3. **Numba JIT 연산 가속**: 상관관계 병목 연산을 100배 이상 가속하여 Optuna 최적화 생산성 극대화.

---

## 2. 대상 파일 및 변경 사항 요약

- `src/domain/futures/strategy/integrity.py`
  - **역할**: 데이터 무결성 검증, Alias 피처 정적 제거, PSI(Population Stability Index) 기반 Drift 검증 및 Leak-free 피처 동적 선택.
- `src/domain/futures/strategy/diagnostics.py`
  - **역할**: Numba JIT 데코레이터가 적용된 고속 rolling_ic 연산부 구현 및 정합성 보장.
- `src/domain/futures/strategy/alpha_evaluation.py`
  - **역할**: Beta-residualized IC 연산 엔진, Effective Breadth ($N_{eff}$) 보정 Breakeven IC 산출 및 다중 Regime 게이팅.
- `src/domain/futures/strategy/ml_builder.py`
  - **역할**: 위 3개 모듈의 기능을 통합하여 학습 및 최적화 루프에 Pipeline 형태로 주입.

---

## 3. 세부 설계 및 계약 (Contracts)

### 3.1 `integrity.py`
```python
def select_features(
    features: FeaturePanel,
    *,
    train_slice: slice,
    oos_slice: slice,
    config: FeatureIntegrityConfig,
) -> tuple[str, ...]:
    """
    Train fold 기준으로 아래 피처를 동적으로 필터링하여 Leak-free 피처 목록을 반환한다.
    1. Constant (std < epsilon)
    2. Drift (PSI > drift_threshold)
    3. Redundant (|Pearson corr| > collinearity_threshold)
    """
```

### 3.2 `diagnostics.py`
```python
@njit(fastmath=True, cache=True)
def _fast_rank1d(x: np.ndarray) -> np.ndarray:
    """1차원 배열의 Numba 기반 초고속 rank 산출 함수"""
    
@njit(fastmath=True, cache=True)
def _fast_pearson_core(x: np.ndarray, y: np.ndarray) -> float:
    """Numba 기반 고속 Pearson 피어슨 상관계수 산출"""
```

### 3.3 `alpha_evaluation.py`
```python
def effective_breadth_corr(corr_matrix: NDArray[np.float64]) -> float:
    """
    자산 간 평균 상관관계 rho를 도출하고, 이에 기반한 유효 거래 횟수 N_eff를 계산한다.
    Formula: N_eff = N / (1 + (N - 1) * rho)
    """

def diagnose_alpha_ic_decomposition(
    alpha: NDArray[np.float64],
    target: NDArray[np.float64],
    beta_factor: NDArray[np.float64],
) -> dict[str, float]:
    """
    raw rank IC와 beta-residualized rank IC를 비교 분석하여 팩터 오염 수준을 수치화한다.
    """
```

---

## 4. 수술 계획 (Surgical Plan)

### Step 1: `src/domain/futures/strategy/integrity.py` 수정
- Constant 피처 판단 시 분산 한계값($\epsilon$)을 설정하고, PSI 연산 결과를 명확히 필터링하여 `FeatureIntegrityReport`를 완성한다.
- 정의상 상관관계가 $\pm 1.0$인 Alias 피처를 정적으로 소거하는 화이트리스트/블랙리스트 체계를 구성한다.

### Step 2: `src/domain/futures/strategy/diagnostics.py`에 Numba 데코레이터 적용
- `rolling_ic`에 Numba `njit` 최적화 구조를 도입하여 루프 오버헤드와 정렬 연산을 C-level 속도로 연산하게 한다.

### Step 3: `src/domain/futures/strategy/alpha_evaluation.py` 내 평가지표 Hardening
- `beta_residualized_rank_ic` 함수를 설계하여 원본 신호에서 베타 팩터 영향력을 1차 잔차화(Regression Residuals) 한 뒤 랭크 상관계수를 도출한다.
- `N_eff_breakeven` 통과 여부를 검증하고, Fold별로 IC 부호 일관성(Sign Consistency) 및 Bear Regime IC 값을 추출하여 최종 게이팅 조건에 결합한다.

---

## 5. 검증 계획 (Verification)
- **속도 검증**: `uv run python -c "import src.domain.futures.strategy.diagnostics as d; ..."` 스피어먼 연산 속도 실측.
- **수치 정합성 검증**: Scipy 대비 오차범위 $10^{-7}$ 만족 여부 테스트.
- **통합 검증**: `uv run pytest tests/unit/domain/futures/optimization/` 모든 단위 테스트가 통과하는지 확인.
