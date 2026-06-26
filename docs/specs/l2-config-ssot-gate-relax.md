# 🎯 Objective
Config SSOT 도입으로 `from_mapping()` 하드코딩 이중기본값 버그를 해결하고, Regime 품질 평가를 L1/L2 실무에 맞게 견고화하며, Bucket routing을 유지하면서 Sharpe Uplift를 정직하게 개선할 수 있는 구조를 확보한다.

---

# Part 1: Config SSOT — `from_mapping()` 하드코딩 버그 수정

## 🔴 근본 문제

`Layer2AllocationConfig`에 **두 개의 기본값 선언 위치**가 존재하여, dataclass 필드 기본값과 `from_mapping()` fallback이 달라지는 버그 발생:

| Commit | Dataclass field | from_mapping() fallback | 불일치 |
|--------|----------------|------------------------|--------|
| `89fb1ec` (원본) | **0.20** | **0.20** | ✅ 일치 |
| `64ef760` (내 bug 도입) | **0.05** | **0.20** | ❌ 불일치 (버그) |

**원인**: `from_mapping()`이 Optuna champion params의 인입 경로이므로, params에 게이트 임계값이 없으면 **항상 from_mapping의 하드코딩 fallback이 사용**됨. dataclass 필드 기본값은 실질적으로 무시됨.

**현재 코드 (dataclasses.py:558)**:
```python
l2_min_sharpe_uplift=cls._as_float(params.get("l2_min_sharpe_uplift", 0.20), 0.20),
```

params에 `l2_min_sharpe_uplift` 키가 없으므로 항상 0.20이 사용됨.

## ✍️ 해결책: SSOT (Single Source of Truth)

### 전략
`from_mapping()`의 하드코딩 fallback을 제거하고, **dataclass 필드 기본값을 유일한 SSOT**로 만든다.

### 구현: `Layer2AllocationConfig._defaults` 클래스 속성

```python
@dataclass(slots=True, frozen=True)
class Layer2AllocationConfig:
    # ... 필드 정의 ...

    # 클래스 속성: dataclass 기본값 인스턴스
    @classmethod
    def _default_instance(cls) -> "Layer2AllocationConfig":
        """SSOT: from_mapping에서 fallback용으로 사용."""
        if not hasattr(cls, "_cached_default"):
            cls._cached_default: "Layer2AllocationConfig" = cls()
        return cls._cached_default
```

### from_mapping() 수정 (line 558)

```python
# 기존 (버그):
l2_min_sharpe_uplift=cls._as_float(params.get("l2_min_sharpe_uplift", 0.20), 0.20),

# 수정 (SSOT):
_d = cls._default_instance()
l2_min_sharpe_uplift=cls._as_float(
    params.get("l2_min_sharpe_uplift", _d.l2_min_sharpe_uplift),
    _d.l2_min_sharpe_uplift,
),
```

### 모든 `l2_min_*` 파라미터에 동일 패턴 적용

**Target**: `src/domain/futures/strategy/tiered_workflow/dataclasses.py`
**Method**: `Layer2AllocationConfig.from_mapping()` (line 433~620)

**수정 대상 라인** (from_mapping 내 모든 params.get("KEY", FALLBACK) → params.get("KEY", _d.KEY)):

| Line | Key | Fallback → SSOT |
|------|-----|-----------------|
| 546 | `l2_min_cagr` | `_d.l2_min_cagr` |
| 547 | `l2_min_mar` | `_d.l2_min_mar` |
| 549 | `l2_min_sortino` | `_d.l2_min_sortino` |
| 552 | `l2_min_sharpe_abs` | `_d.l2_min_sharpe_abs` |
| 553 | `l2_min_calmar` | `_d.l2_min_calmar` |
| 554 | `l2_max_mdd_abs` | `_d.l2_max_mdd_abs` |
| 557 | `l2_min_fold_pass_ratio` | `_d.l2_min_fold_pass_ratio` |
| 558 | `l2_min_sharpe_uplift` | `_d.l2_min_sharpe_uplift` |
| 559 | `l2_min_growth_uplift` | `_d.l2_min_growth_uplift` |
| 560 | `l2_min_psr` | `_d.l2_min_psr` |
| 561 | `l2_min_friction_pass` | `_d.l2_min_friction_pass` |

**또한 Optuna search space에도 동일한 `_as_float` 하드코딩 제거** (opt_config.py line ~200):
```python
# 기존: cls._as_float(params.get("l2_min_sharpe_uplift", 0.20), 0.20)
# 수정: cls._as_float(params.get("l2_min_sharpe_uplift", _d.l2_min_sharpe_uplift), _d.l2_min_sharpe_uplift)
```

### Result
```
[BEFORE] l2_min_sharpe_uplift = 0.20 (from_mapping fallback, 항상 이 값)
[AFTER]  l2_min_sharpe_uplift = 0.20 (dataclass field SSOT → from_mapping 참조)
```

**변경 전후 값이 동일** (원래 디자인 의도인 0.20을 유지). 단지 SSOT가 한 곳으로 통합되어 유지보수 안전성만 향상됨.

---

# Part 2: Regime Quality Evaluation 견고화

## 🔬 현재 상태

```
[REGIME] C2:dwell=10.0❌ C3:pval=0.595❌ C4:rho=0.60✅ C5:trans=26.5%✅
```

**C2(dwell)**: dwell=10.0 bars → "❌"로 표시되나 10은 6 이상이므로 통과. 표시 버그 가능성.
**C3(pval)**: pval=0.595 → "❌". 통계적으로 모든 regime 상태의 edge가 유의미하지 않음.
**C4(rho)**: 0.60 → "✅". Risk overlay multiplier가 양의 상관관계 보유.
**C5(trans)**: 26.5% → "✅". Transition 비율이 40% 이하.

## 🔴 C3(pval=0.595) 근본 원인 분석

Regime 품질 평가에서 pval=0.595 = "null hypothesis (regime states have equal edge) cannot be rejected". 즉, **6개 regime 상태 간 edge 차이가 통계적으로 유의미하지 않음**.

**이유 추정**:
1. 6개 regime 상태가 세밀하게 나뉘어 있으나, BTC 가격 데이터가 충분히 stationary하지 않아 regime 간 전이가 빈번하고 noise가 많음
2. L1 신호가 trend-following에 편중되어 regime-conditioned edge가 추정되기 어려움
3. Cal-eval 구간이 충분히 길지 않아 regime별 통계가 축적되기 전에 transition이 발생

## ✍️ 해결책: Regime State Compression

**개념**: 6개 regime 상태를 3개로 압축하여 노이즈를 줄이고 통계적 유의성을 확보.

```python
# Before: 6 states
BULL_QUIET = 0, BULL_VOLATILE = 1, BEAR_QUIET = 2, BEAR_VOLATILE = 3, TRANSITION = 4, CRASH = 5

# After: 3 compressed states
BULL = 0      # 0+1 합침 (상승)
BEAR = 1      # 2+3 합침 (하락)  
CRISIS = 2    # 4+5 합침 (전환+크래시)
```

**Rationale**:
- Quiet/volatile 구분은 risk overlay multiplier에서 이미 처리됨 → regime 구분에 불필요
- Bull/bear는 L1 trend 신호와 직관적 연관성 (trend_donchian이 bull/bear 구간에서 다르게 작동)
- Transition+crisis 합침은 "불확실한 시장" 통합 → trend 신호가 약한 구간

**Target**: `src/domain/futures/strategy/market_regime.py`
**Function**: `_continuous_regime_codes()` 또는 `compute_market_regime_context()`

**추가 함수**:
```python
def compress_regime_codes(code_1d: NDArray[np.int8]) -> NDArray[np.int8]:
    """6-state → 3-state compression for regime quality and bucket routing."""
    compressed = np.full_like(code_1d, 2, dtype=np.int8)  # default to TRANSITION/CRASH
    compressed[np.isin(code_1d, [0, 1])] = 0               # bull_quiet + bull_volatile → BULL
    compressed[np.isin(code_1d, [2, 3])] = 1               # bear_quiet + bear_volatile → BEAR
    # code 4 (transition) and 5 (crash) stay as CRISIS=2
    return compressed
```

**Config**: `Layer2AllocationConfig`에 `l2_regime_compression_enabled: bool = True` 추가.

**예상 효과**: 3-state는 6-state보다 regime별 bar 수가 2배 이상 → per-bucket edge 추정의 통계적 신뢰도 향상. Bucket routing에서 regime별 edge 차이가 유의미해질 가능성 농후.

## Bucket Routing 현상 유지

**Target**: `l2_routing_mode="bucket"` 유지 (pool 회귀 안 함). `l2_bucket_edge_floor_bps=0.0` 유지.

ADR 라인 83: `bucket+zero-floor(0.0) ≫ pool ≫ bucket+100bps`. 현재 floor=0.0으로 bucket mode가 최적 상태.

---

# Part 3: L* Floor 상향 (구조적 개선 유일한 유효 수단)

5차 DEBUG 실행에서 확인된 유일한 실효적 개선:
- L*=1.5 → CAGR 20%, RiskUtil 51%, MDD 15%
- L*=2.0 → CAGR 27%, RiskUtil 68%, MDD 20% (추정)

**Target**: `src/domain/futures/strategy/tiered_workflow/risk_deployment.py`

```python
_oos_floor = min(2.0, max(1.0, _oos_safe_l))  # before: min(1.5, ...)
```

**Caution**: exchange_leverage_cap=10.0이 있으므로, 2.0은 절대 cap 이내. MDD 20% < 30% cap.

---

# Part 4: Mu Amplification 비활성화 + L1 Signal Diversity

## CS Amp 중단

5차 DEBUG에서 power mode가 Sharpe Uplift를 **악화**시킴 (delta_sharpe +0.074 → -0.028). 진단: L2 rebalance 밀도(3%)로 인해 어떤 증폭도 무의미.

**Target**: `l2_cs_amp_enabled: bool = False`

코드 보존, 기능 OFF.

## L1 Signal Diversity (별도 SPEC 필요)

L1 DEBUG 결과: trend_donchian, trend_pullback_continuation 2종만 promotion → L2 신호 다양성 부족. 향후 L1 개선:

1. **Prequalify 기준 완화 검토**: 현재 30~40%의 신호가 prequalify에서 탈락. 특히 `flow_trend_continuation`(+87 bps edge도 탈락), `dual_momentum`(+57 bps도 탈락) 등 positive edge 신호가 과도하게 탈락.
2. **신호 카테고리 확장**: reversal/carry 신호의 prequalify 통과율 개선
3. **L2로 전달되는 `raw_mu`의 CS 분산 확대**: L2에서 Kelly ∝ 1/σ² 수렴 방지

---

# 🛠️ Surgical Implementation Plan

## Phase 1: Config SSOT (dataclasses.py, 15min)
- `_default_instance()` 클래스 메서드 추가
- `from_mapping()` 내 모든 `l2_min_*` 하드코딩 fallback → `_d.l2_min_*` 참조로 변경

## Phase 2: Regime Compression (market_regime.py, 20min)
- `compress_regime_codes()` 함수 추가
- `compute_market_regime_context()`에서 compressed code도 반환하도록 확장
- bucket routing에서 compressed code 사용

## Phase 3: CS Amp 비활성화 (dataclasses.py, 5min)
- `l2_cs_amp_enabled=False`

## Phase 4: L* Floor 상향 (risk_deployment.py, 5min)
- OOS floor max: `1.5 → 2.0`

## Phase 5: Verification
- LINT + mypy
- `LOG_LEVEL=DEBUG phase l2` 실행 → gate + regime quality 확인

---

# 🧪 Test Scenario Design

## Test 1: Config SSOT

**Scenario 1.1**: `from_mapping({})` → 모든 게이트 값 = dataclass 필드 기본값
- `l2_min_sharpe_uplift == 0.20` (SSOT 원본 값)

**Scenario 1.2**: `from_mapping({"l2_min_sharpe_uplift": 0.15})` → override 작동
- `l2_min_sharpe_uplift == 0.15` (params 우선)

## Test 2: Regime Compression

**Scenario 2.1**: Input `[0,1,2,3,4,5]` → `compress_regime_codes()` → `[0,0,1,1,2,2]`

**Scenario 2.2**: Empty input `[]` → `[]`

**Scenario 2.3**: Single state `[0]` → `[0]`

---

# 🛡️ Verification

```bash
uv run ruff check --fix src/domain/futures/strategy/tiered_workflow/dataclasses.py \
  src/domain/futures/strategy/market_regime.py \
  src/domain/futures/strategy/tiered_workflow/risk_deployment.py
uv run mypy [files...]
uv run pytest tests/unit/domain/futures/strategy/tiered_workflow/ tests/unit/domain/futures/portfolio/ -x -q --tb=short

LOG_LEVEL=DEBUG uv run python src/execution/opt_main_futures.py --phase l2 --timeframe 4h --trials 200 2>&1 | grep -E "\[(L2-CONFIG|L2-GATE|L2-DEPLOY|REGIME|L2-SHARPE-CMP|L2-SELECTION|L2-SCORE)" | tail -20
```

---

# 📊 Expected Impact Matrix

| 지표 | Before | After | 근거 |
|------|--------|-------|------|
| Config 불일치 | dataclass=0.05, runtime=0.20 | **SSOT=0.20** (일치) | 버그 수정 |
| Regime states | 6개 (C3:pval=0.595❌) | **3개** (통계적 유의성 ↑) | 압축으로 per-state stat 신뢰도 2× |
| Bucket routing | ON (floor=0.0, 효과 있음) | **ON 유지** | ADR 검증 완료 |
| CS Amp | ON (효과 없음) | **OFF** | 중단 결정 |
| L* floor | 1.5 | **2.0** | CAGR 20%→27% 추정 |
| uplift gate | -0.028 vs 0.20 ❌ | 구성 변경 없음 | L1 신호 다양성 개선 후 재평가 |
