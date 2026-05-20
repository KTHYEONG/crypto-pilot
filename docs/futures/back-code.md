# 백테스트 v3.0 코드 변환 사양서

**작성일**: 2026-05-20  
**기준 아키텍처**: `docs/futures/backtest-logic.md` v3.0  
**전략**: Test-First. 테스트를 먼저 작성하고 통과시키며 구현을 확정한다.

---

## 0. 현황 Gap 분석

| 모듈 | 현재 상태 | v3.0 요구 | 조치 |
|---|---|---|---|
| `execution_sim.py` / intrabar | 청산 기준 `path_low_2d` | `mark_price_1m` HARD | **파라미터 추가 + 로직 수정** |
| `evaluator.py` / score | `mean - λ_mad*semi_dev - ψ*MDD` (2항) | 6항 고정 λ | **신규 함수 `compute_v3_score`** |
| `evaluator.py` / DSR | `n_trials_eff = min(n, 50)` hardcap | entropy effective rank | **신규 함수 `calc_n_trials_eff_entropy`** |
| `walk_forward.py` | K=10, gates(0.70/0.95/1.00), DSR 없음 | K=8, gates(0.55/0.85/1.015), DSR≥0.60 | **상수 교체 + 신규 gate** |
| `walk_forward.py` | atomic block 없음 | non-overlap 6M blocks | **신규 `atomic_blocks.py`** |
| `portfolio_constructor.py` | Kelly cap만 존재, fractional 없음 | 0.25x fractional Kelly | **`_kelly_raw` 호출부 수정** |
| `portfolio_constructor.py` | gross/per-symbol cap만 | 5 caps + minNotional 양자화 | **cap 투영 확장** |
| `backtest_preparation.py` | mark_price 정렬 없음 | `mark_price_1m` 배열 추가 | **준비 로직 확장** |
| `validation/unified_gates.py` | 구 gate 집합 | 신규 8-gate 체계 | **상수 업데이트** |

---

## 1. 구현 순서 (테스트 기준)

```
Phase 1  execution_sim — mark_price 청산, 회계 무결성
Phase 2  evaluator     — score 공식, DSR entropy, ergodicity
Phase 3  hard_gates    — 8 gate 체계
Phase 4  atomic_blocks — non-overlap 6M pass_ratio
Phase 5  portfolio     — fractional Kelly, 5 caps, quantization
Phase 6  boundary      — purge_bars seam 등록 계약
```

---

## 2. 테스트 파일 구조

```
tests/unit/domain/futures/
├── backtest/
│   ├── test_execution_sim_math.py          # 기존 — 유지
│   ├── test_mark_price_liquidation.py      # NEW — Phase 1
│   ├── test_funding_timing.py              # NEW — Phase 1
│   └── test_conservation_identity.py      # NEW — Phase 1
├── optimization/
│   ├── test_score_v3.py                   # NEW — Phase 2
│   ├── test_dsr_entropy.py                # NEW — Phase 2
│   └── test_ergodicity_gate.py            # NEW — Phase 2
├── validation/
│   ├── test_hard_gates_v3.py              # NEW — Phase 3
│   ├── test_atomic_blocks.py              # NEW — Phase 4
│   └── test_boundary_contract.py          # NEW — Phase 6
└── portfolio/
    ├── test_fractional_kelly.py           # NEW — Phase 5
    ├── test_caps_projection.py            # NEW — Phase 5
    └── test_quantization.py              # NEW — Phase 5
```

---

## 3. Phase 1: execution_sim — mark_price 청산

### 3.1 수정 대상
`src/domain/futures/portfolio/execution_sim.py`  
함수: `backtest_target_weights_intrabar_numba`

**현재 청산 로직 (line ~600 근처)**:
```python
# 현재: exec_low_1m 기준
if path_low_2d[j, s] <= liq_p[s]:   # Long 청산
```

**변경 후**:
```python
# 신규 파라미터 추가
mark_price_1m: np.ndarray | None = None   # shape [B_1m, N]

# 청산 판정 기준 교체
mark = mark_price_1m[j, s] if mark_price_1m is not None else path_low_2d[j, s]
if mark <= liq_p[s]:   # Long
if mark >= liq_p[s]:   # Short
```

**펀딩 정산 타이밍 수정**:
```python
# 현재: 바 종료 시 적용 여부 불분명
# 변경: 바 시작 시점(open 처리 전) 보유 포지션에만 적용
if funding_event_mask_1m[j, s] == 1 and in_pos[s]:  # 바 시작 시 검사
    fund_fee_stored[s] += amount[s] * path_open_2d[j, s] * funding_rate_1m[j, s] * side
```

### 3.2 테스트 파일: `test_mark_price_liquidation.py`

```python
# 테스트 1: mark_price ≤ liq_price 조건에서만 청산
# - exec_low가 liq_price 이하이나 mark_price는 이상인 경우 → 청산 미발생
# - mark_price가 liq_price 이하인 경우 → 청산 발생

# 테스트 2: mark_price vs exec_low 차이 측정
# - 동일 시나리오, mark_price > exec_low 설정
# - mark_price 기준: 청산 미발생 → equity 더 높음
# - exec_low 기준: 청산 발생 → equity 낮음
# assert equity_mark > equity_exec_low

# 테스트 3: mark_price_1m=None fallback
# - None 전달 시 exec_low 대리 사용, 정상 작동 확인
```

### 3.3 테스트 파일: `test_funding_timing.py`

```python
# 테스트 1: 8h 이벤트 바에서 포지션 보유 시 펀딩 적용
# 테스트 2: 이벤트 바에서 포지션 진입 후 즉시 청산 시 펀딩 미적용
#           (바 시작 시 포지션 없음 → fund_fee = 0)
# 테스트 3: funding_rate 양수 환경 Long: PnL이 funding만큼 감소
#           Short: PnL이 funding만큼 증가
# assert abs(final_equity_long - expected) < 0.01
```

### 3.4 테스트 파일: `test_conservation_identity.py`

```python
# 회계 항등식: Final = Initial - Σfees - Σcarry + Σrealized_PnL
# 시나리오별 검증:
#   A. 단일 심볼 Long 진입 → 수익 청산
#   B. 단일 심볼 Short 진입 → 손실 청산
#   C. stop-loss gap-down 강제 체결
#   D. 격리 청산 발생 (mark_price 기준)
#   E. 펀딩비 누적 후 청산
# assert abs(final_equity - computed_identity) < 1e-6  # 부동소수점 허용
```

---

## 4. Phase 2: evaluator — score, DSR, ergodicity

### 4.1 신규 함수: `compute_v3_score`
`src/domain/futures/optimization/evaluator.py` 에 추가

```python
V3_LAMBDA = {
    "down":     0.50,  # downside_semidev
    "mdd":      1.00,  # worst_MDD
    "cvar":     0.30,  # CVaR_5
    "turnover": 0.20,  # excess_turnover
    "funding":  0.50,  # funding_drag
    "capacity": 0.40,  # AUM_impact_penalty
}

def compute_v3_score(
    leg_log_tw: np.ndarray,          # shape [K]
    worst_mdd: float,                 # 0~1
    cvar_5: float,                    # 0~1 (loss pct)
    excess_turnover: float,           # normalized
    funding_drag: float,              # 0~1
    aum_impact_penalty: float,        # 0~1
) -> float:
    # 고정 λ 사용, 외부 주입 금지
    ...
```

### 4.2 신규 함수: `calc_n_trials_eff_entropy`
```python
def calc_n_trials_eff_entropy(
    signatures: np.ndarray,   # shape [n_trials, 11] (K=8 log_tw + 3 stats)
    weights: np.ndarray,      # shape [n_trials] (completed_legs/8, pruned는 <1)
) -> float:
    # C = weighted_corr(signatures)
    # λ_i = eigenvalues(C), p_i = λ_i / Σλ_i
    # return exp(-Σ p_i * log(p_i))
    ...
```

**Pruned trial 처리 규칙**:
- `completed_legs < K`: 미완료 leg를 cross-sectional median으로 impute
- `weight = completed_legs / K`
- weight=0 trial(즉시 pruned)은 포함하되 기여 최소화

### 4.3 테스트 파일: `test_score_v3.py`

```python
# 테스트 1: 알려진 입력에 대한 score 검증
leg_tw = np.log([1.05] * 8)           # 모든 leg +5%
score = compute_v3_score(leg_tw, mdd=0.10, cvar=0.05, ...)
# 수동 계산값과 비교 (tolerance 1e-9)

# 테스트 2: λ 고정 불변 검증
# compute_v3_score에 λ 파라미터가 없어야 함 (외부 주입 불가)
import inspect
sig = inspect.signature(compute_v3_score)
assert "lambda_down" not in sig.parameters

# 테스트 3: 음의 score 전략 vs 양의 score 전략 순위
# 랜덤 손실 전략 < 양의 수익 전략 항상 성립
```

### 4.4 테스트 파일: `test_dsr_entropy.py`

```python
# 테스트 1: 동일한 signature를 가진 trial 100개 → n_trials_eff ≈ 1.0
identical = np.tile(np.array([0.05]*11), (100, 1))
n_eff = calc_n_trials_eff_entropy(identical, np.ones(100))
assert n_eff < 2.0  # 동일 basin → 유효 독립 검정 최소

# 테스트 2: 완전 독립 signature → n_trials_eff ≈ n_trials
rng = np.random.default_rng(0)
indep = rng.normal(0, 1, (100, 11))
n_eff = calc_n_trials_eff_entropy(indep, np.ones(100))
assert n_eff > 50.0  # 충분히 다양

# 테스트 3: pruned trial weight < 1 적용 시 n_eff 감소
w_partial = np.array([0.5] * 50 + [1.0] * 50)
n_eff_partial = calc_n_trials_eff_entropy(sigs, w_partial)
n_eff_full = calc_n_trials_eff_entropy(sigs, np.ones(100))
assert n_eff_partial <= n_eff_full
```

### 4.5 테스트 파일: `test_ergodicity_gate.py`

```python
# ergodicity_deviation = |log(ensemble_mean) - mean(log_TW_legs)|
# ensemble_mean = mean(exp(log_TW_legs))

# 테스트 1: 낮은 변동성 → deviation 작음 → gate 통과
# 테스트 2: 높은 변동성 (큰 손실 leg 포함) → deviation 큼 → gate 실패 (>15%)
# 테스트 3: deviation 정의 수식 검증 (known example)
```

---

## 5. Phase 3: hard_gates — 8-gate 체계

### 5.1 수정 대상
`src/domain/futures/validation/unified_gates.py`

**v3.0 확정 상수**:
```python
class V3HardGates:
    MIN_POSITIVE_LEG_RATIO: float = 0.55
    WORST_LEG_TW_FLOOR: float = 0.85
    MEAN_LEG_TW_FLOOR: float = 1.015       # 3M leg 기준
    ERGODICITY_PCT: float = 15.0
    EV_COST_FLOOR: float = 3.0
    DSR_FLOOR: float = 0.60
    FUNDING_DRAG_CEILING: float = 0.30     # drag/gross_return
    CAPACITY_REQUIRED_TIERS: tuple = (50_000, 100_000, 250_000)  # 전부 pass 필수
```

**gate 평가 함수**:
```python
@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    metrics: dict[str, float]

def evaluate_v3_hard_gates(
    leg_log_tw: np.ndarray,
    worst_mdd: float,
    dsr: float,
    ev_cost: float,
    funding_drag_ratio: float,
    ergodicity_dev_pct: float,
    capacity_results: dict[int, bool],     # {aum: pass/fail}
) -> GateResult:
    ...
```

### 5.2 테스트 파일: `test_hard_gates_v3.py`

```python
# 각 gate 독립 검증 (나머지는 통과, 해당 gate만 경계값 테스트)

# 테스트 1: min_positive_leg_ratio
# - 8 leg 중 4개 양수 (4/8=0.50) → FAIL
# - 8 leg 중 5개 양수 (5/8=0.625) → PASS (0.55 초과)

# 테스트 2: worst_leg_tw_floor
# - worst leg TW = 0.84 → FAIL
# - worst leg TW = 0.86 → PASS

# 테스트 3: mean_leg_tw_floor
# - mean TW = 1.014 → FAIL
# - mean TW = 1.016 → PASS

# 테스트 4: DSR_FLOOR
# - DSR = 0.59 → FAIL
# - DSR = 0.61 → PASS

# 테스트 5: funding_drag_ceiling
# - drag/return = 0.31 → FAIL
# - drag/return = 0.29 → PASS

# 테스트 6: capacity gate
# - 50k pass, 100k FAIL, 250k FAIL → FAIL (3개 전부 필요)
# - 50k pass, 100k pass, 250k pass → PASS
# - 10k FAIL, 나머지 모두 pass → PASS (sanity only)

# 테스트 7: 모든 gate 통과 시 GateResult.passed == True
# 테스트 8: 복수 gate 실패 시 failures 리스트에 모두 포함
```

---

## 6. Phase 4: atomic_blocks — non-overlap 6M pass_ratio

### 6.1 신규 파일
`src/domain/futures/validation/atomic_blocks.py`

```python
@dataclass(frozen=True)
class AtomicBlockConfig:
    block_months: int = 6          # 비변경 상수
    min_pass_ratio: float = 0.70   # 비변경 상수
    required_min_blocks: int = 3   # 판정을 위한 최소 blocks 수

@dataclass
class AtomicBlockResult:
    n_blocks: int
    n_passed: int
    pass_ratio: float
    passed: bool
    block_log_tws: list[float]
    worst_block_mdd: float
    median_log_growth: float

def build_atomic_blocks(
    timestamps: np.ndarray,        # decision bar timestamps (UTC unix ms)
    is_end_ts: int,                # IS 종료 시점 이후부터 block 시작
    block_months: int = 6,
) -> list[tuple[int, int]]:
    """Non-overlapping 6M 시작/끝 인덱스 쌍 반환. IS 이후부터 시작."""
    ...

def evaluate_atomic_blocks(
    equity_curves: list[np.ndarray],   # 각 block별 equity curve
    gate: V3HardGates = V3HardGates(),
) -> AtomicBlockResult:
    ...
```

### 6.2 테스트 파일: `test_atomic_blocks.py`

```python
# 테스트 1: block 생성 non-overlap 검증
# - 연속 block 간 시간 겹침 없음
# - 각 block이 정확히 6M (±1 bar 허용)

# 테스트 2: IS 기간 데이터가 block에 포함되지 않음
# - build_atomic_blocks(is_end_ts=T)의 첫 block 시작 ≥ T

# 테스트 3: pass_ratio 계산
# - 11 blocks 중 8 pass → pass_ratio = 0.727 > 0.70 → passed=True
# - 11 blocks 중 7 pass → pass_ratio = 0.636 < 0.70 → passed=False

# 테스트 4: double-counting 없음
# - 전체 OOS 기간을 blocks로 분할 시 Σ(block_bars) == total_oos_bars

# 테스트 5: 데이터 부족 (2 blocks만 확보)
# - n_blocks < required_min_blocks → passed=False, 사유 명시
```

---

## 7. Phase 5: portfolio — Kelly, caps, quantization

### 7.1 Fractional Kelly 수정
`src/domain/futures/portfolio/portfolio_constructor.py`

`_kelly_raw` 반환값에 `× KELLY_FRACTION` 적용:
```python
KELLY_FRACTION: float = 0.25  # 모듈 상수, 변경 금지

def _kelly_scaled(
    mu: np.ndarray, sigma_diag: np.ndarray, *, f_kelly_max: float
) -> np.ndarray:
    raw = _kelly_raw(mu, sigma_diag, f_kelly_max=f_kelly_max)
    return raw * KELLY_FRACTION
```

### 7.2 5-cap 투영 확장
현재 gross/per-symbol cap에 net/beta/vol cap 추가:

```python
@dataclass(frozen=True)
class PortfolioCaps:
    gross: float = 3.0
    per_symbol: float = 0.10
    net: float = 0.30        # abs(Σw) ≤ 0.30
    beta: float = 0.50       # abs(w @ btc_beta) ≤ 0.50
    target_ann_vol: float = 0.20

def project_all_caps(
    w: np.ndarray,
    btc_beta: np.ndarray,    # shape [N]
    sigma_port: float,       # realized 1-bar vol
    bars_per_year: float,
    caps: PortfolioCaps = PortfolioCaps(),
) -> np.ndarray:
    ...
```

### 7.3 minNotional 양자화
```python
def quantize_weights(
    w: np.ndarray,           # shape [N]
    equity: float,
    prices: np.ndarray,      # shape [N]
    step_sizes: np.ndarray,  # shape [N] (exchangeInfo)
    min_notional: float = 20.0,
) -> np.ndarray:
    qty = np.floor(w * equity / (prices * step_sizes)) * step_sizes
    notional = qty * prices
    qty = np.where(notional < min_notional, 0.0, qty)
    return qty * prices / equity   # 비중으로 재변환
```

### 7.4 테스트 파일: `test_fractional_kelly.py`

```python
# 테스트 1: full Kelly 대비 0.25x 검증
full = _kelly_raw(mu, sigma_diag, f_kelly_max=10.0)
scaled = _kelly_scaled(mu, sigma_diag, f_kelly_max=10.0)
np.testing.assert_allclose(scaled, full * 0.25)

# 테스트 2: KELLY_FRACTION이 모듈 상수임을 확인 (외부 주입 불가)
import inspect
sig = inspect.signature(_kelly_scaled)
assert "fraction" not in sig.parameters
```

### 7.5 테스트 파일: `test_caps_projection.py`

```python
# 테스트 1: gross cap
w = np.array([0.6] * 6)  # gross = 3.6 > 3.0
w_proj = project_all_caps(w, ...)
assert np.sum(np.abs(w_proj)) <= 3.0 + 1e-6

# 테스트 2: net cap
w = np.array([0.2] * 8)  # net = 1.6 >> 0.30
w_proj = project_all_caps(w, ...)
assert abs(np.sum(w_proj)) <= 0.30 + 1e-6

# 테스트 3: beta cap
w = np.ones(10) * 0.1    # all beta=1.0 → beta_exposure=1.0 > 0.50
w_proj = project_all_caps(w, btc_beta=np.ones(10), ...)
assert abs(np.dot(w_proj, np.ones(10))) <= 0.50 + 1e-6

# 테스트 4: per_symbol cap
w = np.array([0.4, 0.1, 0.1])
w_proj = project_all_caps(w, ...)
assert np.max(np.abs(w_proj)) <= 0.10 + 1e-6
```

### 7.6 테스트 파일: `test_quantization.py`

```python
# 테스트 1: minNotional 미달 주문 → 0
# equity=10000, price=50000, step_size=0.001, w=0.0001
# notional = 0.0001*10000 = 1 USDT < 20 → qty=0

# 테스트 2: step_size 양자화 잔여 처리
# w=0.055, equity=1000, price=100, step_size=0.1
# raw_qty = 0.55 → floor → 0.5 → notional=50 USDT
# 잔여 0.05 * 1000 / 100 = 0.5 unit 미반영 (현금 보유)

# 테스트 3: AUM 10k에서 50 심볼 × w=0.02
# price=500 USDT인 심볼: notional=200 → 통과
# price=5000 USDT인 심볼: notional=20 → 경계값
# price=50000 USDT인 심볼: step_size=0.001, w=0.02*10000/50000=0.004 qty
# → floor(0.004/0.001)*0.001=0.004, notional=200 → 통과
```

---

## 8. Phase 6: boundary_contract — purge_bars seam

### 8.1 신규 인터페이스
`src/domain/futures/validation/boundary_contract.py`

```python
@dataclass
class ModulePurgeBarsMeta:
    module_name: str
    purge_bars: int
    reason: str            # 왜 이 값인지 (label_horizon / fit_window / etc.)

class PurgeBarsRegistry:
    """모든 signal/feature 모듈이 purge_bars를 등록하는 중앙 레지스트리."""
    _registry: dict[str, ModulePurgeBarsMeta] = {}

    def register(self, meta: ModulePurgeBarsMeta) -> None:
        self._registry[meta.module_name] = meta

    def get_boundary_purge_bars(self) -> int:
        """max(all registered purge_bars). 등록 모듈 0개면 RuntimeError."""
        if not self._registry:
            raise RuntimeError("No modules registered purge_bars. Fail-fast.")
        return max(m.purge_bars for m in self._registry.values())

    def validate(self) -> None:
        """미등록 상태면 backtest 진입 거부."""
        _ = self.get_boundary_purge_bars()  # RuntimeError 발생
```

### 8.2 테스트 파일: `test_boundary_contract.py`

```python
# 테스트 1: 빈 레지스트리에서 get_boundary_purge_bars → RuntimeError
reg = PurgeBarsRegistry()
with pytest.raises(RuntimeError, match="Fail-fast"):
    reg.get_boundary_purge_bars()

# 테스트 2: 여러 모듈 등록 시 max 반환
reg.register(ModulePurgeBarsMeta("scaler", 24, "fit_window=24bars"))
reg.register(ModulePurgeBarsMeta("label", 6, "label_horizon=6bars"))
assert reg.get_boundary_purge_bars() == 24

# 테스트 3: purge 적용 후 IS 말단 ~ OOS 시작 사이 데이터 사용 불가
# - IS 마지막 bar = T
# - purge_bars = 24
# - OOS 첫 사용 가능 bar = T + 24
# purge_start, purge_end 인덱스 검증

# 테스트 4: 단순 rolling indicator는 purge 없이 사용 가능
# (IS history 연속 사용 허용 검증)
```

---

## 9. walk_forward.py 상수 업데이트

`src/domain/futures/validation/walk_forward.py`의 `WalkForwardConfig` 기본값 교체:

```python
@dataclass(frozen=True)
class WalkForwardConfig:
    n_legs: int = 8                        # 6 → 8
    purge_bars: int = 24
    min_positive_leg_ratio: float = 0.55   # 0.70 → 0.55
    worst_leg_tw_floor: float = 0.85       # 0.95 → 0.85
    mean_leg_tw_floor: float = 1.015       # 1.00 → 1.015
    ergodicity_guideline_pct: float = 15.0
    dsr_floor: float = 0.60                # NEW
    funding_drag_ceiling: float = 0.30     # NEW
    ergodicity_hard_gate_enabled: bool = True
```

**영향 범위 확인 필요**:
```bash
rg "WalkForwardConfig\|n_legs\|min_positive_leg_ratio" src/ --include="*.py" -l
```

기존 테스트 `test_backtest_engine.py`, `test_final_evaluation.py` 에서 구 상수 하드코딩 여부 확인 후 수정.

---

## 10. 테스트 공통 픽스처

`tests/unit/domain/futures/conftest.py` (신규 or 추가):

```python
import numpy as np
import pytest

@pytest.fixture
def flat_market():
    """가격 불변, 펀딩 없는 기본 환경."""
    n_bars, n_syms = 200, 5
    price = 100.0
    return {
        "open_1m":  np.full((n_bars * 240, n_syms), price),
        "high_1m":  np.full((n_bars * 240, n_syms), price * 1.001),
        "low_1m":   np.full((n_bars * 240, n_syms), price * 0.999),
        "close_1m": np.full((n_bars * 240, n_syms), price),
        "mark_1m":  np.full((n_bars * 240, n_syms), price),
        "funding_mask": np.zeros((n_bars * 240, n_syms), dtype=np.int8),
        "funding_rate": np.zeros((n_bars * 240, n_syms)),
        "kill":     np.zeros((n_bars, n_syms)),
        "n_bars_4h": n_bars,
        "n_syms":    n_syms,
    }

@pytest.fixture
def luna_crash_scenario():
    """LUNA 2022-05-09 급락 재현 (mark vs last 괴리 시나리오)."""
    # mark_price는 last_price보다 높게 유지 (실제 LUNA 현상)
    n = 2880  # 48시간 × 60
    price = np.linspace(80.0, 0.01, n)
    mark  = np.clip(price * 1.05, 0.01, None)  # mark > last
    return {"price_1m": price, "mark_1m": mark}

@pytest.fixture
def leg_log_tw_healthy():
    """K=8 legs, 모두 양수, hard gate 모두 통과하는 기본 케이스."""
    return np.array([0.04, 0.06, 0.03, 0.05, 0.07, 0.04, 0.05, 0.06])

@pytest.fixture
def leg_log_tw_borderline():
    """경계값: min_positive_leg_ratio=0.55 경계 (5/8=0.625)."""
    return np.array([0.04, -0.01, 0.03, 0.05, -0.02, 0.04, 0.05, 0.06])
```

---

## 11. CI 실행 방법

```bash
# Phase별 실행
uv run pytest tests/unit/domain/futures/backtest/ -k "mark_price or funding or conservation" --tb=short
uv run pytest tests/unit/domain/futures/optimization/test_score_v3.py --tb=short
uv run pytest tests/unit/domain/futures/validation/ --tb=short
uv run pytest tests/unit/domain/futures/portfolio/ --tb=short

# 전체 실행
uv run pytest tests/unit/domain/futures/ --tb=short -q

# 기존 테스트 회귀 확인
uv run pytest tests/unit/domain/futures/backtest/test_execution_sim_math.py --tb=short
```

---

## 12. 구현 완료 기준

| Phase | 완료 조건 |
|---|---|
| 1 | `test_mark_price_liquidation` 3개, `test_conservation_identity` 5개 ALL PASS |
| 2 | `test_score_v3` λ 고정 검증, `test_dsr_entropy` 3개 ALL PASS |
| 3 | `test_hard_gates_v3` 8개 gate 독립 검증 ALL PASS |
| 4 | `test_atomic_blocks` non-overlap + pass_ratio + double-count 검증 ALL PASS |
| 5 | `test_fractional_kelly` + `test_caps_projection` + `test_quantization` ALL PASS |
| 6 | `test_boundary_contract` fail-fast + max 반환 ALL PASS |
| 전체 | 기존 테스트 (`test_execution_sim_math`, `test_final_evaluation`) 회귀 없음 |
