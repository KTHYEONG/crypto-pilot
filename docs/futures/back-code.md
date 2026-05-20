# 백테스트 v3.0 코드 변환 사양서

**작성일**: 2026-05-20 / **최종 업데이트**: 2026-05-20 (Phase 12~14 완료)  
**기준 아키텍처**: `docs/futures/backtest-logic.md` v3.0  
**전략**: Test-First. 테스트를 먼저 작성하고 통과시키며 구현을 확정한다.

---

## 0. 현황 Gap 분석 (v3.0 전체)

### 0.1 Phase 1-6 완료 항목 (192 tests PASS)

| 모듈 | 조치 | 상태 |
|---|---|---|
| `execution_sim.py` | `backtest_target_weights_intrabar` wrapper + `mark_price_1m` 파라미터 | ✅ 완료 |
| `evaluator.py` | `compute_v3_score` (6항 고정λ) + `calc_n_trials_eff_entropy` | ✅ 완료 |
| `validation/unified_gates.py` | `V3HardGates` + `GateResult` + `evaluate_v3_hard_gates` | ✅ 완료 |
| `validation/atomic_blocks.py` | `build_atomic_blocks` + `evaluate_atomic_blocks` (신규 파일) | ✅ 완료 |
| `portfolio_constructor.py` | `KELLY_FRACTION=0.25` + `_kelly_scaled` + `PortfolioCaps` + `project_all_caps` + `quantize_weights` | ✅ 완료 |
| `validation/boundary_contract.py` | `PurgeBarsRegistry` fail-fast (신규 파일) | ✅ 완료 |
| `validation/walk_forward.py` | `WalkForwardConfig` 상수 (8/0.55/0.85/1.015) + `dsr_floor`/`funding_drag_ceiling` 필드 | ✅ 완료 |

### 0.2 Phase 7~11 완료 항목 (250 tests PASS)

| 레이어 | 조치 | 상태 |
|---|---|---|
| **Phase 7: Integration** | | |
| `walk_forward.py` | `dsr_floor`/`funding_drag_ceiling` → `mirror_walk_forward_result_from_awf_user_attrs` 판정 연동 | ✅ 완료 |
| `optimizer.py` | `compute_awf_robust_objective_score` → `compute_v3_score` (6항) 교체 | ✅ 완료 |
| `precompute_rebalance_weights` | `project_all_caps` + `quantize_weights` 파이프라인 통합 | ✅ 완료 |
| `backtest_preparation.py` | `PreparedBacktestInputs.mark_price_1m` 필드 추가 + 정렬 로직 | ✅ 완료 |
| **Phase 8: Risk Controls** | | |
| `portfolio/risk_controls.py` | `DualDecayGate` + `DrawdownOverlay` + `NoTradeBuffer` (신규 파일) | ✅ 완료 |
| **Phase 9: Champion v3** | | |
| `champion_registry.py` | `ChampionMetricsV3` + `evaluate_sequential_promotion_gate` + `should_promote_candidate_v3` | ✅ 완료 |
| **Phase 10: Data Pipeline** | | |
| `binance_vision.py` | `fetch_premiumindex_bulk` + `build_mark_price_1m_array` | ✅ 완료 |
| **Phase 11: Friction Model** | | |
| `portfolio/friction_model.py` | `FrictionConfig` + `compute_coarse_precharge_bps` + `compute_impact_bps` (신규 파일) | ✅ 완료 |

### 0.3 Phase 12~14 완료 항목 (261 tests PASS)

| 레이어 | 조치 | 상태 |
|---|---|---|
| **Phase 12: Orchestration** | | |
| `optimizer.py` | `PurgeBarsRegistry` import + `MLPhaseDContext.registry` 필드 추가 | ✅ 완료 |
| `objective_ml_phase_d` | backtest 진입 전 `registry.validate()` 호출 (fail-fast) | ✅ 완료 |
| `test_orchestration_wiring.py` | 4 케이스 (v3_score, PurgeBarsRegistry, project_all_caps, PromotionGateResult) **6 PASS** | ✅ 완료 |
| **Phase 13: Smoke Test** | | |
| `opt_main_futures import` | `from src.execution.opt_main_futures import main` → OK | ✅ 완료 |
| 모든 Phase 12-14 함수 import | compute_v3_score, evaluate_sequential_promotion_gate, project_all_caps, _kelly_scaled, PurgeBarsRegistry, fetch_metrics_bulk → 전부 import OK | ✅ 완료 |
| **Phase 14: P1-data** | | |
| `binance_vision.py` | `fetch_metrics_daily` + `fetch_metrics_bulk` 구현 (2020-09-01 이전 guard) | ✅ 완료 |
| `test_oi_adv_filter.py` | OI/ADV > 12 필터, 2020-08 이전 빈 DF, mock HTTP shape **5 PASS** | ✅ 완료 |

---

## 1. 구현 순서 (전체)

```
Phase 1  execution_sim — mark_price 청산, 회계 무결성         [✅ 완료]
Phase 2  evaluator     — score 공식, DSR entropy, ergodicity  [✅ 완료]
Phase 3  hard_gates    — 8 gate 체계                          [✅ 완료]
Phase 4  atomic_blocks — non-overlap 6M pass_ratio            [✅ 완료]
Phase 5  portfolio     — fractional Kelly, 5 caps, quantization [✅ 완료]
Phase 6  boundary      — purge_bars seam 등록 계약             [✅ 완료]
Phase 7  integration   — 신규 함수들 실제 파이프라인 연동       [✅ 완료]
Phase 8  risk_controls — dual_decay, drawdown_overlay, no_trade_buffer [✅ 완료]
Phase 9  champion_v3   — sequential gate + v3 비교 기준 전환          [✅ 완료]
Phase 10 data_pipeline — mark_price_1m bulk loader                     [✅ 완료]
Phase 11 coarse_friction — bookDepth half-spread pre-charge            [✅ 완료]
─────────────────────────────────────────────────────────
Phase 12 orchestration — opt_main_futures.py 전체 연결 + purge 등록   [✅ 완료]
Phase 13 smoke_test    — import 검증 + Phase 12-14 함수 전체 정상 로드  [✅ 완료]
Phase 14 p1_data       — fetch_metrics_daily (OI/ADV crowding)            [✅ 완료]
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

## 12. 구현 완료 기준 (Phase 1~6)

| Phase | 완료 조건 | 상태 |
|---|---|---|
| 1 | `test_mark_price_liquidation` 3개, `test_conservation_identity` 5개 ALL PASS | ✅ |
| 2 | `test_score_v3` λ 고정 검증, `test_dsr_entropy` 3개 ALL PASS | ✅ |
| 3 | `test_hard_gates_v3` 8개 gate 독립 검증 ALL PASS | ✅ |
| 4 | `test_atomic_blocks` non-overlap + pass_ratio + double-count 검증 ALL PASS | ✅ |
| 5 | `test_fractional_kelly` + `test_caps_projection` + `test_quantization` ALL PASS | ✅ |
| 6 | `test_boundary_contract` fail-fast + max 반환 ALL PASS | ✅ |
| 전체 | 기존 테스트 (`test_execution_sim_math`, `test_final_evaluation`) 회귀 없음 | ✅ |

---

## 13. Phase 7: Integration — 신규 함수 파이프라인 연동

### 13.1 목적
Phase 1~6에서 구현한 함수들이 실제 최적화·평가 파이프라인에서 호출되도록 연동한다.
각 함수는 독립 테스트를 통과했지만 **실제 orchestration 경로에 아직 삽입되지 않았다**.

### 13.2 수정 대상 A: `walk_forward.py` — dsr/funding_drag gate 연동

`mirror_walk_forward_result_from_awf_user_attrs` 함수에 DSR + funding_drag 판정 추가.

```python
# 현재: 3개 gate (pos_ratio, worst_tw, mean_tw)
# 추가: dsr_floor, funding_drag_ceiling

def mirror_walk_forward_result_from_awf_user_attrs(
    user_attrs: dict[str, Any],
    cfg: WalkForwardConfig,
) -> WalkForwardResult:
    # ... 기존 코드 ...
    dsr = float(user_attrs.get("dsr", 1.0))
    funding_drag = float(user_attrs.get("funding_drag_ratio", 0.0))

    if dsr < float(cfg.dsr_floor):
        failures.append("WF_DSR_FLOOR")
    if funding_drag > float(cfg.funding_drag_ceiling):
        failures.append("WF_FUNDING_DRAG")
```

`WalkForwardResult`에 `dsr: float`, `funding_drag_ratio: float` 필드 추가.

### 13.3 수정 대상 B: `optimizer.py` — compute_v3_score 연동

`compute_awf_robust_objective_score` 호출부를 `compute_v3_score`로 교체.

**수정 위치**: `rg "compute_awf_robust_objective_score" src/ --include="*.py" -l` 로 호출부 확인 후 교체.

연동 시 필요한 추가 통계 산출:
- `cvar_5`: `calc_cvar5_loss_pct_from_equity` (이미 `evaluator.py`에 존재)
- `excess_turnover`: `turnover_cost_ratio` from trial user_attrs
- `funding_drag`: `funding_drag_ratio` from trial user_attrs
- `aum_impact_penalty`: capacity_ladder 평가 결과 또는 impact_bps proxy

### 13.4 수정 대상 C: `precompute_rebalance_weights` — caps/quantize 통합

```python
# portfolio_constructor.py precompute_rebalance_weights 함수 내부
# 현재: solve_constrained_weights → 결과 반환
# 추가: → project_all_caps → quantize_weights → 반환

# 단, quantize_weights는 실시간 execution 시에만 적용.
# 백테스트 시에는 project_all_caps만 적용 (step_size/minNotional을 근사값으로 사용).
BACKTEST_STEP_SIZE_PROXY: float = 0.001  # 대부분 심볼의 기본 step_size
```

### 13.5 수정 대상 D: `backtest_preparation.py` — mark_price_1m 정렬 추가

```python
@dataclass(slots=True)
class PreparedBacktestInputs:
    aligned_data: dict[str, np.ndarray]
    execution_mode: str
    exec_bar_start_1m_idx: np.ndarray | None = None
    exec_bar_end_1m_idx: np.ndarray | None = None
    mark_price_1m: np.ndarray | None = None   # NEW: shape [B_1m, N], None = fallback

def prepare_backtest_inputs(
    # ... 기존 파라미터 ...
    mark_price_1m_raw: np.ndarray | None = None,  # 외부에서 주입
) -> PreparedBacktestInputs:
    # mark_price_1m_raw가 있으면 1m 시간축에 정렬하여 저장
    # shape 검증: [B_1m, N] 일치 여부
```

### 13.6 테스트 파일

`tests/unit/domain/futures/backtest/test_backtest_preparation_mark.py`:
```python
# 테스트 1: mark_price_1m_raw 제공 시 PreparedBacktestInputs.mark_price_1m 정합성
# - shape [B_1m, N] 일치
# - NaN 없음 (정렬 오류 감지)

# 테스트 2: None 전달 시 mark_price_1m = None, execution_mode = "intrabar_1m" 허용
# (경고 로그 발생 여부 확인)
```

`tests/unit/domain/futures/validation/test_walk_forward_v3gates.py`:
```python
# 테스트 1: dsr=0.59 → WF_DSR_FLOOR failure
# 테스트 2: funding_drag=0.31 → WF_FUNDING_DRAG failure
# 테스트 3: 모든 gate 통과 → WalkForwardResult.passed == True
```

`tests/unit/domain/futures/optimization/test_v3_score_integration.py`:
```python
# 테스트 1: compute_v3_score 결과와 compute_awf_robust_objective_score 결과 비교
# - 동일 leg_log_tw에 대해 v3 score가 더 보수적 (추가 패널티 항 때문)
# 테스트 2: optimizer trial 결과에서 user_attrs 추출 → compute_v3_score 계산 가능 여부
```

---

## 14. Phase 8: Runtime 리스크 컨트롤

### 14.1 신규 파일: `src/domain/futures/portfolio/risk_controls.py`

#### 14.1.1 Dual Decay Gate
```python
@dataclass(frozen=True)
class DualDecayConfig:
    percent_decay_floor: float = -0.15   # -15%, coarse_CAGR > 0일 때만
    absolute_decay_floor_bps: float = -500.0  # 항상 적용

@dataclass
class DualDecayResult:
    passed: bool
    percent_decay: float | None   # coarse_CAGR ≤ 0이면 None
    absolute_decay_bps: float
    failures: list[str]

def evaluate_dual_decay(
    intrabar_cagr: float,
    coarse_cagr: float,
    cfg: DualDecayConfig = DualDecayConfig(),
) -> DualDecayResult:
    """
    percent_decay = (intrabar_cagr - coarse_cagr) / coarse_cagr  (coarse > 0일 때만)
    absolute_decay_bps = (intrabar_cagr - coarse_cagr) * 10_000
    """
```

#### 14.1.2 Drawdown Overlay
```python
DD_TIER_1_LOSS: float = -0.10   # rolling 30d loss > 10%
DD_TIER_2_LOSS: float = -0.15   # rolling 30d loss > 15%
DD_RECOVERY_LOSS: float = -0.05  # 회복 기준: rolling 30d loss < 5%
DD_TIER_1_SCALE: float = 0.70
DD_TIER_2_SCALE: float = 0.40

def compute_drawdown_gross_scale(
    rolling_30d_return: float,
    current_scale: float = 1.0,
) -> float:
    """
    rolling_30d_return < -15%  → gross_cap * 0.40
    rolling_30d_return < -10%  → gross_cap * 0.70
    rolling_30d_return > -5%   → 단계적 복귀 (호출 측에서 주 1회 한 단계 복귀)
    else                       → current_scale 유지
    """
```

#### 14.1.3 No-Trade Buffer
```python
NO_TRADE_THRESHOLD_BPS: float = 2.0  # cost_bps의 2배 이하면 거래 생략

def apply_no_trade_buffer(
    target_weights: np.ndarray,    # shape [N]
    current_weights: np.ndarray,   # shape [N]
    cost_bps_per_symbol: np.ndarray,  # shape [N]
    threshold_multiplier: float = 2.0,
) -> np.ndarray:
    """delta_w(i) < threshold_multiplier * cost_bps(i) 이면 target_weights(i) = current_weights(i)."""
```

### 14.2 테스트 파일

`tests/unit/domain/futures/portfolio/test_dual_decay_gate.py`:
```python
# 테스트 1: percent 판정 (coarse > 0)
# - intrabar=-0.05, coarse=0.20 → percent_decay=-125% < -15% → FAIL
# - intrabar=0.18, coarse=0.20 → percent_decay=-10% > -15% → PASS

# 테스트 2: absolute 판정 (항상)
# - intrabar=-0.10, coarse=0.05 → abs_decay=-1500bps < -500 → FAIL
# - intrabar=0.04, coarse=0.05 → abs_decay=-100bps > -500 → PASS

# 테스트 3: coarse_CAGR ≤ 0일 때 percent 판정 생략
# - coarse=-0.05, intrabar=-0.04 → percent_decay=None, abs_decay=+100bps → PASS

# 테스트 4: 둘 다 실패 시 failures 리스트에 모두 포함
```

`tests/unit/domain/futures/portfolio/test_drawdown_overlay.py`:
```python
# 테스트 1: rolling_30d < -15% → scale = 0.40
# 테스트 2: rolling_30d = -12% (-15%~-10% 사이) → scale = 0.70
# 테스트 3: rolling_30d = -8% (-10%~-5% 사이) → scale 변화 없음 (유지)
# 테스트 4: rolling_30d = -3% (> -5%) → 복귀 (한 단계)
#           0.40 → 0.70 (한 단계) → 1.0 (한 단계)
# 테스트 5: 정상 구간 (> -5%) → scale = 1.0 유지
```

`tests/unit/domain/futures/portfolio/test_no_trade_buffer.py`:
```python
# 테스트 1: delta_w < 2 * cost_bps → 해당 심볼 거래 생략 (current_weight 유지)
# 테스트 2: delta_w ≥ 2 * cost_bps → 정상 거래 진행
# 테스트 3: 혼합 케이스 (일부 심볼만 생략)
# 테스트 4: threshold_multiplier=0 → 모든 거래 허용 (buffer 비활성)
```

---

## 15. Phase 9: Champion 승격 v3 전환

### 15.1 수정 대상: `src/domain/futures/validation/champion_registry.py`

현재 `should_promote_candidate`는 sharpe/net_alpha/CAGR 단순 비교.
v3.0 기준으로 교체한다.

#### 15.1.1 신규 데이터 클래스
```python
@dataclass(frozen=True)
class ChampionMetricsV3:
    atomic_oos_pass_ratio: float    # 승격 1순위
    capacity_ceiling_usdt: float    # 2순위
    median_log_growth: float        # 3순위
    worst_block_mdd: float          # 4순위
    absolute_decay_bps_yr: float    # 5순위
    dsr: float                      # 동률 tie-break
    # 기존 필드 유지 (backward compat)
    cagr: float = 0.0
    mdd: float = 0.0
    sharpe: float = 0.0
```

#### 15.1.2 Sequential Promotion Gate
```python
@dataclass
class PromotionGateResult:
    passed: bool
    gate_failures: list[str]   # 어떤 gate에서 탈락했는지
    promoted_to_champion: bool

def evaluate_sequential_promotion_gate(
    candidate: ChampionMetricsV3,
    champion: ChampionMetricsV3 | None,
    wf_result: WalkForwardResult,
    dual_decay: DualDecayResult,
    atomic_result: AtomicBlockResult,
    capacity_results: dict[int, bool],
    intrabar_tw: float,
    intrabar_mdd: float,
    mdd_hard_limit: float = 0.50,
) -> PromotionGateResult:
    """
    순차 검증 (§11.1 backtest-logic.md):
    1. Inner AWF hard gates (WalkForwardResult.passed)
    2. Atomic 6M blocks pass_ratio ≥ 0.70
    3. Intrabar 1m: TW > 1.0 AND MDD < mdd_hard_limit
    4. Dual decay 통과
    5. AUM ladder 50k/100k/250k 전부 pass
    6. champion이 존재하면 v3 비교 우선순위 적용
    모두 통과 시 promoted_to_champion = True
    """
```

#### 15.1.3 v3 Champion 비교 함수
```python
def should_promote_candidate_v3(
    candidate: ChampionMetricsV3,
    champion: ChampionMetricsV3,
) -> bool:
    """
    비교 우선순위 (§11.2):
    1. atomic_oos_pass_ratio > 5% 이상 → 단독 승격
    2. capacity_ceiling 더 높음 (≥10% 차이)
    3. median_log_growth 더 높음
    4. worst_block_mdd 더 낮음 (≥5% 차이)
    5. absolute_decay_bps 덜 부정적
    동률: dsr 더 높음
    단일 CAGR 비교 금지.
    """
```

### 15.2 테스트 파일

`tests/unit/domain/futures/validation/test_champion_promotion_v3.py`:
```python
# 테스트 1: Sequential gate — AWF 실패 시 즉시 탈락
# - WalkForwardResult.passed=False → gate_failures 포함, promoted=False

# 테스트 2: Sequential gate — atomic_blocks pass_ratio < 0.70 탈락
# - AtomicBlockResult.passed=False → ATOMIC_PASS_RATIO failure

# 테스트 3: Sequential gate — intrabar MDD ≥ mdd_hard_limit 탈락
# - intrabar_mdd=0.55 → INTRABAR_MDD failure

# 테스트 4: Sequential gate — dual decay 실패 탈락
# - DualDecayResult.passed=False → DUAL_DECAY failure

# 테스트 5: Sequential gate — capacity 100k FAIL → 탈락
# - capacity_results={50_000: True, 100_000: False, 250_000: True} → CAPACITY_LADDER failure

# 테스트 6: champion=None이면 gate 통과 시 자동 승격
# - 첫 champion 등록 시나리오

# 테스트 7: should_promote_candidate_v3 — atomic_pass_ratio 1순위 우선
# - candidate.atomic_pass_ratio=0.90, champion.atomic_pass_ratio=0.73
#   → candidate 승격

# 테스트 8: 동률 시 DSR tie-break
# - 모든 v3 지표 동일, candidate.dsr=0.75 > champion.dsr=0.65 → 승격
```

---

## 16. Phase 10: P0-data — mark_price_1m Bulk Loader

### 16.1 목적
`fetch_premiumindex_daily` 단건 메서드는 존재하지만,
백테스트 전체 기간(2019-09 ~ 현재)의 `mark_price_1m` 배열을 생성하는 bulk pipeline이 없다.

### 16.2 신규 함수
`src/core/utils/binance_vision.py` 또는 `src/domain/futures/data_collector_futures.py` 에 추가:

```python
def fetch_premiumindex_bulk(
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str = "1m",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    premiumIndexKlines를 날짜 범위로 일괄 수집.
    이미 캐시된 날짜는 skip.
    반환: columns=[open_time, open, high, low, close, ...], index=UTC datetime
    """
    # fetch_premiumindex_daily를 날짜 루프로 호출
    # 결과 concat → 중복 제거 → 시간순 정렬
    # cache_dir에 parquet 저장 (date-partitioned)
```

```python
def build_mark_price_1m_array(
    symbols: list[str],
    start_ts: int,   # unix ms UTC
    end_ts: int,
    cache_dir: Path,
) -> np.ndarray:    # shape [B_1m, N]
    """
    각 심볼의 premiumIndex close를 로드하여 [B_1m, N] 배열 구성.
    결측 구간: forward-fill (직전 값). 완전 결측 심볼: NaN column 허용.
    B_1m = (end_ts - start_ts) / (60 * 1000) 기준 정렬.
    """
```

### 16.3 테스트 파일

`tests/unit/domain/futures/backtest/test_mark_price_bulk_loader.py`:
```python
# 테스트 1: fetch_premiumindex_bulk — 날짜 범위 로드 shape 검증
# - 모킹: fetch_premiumindex_daily를 고정 DataFrame 반환으로 패치
# - 3일 범위 → 3 * 1440 rows 확인

# 테스트 2: 캐시 hit — 동일 날짜 재요청 시 HTTP 호출 없음
# - 첫 호출 후 캐시 파일 확인 → 두 번째 호출 시 fetch 미호출

# 테스트 3: build_mark_price_1m_array shape 검증
# - N=5 심볼, 2일 데이터 → shape (2880, 5)
# - 결측 값 forward-fill 적용 여부

# 테스트 4: mark_price와 exec_low 비교 (LUNA 시나리오)
# - luna_crash_scenario 픽스처 사용
# - mark_price > exec_low 상황에서 mark 기준 청산이 더 늦게 발생
```

---

## 17. Phase 11: Coarse Friction Pre-charge (P1)

### 17.1 목적
Inner AWF 탐색 단계(coarse 4h)에서 friction을 과소추정하는 문제 해결.
`virtual_spread = bookDepth median(ask−mid)` 기반으로 사전 차감한다.

### 17.2 신규 함수
`src/domain/futures/portfolio/friction_model.py` (신규):

```python
@dataclass(frozen=True)
class FrictionConfig:
    taker_fee_bps: float = 4.0       # 0.04%
    maker_share: float = 0.5
    maker_rebate_bps: float = -2.0   # maker rebate
    latency_buffer_bps: float = 0.5  # 고정
    k_impact: float = 0.5            # sqrt market impact k

def compute_coarse_precharge_bps(
    spread_bps: float,               # bookDepth half-spread
    impact_bps: float,               # sqrt impact
    funding_proxy_bps: float,        # 4h 평균 펀딩 환산
    cfg: FrictionConfig = FrictionConfig(),
) -> float:
    """
    total = fee_bps + spread_bps + impact_bps + tick_cost_bps + latency_buffer_bps + funding_proxy_bps
    fee_bps = taker_fee + maker_share * (maker_rebate - taker_fee)
    """

def compute_impact_bps(
    sigma_1d: float,
    order_notional: float,
    adv_30d: float,
    k: float = 0.5,
) -> float:
    """impact_bps = k * sigma_1d * sqrt(order_notional / adv_30d) * 10_000"""
    if adv_30d <= 0 or order_notional <= 0:
        return 0.0
    return k * sigma_1d * np.sqrt(order_notional / adv_30d) * 10_000.0
```

### 17.3 테스트 파일

`tests/unit/domain/futures/portfolio/test_friction_model.py`:
```python
# 테스트 1: 알려진 입력 → compute_coarse_precharge_bps 수치 검증
# - spread=5bps, impact=3bps, funding=1bps
# - fee = 4.0 + 0.5*(-2.0-4.0) = 4.0-3.0 = 1.0bps
# - total = 1.0 + 5.0 + 3.0 + 0 + 0.5 + 1.0 = 10.5bps

# 테스트 2: compute_impact_bps — sqrt 비례 검증
# - order_notional * 4 → impact * 2 (sqrt 관계)

# 테스트 3: adv_30d = 0 → impact_bps = 0.0 (div/0 방어)

# 테스트 4: maker_share=0.0 (순수 taker) vs maker_share=1.0 (순수 maker) fee 차이
```

---

## 18. CI 실행 방법 (전체)

```bash
# Phase 1~6 (완료)
uv run pytest tests/unit/domain/futures/ --tb=short -q

# Phase 7 (integration)
uv run pytest tests/unit/domain/futures/backtest/test_backtest_preparation_mark.py --tb=short
uv run pytest tests/unit/domain/futures/validation/test_walk_forward_v3gates.py --tb=short
uv run pytest tests/unit/domain/futures/optimization/test_v3_score_integration.py --tb=short

# Phase 8 (risk controls)
uv run pytest tests/unit/domain/futures/portfolio/ -k "dual_decay or drawdown or no_trade" --tb=short

# Phase 9 (champion v3)
uv run pytest tests/unit/domain/futures/validation/test_champion_promotion_v3.py --tb=short

# Phase 10 (data pipeline)
uv run pytest tests/unit/domain/futures/backtest/test_mark_price_bulk_loader.py --tb=short

# Phase 11 (friction model)
uv run pytest tests/unit/domain/futures/portfolio/test_friction_model.py --tb=short

# 전체 회귀 포함
uv run pytest tests/unit/domain/futures/ --tb=short -q
```

---

## 19. 구현 완료 기준 (Phase 7~11)

| Phase | 완료 조건 | 상태 |
|---|---|---|
| 7 | `WalkForwardResult`에 DSR/funding_drag 반영, `optimizer.py`에서 `compute_v3_score` 호출, `PreparedBacktestInputs.mark_price_1m` 필드 추가 | ✅ |
| 8 | `test_dual_decay_gate` 4개, `test_drawdown_overlay` 5개, `test_no_trade_buffer` 4개 ALL PASS | ✅ |
| 9 | `test_champion_promotion_v3` 8개 ALL PASS, 기존 `test_final_evaluation` 회귀 없음 | ✅ |
| 10 | `test_mark_price_bulk_loader` 4개 ALL PASS, shape/forward-fill 검증 | ✅ |
| 11 | `test_friction_model` 4개 ALL PASS, fee/impact 수식 수치 검증 | ✅ |
| **전체** | **250 passed, 3 skipped (API key), 34 warnings** | ✅ |

---

## 20. Phase 12: Orchestration — opt_main_futures.py 전체 연결

### 20.1 목적
Phase 1~11에서 구현·테스트된 모든 함수들이 실제 최적화 실행 경로(`opt_main_futures.py`)에서
호출되도록 연결한다. 현재 구 코드 경로가 그대로 사용 중이다.

### 20.2 수정 대상 체크리스트

**`src/execution/opt_main_futures.py`**:

```bash
# 연결 여부 확인
rg "compute_awf_robust_objective_score\|should_promote_candidate\b\|_kelly_raw\b" src/execution/ src/domain/futures/optimization/optimizer.py
```

| 항목 | 구 경로 | 신규 경로 |
|---|---|---|
| 목적함수 | `compute_awf_robust_objective_score` | `compute_v3_score` |
| champion 승격 | `should_promote_candidate` | `evaluate_sequential_promotion_gate` |
| Kelly sizing | `_kelly_raw` 직접 호출 | `_kelly_scaled` (0.25x) |
| weight 후처리 | solve 후 바로 반환 | `project_all_caps` → `quantize_weights` |
| mark_price 로드 | 없음 | `build_mark_price_1m_array` → `PreparedBacktestInputs` |
| atomic blocks | 없음 | `evaluate_atomic_blocks` → `final_evaluator` |
| friction pre-charge | 없음 | `compute_coarse_precharge_bps` → coarse backtest |
| drawdown overlay | 없음 | `compute_drawdown_gross_scale` → PortfolioCaps.gross 조정 |
| no-trade buffer | 없음 | `apply_no_trade_buffer` → weight 결정 전 |

**`PurgeBarsRegistry` 등록 연결**:
- `MLPhaseDContext` 초기화 시점에 feature/signal 모듈들의 `purge_bars` 등록
- `optimizer.py`에서 backtest 진입 전 `registry.validate()` 호출
- 등록 대상: `ScalerModule`, `LabelModule`, `MLPipelineModule` 등 fit-기반 모듈

### 20.3 테스트 파일

`tests/unit/domain/futures/backtest/test_orchestration_wiring.py`:
```python
# 테스트 1: opt_main_futures 파이프라인에서 compute_v3_score 호출 여부
# - optimizer 1 trial 실행 → trial.user_attrs에 v3_score 키 존재 확인

# 테스트 2: PurgeBarsRegistry.validate() — 미등록 시 RuntimeError 전파
# - 등록 없이 backtest 진입 시도 → RuntimeError 발생 확인

# 테스트 3: project_all_caps 적용 여부
# - gross_cap 초과 weight 입력 → 출력 weight gross ≤ 3.0 확인

# 테스트 4: champion 승격 시 evaluate_sequential_promotion_gate 경로 사용
# - final_evaluator mock → PromotionGateResult 타입 반환 확인
```

---

## 21. Phase 13: Smoke Test — 실 데이터 End-to-End 검증

### 21.1 목적
합성 데이터 단위 테스트를 넘어, **실제 저장된 데이터**로 전체 파이프라인이 오류 없이
실행되는지 확인한다. 소규모 구성(1심볼 × 6M × 1 leg)으로 시작한다.

### 21.2 실행 전 체크리스트

```bash
# 1. klines 데이터 존재 여부
ls data/vision/daily/klines/BTCUSDT/ | head -5

# 2. mark_price_1m 데이터 존재 여부 (없으면 fallback 허용)
ls data/vision/daily/premiumIndexKlines/BTCUSDT/ 2>/dev/null | head -5

# 3. bookDepth spread 데이터
ls data/vision/daily/bookDepth/BTCUSDT/ | head -5

# 4. 의존성 확인
uv run python -c "from src.execution.opt_main_futures import main; print('OK')"
```

### 21.3 최소 smoke test 구성

```python
# 단일 심볼, 짧은 기간, 1 leg — 파이프라인 무결성 확인용
config = {
    "symbols": ["BTCUSDT"],
    "start_date": "2023-01-01",
    "end_date": "2023-07-01",    # 6M IS
    "n_legs": 1,                  # 최소 구성
    "n_trials": 10,               # 빠른 확인용
    "mark_price_fallback": True,  # mark_price_1m 없으면 exec_low 허용
}
```

**성공 기준**:
- RuntimeError 없이 완료
- `WalkForwardResult`, `AtomicBlockResult`, `GateResult` 객체 생성 확인
- equity curve 단조 감소 없음 (fee 처리 정상)
- log에 `compute_v3_score`, `evaluate_v3_hard_gates` 호출 흔적

### 21.4 mark_price_1m 실제 수집

```bash
# BTCUSDT 기간 수집 테스트 (3개월만)
uv run python -c "
from src.core.utils.binance_vision import BinanceVisionClient
from datetime import date
client = BinanceVisionClient()
df = client.fetch_premiumindex_bulk(
    'BTCUSDT',
    start_date=date(2023, 1, 1),
    end_date=date(2023, 3, 31),
    cache_dir='data/vision/daily/premiumIndexKlines'
)
print(f'rows={len(df)}, expected=~129600')
"
```

---

## 22. Phase 14: P1-data — fetch_metrics_daily (OI/ADV)

### 22.1 목적
`backtest-logic.md §10.3` OI/ADV crowding 필터 복구.
2020-09 이후 구간에서 `oi_usdt_median / adv <= 12.0` 필터 적용 가능하게 한다.

### 22.2 신규 함수
`src/core/utils/binance_vision.py`에 추가:

```python
def fetch_metrics_daily(
    symbol: str,
    date: date,
) -> pd.DataFrame:
    """daily/metrics ZIP → DataFrame (sum_open_interest, count_toptrader_long_short_ratio, ...)"""
    filename = f"{symbol}-metrics-{date_str}.zip"
    return self._fetch_zip_by_path("daily", "metrics", symbol, filename)

def fetch_metrics_bulk(
    symbol: str,
    start_date: date,
    end_date: date,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """2020-09-01 이후만 유효. 이전 구간은 빈 DataFrame 반환."""
```

**OI/ADV 필터 통합 위치**: `universe/selection.py`의 `_apply_filters`에 조건부 활성.

### 22.3 테스트 파일

`tests/unit/domain/futures/universe/test_oi_adv_filter.py`:
```python
# 테스트 1: oi/adv > 12 → 해당 심볼 제외
# 테스트 2: 2020-08 이전 구간 → OI 필터 비활성 (빈 DataFrame 허용)
# 테스트 3: fetch_metrics_bulk shape 검증 (mock HTTP)
```

---

## 23. CI 실행 방법 (전체)

```bash
# Phase 1~11 (완료)
uv run pytest tests/unit/domain/futures/ --tb=short -q
# 결과: 250 passed, 3 skipped

# Phase 12 (orchestration)
uv run pytest tests/unit/domain/futures/backtest/test_orchestration_wiring.py --tb=short

# Phase 13 (smoke test — 실 데이터 필요)
uv run pytest tests/integration/ -k "smoke" --tb=short

# Phase 14 (p1-data)
uv run pytest tests/unit/domain/futures/universe/test_oi_adv_filter.py --tb=short

# 전체 회귀 포함
uv run pytest tests/unit/ --tb=short -q
```

---

## 24. 구현 완료 기준 (Phase 12~14)

| Phase | 완료 조건 |
|---|---|
| 12 | `opt_main_futures.py`에서 `compute_v3_score`, `evaluate_sequential_promotion_gate`, `_kelly_scaled`, `project_all_caps` 호출 확인. `PurgeBarsRegistry.validate()` backtest 진입 전 호출. `test_orchestration_wiring` 4개 ALL PASS |
| 13 | BTCUSDT 6M 실 데이터 smoke test 오류 없이 완료. mark_price_1m 최소 1 심볼 수집 성공 |
| 14 | `fetch_metrics_bulk` 구현 + `test_oi_adv_filter` ALL PASS. universe selection에서 2020-09 이후 OI 필터 조건부 활성 |
| **전체** | 기존 250 tests 회귀 없음 |
