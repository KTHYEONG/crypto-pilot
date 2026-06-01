---
title: Forecast Layer Separation (Cost/Risk)
domain: forecast
type: domain-spec
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/forecast/contracts.py
  - src/domain/futures/forecast/cost.py
  - src/domain/futures/forecast/risk.py
  - src/domain/futures/forecast/diagnostics.py
  - src/domain/futures/optimization/objectives.py
  - src/domain/futures/optimization/ml_context.py
change_triggers:
  - "src/domain/futures/forecast/**/*.py"
  - "src/domain/futures/optimization/{objectives,ml_context}.py"
dependencies:
  documents:
    - universe.md
last_verified: "2026-06-01"
---

# Forecast Layer Separation: Cost & Risk (Active)

> **Deprecation Notice (2026-06-01):** `AlphaForecast`/`AlphaArtifactHash`/`to_alpha_forecast`(`forecast/alpha.py`)와
> `compose_mu`(`forecast/compose.py`)는 `candidate_ml` 아키텍처 전환으로 완전히 제거되었다.
> 현재 유효한 계약은 **`CostForecast`(cost.py), `RiskForecast`(risk.py), `LabelDiagnostics`(diagnostics.py)** 뿐이다.

## 1. Overview

**Problem:** Single ML panel (DataFrame + attrs) mixed cost/risk responsibilities. Cost was static (universe-selection time), residual_var always None.

**Solution:** Separate cost and risk into 2 typed Forecast contracts (CostForecast, RiskForecast) + single SSOT per layer. Enables clean decoupling and deterministic backtest consistency.

**Status:** Phase 1 (Active)

---

## 2. Core Components

### 2.1 CostForecast (typed contract)

```python
@dataclass(slots=True, frozen=True)
class CostForecast:
    execution_cost_bps_2d: np.ndarray      # [T,N]
    execution_cost_fraction_2d: np.ndarray # [T,N]
    uncertainty_bps_2d: np.ndarray         # [T,N]
    capacity_notional_2d: np.ndarray | None
    source: str                            # "parametric_dynamic" | "universe_static" | "fallback_global"
    components: dict[str, np.ndarray] = field(default_factory=dict)
```

**Phase 1 (Active):** Floor-only (universe_static acts as minimum).
**Phase 2 (Pending):** per-bar volatility_buffer, dynamic impact, funding_buffer (when COST_FORECAST_DYNAMIC=True).

### 2.2 RiskForecast (typed contract)

```python
@dataclass(slots=True, frozen=True)
class RiskForecast:
    covariance_3d: np.ndarray          # [T,N,N] Ledoit-Wolf
    beta_2d: np.ndarray | None         # [T,N] BTC beta, strict causal
    residual_var_2d: np.ndarray | None # [T,N]
    forecast_vol_2d: np.ndarray        # [T,N]
    beta_source: str                   # "trailing_btc" | "unavailable"
    source: str
```

**Phase 1 (Active):** residual_var computed via build_risk_forecast, injected into RiskSnapshot/PortfolioPolicyInputs.
**Phase 2 (Pending):** KELLY_USE_RESIDUAL_VAR flag to feed idiosyncratic risk into Kelly sizing.

### 2.3 LabelDiagnostics

```python
@dataclass(frozen=True)
class LabelDiagnostics:
    cost_clearance_target: np.ndarray      # [T,N]
    cost_clearance_target_long: np.ndarray
    cost_clearance_target_short: np.ndarray
```

Isolated from LabelPanel (separated in candidate_ml architecture).

---

## 3. Data Flow (Active Path)

```
ml_context._attach_risk_snapshot_slice
  → build_risk_forecast(close_2d_full[slice:], symbols, ...)
    → RiskForecast {covariance_3d, beta_2d, residual_var_2d, forecast_vol_2d, beta_source}
  → RiskSnapshot {covariance_3d, beta_2d, residual_var_2d}
  → aligned["residual_var_2d"] = residual_var_slice (diagnostic)
  → aligned["_beta_source"] = beta_source

objectives.compute_strategy_scores
  → tw_blk = candidate target_weights (pre-merged via candidate_ml pipeline)
  → portfolio construction via precompute_rebalance_weights
```

---

## 4. Business Rules (Invariants)

### Phase 1 (Behavior-Preserving, Active)
1. **Kelly sizing:** ks_diag derives from composer_sigma_2d only (not residual_var).
2. **residual_var diagnostic:** Populated but not consumed for Kelly in Phase 1.
3. **cost forecast static:** COST_FORECAST_DYNAMIC defaults False.

### Phase 2 (Pending, Behavior-Changing)

#### W2: Dynamic CostForecast
- Activation: COST_FORECAST_DYNAMIC = True
- Adds: per-bar volatility_buffer, dynamic impact (σ·√(notional/ADV)), funding_buffer
- Floor: universe_static remains minimum

#### W12: Residual Var → Kelly
- Activation: KELLY_USE_RESIDUAL_VAR = True
- Effect: ks_diag = composer_sigma_bar + residual_var (idiosyncratic risk control)

---

## 5. Testing & Verification

### Phase 1 Coverage
- `test_cost_forecast.py`: floor equivalence, uncertainty >= 0
- `test_risk_forecast.py`: strict causal (future data mutation doesn't change past beta), PSD
- Coverage: 98% (forecast package after Alpha/compose removal)

### Phase 2 A/B Validation
- IS: baseline metrics (CAGR/Sharpe/MDD)
- OOS: metric deltas (flag on vs off)
- Decision rule: >2% CAGR improvement + Calmar stable → flip flag default

---

## 6. Edge Cases & Risks

### Warmup Starvation
- **Risk:** Per-slice build_risk_forecast restarts rolling windows at leg boundaries.
- **Mitigation (Phase 1):** residual_var diagnostic only; Kelly untouched.
- **Action (Phase 2):** When KELLY_USE_RESIDUAL_VAR activates, warmup zeros must be clipped or ignored.

### Parametric Cost Stability
- **Risk:** Parametric cost (vol_buffer, impact) sensitive to outliers.
- **Mitigation:** Parametric floor = universe_static (conservative baseline).
- **Action:** Cap dynamic_cost <= universe_static + margin; monitor cost_underestimation_rate.

---

## 7. Related Decisions

- **Constraint:** Phase 1 metric invariance (no backtest number changes).
- **Phase 2 gating:** All behavior-changing flags default False (opt-in A/B).
- **Architecture:** Alpha/compose logic completely removed (candidate_ml pipeline replacement).
