---
title: Forecast Layer Separation (Alpha/Cost/Risk)
domain: forecast
type: domain-spec
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/forecast/
  - src/domain/futures/optimization/objectives.py
  - src/domain/futures/optimization/ml_context.py
  - src/domain/futures/optimization/final_evaluator.py
  - src/domain/futures/strategy/ml_builder.py
change_triggers:
  - "src/domain/futures/forecast/**/*.py"
  - "src/domain/futures/optimization/{objectives,ml_context,final_evaluator}.py"
  - "src/domain/futures/strategy/{ml_builder,labels,contracts}.py"
dependencies:
  documents:
    - strategy-ml.md
    - universe.md
last_verified: "2026-05-27"
---

# Forecast Layer Separation: Alpha/Cost/Risk

## 1. Overview

**Problem:** Single ML panel (DataFrame + attrs) mixed alpha/cost/risk responsibilities implicitly. Cost was static (universe-selection time), residual_var always None, compose was embedded in objectives, cost_clearance was dead code.

**Solution:** Separate concerns into 3 typed Forecast contracts (AlphaForecast, CostForecast, RiskForecast) + single compose_mu SSOT. Enables module synergy and correct backtest consistency.

**Delivery:** 2-phase
- **Phase 1 (ACTIVE):** Behavior-preserving, foundation laid
- **Phase 2 (PENDING):** Flag-gated behavior-changing improvements (W2/W7/W12)

---

## 2. Core Components

### 2.1 AlphaForecast (typed contract)

```python
@dataclass(slots=True, frozen=True)
class AlphaArtifactHash:
    alpha_config_hash: str           # asdict(cfg.ml)
    feature_config_hash: str         # sorted feature_names
    label_config_hash: str           # horizon + calibrator_target
    train_window_hash: str           # fold window [start:end]
    fold_spec_hash: str              # fold spec + count
    model_family: str                # "lightgbm_dual_side_quantile"
    selected_horizon: int
    
    def combined(self) -> str        # 7-field full provenance
    def structural_hash(self) -> str # config-only, window-independent (IS/HO/OOS validation)

@dataclass(slots=True, frozen=True)
class AlphaForecast:
    datetimes: np.ndarray            # [T]
    symbols: tuple[str, ...]         # [N]
    alpha_long_2d: np.ndarray        # [T,N] gross EV, >= 0
    alpha_short_2d: np.ndarray       # [T,N] gross EV, >= 0
    q10_long_2d: np.ndarray | None
    q50_long_2d: np.ndarray | None
    q90_long_2d: np.ndarray | None
    q10_short_2d: np.ndarray | None
    q50_short_2d: np.ndarray | None
    q90_short_2d: np.ndarray | None
    confidence_long_2d: np.ndarray | None
    confidence_short_2d: np.ndarray | None
    eligible_mask: np.ndarray        # [T,N] bool
    source: str
    artifact_hash: AlphaArtifactHash
```

**W4 Resolution:** AlphaArtifactHash(7 fields) + structural_hash() ensures config identity across IS/HO/OOS.

### 2.2 CostForecast (typed contract)

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

**Phase 1:** Floor-only (universe_static acts as minimum).
**Phase 2 (W2):** per-bar volatility_buffer, dynamic impact, funding_buffer (when COST_FORECAST_DYNAMIC=True).

### 2.3 RiskForecast (typed contract)

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

**W3 Resolution:** residual_var now computed via build_risk_forecast, injected into RiskSnapshot/PortfolioPolicyInputs.
**Phase 2 (W12):** KELLY_USE_RESIDUAL_VAR flag to feed idiosyncratic risk into Kelly sizing.

### 2.4 LabelDiagnostics (W5 separation)

```python
@dataclass(frozen=True)
class LabelDiagnostics:
    cost_clearance_target: np.ndarray      # [T,N]
    cost_clearance_target_long: np.ndarray
    cost_clearance_target_short: np.ndarray
```

Isolated from LabelPanel (which previously mixed training labels + cost diagnostics).

---

## 3. Data Flow

```
build_ml_strategy_alpha
  → panel (DataFrame) + attrs
    {config_hash, feature_config_hash, label_config_hash,
     train_window_hash, fold_spec_hash, model_family, selected_horizon,
     alpha_artifact_combined_hash, alpha_artifact_structural_hash}
  → to_alpha_forecast(panel) → AlphaForecast (typed)
  → merge_ml_output_into_data_maps

ml_context._attach_risk_snapshot_slice
  → build_risk_forecast(close_2d_full[slice:], symbols, ...)
    → RiskForecast {covariance_3d, beta_2d, residual_var_2d, forecast_vol_2d, beta_source}
  → RiskSnapshot {covariance_3d, beta_2d, residual_var_2d}
  → aligned["residual_var_2d"] = residual_var_slice (diagnostic, Phase 1)
  → aligned["_beta_source"] = beta_source

objectives._compose_strategy_scores_inplace
  → compose_mu(alpha, cost, params, holding_bars=rebalance_bars)
    → mu_long = BETA_ALPHA * alpha_long - cost_frac
    → mu_short = BETA_ALPHA * alpha_short - cost_frac
    → [OPTIONAL] if COST_GATE_AMORTIZE: cost_frac /= holding_bars
    → xs_long = where(mu_long >= HURDLE, mu_long, 0)
    → xs_short = where(mu_short >= HURDLE, mu_short, 0)

precompute_rebalance_weights
  → [Phase 1] ks_diag = composer_sigma_2d (unchanged)
  → [Phase 2 if KELLY_USE_RESIDUAL_VAR] ks_diag = composer_sigma_bar + residual_var (optional)
```

---

## 4. Business Rules (Invariants)

### Phase 1 (Behavior-Preserving)
1. **compose_mu formula unchanged:** mu = BETA_ALPHA * alpha - cost; xs = hurdle gate.
2. **Kelly sizing untouched:** ks_diag derives from composer_sigma_2d only (not residual_var).
3. **residual_var diagnostic:** Populated but not consumed in Phase 1.
4. **cost amortize off:** COST_GATE_AMORTIZE defaults False (compose_mu L43).
5. **cost forecast static:** COST_FORECAST_DYNAMIC defaults False.
6. **artifact hash validation:** IS/HO/OOS structural_hash must match (final_evaluator.py:756).

### Phase 2 (Behavior-Changing, Flag-Gated)

#### W2: Dynamic CostForecast
- Activation: COST_FORECAST_DYNAMIC = True
- Adds: per-bar volatility_buffer, dynamic impact (σ·√(notional/ADV)), funding_buffer
- Floor: universe_static remains minimum
- When to use: OOS cost underestimation > 10% → flip flag, A/B test

#### W7: Cost Amortize
- Activation: COST_GATE_AMORTIZE = True + holding_bars provided
- Effect: cost_frac /= holding_bars in compose_mu (L43)
- When to use: Marginal trades failing hurdle due to full round-trip cost over 1-bar horizon

#### W12: Residual Var → Kelly
- Activation: KELLY_USE_RESIDUAL_VAR = True
- Effect: ks_diag = composer_sigma_bar + residual_var (idiosyncratic risk control)
- When to use: Risk concentration in single symbols → enhance diversification penalty

---

## 5. Testing & Verification

### Phase 1 Coverage
- `test_compose.py`: hurdle boundary, amortize on/off, double-cost absence
- `test_cost_forecast.py`: floor equivalence, uncertainty >= 0
- `test_risk_forecast.py`: strict causal (future data mutation doesn't change past beta), PSD
- `test_alpha_forecast.py`: lossless wrapping, artifact hash determinism
- `test_artifact_hash_consistency.py`: structural_hash stable across IS/HO/OOS, combined_hash differs
- Coverage: 97% (forecast package)

### Phase 2 A/B Validation
- IS: baseline metrics (CAGR/Sharpe/MDD)
- OOS: metric deltas (flag on vs off)
- Decision rule: >2% CAGR improvement + Calmar stable → flip flag default
- Acceptance: at least 1 Phase 2 flag deployed with positive A/B result

---

## 6. Edge Cases & Risks

### W3 Warmup Starvation
- **Risk:** Per-slice build_risk_forecast restarts rolling windows at leg boundaries.
- **Mitigation (Phase 1):** residual_var diagnostic only; Kelly untouched.
- **Action (Phase 2):** When KELLY_USE_RESIDUAL_VAR activates, warmup zeros must be clipped or ignored.

### W2 Parametric Cost Stability
- **Risk:** Parametric cost (vol_buffer, impact) sensitive to outliers.
- **Mitigation:** Parametric floor = universe_static (conservative baseline).
- **Action:** Cap dynamic_cost <= universe_static + margin; monitor cost_underestimation_rate.

---

## 7. Examples

### Phase 1: Compose without amortization
```python
af = to_alpha_forecast(ml_panel)
cf = build_cost_forecast(close_2d, ..., universe_cost_bps_2d=universe_bps, cfg=CostModelConfig())
xs_l, xs_s, mu_l, mu_s = compose_mu(af, cf, params={"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 10.0})
# mu = 1.0 * alpha - cost; xs = where(mu >= 0.001, mu, 0)
```

### Phase 2: Compose with amortization
```python
xs_l, xs_s, mu_l, mu_s = compose_mu(
    af, cf, 
    params={"BETA_ALPHA": 1.0, "EV_HURDLE_BPS": 10.0, "COST_GATE_AMORTIZE": True},
    holding_bars=5  # rebalance_bars
)
# mu = 1.0 * alpha - (cost / 5); xs = where(mu >= 0.001, mu, 0)
```

---

## 8. Related Decisions

- **ADR:** Use of `structural_hash()` instead of `combined()` for IS/HO/OOS validation (window-independent config identity).
- **Constraint:** Phase 1 metric invariance (no backtest number changes).
- **Phase 2 gating:** All behavior-changing flags default False (opt-in A/B).
