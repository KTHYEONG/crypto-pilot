---
title: Candidate ML Edge Recovery Plan
domain: futures-strategy
type: refactor
status: ready_for_implementation
priority: critical
created_at: 2026-06-03
related_paths:
  - docs/results/result.md
  - docs/specs/candidate_ml_strategy.md
  - docs/specs/candidate_ml_compounding_improvement.md
  - src/domain/futures/strategy/rule_diagnostics.py
  - src/domain/futures/strategy/candidate_edge.py
  - src/domain/futures/strategy/candidate_dataset.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/strategy/ablation.py
---

# Candidate ML Edge Recovery Plan

## 1. Current Diagnosis

Reproduced command:

```bash
UV_CACHE_DIR=/tmp/uv-cache FUTURES_STRATEGY_NAME=candidate_ml PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase alpha --sync skip --timeframe 4h --trials 1 --date 2026-05-01
```

Current blocker is not the gate threshold. It is edge prediction and selection:

- `Active Signals: 0`
- `zero_reason=no_eligible_after_breakeven_floor`
- `p_pass`: median `0.4266`, p90 `0.5632`, `pct_ge40=0.618`
- `mu_net_decision_bps`: median `-18.3`, p90 `-12.8`, max `-8.9`
- `breakeven_floor=12.0bps`
- `edge_fail=11101/11101`, `eligible=0`

The user's judgment is partially correct: the current strategy universe does not provide enough stable, learnable alpha for the ML edge model. However, the immediate problem is more specific than "strategy count is low":

1. Rule diagnostics are selecting some candidates from the same OOS region used for final evaluation.
2. Several top OOS rules are regime-shift winners: they are negative or unstable in fit/calibration, so supervised ML has no valid pre-OOS evidence to learn them.
3. The edge regressor collapses all OOS variants to a negative band, even when some grouped rule diagnostics show positive OOS edge.
4. Current tests do not assert that positive calibration edge is preserved in OOS prediction or that OOS-selected rules are not reused for inference.

## 2. Exact Problems

### P0-1. Promotion filter still uses inference OOS for recommendations

`compute_rule_diagnostics()` uses a fixed `0.8` split internally and builds `recommended_keep_variants` from OOS metrics. `bridge.py` then applies those recommendations before building ML fit/calibration/OOS datasets.

This means `CANDIDATE TOP STRATEGIES` is an ex-post report, not a clean training signal. The promotion filter must not choose variants using the same OOS rows later used for inference/backtest.

Impact:

- Reported `KEEP` variants are not proof of deployable alpha.
- ML can be asked to trade variants whose positive edge only appears in unseen OOS.
- If the ML path ever produces trades, this promotion step becomes look-ahead leakage.

### P0-2. Rule candidates are not consistently learnable before OOS

DEBUG reproduction shows promoted fit/calibration targets are mostly negative:

```text
EDGE_TARGET train mean=-15.7 median=-124.0 pct_pos=0.439
EDGE_TARGET valid mean=-25.0 median=-195.1 pct_pos=0.422
```

Variant examples:

- `btc_regime_pullback:btc_pullback_50`: train `-68.1bps`, valid `+12.1bps`, report OOS `+29.8bps`
- `funding_zscore_carry:fzs_168`: train `-62.2bps`, valid `-30.6bps`, report OOS `+28.3bps`
- `cross_sectional_momentum:cs_mom_10`: train `+24.5bps`, valid `-29.8bps`, report OOS `+5.0bps`

This is not a normal supervised-learning setup. Positive OOS results are often not supported by positive pre-OOS targets, so the model correctly learns weak or negative expected edge.

### P0-3. Edge model has no safety check for prediction collapse

`fit_candidate_edge_models()` trains Huber/quantile LightGBM models but does not validate whether predictions preserve target dispersion or variant-level positive edge. Current OOS predictions collapse:

```text
EDGE_VARIANT_TOP mean_mu ~= -16.7 to -17.7
max_mu <= -8.9
pct_mu_pass=0.000 for every promoted variant
```

This is a model failure mode, not a selection-threshold issue. Lowering `min_expected_net_bps` or `min_net_floor_cost_fraction` would only hide the failure and create forced trades with negative expected edge.

### P0-4. Selection policy is behaving correctly

`utility_topk` blocks all trades because no event reaches the breakeven floor. This is desirable fail-closed behavior:

```python
eligible = catastrophic_mask & (df["mu_net_decision_bps"] >= breakeven_floor)
```

Do not loosen this as the primary fix. Selection should only be relaxed in diagnostic ablations, never as the default production path.

### P1. Strategy expansion is necessary but should be gated

Current top rule variants are too few and too correlated around broad crypto beta, funding, and simple mean-reversion/momentum. More strategies are needed, but adding families before fixing leakage and model collapse will produce another ex-post report rather than a tradable system.

## 3. Target Files

Modify:

- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/rule_diagnostics.py`
- `src/domain/futures/strategy/candidate_edge.py`
- `src/domain/futures/strategy/candidate_contracts.py`
- `src/domain/futures/strategy/candidate_portfolio.py`
- `src/domain/futures/strategy_runtime/bridge.py`
- `src/domain/futures/strategy/ablation.py`
- `tests/unit/domain/futures/strategy/test_rule_diagnostics.py`
- `tests/unit/domain/futures/strategy/test_candidate_edge.py`
- `tests/unit/domain/futures/strategy/test_candidate_portfolio.py`
- `tests/unit/domain/futures/strategy/test_ablation.py`

Reference:

- `src/domain/futures/strategy/rule_signals.py`
- `src/domain/futures/strategy/candidate_dataset.py`
- `docs/architecture/backtest-logic.md`

## 4. Contracts

### 4.1 CandidateStrategyConfig

Add fields:

```python
promotion_decision_split: Literal["fit", "calibration", "fit_calibration"] = "fit_calibration"
min_promotion_calibration_edge_bps: float = 1.0
min_promotion_calibration_obs: int = 100
edge_prediction_min_std_bps: float = 3.0
edge_prediction_min_positive_rate: float = 0.01
edge_prior_enabled: bool = True
edge_prior_min_obs: int = 100
edge_prior_shrinkage_obs: int = 500
edge_residual_model_enabled: bool = True
```

Validation:

- `promotion_decision_split` in `{"fit", "calibration", "fit_calibration"}`
- all obs fields `>= 1`
- all bps/std/rate fields non-negative
- `edge_prediction_min_positive_rate <= 1.0`

### 4.2 RuleDiagnosticsResult

Extend dataclass:

```python
recommendation_basis: str
recommendation_split: tuple[int, int]
report_split: tuple[int, int]
```

### 4.3 compute_rule_diagnostics

Replace signature:

```python
def compute_rule_diagnostics(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    min_obs: int = 100,
    silent: bool = False,
) -> RuleDiagnosticsResult:
```

with:

```python
def compute_rule_diagnostics(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    min_obs: int = 100,
    silent: bool = False,
    recommendation_start: int | None = None,
    recommendation_end: int | None = None,
    report_start: int | None = None,
    report_end: int | None = None,
) -> RuleDiagnosticsResult:
```

Rules:

- `recommended_keep_variants` and `recommended_flip_variants` must be computed only from `[recommendation_start, recommendation_end)`.
- The display table may still report `[report_start, report_end)`.
- If recommendation bounds are not passed, preserve current behavior for backward compatibility in tests, but bridge/ablation must always pass explicit bounds.

### 4.4 CandidateEdgeModels

Replace:

```python
@dataclass(slots=True, frozen=True)
class CandidateEdgeModels:
    center_model: LGBMRegressor
    q10_model: LGBMRegressor
    q90_model: LGBMRegressor
    feature_names: tuple[str, ...]
```

with:

```python
@dataclass(slots=True, frozen=True)
class CandidateEdgeModels:
    center_model: LGBMRegressor
    q10_model: LGBMRegressor
    q90_model: LGBMRegressor
    feature_names: tuple[str, ...]
    variant_prior_bps: dict[str, float]
    variant_prior_obs: dict[str, int]
    global_prior_bps: float
    target_mode: Literal["direct", "prior_residual"]
    prediction_diagnostics: dict[str, float | int | str]
```

### 4.5 Edge Prediction Diagnostics

`predict_candidate_edges()` must populate `CandidateModelOutput.selection_thresholds` with:

```python
{
    "utility_min": float,
    "p_pass_min": float,
    "edge_min": float,
    "q10_catastrophic_min": float,
    "mu_std_bps": float,
    "mu_positive_rate": float,
    "mu_floor_pass_rate": float,
    "prediction_collapse": bool,
}
```

## 5. Step-by-Step Logic

### Step A. Remove promotion leakage

In `bridge.py`:

1. Compute `fit/calibration/oos` indices before `compute_rule_diagnostics`.
2. Derive recommendation bounds:
   - `fit`: `[fit_start, fit_end)`
   - `calibration`: `[calibration_start, calibration_end)`
   - `fit_calibration`: `[fit_start, calibration_end)` excluding purge gap by filtering event rows.
3. Call `compute_rule_diagnostics(..., recommendation_start=..., recommendation_end=..., report_start=oos_start, report_end=oos_end)`.
4. Apply promotions from recommendation period only.
5. Keep `CANDIDATE TOP STRATEGIES` as OOS report, but label it as report-only.

### Step B. Add no-leak promotion ablation

In `ablation.py`, add rows:

- `Rule Promo No-Leak`: rule promotion decided on fit/calibration, evaluated on OOS.
- `Rule Promo OOS Oracle`: current ex-post promotion, explicitly marked oracle.
- `Variant Prior`: no event-level ML; use calibration variant mean edge prior and current portfolio selection.

Expected interpretation:

- If `Rule Promo No-Leak` is near zero or negative while `OOS Oracle` is positive, the issue is strategy stability, not ML architecture.
- If `Variant Prior` produces nonzero OOS trades but ML does not, the issue is edge model collapse.

### Step C. Add empirical-Bayes variant prior for edge

In `fit_candidate_edge_models()`:

1. Build `variant_key = family + ":" + variant` from `train.event_index`.
2. Compute `global_prior_bps = weighted_mean(train.y_edge_bps)`.
3. For each variant:
   - `variant_mean = weighted_mean(y_edge)`
   - `n = obs`
   - `shrink = n / (n + cfg.edge_prior_shrinkage_obs)`
   - `prior = shrink * variant_mean + (1 - shrink) * global_prior_bps`
4. If `cfg.edge_prior_enabled`, train center model on residual:
   - `y_center_train = train.y_edge_bps - prior_for_train_event`
   - `target_mode="prior_residual"`
5. Otherwise keep direct target.

In `predict_candidate_edges()`:

1. Compute `prior_for_oos_event` by variant key; fallback to `global_prior_bps`.
2. `mu_net_decision_bps = prior + residual_pred` when `target_mode == "prior_residual"`.
3. Keep q10/q90 direct initially; do not add a q10 prior until center edge recovery is validated.

### Step D. Add prediction collapse guard

After prediction:

```python
finite_mu = mu_net_decision_bps[np.isfinite(mu_net_decision_bps)]
prediction_collapse = (
    finite_mu.size > 0
    and np.std(finite_mu) < cfg.edge_prediction_min_std_bps
    and (finite_mu > 0.0).mean() < cfg.edge_prediction_min_positive_rate
)
```

Behavior:

- Log `[DIAG][EDGE_COLLAPSE]` at `WARNING`.
- Do not force trades.
- Store the flag in `selection_thresholds`.
- Add ablation rows to compare direct model vs prior-residual model.

### Step E. Keep selection fail-closed

Do not change default `utility_topk` eligibility:

```python
eligible = catastrophic_mask & (df["mu_net_decision_bps"] >= breakeven_floor)
```

Only add diagnostic sensitivity rows for:

- `min_net_floor_cost_fraction=0.0`
- `selection_policy="validation_quantile"`
- `q10` catastrophic-only mode

These are diagnostics, not production defaults.

### Step F. Strategy expansion after no-leak baseline

Add new families only after Step A-D show whether no-leak rules can produce positive OOS trades.

Priority families:

1. `funding_acceleration_carry`: funding z-score slope and persistence, not just level.
2. `btc_residual_momentum`: alt return residual after BTC beta adjustment.
3. `oi_volume_confirmed_breakout`: OI impulse + volume z-score + range breakout.
4. `basis_funding_dislocation`: high funding with adverse/flat price confirmation.
5. `low_vol_carry_reversion`: volatility compression plus funding carry unwind.

Family acceptance criteria:

- No feature may use future bars.
- Fit/calibration edge must be positive after `max(cost_floor_bps, execution_cost_bps_2d)`.
- OOS edge must remain positive in walk-forward report.
- Spearman IC is diagnostic only unless score is truly continuous and side-aware.
- Family must improve no-leak `Rule Promo` or `Variant Prior` ablation, not only OOS oracle.

## 6. Surgical Plan

### `src/domain/futures/strategy/config.py`

Action: ADD config fields and validation from section 4.1.

### `src/domain/futures/strategy/rule_diagnostics.py`

Action: REPLACE fixed recommendation split logic with explicit recommendation/report windows.

Action: ADD log line:

```text
[DIAG][RULE_RECOMMEND_BASIS] basis=fit_calibration recommend=[0,5260) report=[5278,6576)
```

### `src/domain/futures/strategy_runtime/bridge.py`

Action: MOVE `_candidate_ml_split_indices()` call before `compute_rule_diagnostics()`.

Action: PASS no-leak recommendation bounds and OOS report bounds into `compute_rule_diagnostics()`.

Action: INCLUDE `recommendation_basis`, `recommendation_split`, and `report_split` in `rule_report`.

### `src/domain/futures/strategy/candidate_edge.py`

Action: REPLACE `CandidateEdgeModels` contract.

Action: ADD variant prior/residual target mode.

Action: ADD prediction collapse diagnostics and warning log.

### `src/domain/futures/strategy/candidate_contracts.py`

Action: UPDATE `CandidateModelOutput.selection_thresholds` expectations only if stricter typing is introduced; otherwise no code change required.

### `src/domain/futures/strategy/ablation.py`

Action: ADD no-leak vs oracle rule promotion ablation rows.

Action: ADD direct edge vs prior-residual edge ablation rows.

### Tests

Action: ADD tests:

- `test_rule_recommendations_use_explicit_recommendation_window_not_report_window`
- `test_bridge_passes_no_leak_recommendation_window`
- `test_edge_prior_residual_preserves_positive_variant_prior`
- `test_predict_candidate_edges_flags_prediction_collapse`
- `test_selection_stays_zero_when_all_mu_below_breakeven_floor`

## 7. Verification

Run L1:

```bash
uv run ruff check --fix src/domain/futures/strategy/config.py src/domain/futures/strategy/rule_diagnostics.py src/domain/futures/strategy/candidate_edge.py src/domain/futures/strategy/candidate_contracts.py src/domain/futures/strategy/candidate_portfolio.py src/domain/futures/strategy_runtime/bridge.py src/domain/futures/strategy/ablation.py tests/unit/domain/futures/strategy/test_rule_diagnostics.py tests/unit/domain/futures/strategy/test_candidate_edge.py tests/unit/domain/futures/strategy/test_candidate_portfolio.py tests/unit/domain/futures/strategy/test_ablation.py
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/rule_diagnostics.py src/domain/futures/strategy/candidate_edge.py src/domain/futures/strategy/candidate_contracts.py src/domain/futures/strategy/candidate_portfolio.py src/domain/futures/strategy_runtime/bridge.py src/domain/futures/strategy/ablation.py
```

Run focused tests:

```bash
uv run pytest tests/unit/domain/futures/strategy/test_rule_diagnostics.py tests/unit/domain/futures/strategy/test_candidate_edge.py tests/unit/domain/futures/strategy/test_candidate_portfolio.py tests/unit/domain/futures/strategy/test_ablation.py --tb=short
```

Run integration:

```bash
UV_CACHE_DIR=/tmp/uv-cache FUTURES_STRATEGY_NAME=candidate_ml PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase alpha --sync skip --timeframe 4h --trials 1 --date 2026-05-01
```

Expected outcomes:

- Rule recommendations are based on fit/calibration, not inference OOS.
- OOS top strategy table remains report-only.
- If selected events remain zero, diagnostics distinguish `no_stable_no_leak_rule_edge` from `edge_prediction_collapse`.
- At least one ablation row shows whether the bottleneck is strategy stability or ML edge model collapse.
- No production path forces trades with `mu_net_decision_bps < breakeven_floor`.

## 8. Acceptance Criteria

- No OOS row used for final inference can decide promoted variants.
- Edge prediction collapse is detected and reported.
- Positive calibration variant edge is preserved by the prior-residual edge model in unit tests.
- The ablation table separates no-leak rules, OOS oracle rules, direct ML edge, and prior-residual ML edge.
- Strategy expansion is started only after no-leak baseline results are measurable.
