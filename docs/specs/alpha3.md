---
title: Alpha production uplift via cost-aware soft portfolio policy
domain: futures-alpha
type: prd
status: ready
priority: critical
ai_read_policy: always
created: 2026-05-31
references:
  - docs/specs/alpha0.md
  - docs/specs/alpha1.md
  - docs/specs/alpha2.md
  - docs/results/re-alpha.md
target_phase: alpha3
---

# Alpha Production Uplift via Cost-Aware Soft Portfolio Policy

## 0. Cold Current-State Read

Latest documented `re-alpha.md` outcome remains `ALPHA_PASS=FALSE`.

The important signal is not "no alpha exists". The current state is narrower:

- Dense ranker skill exists: documented `dense_ranker ic=0.0347`, `t=4.51`.
- Residual IC is positive: documented `resid=0.0361`.
- Effective tradable breadth is weak: documented `emit_breadth≈1`, `target_breadth>=8`, and dense breadth around `3.7`.
- Validation policy emits no trade because `validation_net_lcb_bps <= 0`.
- The final blocker is `policy_economics.validation_net_lcb_non_positive`, not the old `alpha_panel empty` mechanical failure.

Therefore alpha is not ready for live capital. It is ready for the next engineering phase: convert weak but real cross-sectional rank skill into a cost-aware, breadth-preserving portfolio policy and validate that policy without OOS leakage.

## 1. Evaluation Criteria Audit

### 1.1 24bps judgment

The 24bps wall is valid, but only at the economics/execution layer.

Correct:

- Apply 24bps or dynamic per-symbol cost to realized or expected portfolio spread.
- Use it inside validation policy objective, basket net LCB, and execution/backtest checks.
- Keep `basket_net_bps_lcb_24bps > 0` as a hard economic claim threshold.

Incorrect:

- Do not compare `rank_weight` magnitude to 24bps.
- Do not use `alpha_active_p95_bps >= 24bps` for `rank_cs_neutral` / `rank_sized` output.

### 1.2 Production-readiness gates

`ALPHA_PASS` must remain strict. Passing alpha mode is still not enough for live deployment.

Add a staged promotion model:

- `diagnostic`: rank skill exists, but portfolio policy can fail.
- `paper`: validation-only policy has positive net LCB and OOS diagnostics agree.
- `shadow`: strategy-mode backtest passes execution realism and turnover constraints.
- `canary`: real-time paper or minimal-notional run stays inside risk limits.

This phase targets `paper` readiness only. Do not claim live readiness from alpha diagnostics alone.

### 1.3 Main diagnosis

The current hard tail policy is the bottleneck. Dense rank IC is positive, but the current validation policy selects too few symbols and fails post-cost LCB. The next implementation should preserve more cross-sectional information through a soft, neutralized, cost-aware weight surface instead of relying only on top/bottom tail selection.

## 2. Target Files

- `src/domain/futures/strategy/contracts.py`
- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/labels.py`
- `src/domain/futures/strategy/rank_selection.py`
- `src/domain/futures/strategy/ml_builder.py`
- `src/domain/futures/strategy/alpha_evaluation.py`
- `src/execution/opt_main_futures.py`
- `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`
- `tests/unit/domain/futures/strategy/test_rank_selection.py`
- `tests/unit/domain/futures/strategy/test_labels.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`
- `docs/results/re-alpha.md`

## 3. Contracts

### 3.1 LabelPanel horizon map

Extend `src/domain/futures/strategy/contracts.py`.

```python
@dataclass(slots=True, frozen=True)
class LabelPanel:
    # existing fields ...
    forward_return_by_horizon: dict[int, np.ndarray] | None = None
```

Rules:

- Keys are horizon bars.
- Values are shape `[T, N]`, dtype `float32`.
- Values must use the same target mode as policy validation:
  - `beta_residualized`: gross forward return minus trailing beta market component minus funding.
  - `gross`: raw gross forward return minus funding.
- No fee or slippage is subtracted here. Execution cost is applied in policy/evaluation.

### 3.2 Horizon label helper

Add to `src/domain/futures/strategy/labels.py`.

```python
def build_forward_return_for_horizon(
    *,
    aligned: AlignedMarketData,
    eligible_2d: NDArray[np.bool_],
    beta_2d: NDArray[np.float64],
    horizon_bars: int,
    target_mode: Literal["beta_residualized", "gross"],
) -> NDArray[np.float32]:
    """Build leak-free forward return labels for one horizon."""
```

Indexing contract:

- Entry price is `open[t + 1]`.
- Exit price is `close[t + horizon_bars]`.
- The last `horizon_bars` rows are `NaN`.
- `beta_2d[t]` must be trailing-only and known at `t`.
- Market forward return may use `[t + 1, t + horizon_bars]` because it is the label, not a feature.
- The function must not read model scores or OOS diagnostics.

Update `build_label_panel()`:

- Build `forward_return_by_horizon` for `cfg.rank_policy_holding_candidates`.
- Include `cfg.label_horizon_bars` even if not listed.
- Keep existing `exec_net_ret` behavior for backward compatibility.

### 3.3 RankSelectionPolicy expansion

Update `src/domain/futures/strategy/rank_selection.py`.

```python
RankSelectionMode = Literal["tail", "soft_cs"]

@dataclass(frozen=True, slots=True)
class RankSelectionPolicy:
    polarity: Literal[1, -1]
    quantile: float
    min_abs_z: float
    weighting: Literal["equal", "zscore", "tanh"]
    weight_k: float
    holding_bars: int
    validation_net_lcb_bps: float
    validation_gross_bps: float
    validation_ir_t: float
    validation_monotonicity: float
    n_obs: int
    selection_mode: RankSelectionMode = "tail"
    validation_turnover: float = float("nan")
    validation_cost_bps: float = float("nan")
    validation_breadth: float = float("nan")
    validation_abs_net_exposure: float = float("nan")
    validation_abs_beta_exposure: float = float("nan")
```

Backward compatibility:

- `policy_to_dict()` and `policy_from_dict()` must tolerate missing new keys.
- Existing tests using the old constructor must continue by relying on defaults.

### 3.4 Soft weight construction

Add to `src/domain/futures/strategy/rank_selection.py`.

```python
def build_signed_rank_weights(
    *,
    signed_score_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    policy: RankSelectionPolicy,
    beta_2d: NDArray[np.float64] | None = None,
    gross_target: float = 1.0,
    max_abs_net_exposure: float = 0.05,
    max_abs_beta_exposure: float = 0.20,
) -> NDArray[np.float64]:
    """Convert rank scores to signed portfolio weights without realized returns."""
```

Rules:

- Shape is `[T, N]`.
- Output is signed: positive long, negative short.
- `tail` mode preserves current top/bottom behavior.
- `soft_cs` mode:
  - Compute row-wise z-score.
  - Apply `polarity`.
  - Apply `tanh(weight_k * z)` for `weighting="tanh"`.
  - Set ineligible cells to `0.0`.
  - Cross-sectionally de-mean eligible weights per bar.
  - Scale gross exposure to `gross_target` when nonzero.
  - Project net and beta exposure using existing `project_all_caps()` from `src/domain/futures/portfolio/portfolio_constructor.py`.
- Do not use realized returns.
- If `policy_is_no_trade(policy)` is true, return all zeros.

Update `apply_rank_selection_policy()`:

```python
def apply_rank_selection_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
    policy: RankSelectionPolicy,
    beta_2d: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Emit alpha_long/alpha_short rank-weight surfaces from signed weights."""
```

Conversion:

- `alpha_long = clip(weights, 0, inf)`.
- `alpha_short = clip(-weights, 0, inf)`.
- These remain rank weights, not return bps.

### 3.5 Cost-aware calibration

Add to `src/domain/futures/strategy/rank_selection.py`.

```python
def calibrate_rank_portfolio_policy(
    *,
    signed_score_2d: NDArray[np.float64],
    realized_fwd_ret_by_horizon: Mapping[int, NDArray[np.float64]],
    eligible_2d: NDArray[np.bool_],
    execution_cost_bps_2d: NDArray[np.float64] | None,
    beta_2d: NDArray[np.float64] | None,
    quantiles: tuple[float, ...],
    min_abs_z_grid: tuple[float, ...],
    holding_bars_candidates: tuple[int, ...],
    selection_modes: tuple[RankSelectionMode, ...],
    cost_bps_fallback: float,
    min_obs: int = 120,
    weight_k: float = 3.0,
    weighting: Literal["equal", "zscore", "tanh"] = "tanh",
    target_breadth_min: int = 8,
    max_turnover: float = 1.25,
    max_abs_net_exposure: float = 0.05,
    max_abs_beta_exposure: float = 0.20,
) -> RankSelectionPolicy:
    """Select validation-only rank-to-portfolio policy after costs and risk constraints."""
```

Validation objective:

1. For each horizon, mode, polarity, quantile, and `min_abs_z` candidate:
   - Build signed weights using only validation fold scores.
   - Compute bar PnL: `sum(weights[t] * realized[t]) * 1e4`.
   - Compute turnover: `sum(abs(weights[t] - weights[t-1]))`.
   - Compute cost: `sum(abs(delta_weight_i) * execution_cost_bps_i)`.
   - Use fallback cost when dynamic cost is unavailable.
   - Net bar bps: `gross_bar_bps - cost_bar_bps`.
2. Compute:
   - `validation_gross_bps = mean(gross_bar_bps)`.
   - `validation_cost_bps = mean(cost_bar_bps)`.
   - `validation_net_lcb_bps = mean(net_bar_bps) - se(net_bar_bps)`.
   - `validation_ir_t = mean(net_bar_bps) / se(net_bar_bps)`.
   - `validation_breadth = mean(count(abs(weight) > 0))`.
   - `validation_turnover = mean(turnover)`.
   - `validation_abs_net_exposure = mean(abs(sum(weights)))`.
   - `validation_abs_beta_exposure = mean(abs(sum(weights * beta)))`.
3. Reject candidate when:
   - `n_obs < min_obs`.
   - `validation_net_lcb_bps <= 0`.
   - `validation_monotonicity <= 0`.
   - `validation_breadth < target_breadth_min`.
   - `validation_turnover > max_turnover`.
   - net or beta exposure exceeds configured limits.
4. Choose max objective:

```text
objective =
  validation_net_lcb_bps
  + 0.10 * validation_gross_bps
  + 0.05 * min(validation_breadth, target_breadth_min)
  - 0.25 * validation_cost_bps
```

Fallback:

- If no candidate passes, return no-trade policy with:
  - `validation_net_lcb_bps = -1.0`
  - `n_obs = 0`
  - `selection_mode = "soft_cs"` when configured, otherwise `"tail"`

Anti-bias:

- Calibration must use validation fold labels only.
- Test/OOS returns, smoke metrics, and `docs/results/re-alpha.md` values must never enter policy selection.

Compatibility wrapper:

- Keep `calibrate_rank_selection_policy()` as a wrapper around `calibrate_rank_portfolio_policy()` with one horizon and static cost so existing callers/tests can migrate incrementally.

### 3.6 Config additions

Update `StrategyMLConfig` in `src/domain/futures/strategy/config.py`.

```python
rank_policy_selection_modes: tuple[Literal["tail", "soft_cs"], ...] = ("soft_cs", "tail")
rank_policy_target_breadth_min: int = 8
rank_policy_max_turnover: float = 1.25
rank_policy_max_abs_net_exposure: float = 0.05
rank_policy_max_abs_beta_exposure: float = 0.20
rank_policy_cost_source: Literal["dynamic", "static"] = "dynamic"
alpha_promotion_min_oos_folds: int = 2
```

Validation:

- `rank_policy_selection_modes` must be non-empty.
- Values must be only `"tail"` or `"soft_cs"`.
- `rank_policy_target_breadth_min >= 2`.
- `rank_policy_max_turnover > 0`.
- Net and beta exposure caps must be non-negative.
- `rank_policy_cost_source` must be `"dynamic"` or `"static"`.

### 3.7 ML builder integration

Update `src/domain/futures/strategy/ml_builder.py`.

For each fold:

- Build validation score grids as today.
- Read realized validation returns from `labels.forward_return_by_horizon`.
- Pass `labels.dynamic_cost_bps_2d` when `cfg.rank_policy_cost_source == "dynamic"`.
- Pass `labels.beta_2d`.
- Call `calibrate_rank_portfolio_policy()`.
- Apply the selected policy to test rows using `apply_rank_selection_policy(..., beta_2d=labels.beta_2d)`.

Panel attrs:

```python
panel.attrs["rank_selection_policy"] = policy_to_dict(aggregate_policy)
panel.attrs["rank_selection_policy_by_fold"] = list(rank_policies_by_fold)
panel.attrs["rank_policy_no_trade"] = bool(rank_policy_no_trade)
panel.attrs["rank_policy_selection_mode"] = str(aggregate_policy["selection_mode"])
panel.attrs["rank_policy_target_breadth_min"] = int(ml_cfg.rank_policy_target_breadth_min)
```

No-trade rule:

- `rank_policy_no_trade=True` only when all fold policies are no-trade.
- A mixed result must emit weights for folds whose policy passed and zeros for failed folds.

### 3.8 Alpha evaluation extension

Update `AlphaEvaluationReport` in `src/domain/futures/strategy/alpha_evaluation.py`.

```python
policy_validation_turnover: float = float("nan")
policy_validation_cost_bps: float = float("nan")
policy_validation_breadth: float = float("nan")
policy_selection_mode: str = ""
promotion_stage: str = "diagnostic"
```

Update `evaluate_alpha()` signature:

```python
def evaluate_alpha(
    *,
    # existing args ...
    policy_validation_turnover: float = float("nan"),
    policy_validation_cost_bps: float = float("nan"),
    policy_validation_breadth: float = float("nan"),
    policy_selection_mode: str = "",
) -> AlphaEvaluationReport:
```

Rules:

- If `policy_no_trade=True`, keep `ALPHA_PASS=False`.
- If `policy_no_trade=False`, enforce:
  - `policy_validation_net_lcb_bps > 0`
  - `policy_validation_breadth >= target_breadth_min` at builder/evaluation call site
  - `basket_net_bps_lcb_24bps > 0`
  - rank skill and robustness gates unchanged
- Add fail reasons:
  - `policy_economics.validation_breadth_below_target`
  - `policy_economics.validation_turnover_too_high`
  - `policy_economics.validation_cost_drag_too_high`

Promotion:

- `diagnostic`: rank IC available but any hard gate fails.
- `paper`: alpha gates pass and policy is not no-trade.
- Do not emit `shadow` or `canary` from `evaluate_alpha()`.

### 3.9 Alpha CLI reporting

Update `src/execution/opt_main_futures.py`.

Add log line:

```text
[ALPHA-POLICY-PORT] mode=<selection_mode> hold=<bars> breadth=<x> turnover=<x> cost=<x> net_lcb=<x> beta=<x> net=<x>
```

Final verdict must report:

- `promotion_stage`
- `policy_selection_mode`
- `policy_validation_breadth`
- `policy_validation_turnover`
- `policy_validation_cost_bps`

Do not claim live tradability. The maximum stage in alpha mode is `paper`.

## 4. Surgical Plan

### 4.1 `src/domain/futures/strategy/contracts.py`

[ACTION: ADD]

- Add `forward_return_by_horizon` to `LabelPanel`.

### 4.2 `src/domain/futures/strategy/labels.py`

[ACTION: ADD]

- Add `build_forward_return_for_horizon()`.
- Populate `forward_return_by_horizon` in `build_label_panel()`.
- Add tests that prove:
  - entry is `open[t + 1]`;
  - exit is `close[t + horizon]`;
  - final horizon rows are `NaN`;
  - different horizons produce different labels on non-flat prices.

### 4.3 `src/domain/futures/strategy/rank_selection.py`

[ACTION: REPLACE]

- Extend `RankSelectionPolicy`.
- Add `RankSelectionMode`.
- Add `build_signed_rank_weights()`.
- Add `calibrate_rank_portfolio_policy()`.
- Update `apply_rank_selection_policy()` to accept optional `beta_2d`.
- Keep `calibrate_rank_selection_policy()` as compatibility wrapper.
- Update serialization helpers.

### 4.4 `src/domain/futures/strategy/config.py`

[ACTION: ADD]

- Add config fields from section 3.6.
- Add validation in `__post_init__`.

### 4.5 `src/domain/futures/strategy/ml_builder.py`

[ACTION: REPLACE]

- Replace direct `calibrate_rank_selection_policy()` call with `calibrate_rank_portfolio_policy()`.
- Use horizon-specific labels from `labels.forward_return_by_horizon`.
- Pass dynamic costs and beta.
- Preserve mixed fold behavior.
- Add new policy attrs to panel.

### 4.6 `src/domain/futures/strategy/alpha_evaluation.py`

[ACTION: ADD]

- Extend report and function arguments.
- Add policy economics fail reasons.
- Keep 24bps basket economics as execution realism.

### 4.7 `src/execution/opt_main_futures.py`

[ACTION: ADD]

- Log `[ALPHA-POLICY-PORT]`.
- Include `promotion_stage` in final alpha verdict.
- Keep `ALPHA_PASS=false` unless every hard gate passes.

### 4.8 `docs/results/re-alpha.md`

[ACTION: ADD]

- Append alpha3 smoke section after implementation.
- Include:
  - policy mode;
  - selected horizon;
  - validation net LCB;
  - validation breadth;
  - validation turnover;
  - alpha pass;
  - promotion stage.

## 5. Verification

### L1

```bash
uv run ruff check --fix \
  src/domain/futures/strategy/contracts.py \
  src/domain/futures/strategy/config.py \
  src/domain/futures/strategy/labels.py \
  src/domain/futures/strategy/rank_selection.py \
  src/domain/futures/strategy/ml_builder.py \
  src/domain/futures/strategy/alpha_evaluation.py \
  src/execution/opt_main_futures.py \
  tests/unit/domain/futures/strategy/test_labels.py \
  tests/unit/domain/futures/strategy/test_rank_selection.py \
  tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
  tests/unit/domain/futures/strategy/test_ml_builder.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py
```

```bash
uv run mypy \
  src/domain/futures/strategy/contracts.py \
  src/domain/futures/strategy/config.py \
  src/domain/futures/strategy/labels.py \
  src/domain/futures/strategy/rank_selection.py \
  src/domain/futures/strategy/ml_builder.py \
  src/domain/futures/strategy/alpha_evaluation.py \
  src/execution/opt_main_futures.py
```

Expected:

- Ruff passes.
- Mypy passes.

### L2

```bash
uv run pytest \
  tests/unit/domain/futures/strategy/test_labels.py \
  tests/unit/domain/futures/strategy/test_rank_selection.py \
  tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
  tests/unit/domain/futures/strategy/test_ml_builder.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py \
  --tb=short
```

Expected:

- All selected tests pass.
- New tests prove horizon candidates are real, not metadata-only.
- New tests prove `soft_cs` produces higher breadth than equivalent `tail` policy on a synthetic cross-section.
- New tests prove no OOS/test return is used during calibration.

### Smoke

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python \
  src/execution/opt_main_futures.py \
  --mode alpha \
  --sync-mode skip \
  --trials 1 \
  --tf 4h \
  --reference-date 2026-05-01
```

Expected minimum:

- Smoke completes without `alpha_panel empty`.
- Logs include `[ALPHA-POLICY-PORT]`.
- If policy remains no-trade, final failure remains `policy_economics.validation_net_lcb_non_positive`.
- If policy emits, `policy_validation_breadth >= 8` or failure reason explicitly says `policy_economics.validation_breadth_below_target`.
- `ALPHA_PASS` remains false unless net LCB, rank skill, statistical robustness, and execution realism all pass.

## 6. Acceptance Checklist

- [ ] 24bps is applied only to policy economics / execution realism, not rank-weight magnitude.
- [ ] Horizon candidates use horizon-specific forward returns.
- [ ] `soft_cs` mode is available and breadth-preserving.
- [ ] Policy calibration uses dynamic cost when available.
- [ ] Policy calibration penalizes turnover and rejects excessive beta/net exposure.
- [ ] No OOS/test labels are used for policy selection.
- [ ] Alpha mode reports `promotion_stage`.
- [ ] `docs/results/re-alpha.md` records alpha3 smoke result.

## 7. Non-Goals

- Do not relax rank IC, t-stat, DSR, or basket net LCB thresholds.
- Do not claim live deployment readiness from alpha mode.
- Do not optimize model hyperparameters in this phase.
- Do not add new external dependencies.
