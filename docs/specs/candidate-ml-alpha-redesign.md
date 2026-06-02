# Candidate ML Alpha Redesign Spec

## Spec Type
`prd` / `quant`

## Context Summary
Current diagnostics in `docs/results/result.md` show that the Candidate ML stack is not yet a production-ready alpha source. Rule-only variants trade but fail growth targets, while ML-assisted variants produce zero trades under production thresholds. The direct blockers are gate support and `q10` shortfall filtering, but the deeper issue is weak and unstable alpha extraction.

Evidence:
- `rule_only_equal_size`: CAGR `-7.44%`, DD `22.95%`.
- `rule_only_fractional_kelly`: CAGR `-0.20%`, DD `1.20%`.
- ML variants: CAGR `0.00%`, DD `0.00%`, trades `0`.
- Gate probabilities max near `0.42`, below the fixed `0.55` threshold.
- `q10_net_bps` is strongly negative and blocks every production candidate at `q10>=-80`.
- Positive edge pockets exist, but the full distribution is fat-tailed and unstable.

Conclusion:
The current strategy should be treated as an alpha research scaffold, not a deployable compounding engine. The next design should focus on extracting robust alpha, not on loosening thresholds to force trades.

## Direct Answers To The Four Questions

### 1. Is the Current Strategy Itself Insufficient?
Yes, based on the current diagnostic report. The evidence is not just zero trades from ML selection. The rule-only baselines also fail to produce meaningful post-cost compounding, and the edge distribution has large downside tails. This means the strategy's candidate alpha quality is insufficient under the measured IS/OOS period.

Important caveat:
One diagnostic window is not enough to declare every rule family permanently invalid. It is enough to reject current production promotion and require walk-forward alpha promotion before deploying.

### 2. Is "Generate Many Candidate Strategies, Then Select High Expected Return Candidates" Effective?
Directionally yes, but only if the selection target is leak-free, OOS-validated, and compounding-aware.

The correct form is:
1. Generate many causal rule candidates.
2. Label each candidate with net-of-cost forward outcomes under an execution policy that matches backtest/live behavior.
3. Promote only candidate families/variants that pass OOS alpha diagnostics.
4. Train ML to rank/select promoted candidates by expected log-growth utility, not raw future return alone.
5. Size selected events with fractional Kelly capped by downside, liquidity, beta, gross/net exposure, and turnover.

The wrong form is:
Selecting candidates because they performed well when looking at the same future window used to choose them. That creates look-ahead bias and will overfit aggressively.

Recommended evaluation criteria:
- Primary: OOS mean log-growth uplift after fees, slippage, funding, and borrow.
- Alpha: Spearman Rank IC between candidate score and net edge.
- Edge: `E[net_bps]`, `P(net_bps > 0)`, payoff ratio, and tail ratio.
- Downside: `q10_net_bps`, CVaR-like expected shortfall, MAE distribution.
- Stability: purged walk-forward block pass ratio, DSR, PBO, and train/OOS edge decay.
- Realism: turnover, capacity/liquidity proxy, drawdown, liquidation count.

### 3. Is the Current ML Gate Market Regime Logic Valid?
It is not a dedicated market regime model. The current gate is a per-event binary classifier trained on `profitable_after_hurdle_label`, with a small set of trailing return, volatility, dispersion, cost, and funding features. It has some context features, but it does not explicitly model regime state, per-family regime validity, or regime-conditioned thresholds.

Current weakness:
- `min_gate_probability=0.55` is absolute while calibrated support maxes near `0.42`.
- The gate target is binary profitability, not compounding utility.
- Candidate family/variant identity is not part of the feature matrix.
- Rule diagnostics are advisory and not wired into production selection.
- Regime behavior is implicit, not validated by OOS regime buckets.

Better approach:
Use the gate as a meta-label layer, but make final selection utility-driven and validation-calibrated:
- Add causal market-state features: BTC trend, BTC volatility, market dispersion, cross-sectional breadth, funding stress, volume/liquidity stress, and recent drawdown.
- Add candidate identity features: family, variant, side, holding horizon, stop/take-profit multipliers.
- Calibrate thresholds from validation OOS support, not fixed constants only.
- Track per-regime OOS alpha diagnostics and disable families in regimes where they have no positive net edge.
- Prefer `utility = E[log(1 + clipped_weight * net_return)] - downside_penalty - turnover_penalty` over pure `p_pass >= threshold`.

### 4. What Is The Gate Target For Stop-Loss / Take-Profit First Hit, And Is ATR Best?
Current labeling uses ATR stop/take-profit thresholds:
- Entry uses `open_2d[entry_idx]`.
- ATR is computed at `decision_idx = entry_idx - 1`.
- TP/SL thresholds are `take_profit_atr_mult * ATR / entry_px` and `stop_atr_mult * ATR / entry_px`.
- For long candidates, favorable path uses high/entry and adverse path uses low/entry.
- For short candidates, favorable path uses entry/low and adverse path uses entry/high.
- If TP and SL occur on the same bar offset, stop-loss wins because `sl_i <= tp_i` is treated as loss.
- If neither barrier is hit, exit is at the last close in the holding horizon.
- `barrier_first_label` records TP-first plus positive edge.
- `profitable_after_hurdle_label` records net profitability after cost/hurdle and is currently used as `y_gate`.

This is reasonable as a conservative labeling primitive, but it is not sufficient as the final strategy logic.

Current mismatch:
The Candidate label path uses per-event `stop_atr_mult` and `take_profit_atr_mult`, but the actual backtest engine consumes `target_weights` plus global `ATR_MULT` and `TRAIL_MULT`. The portfolio builder forward-fills weights over `expected_holding_bars`; it does not pass per-event take-profit logic to the engine. Therefore, label exit assumptions and execution exit behavior can diverge.

ATR is acceptable as a baseline volatility-normalized stop, but not necessarily optimal:
- Static ATR multiples ignore regime, liquidity, volatility clustering, and candidate family behavior.
- Same ATR policy for trend breakout and mean reversion is usually too crude.
- Stop/take-profit should be candidate-family and regime aware.
- The better target is to learn/validate exit policy using MAE/MFE, realized volatility, and time-to-exit distributions under purged walk-forward validation.

## Target Files

Primary modifications:
- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/candidate_contracts.py`
- `src/domain/futures/strategy/rule_signals.py`
- `src/domain/futures/strategy/candidate_labels.py`
- `src/domain/futures/strategy/candidate_dataset.py`
- `src/domain/futures/strategy/candidate_gate.py`
- `src/domain/futures/strategy/candidate_edge.py`
- `src/domain/futures/strategy/candidate_portfolio.py`
- `src/domain/futures/strategy/rule_diagnostics.py`
- `src/domain/futures/strategy/ablation.py`
- `src/domain/futures/strategy_runtime/bridge.py`
- `src/domain/futures/backtest/engine.py`
- `src/domain/futures/portfolio/execution_sim.py`

Tests:
- `tests/unit/domain/futures/strategy/test_candidate_labels.py`
- `tests/unit/domain/futures/strategy/test_candidate_dataset.py`
- `tests/unit/domain/futures/strategy/test_candidate_gate.py`
- `tests/unit/domain/futures/strategy/test_candidate_edge.py`
- `tests/unit/domain/futures/strategy/test_candidate_portfolio.py`
- `tests/unit/domain/futures/strategy/test_rule_diagnostics.py`
- `tests/unit/domain/futures/strategy/test_ablation.py`
- `tests/unit/domain/futures/backtest/test_backtest_engine.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`

Documentation:
- `docs/architecture/candidate-ml-architecture.md`
- `docs/results/result.md`

## Contracts

### Keep Existing Public Function Signatures
Keep these signatures stable unless a call-site update is included in the same patch:

```python
def build_rule_signal_panels(
    *,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
) -> tuple[CandidateSignalPanel, ...]:
    ...
```

```python
def candidate_panels_to_events(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    min_abs_score: float,
    side_flip_variants: tuple[str, ...] = (),
) -> pd.DataFrame:
    ...
```

```python
def label_candidate_events(
    *,
    events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    ...
```

```python
def build_candidate_dataset(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    split_start: int,
    split_end: int,
) -> CandidateDataset:
    ...
```

```python
def fit_candidate_gate(
    *,
    train: CandidateDataset,
    valid: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateGateModel:
    ...
```

```python
def predict_candidate_gate(
    *,
    model: CandidateGateModel,
    dataset: CandidateDataset,
) -> NDArray[np.float64]:
    ...
```

```python
def fit_candidate_edge_models(
    *,
    train: CandidateDataset,
    valid: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateEdgeModels:
    ...
```

```python
def predict_candidate_edges(
    *,
    models: CandidateEdgeModels,
    dataset: CandidateDataset,
    p_pass: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> CandidateModelOutput:
    ...
```

```python
def select_candidate_events_for_portfolio(
    *,
    model_output: CandidateModelOutput,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    ...
```

```python
def build_candidate_target_weights(
    *,
    selected_events: pd.DataFrame,
    close_2d: NDArray[np.float64],
    symbols: tuple[str, ...],
    beta_2d: NDArray[np.float64] | None,
    sigma_3d: NDArray[np.float64] | None,
    cfg: CandidateStrategyConfig,
) -> NDArray[np.float64]:
    ...
```

### Add Config Fields
Add to `CandidateStrategyConfig`:

```python
candidate_identity_features_enabled: bool = True
market_state_features_enabled: bool = True
promotion_filter_enabled: bool = True
selection_policy: Literal["hard", "validation_quantile", "utility_topk"] = "validation_quantile"
selection_top_quantile: float = 0.10
min_oos_rank_ic: float = 0.01
min_oos_log_growth_uplift: float = 0.0
max_oos_edge_decay_bps: float = 50.0
exit_policy_mode: Literal["label_only", "engine_aligned"] = "engine_aligned"
```

Validation:
- `0.0 < selection_top_quantile <= 1.0`
- `selection_policy in {"hard", "validation_quantile", "utility_topk"}`
- `exit_policy_mode in {"label_only", "engine_aligned"}`
- `min_oos_rank_ic >= -1.0 and <= 1.0`

### CandidateDataset Contract Extension
Extend `CandidateDataset` without changing existing field names:

```python
@dataclass(slots=True, frozen=True)
class CandidateDataset:
    X: NDArray[np.float32]
    y_gate: NDArray[np.int8]
    y_edge_bps: NDArray[np.float32]
    y_q10_bps: NDArray[np.float32]
    y_mfe_bps: NDArray[np.float32]
    sample_weight: NDArray[np.float32]
    groups: NDArray[np.int32]
    event_index: pd.DataFrame
    feature_names: tuple[str, ...]
    feature_schema_version: str = "candidate_v2"
```

### Required Event Columns After Labeling
`label_candidate_events()` must return:
- Existing columns: `gross_fwd_bps`, `ex_ante_cost_bps`, `edge_after_hurdle_bps`, `barrier_first_label`, `profitable_after_hurdle_label`, `triple_barrier_label`, `time_to_exit_bars`, `mae_bps`, `mfe_bps`, `realized_vol_bps`.
- New columns:
  - `exit_reason`: one of `take_profit`, `stop_loss`, `time_exit`, `invalid`.
  - `exit_idx`: integer bar index.
  - `exit_policy_version`: string.
  - `same_bar_collision`: int8, `1` when TP and SL hit the same bar offset.

### CandidateModelOutput Contract Extension
Keep existing fields and add optional diagnostics:

```python
@dataclass(slots=True, frozen=True)
class CandidateModelOutput:
    events: pd.DataFrame
    p_pass: NDArray[np.float64]
    mu_gross_bps: NDArray[np.float64]
    mu_net_decision_bps: NDArray[np.float64]
    q10_net_bps: NDArray[np.float64]
    q90_net_bps: NDArray[np.float64]
    utility_score: NDArray[np.float64]
    selection_thresholds: dict[str, float] = field(default_factory=dict)
```

## Step-By-Step Logic

### Phase 1: Stop Proving Alpha With Zero-Trade Outputs
1. Keep production hard thresholds as safety gates.
2. Add a validation-calibrated selection mode:
   - Compute validation utility distribution.
   - Select candidates in OOS if their utility is above the validation top-quantile threshold.
   - Still enforce catastrophic shortfall and liquidity/cost filters.
3. Log both hard-threshold pass count and validation-quantile pass count.

### Phase 2: Make Candidate Identity Learnable
1. Build feature schema from all labeled events before split masking.
2. Append one-hot features:
   - `family=<family>`
   - `variant=<family>:<variant>`
   - `side=<long|short>`
3. Ensure train, valid, full datasets have identical `feature_names`.
4. Add tests proving feature matrices align across splits even when one split lacks a variant.

### Phase 3: Add Causal Market-State Features
For each event at `t = entry_idx - 1`, add only trailing features:
- `btc_ret_1`, `btc_ret_5`, `btc_trend_20_100`
- `mkt_vol_20`, `mkt_vol_z120`
- `mkt_dispersion_20`, `mkt_dispersion_z120`
- `market_breadth_20`
- `symbol_ret_rank_20`
- `symbol_vol_z120`
- `funding_cross_section_z`
- `cost_to_vol_bps`

No feature may use `entry_idx` forward bars or OOS labels.

### Phase 4: Align Label Exit Policy With Execution
1. Keep conservative same-bar collision logic: same bar TP/SL means stop-loss first.
2. Add `exit_reason`, `exit_idx`, and `same_bar_collision`.
3. Decide implementation path:
   - Minimum viable path: set `exit_policy_mode="label_only"` and document that labels are research-only.
   - Preferred path: implement `exit_policy_mode="engine_aligned"` by passing per-event stop/take-profit metadata into the target/execution path.
4. If the engine cannot consume per-event take-profit, do not train selection as if per-event take-profit is enforced live.

### Phase 5: Promote Candidate Families Before ML Complexity
1. Run `compute_rule_diagnostics()` on labeled events.
2. Require OOS metrics before enabling a variant:
   - `oos_n >= min_variant_oos_obs`
   - `oos_mean_edge_bps >= min_variant_oos_edge_bps`
   - `oos_q10_shortfall_fail_rate <= max_variant_oos_q10_fail_rate`
   - `oos_rank_ic >= min_oos_rank_ic`
   - `edge_stability_bps >= -max_oos_edge_decay_bps`
3. Wire `recommended_keep_variants` and `recommended_flip_variants` into the ablation/full production candidate panel filter when `promotion_filter_enabled=True`.

### Phase 6: Change Final Selection From Gate-Only To Utility-First
1. Keep `p_pass` as meta-label confidence.
2. Fit edge and quantile models as current code does.
3. Compute utility:
   - `kelly_proxy = clipped_fractional_kelly(mu_net_decision_bps, variance)`
   - `expected_log_growth_proxy = p_pass * log1p(kelly_proxy * mu_net_return) - downside_penalty`
   - subtract turnover and concentration penalties.
4. Selection order:
   - reject catastrophic q10;
   - reject non-positive expected net edge;
   - rank by expected log-growth utility;
   - pick at most one event per `(datetime, symbol)`.

## Surgical Plan

### `src/domain/futures/strategy/config.py`
`[ACTION: REPLACE]`
- Add the config fields listed in `Add Config Fields`.
- Extend `__post_init__()` validation.

### `src/domain/futures/strategy/candidate_contracts.py`
`[ACTION: REPLACE]`
- Add `selection_thresholds` to `CandidateModelOutput`.
- Keep backward-compatible field order as much as possible; add the new field last with `default_factory=dict`.

### `src/domain/futures/strategy/candidate_labels.py`
`[ACTION: REPLACE]`
- Add `exit_reason`, `exit_idx`, `same_bar_collision`, `exit_policy_version`.
- Preserve current ATR logic and conservative same-bar collision.
- Add a unit test for same-bar collision.

### `src/domain/futures/strategy/candidate_dataset.py`
`[ACTION: REPLACE]`
- Add stable identity one-hot feature schema generated from full `labeled_events`.
- Add market-state feature builder using only `t = entry_idx - 1` and trailing windows.
- Do not use future returns or labels in features.
- Add `feature_schema_version`.

### `src/domain/futures/strategy/rule_diagnostics.py`
`[ACTION: REPLACE]`
- Add `oos_rank_ic`.
- Add `oos_log_growth_proxy` only if it can be computed from available event-level data without running the engine.
- Update recommendation thresholds to include rank IC and edge decay.

### `src/domain/futures/strategy/candidate_edge.py`
`[ACTION: REPLACE]`
- Keep models, but add validation threshold diagnostics.
- Include `selection_thresholds` in `CandidateModelOutput`.
- Ensure utility is logged by variant and regime bucket.

### `src/domain/futures/strategy/candidate_portfolio.py`
`[ACTION: REPLACE]`
- Add `validation_quantile` and `utility_topk` selection policies.
- Keep `hard` mode for regression compatibility.
- Continue enforcing catastrophic shortfall as a safety gate.

### `src/domain/futures/strategy/ablation.py`
`[ACTION: REPLACE]`
- Wire `promotion_filter_enabled` into candidate panel filtering.
- Add ablation rows:
  - `candidate_ml_identity_features`
  - `candidate_ml_market_state_features`
  - `candidate_ml_promotion_filter`
  - `candidate_ml_validation_quantile_selection`
- Require OOS-only evaluation for final rows.

### `src/domain/futures/backtest/engine.py` and `src/domain/futures/portfolio/execution_sim.py`
`[ACTION: ADD/REPLACE]`
- Only modify if implementing `exit_policy_mode="engine_aligned"`.
- Add per-symbol/per-bar stop/take-profit arrays if the execution simulator can safely consume them.
- If this is too large for one patch, keep it as a separate follow-up and force `exit_policy_mode="label_only"` in tests.

## Verification

### L1
Run after Python modifications:

```bash
uv run ruff check --fix src/domain/futures/strategy/config.py src/domain/futures/strategy/candidate_contracts.py src/domain/futures/strategy/candidate_labels.py src/domain/futures/strategy/candidate_dataset.py src/domain/futures/strategy/candidate_edge.py src/domain/futures/strategy/candidate_portfolio.py src/domain/futures/strategy/rule_diagnostics.py src/domain/futures/strategy/ablation.py
```

```bash
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/candidate_contracts.py src/domain/futures/strategy/candidate_labels.py src/domain/futures/strategy/candidate_dataset.py src/domain/futures/strategy/candidate_edge.py src/domain/futures/strategy/candidate_portfolio.py src/domain/futures/strategy/rule_diagnostics.py src/domain/futures/strategy/ablation.py
```

Expected:
- ruff: no remaining issues.
- mypy: no errors in modified files.

### L2
Run targeted regression:

```bash
uv run pytest tests/unit/domain/futures/strategy --tb=short
```

```bash
uv run pytest tests/unit/domain/futures/backtest/test_backtest_engine.py tests/unit/execution/test_opt_main_futures_strategy_mode.py --tb=short
```

Expected:
- Existing strategy tests pass.
- New tests prove split feature-schema alignment, same-bar stop/take collision handling, and validation-quantile selection.

### Quant Verification
Run a smoke strategy diagnostic after implementation:

```bash
uv run python src/execution/opt_main_futures.py --mode strategy-smoke --skip-universe --skip-data-sync --symbols BTCUSDT --trials 1 --tf 4h --reference-date 2026-05-01 --strategy candidate_ml
```

Expected:
- Logs include hard-threshold pass counts and validation-quantile pass counts.
- No selected candidate is formed from future features.
- Final OOS evaluation is non-leaky and post-cost.

## Acceptance Criteria
Implementation is ready only when:
- Candidate identity features are present and schema-aligned across train/valid/full splits.
- Market-state features are causal and tested.
- Rule promotion can filter candidate variants before ML selection.
- Selection no longer depends only on a fixed `p_pass >= 0.55` gate.
- Label exit policy is either execution-aligned or explicitly marked label-only.
- Strategy ablation reports whether each added layer improves OOS compound metrics after costs.

## Implementation Priority
1. Candidate identity features and schema alignment.
2. Market-state features.
3. Promotion filter wiring.
4. Validation-calibrated utility selection.
5. Exit-policy engine alignment.

Do not implement large candidate-grid expansion before the promotion/filtering layer works. More candidates without deflated validation increase overfit risk and can make alpha extraction worse.
