# Re-Alpha Practical Alpha Extraction Spec

## Spec Type
- `prd` + `refactor`
- Goal: turn current `paper` alpha into an `ALPHA_PASS=True` candidate by preserving rank skill after selection and portfolio caps.

## Current Evidence
- Latest result: `ALPHA_PASS=FALSE` with single blocker `signal_lost_after_selection`.
- Current skill metrics are already strong:
  - `gating_ic=0.0347`
  - `RESID_IC=0.0425`
  - `T-STAT=2.22`
  - `DSR=0.9804`
  - `EXEC_DIAG=PASS`
  - `PROMOTION=paper`
- Failure is downstream:
  - `clip_preservation_ratio < 0.70`
  - Current report says `presv=0.46`.
  - `clip_preservation_ratio = post_clip_IC / pre_clip_IC` in `evaluate_alpha()`.

## Design Decision
Do not start with more model complexity. The current blocker is not raw rank skill. The first production-grade improvement must make policy calibration optimize the same quantity that gates production: post-cap signal preservation.

MHE and hybrid rank/regression blending are explicitly not baseline candidates because the latest isolation result showed OOS IC collapse to `0.0096`.

## Target Files
- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/rank_selection.py`
- `src/domain/futures/strategy/ml_builder.py`
- `src/domain/futures/strategy/alpha_evaluation.py`
- `src/execution/opt_main_futures.py`
- `tests/unit/domain/futures/strategy/test_rank_selection.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`
- `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`

## Existing Contracts Verified

### `StrategyMLConfig`
Path: `src/domain/futures/strategy/config.py`

Current relevant fields:
```python
soft_beta_neutralize: bool = True
soft_beta_neutralize_weight: float = 0.40
smoothing_method: Literal["ema", "dema"] = "dema"
adaptive_smoothing: bool = False
rank_policy_max_abs_net_exposure: float = 0.05
rank_policy_max_abs_beta_exposure: float = 0.20
rank_policy_selection_modes: tuple[Literal["tail", "soft_cs"], ...] = ("soft_cs", "tail")
rank_policy_weighting: Literal["equal", "zscore", "tanh"] = "tanh"
```

### `RankSelectionPolicy`
Path: `src/domain/futures/strategy/rank_selection.py`

Current relevant fields:
```python
soft_beta_neutralize: bool = False
soft_beta_neutralize_weight: float = 0.7
smoothing_method: Literal["ema", "dema"] = "dema"
adaptive_smoothing: bool = False
```

Gap: `calibrate_rank_portfolio_policy()` does not accept or pass these config values into candidate `RankSelectionPolicy` objects. Therefore current policy calibration can silently ignore the intended pre-cap beta reduction and smoothing settings.

### `calibrate_rank_portfolio_policy`
Path: `src/domain/futures/strategy/rank_selection.py`

Current signature:
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
    max_abs_net_exposure: float = 0.10,
    max_abs_beta_exposure: float = 0.20,
) -> RankSelectionPolicy:
```

### `evaluate_alpha`
Path: `src/domain/futures/strategy/alpha_evaluation.py`

Current signature:
```python
def evaluate_alpha(
    *,
    alpha_long_2d: NDArray[np.float64],
    alpha_short_2d: NDArray[np.float64],
    realized_fwd_ret_2d: NDArray[np.float64],
    inference_signed_2d: NDArray[np.float64] | None = None,
    q10_2d: NDArray[np.float64] | None = None,
    q90_2d: NDArray[np.float64] | None = None,
    q50_2d: NDArray[np.float64] | None = None,
    btc_close_1d: NDArray[np.float64] | None = None,
    net_daily_returns: NDArray[np.float64] | None = None,
    cost_floor_bps: float = 24.0,
    n_trials: int = 1,
    horizon_bars: int = 6,
    basket_quantile: float = 0.35,
    trading_mask: NDArray[np.bool_] | None = None,
    policy_validation_net_lcb_bps: float = float("nan"),
    policy_validation_gross_bps: float = float("nan"),
    policy_validation_ir_t: float = float("nan"),
    policy_validation_monotonicity: float = float("nan"),
    policy_validation_turnover: float = float("nan"),
    policy_validation_cost_bps: float = float("nan"),
    policy_validation_breadth: float = float("nan"),
    policy_selection_mode: str = "",
    policy_no_trade: bool = False,
    policy_min_breadth: float = 8.0,
    policy_max_turnover: float = 1.25,
    policy_max_cost_bps: float | None = None,
    promotion_min_oos_folds: int = 2,
    observed_oos_folds: int = 0,
) -> AlphaEvaluationReport:
```

Current `clip_preservation_ratio` contract:
```python
clip_preservation_ratio = net_ic_dict["mean_ic"] / _gating_ic
```

### `build_ml_strategy_alpha`
Path: `src/domain/futures/strategy/ml_builder.py`

Current signature:
```python
def build_ml_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
    trading_symbols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
```

### `build_ml_strategy_alpha_anchored`
Path: `src/domain/futures/strategy/ml_builder.py`

Current signature:
```python
def build_ml_strategy_alpha_anchored(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
    anchor_end_idx: int,
    target_start: int,
    target_end: int,
    precomputed_panels: AnchoredMLPrecomputedPanels | None = None,
    trading_symbols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
```

## Contract Changes

### Add `StrategyMLConfig` Fields
```python
rank_policy_min_clip_preservation: float = 0.70
rank_policy_preservation_weight: float = 12.0
rank_policy_post_ic_weight: float = 100.0
rank_policy_allow_preservation_fallback: bool = True
rank_policy_soft_beta_weights: tuple[float, ...] = (0.0, 0.25, 0.40, 0.60)
```

Validation:
```python
if not (0.0 <= self.rank_policy_min_clip_preservation <= 2.0):
    raise ValueError("rank_policy_min_clip_preservation must satisfy 0 <= value <= 2")
if self.rank_policy_preservation_weight < 0.0:
    raise ValueError("rank_policy_preservation_weight must be >= 0")
if self.rank_policy_post_ic_weight < 0.0:
    raise ValueError("rank_policy_post_ic_weight must be >= 0")
if len(self.rank_policy_soft_beta_weights) == 0:
    raise ValueError("rank_policy_soft_beta_weights must be non-empty")
if any(w < 0.0 or w > 1.0 for w in self.rank_policy_soft_beta_weights):
    raise ValueError("rank_policy_soft_beta_weights must contain values in [0, 1]")
```

### Extend `RankSelectionPolicy`
```python
validation_pre_ic: float = float("nan")
validation_post_ic: float = float("nan")
validation_clip_preservation: float = float("nan")
validation_objective: float = float("nan")
```

Update `policy_to_dict()` and `policy_from_dict()` to round-trip these fields.

### Extend `calibrate_rank_portfolio_policy`
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
    max_abs_net_exposure: float = 0.10,
    max_abs_beta_exposure: float = 0.20,
    min_clip_preservation: float = 0.70,
    preservation_weight: float = 12.0,
    post_ic_weight: float = 100.0,
    allow_preservation_fallback: bool = True,
    smoothing_method: Literal["ema", "dema"] = "dema",
    adaptive_smoothing: bool = False,
    soft_beta_neutralize: bool = False,
    soft_beta_weights: tuple[float, ...] = (0.0,),
) -> RankSelectionPolicy:
```

Keep `calibrate_rank_selection_policy()` backward-compatible by relying on defaults.

## Step-by-Step Logic

### Phase 1: Cap-Aware Policy Calibration
1. Add helper `_mean_spearman_ic_2d(signal, realized, eligible) -> float`.
2. Use bar-wise Spearman rank IC; skip bars with fewer than 5 finite assets.
3. In `_estimate_policy_metrics()`:
   - Existing `score` argument is pre-cap rank signal.
   - Add:
     - `validation_pre_ic = _mean_spearman_ic_2d(score, realized, eligible)`
     - `validation_post_ic = _mean_spearman_ic_2d(weights, realized, eligible)`
     - `validation_clip_preservation = validation_post_ic / validation_pre_ic` when `abs(pre_ic) > 1e-9`, else `nan`.
4. In `calibrate_rank_portfolio_policy()`:
   - Iterate `soft_beta_weight` over `soft_beta_weights`.
   - Candidate policy must include:
     - `smoothing_method`
     - `adaptive_smoothing`
     - `soft_beta_neutralize=soft_beta_neutralize and soft_beta_weight > 0.0`
     - `soft_beta_neutralize_weight=soft_beta_weight`
   - Keep existing economic filters:
     - validation net LCB > 0
     - monotonicity > 0
     - breadth >= target
     - turnover <= max
     - abs net <= cap
     - abs beta <= cap
   - Add preservation filter:
     - If `validation_clip_preservation >= min_clip_preservation`, candidate is eligible.
     - If no candidate passes and `allow_preservation_fallback=True`, choose the highest preservation candidate only if:
       - `validation_post_ic > 0`
       - `validation_net_lcb_bps > 0`
       - `validation_breadth >= target_breadth_min`
5. Objective:
```python
objective = (
    metrics["validation_net_lcb_bps"]
    + 0.10 * metrics["validation_gross_bps"]
    + 0.05 * min(metrics["validation_breadth"], float(target_breadth_min))
    - 0.25 * metrics["validation_cost_bps"]
    + post_ic_weight * metrics["validation_post_ic"]
    + preservation_weight * min(metrics["validation_clip_preservation"], 1.25)
)
```
6. Store `validation_objective` in the selected policy.

### Phase 2: Wire Config Into Calibration
In both `build_ml_strategy_alpha()` and `build_ml_strategy_alpha_anchored()`, pass these new arguments:
```python
min_clip_preservation=float(ml_cfg.rank_policy_min_clip_preservation),
preservation_weight=float(ml_cfg.rank_policy_preservation_weight),
post_ic_weight=float(ml_cfg.rank_policy_post_ic_weight),
allow_preservation_fallback=bool(ml_cfg.rank_policy_allow_preservation_fallback),
smoothing_method=ml_cfg.smoothing_method,
adaptive_smoothing=bool(ml_cfg.adaptive_smoothing),
soft_beta_neutralize=bool(ml_cfg.soft_beta_neutralize),
soft_beta_weights=tuple(float(w) for w in ml_cfg.rank_policy_soft_beta_weights),
```

### Phase 3: Expose Diagnostics
1. Add policy fields to `alpha_panel.attrs["rank_selection_policy"]`.
2. In `opt_main_futures.py`, log:
   - `validation_pre_ic`
   - `validation_post_ic`
   - `validation_clip_preservation`
   - `soft_beta_neutralize`
   - `soft_beta_neutralize_weight`
3. Do not relax `ALPHA_PASS` threshold. The target remains `clip_preservation_ratio >= 0.70`.

## Alternative Improvements Considered

### Long/Short Model Separation
Status: candidate after Phase 1.

Current flow effectively uses a signed single ranker and derives long/short surfaces from the same dense score. If Phase 1 does not reach OOS `clip_preservation_ratio >= 0.70`, add:
```python
ranker_side_mode: Literal["signed_single", "separate_long_short"] = "signed_single"
```

Implementation rule:
- Train separate `ranker_long` and `ranker_short` only inside each fold.
- Never fit side-specific transforms on full panel.
- Promote only when side separation improves:
  - C3 post-clip IC
  - validation clip preservation
  - basket net LCB
  - bear basket safety

### Model Ensemble
Status: deferred ablation.

Allowed:
- Same-family LightGBM seed ensemble already exists through `ensemble_seeds`.
- CatBoost can be tested because `catboost>=1.2.0` exists in `pyproject.toml`.

Rejected as baseline:
- MHE: current experiment showed OOS IC collapse.
- Hybrid rank/regression blend: current evidence says it overfits this pipeline.

Promotion rule:
- Any ensemble must be selected by walk-forward validation using post-cap IC and clip preservation, not by raw rank IC.

### Feature Engineering
Status: useful, but secondary to cap-aware calibration.

Allowed low-risk additions:
- Trailing-only beta instability features.
- Liquidity/cost regime interaction features.
- Rank-score volatility and dispersion features.
- Funding/basis carry persistence interaction features.

PCA:
- Allowed only as fold-local transformer or strictly trailing rolling PCA.
- Do not fit PCA on the full precomputed feature panel because that leaks OOS covariance structure.

t-SNE:
- Do not use for production alpha features.
- Reason: non-parametric, unstable OOS mapping, expensive, and not naturally point-in-time safe.

UMAP:
- Same restriction as t-SNE unless a fitted transformer is trained fold-locally and validated under purge/embargo. Not a first-line candidate.

### LightGBM Objective / Smoothing
Status: ablation only after Phase 1.

Recommended tuning grid:
- Keep `objective="lambdarank"` as baseline.
- Sweep:
  - `ranker_lambda_l2`: `(20.0, 30.0, 50.0)`
  - `ranker_reg_alpha`: `(3.0, 5.0, 8.0)`
  - `num_leaves`: `(5, 7, 9)`
  - `ranker_feature_fraction`: `(0.65, 0.80)`
  - `ranker_bagging_fraction`: `(0.70, 0.80)`
  - `rank_policy_soft_beta_weights`: `(0.0, 0.25, 0.40, 0.60)`
  - `rank_policy_quantiles`: `(0.20, 0.25, 0.30, 0.35)`
  - `rank_policy_holding_candidates`: `(12, 18)`

Do not optimize on raw OOS IC alone. Required objective is:
```text
PASS if:
  signal_skill_passes=True
  portfolio_ic_above_breakeven=True
  basket_net_positive=True
  signal_preserved_after_selection=True
  multi_horizon_sweep_passes=True
  bear_market_basket_safe=True
```

## Surgical Plan

### `src/domain/futures/strategy/config.py`
- ACTION: ADD fields listed in "Contract Changes".
- ACTION: ADD validation checks in `StrategyMLConfig.__post_init__`.

### `src/domain/futures/strategy/rank_selection.py`
- ACTION: ADD `_mean_spearman_ic_2d()`.
- ACTION: EXTEND `RankSelectionPolicy` with validation IC/preservation fields.
- ACTION: EXTEND `policy_to_dict()` / `policy_from_dict()`.
- ACTION: EXTEND `_estimate_policy_metrics()` to return pre/post IC and preservation.
- ACTION: EXTEND `calibrate_rank_portfolio_policy()` signature and candidate search.
- ACTION: KEEP `calibrate_rank_selection_policy()` backward-compatible.

### `src/domain/futures/strategy/ml_builder.py`
- ACTION: REPLACE both `calibrate_rank_portfolio_policy()` call sites to pass new config fields.
- ACTION: Ensure fallback policy also receives smoothing and soft-beta settings.

### `src/execution/opt_main_futures.py`
- ACTION: ADD diagnostic logging for policy pre/post IC and preservation.
- ACTION: Do not change `signal_preserved_after_selection` threshold.

### Tests
- ACTION: ADD test that config soft-beta/smoothing fields are serialized into selected policy.
- ACTION: ADD test that policy calibration prefers higher post-cap IC over higher raw net LCB when both are profitable.
- ACTION: ADD test that policy dict round-trips new fields.
- ACTION: ADD regression test that `clip_preservation_ratio` remains computed as post/pre IC.

## Verification

Run L1 checks for modified Python files:
```bash
uv run ruff check --fix src/domain/futures/strategy/config.py src/domain/futures/strategy/rank_selection.py src/domain/futures/strategy/ml_builder.py src/domain/futures/strategy/alpha_evaluation.py src/execution/opt_main_futures.py
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/rank_selection.py src/domain/futures/strategy/ml_builder.py src/domain/futures/strategy/alpha_evaluation.py src/execution/opt_main_futures.py
```

Run targeted tests:
```bash
uv run pytest tests/unit/domain/futures/strategy/test_rank_selection.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/strategy/test_alpha_evaluation.py tests/unit/execution/test_opt_main_futures_strategy_mode.py --tb=short
```

Run full unit suite:
```bash
uv run pytest tests/unit --tb=short
```

Run alpha smoke after tests pass:
```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
```

Expected outcome:
- Unit tests pass.
- `EXEC_DIAG=PASS`.
- `PROMOTION: stage=paper` or better.
- `clip_preservation_ratio >= 0.70`.
- `ALPHA_PASS=TRUE`.

## Risk
- A strict preservation gate may select lower-return policies. The fallback must require positive post-cap IC and positive validation net LCB.
- Soft beta neutralization can reduce raw edge if overweighted. Search over weights and select by validation post-cap IC.
- PCA/t-SNE-style transforms can leak OOS structure if fit globally. They are excluded from the first implementation.
