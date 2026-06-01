# Compound ML Strategy Architecture Spec

> last_verified: 2026-06-01

## Spec Type
- `prd` + `refactor`
- Goal: replace the current ML alpha ranking stack with a candidate-driven strategy architecture that optimizes post-cost geometric capital growth in Binance futures backtests.

## Strategic Decision
The existing ML alpha path must be retired:
- no direct multi-symbol LambdaRank alpha extraction
- no rank-selection policy as the production decision layer
- no alpha pass based primarily on IC
- no ML module controlling leverage or orders

The new architecture uses this flow:
```text
PIT universe filter
  -> per-symbol rule candidate generation
  -> per-symbol candidate backtest/evaluation
  -> ML trade gate
  -> ML edge/downside estimator
  -> fractional Kelly portfolio constructor
  -> existing intrabar futures backtest engine
  -> OOS block promotion and ablation
```

Primary objective:
```text
maximize mean(log(equity_t / equity_{t-1}))
subject to liquidation-free survival, drawdown, turnover, capacity, beta, gross, net, and per-symbol caps
```

Secondary diagnostics:
- CAGR
- MAR / Calmar
- max drawdown
- post-cost net bps
- turnover
- block pass ratio
- DSR / PBO / bootstrap robustness
- IC only as a secondary signal-quality diagnostic

## Current Contracts Verified

### Backtest Engine
Path: `src/domain/futures/backtest/engine.py`

Current production entry:
```python
class FuturesBacktestEngine:
    @staticmethod
    def run_multi(
        aligned_data: dict[str, np.ndarray],
        symbol_names: list[str],
        strategy_params: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, np.ndarray, float, np.ndarray]:
```

Relevant behavior:
- `PortfolioBacktestEngine.run()` consumes `aligned_data["target_weights"]` when present.
- If `target_weights` is absent, it falls back to legacy `xs_score_long/xs_score_short`.
- Therefore new architecture must inject `target_weights` directly and avoid the legacy cross-sectional score fallback.

### Execution Semantics
Paths:
- `docs/architecture/backtest-logic.md`
- `docs/architecture/backtest-engine.md`
- `src/domain/futures/portfolio/execution_sim.py`

Hard constraints:
- T signal can only execute from T+1.
- execution is target-weight driven.
- intrabar 1m path applies conservative fill semantics.
- costs are single-source through settings/friction/execution layers.
- drawdown scaling is path-aware only inside execution simulation.
- ML must remain an alpha supplier, not an order/leverage controller.

### Portfolio Constructor
Path: `src/domain/futures/portfolio/portfolio_constructor.py`

Current reusable contracts:
```python
def precompute_rebalance_weights(
    close_2d: np.ndarray,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    *,
    rebalance_bars: int,
    lookback: int,
    bars_per_year: float,
    kappa: float,
    f_kelly_max: float,
    sigma_target_ann: float,
    gross_cap: float,
    per_symbol_cap: float,
    current_dd: float = 0.0,
    min_obs: int = 20,
    composer_sigma_2d: np.ndarray | None = None,
    sigma_3d: np.ndarray | None = None,
    risk_snapshot: RiskSnapshot | None = None,
    btc_beta_2d: np.ndarray | None = None,
    policy_inputs: PortfolioPolicyInputs | None = None,
    use_residual_var_for_kelly: bool = False,
) -> np.ndarray:
```

```python
@dataclass(frozen=True)
class PortfolioCaps:
    gross: float = 3.0
    per_symbol: float = 0.10
    net: float = 0.30
    beta: float = 0.50
    target_ann_vol: float = 0.20
```

```python
def project_all_caps(
    w: np.ndarray,
    btc_beta: np.ndarray,
    sigma_port: float,
    bars_per_year: float,
    caps: PortfolioCaps | None = None,
) -> np.ndarray:
```

Decision:
- reuse `project_all_caps()`, `quantize_weights()`, covariance utilities, and the execution engine.
- add a candidate-specific portfolio layer that produces `target_weights`.
- do not let the ML router emit final weights.

### Universe Filter
Paths:
- `src/domain/futures/universe/config.py`
- `src/domain/futures/universe/selection.py`
- `src/domain/futures/universe/pipeline.py`
- `src/domain/futures/universe/models.py`

Current Stage6 behavior:
```python
def apply_selection_stage(
    frame: pd.DataFrame,
    *,
    config: Stage6Config | None = None,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    previous_selection: tuple[str, ...] | None = None,
    k_in: int = DEFAULT_K_IN,
    k_out: int = DEFAULT_K_OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

Current Stage6 ranks by `tradeable_score` and selects `k_in`.

Decision:
- keep Stage0 to Stage5 as PIT tradability and risk filters.
- change the role of Stage6 from "alpha trading rank" to "execution pool and capacity metadata".
- per-symbol strategy evaluation must run on Stage5-passed or Stage6-eligible symbols independently.
- final portfolio selection is done by candidate expected utility, not universe rank.

### Existing ML Alpha Legacy Scope
Move to `legacy/` folder (to completely separate rather than delete) after new architecture is green:
- `src/domain/futures/strategy/ml_builder.py` -> `legacy/strategy/ml_builder.py`
- `src/domain/futures/strategy/ranker.py` -> `legacy/strategy/ranker.py`
- `src/domain/futures/strategy/calibrator.py` -> `legacy/strategy/calibrator.py`
- `src/domain/futures/strategy/rank_selection.py` -> `legacy/strategy/rank_selection.py`
- `src/domain/futures/strategy/alpha_evaluation.py` -> `legacy/strategy/alpha_evaluation.py`
- `src/domain/futures/strategy/features.py` -> `legacy/strategy/features.py`
- `src/domain/futures/strategy/labels.py` -> `legacy/strategy/labels.py`
- `src/domain/futures/strategy/dataset.py` -> `legacy/strategy/dataset.py`
- `src/domain/futures/strategy/inference.py` -> `legacy/strategy/inference.py`
- `src/domain/futures/strategy/cache.py` -> `legacy/strategy/cache.py`
- old `StrategyMLConfig`
- old tests named `test_ml_*`, `test_rank_selection.py`, `test_alpha_evaluation.py`

Do not delete these foundations:
- `src/domain/futures/strategy/common/alignment.py`
- `src/domain/futures/strategy/common/normalization.py`
- `src/domain/futures/strategy/diagnostics.py`, if generic IC/return diagnostics are still useful
- `src/domain/futures/portfolio/*`
- `src/domain/futures/backtest/*`
- `src/domain/futures/universe/*`

## New Target Files

### Strategy Layer
- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/builder.py`
- `src/domain/futures/strategy/contracts.py`
- `src/domain/futures/strategy/candidate_contracts.py`
- `src/domain/futures/strategy/rule_signals.py`
- `src/domain/futures/strategy/candidate_labels.py`
- `src/domain/futures/strategy/candidate_dataset.py`
- `src/domain/futures/strategy/candidate_gate.py`
- `src/domain/futures/strategy/candidate_edge.py`
- `src/domain/futures/strategy/candidate_portfolio.py`
- `src/domain/futures/strategy/candidate_backtest.py`
- `src/domain/futures/strategy/candidate_evaluation.py`
- `src/domain/futures/strategy/ablation.py`

### Runtime Layer
- `src/domain/futures/strategy_runtime/bridge.py`
- `src/execution/opt_main_futures.py`

### Universe Layer
- `src/domain/futures/universe/config.py`
- `src/domain/futures/universe/selection.py`
- `src/domain/futures/universe/models.py`
- `src/domain/futures/universe/pipeline.py`

### Forecast Compatibility
- `src/domain/futures/forecast/contracts.py` — CostForecast, RiskForecast (유효)
- `src/domain/futures/forecast/cost.py` — build_cost_forecast (유효)
- `src/domain/futures/forecast/risk.py` — build_risk_forecast (유효)

### Tests
- `tests/unit/domain/futures/strategy/test_rule_signals.py`
- `tests/unit/domain/futures/strategy/test_candidate_labels.py`
- `tests/unit/domain/futures/strategy/test_candidate_dataset.py`
- `tests/unit/domain/futures/strategy/test_candidate_gate.py`
- `tests/unit/domain/futures/strategy/test_candidate_edge.py`
- `tests/unit/domain/futures/strategy/test_candidate_portfolio.py`
- `tests/unit/domain/futures/strategy/test_candidate_backtest.py`
- `tests/unit/domain/futures/strategy/test_candidate_evaluation.py`
- `tests/unit/domain/futures/strategy/test_ablation.py`
- `tests/unit/domain/futures/universe/test_strategy_pool_selection.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py`

## Technology Stack Decision

### Keep
- `numpy`: vectorized arrays and Numba inputs
- `pandas`: time-index alignment and result tables
- `numba`: execution loops and heavy array transforms where needed
- `scikit-learn`: calibration, metrics, Ledoit-Wolf covariance already present
- `lightgbm`: first-line tabular ML for gate and edge models
- `catboost`: optional second-line benchmark, not first implementation
- `optuna`: hyperparameter optimization, only after fixed OOS protocol exists

### Do Not Add Initially
- PyTorch / TensorFlow: not justified for tabular candidate data and increases overfit surface
- RL libraries: wrong first tool because reward is sparse, non-stationary, and easy to overfit to simulator mechanics
- t-SNE / UMAP / PCA production transforms: not point-in-time safe unless fitted fold-locally and still not central to the objective
- vectorbt/backtrader: existing engine has stricter futures semantics and must remain source of truth

### ML Algorithms
Phase 3 gate:
- `lightgbm.LGBMClassifier`
- objective: `binary`
- calibration: validation-fold Platt or isotonic using `sklearn.calibration.CalibratedClassifierCV` only on fold-local validation data

Phase 4 edge:
- `lightgbm.LGBMRegressor`
- objectives:
  - `regression_l1` or `huber` for robust center
  - `quantile` for q10 downside and q90 upside
- final score uses expected utility, not raw predicted return.

Why LightGBM:
- strong on tabular financial features
- handles nonlinear interactions
- handles missing values
- deterministic with fixed seed
- fast enough for walk-forward and ablation
- already in `pyproject.toml`

## Core Data Contracts

### `CandidateStrategyConfig`
Path: `src/domain/futures/strategy/config.py`

```python
@dataclass(slots=True, frozen=True)
class CandidateStrategyConfig:
    name: Literal["candidate_ml"] = "candidate_ml"
    timeframe: str = "4h"
    seed: int = 42
    train_months: int = 24
    valid_months: int = 3
    test_months: int = 6
    purge_bars: int = 18
    embargo_bars: int = 18
    cost_floor_bps: float = 24.0
    min_listing_age_days: int = 90
    min_candidate_obs: int = 200
    min_symbol_oos_blocks: int = 3
    min_rule_net_bps: float = 0.0
    min_rule_ir_t: float = 1.0
    min_rule_hit_rate: float = 0.50
    max_rule_turnover_per_bar: float = 0.50
    max_symbol_weight: float = 0.10
    gross_cap: float = 1.20
    net_cap: float = 0.30
    beta_cap: float = 0.50
    target_ann_vol: float = 0.35
    kelly_fraction: float = 0.25
    min_gate_probability: float = 0.55
    min_expected_net_bps: float = 1.0
    max_expected_shortfall_bps: float = 80.0
    candidate_families: tuple[str, ...] = (
        "trend_ma",
        "trend_donchian",
        "vol_breakout",
        "bollinger_reversion",
        "rsi_reversion",
        "funding_carry",
        "oi_volume_impulse",
        "btc_regime_pullback",
    )
```

Validation:
- all month and bar windows must be positive.
- `purge_bars >= max_label_horizon_bars`.
- `embargo_bars >= max_label_horizon_bars`.
- `0 < kelly_fraction <= 0.25`.
- `cost_floor_bps >= 0`.
- cap values must be non-negative and gross cap must be at least per-symbol cap.

### `StrategyConfig`
Path: `src/domain/futures/strategy/config.py`

Replace:
```python
ml: StrategyMLConfig = field(default_factory=lambda: StrategyMLConfig())
```

With:
```python
candidate: CandidateStrategyConfig = field(default_factory=CandidateStrategyConfig)
```

Change allowed names:
```python
if self.name not in {"candidate_ml", "rule_baseline", "momentum", "xs_reversal"}:
    raise ValueError(...)
```

### `CandidateSignalPanel`
Path: `src/domain/futures/strategy/candidate_contracts.py`

```python
@dataclass(slots=True, frozen=True)
class CandidateSignalPanel:
    family: str
    variant: str
    params: dict[str, float | int | str]
    datetimes: np.ndarray
    symbols: tuple[str, ...]
    signed_score_2d: NDArray[np.float64]
    side_hint_2d: NDArray[np.int8]
    expected_holding_bars: int
    min_holding_bars: int
    stop_atr_mult: float
    take_profit_atr_mult: float
    turnover_proxy_2d: NDArray[np.float64]
    valid_mask_2d: NDArray[np.bool_]
```

Semantics:
- `signed_score_2d > 0`: long candidate strength
- `signed_score_2d < 0`: short candidate strength
- `side_hint_2d`: `1` long, `-1` short, `0` no candidate
- output at T is based only on data up to T
- entry can only occur at T+1 open

### `CandidateEventFrame`
Path: `src/domain/futures/strategy/candidate_contracts.py`

Use a `pd.DataFrame` with columns:
```text
datetime
symbol
family
variant
side
raw_score
score_z
expected_holding_bars
min_holding_bars
stop_atr_mult
take_profit_atr_mult
turnover_proxy
cost_floor_bps
entry_idx
```

Index policy:
- `datetime` is decision timestamp T.
- `entry_idx = t + 1`.
- labels and backtest use T+1 execution, never T close fill.

### `CandidateLabelFrame`
Path: `src/domain/futures/strategy/candidate_labels.py`

Add columns to candidate event frame:
```text
gross_fwd_bps
ex_ante_cost_bps
edge_after_hurdle_bps
triple_barrier_label
time_to_exit_bars
mae_bps
mfe_bps
realized_vol_bps
```

Important B1 rule:
- `edge_after_hurdle_bps` is a decision label only.
- emitted alpha/target weights must not subtract this cost again as realized PnL.
- final execution cost is charged once by the backtest engine.

### `CandidateDataset`
Path: `src/domain/futures/strategy/candidate_dataset.py`

```python
@dataclass(slots=True, frozen=True)
class CandidateDataset:
    X: NDArray[np.float32]
    y_gate: NDArray[np.int8]
    y_edge_bps: NDArray[np.float32]
    y_q10_bps: NDArray[np.float32]
    sample_weight: NDArray[np.float32]
    groups: NDArray[np.int32]
    event_index: pd.DataFrame
    feature_names: tuple[str, ...]
```

### `CandidateModelOutput`
Path: `src/domain/futures/strategy/candidate_contracts.py`

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
```

### `CandidatePortfolioResult`
Path: `src/domain/futures/strategy/candidate_portfolio.py`

```python
@dataclass(slots=True, frozen=True)
class CandidatePortfolioResult:
    alpha_panel: pd.DataFrame
    target_weights_2d: NDArray[np.float64]
    selected_events: pd.DataFrame
    diagnostics: dict[str, float | int | str]
```

`alpha_panel` columns:
```text
alpha_long
alpha_short
target_weight
candidate_family
candidate_variant
p_pass
mu_net_decision_bps
q10_net_bps
utility_score
```

## Phase 1: Rule Baseline

### Purpose
Build a real non-ML control group and candidate generator. This is the most important phase because every ML result must beat this baseline.

### Rule Families For Coin Futures

#### 1. Trend MA Cross With Hysteresis
Use:
- EMA fast: 6, 12, 18 bars
- EMA slow: 36, 72, 108 bars
- hysteresis band: 0.25 to 0.75 ATR-normalized score
- minimum holding: 6 to 12 bars

Signal:
```text
score = (ema_fast - ema_slow) / ATR
long if score > enter_threshold
short if score < -enter_threshold
flat only if abs(score) < exit_threshold
```

Why:
- crypto futures trend regimes persist after large moves.
- hysteresis reduces churn and helps the 24bps cost wall.
- ATR normalization makes scores comparable across symbols without ranking.

Risks:
- late entry after trend exhaustion
- high loss during chop
- crowded public signal

Mitigation:
- regime filter using BTC trend and realized volatility
- no-trade during low dispersion chop
- ATR stop and time barrier

#### 2. Donchian Breakout
Use:
- high/low lookback: 18, 36, 72 bars
- confirmation: close above channel and volume above trailing median
- stop: 1.5 to 3.0 ATR
- time barrier: 12 to 36 bars

Signal:
```text
long if close_t > max(high_{t-L:t-1})
short if close_t < min(low_{t-L:t-1})
score = distance_from_channel / ATR
```

Why:
- crypto futures often trend after range breaks.
- works better than MA cross in sudden volatility expansion.

Risks:
- false breakouts near funding events
- exchange-specific wick noise

Mitigation:
- require close confirmation, not intrabar high/low
- volume or range expansion filter
- use T close signal and T+1 open execution only

#### 3. Volatility Compression Breakout
Use:
- Bollinger bandwidth percentile over 90 to 180 bars
- trigger when bandwidth is below p20 then price exits range
- direction from breakout side

Signal:
```text
compression = bb_width_percentile < 0.20
breakout_up = close > upper_band
breakout_down = close < lower_band
```

Why:
- crypto volatility clusters. Low-vol compression can precede strong expansion.
- better suited for futures than pure spot because short side is available.

Risks:
- compression can persist.
- breakout may reverse immediately.

Mitigation:
- time stop
- MFE/MAE diagnostics
- require market breadth confirmation

#### 4. Bollinger Mean Reversion
Use only in non-trending regimes:
- z-window: 18, 36 bars
- entry z: 1.5, 2.0, 2.5
- exit z: 0.25 to 0.50

Signal:
```text
z = (close - rolling_mean) / rolling_std
long if z < -entry_z and market_regime != trend_down_panic
short if z > entry_z and market_regime != trend_up_breakout
```

Why:
- many alt futures mean-revert after liquidity shocks.
- can be high hit-rate when filtered by regime.

Risks:
- catches falling knives in liquidation cascades.
- short squeeze risk.

Mitigation:
- disallow long mean reversion when BTC trend and breadth are both strongly down.
- disallow short mean reversion during broad market breakout.
- ATR stop and liquidation-aware stop distance.

#### 5. RSI Reversion
Use:
- RSI window: 6, 12, 18 bars
- long threshold: 20, 30
- short threshold: 70, 80
- confirmation: RSI turns back through threshold

Signal:
```text
long if rsi_{t-1} < low_threshold and rsi_t > rsi_{t-1}
short if rsi_{t-1} > high_threshold and rsi_t < rsi_{t-1}
```

Why:
- avoids entering purely because something is oversold.
- confirmation reduces one-way liquidation losses.

Risks:
- repeated entries in strong trend.

Mitigation:
- cooldown bars after stop
- trend regime veto
- no averaging down

#### 6. Funding Carry
Use:
- funding rate level
- funding z-score
- funding persistence over 3 to 12 funding events
- price trend veto

Signal:
```text
short if funding is high positive and price trend is weak/down
long if funding is high negative and price trend is weak/up
```

Why:
- futures-specific alpha candidate.
- directly monetizes crowded positioning.

Risks:
- high funding can persist during strong trend, causing adverse price move.

Mitigation:
- carry trades are forbidden against strong trend.
- require expected funding benefit to exceed price volatility hurdle.
- cap size more aggressively than trend trades.

#### 7. OI/Volume Impulse
Use if columns are available:
- open interest return
- volume shock
- range expansion
- price direction

Signal:
```text
long if price_up and oi_up and volume_shock and not overextended
short if price_down and oi_up and volume_shock and not oversold
```

Why:
- new positioning plus volume confirms directional pressure.

Risks:
- data quality and missing OI.
- liquidation noise.

Mitigation:
- feature availability mask.
- fallback to no signal when OI is missing.

#### 8. BTC Regime Pullback
Use:
- BTC trend filter
- symbol pullback z-score
- relative strength vs BTC

Signal:
```text
long alt pullback in BTC uptrend if symbol relative strength remains positive
short weak symbol bounce in BTC downtrend if relative strength remains negative
```

Why:
- crypto market has dominant BTC regime factor.
- per-symbol strategy can still use market context without cross-sectional ranking.

Risks:
- factor crowding.
- market-wide crash overwhelms per-symbol signal.

Mitigation:
- beta cap in portfolio.
- bear-regime notional cap.

### Phase 1 Contracts

Path: `src/domain/futures/strategy/rule_signals.py`

```python
def build_rule_signal_panels(
    *,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
) -> tuple[CandidateSignalPanel, ...]:
    """Build trailing-only rule candidates for all symbols."""
```

```python
def candidate_panels_to_events(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    min_abs_score: float,
) -> pd.DataFrame:
    """Convert dense [T,N] panels into sparse candidate event rows."""
```

### Phase 1 Evaluation

Path: `src/domain/futures/strategy/candidate_backtest.py`

```python
def run_single_symbol_rule_backtests(
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: tuple[str, ...],
    tf: str,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Backtest each symbol and rule family independently using the existing engine."""
```

Single-symbol backtest policy:
- for each symbol, build `target_weights_2d` shape `[T, 1]`.
- run `FuturesBacktestEngine.run_multi()` with one symbol.
- preserve exact execution semantics.
- no cross-sectional ranking.
- use same cost, stop, funding, liquidation behavior.

Rule baseline report columns:
```text
symbol
family
variant
params_hash
n_trades
gross_bps
net_bps
cagr
max_dd
mar
mean_log_growth
turnover
fee_bps
funding_bps
hit_rate
avg_hold_bars
liquidation_count
pass_rule_gate
```

### Phase 1 Acceptance
Rule candidate becomes eligible for ML dataset only if:
- `n_trades >= min_candidate_obs`
- `net_bps > min_rule_net_bps`
- `ir_t >= min_rule_ir_t`
- `turnover <= max_rule_turnover_per_bar`
- no liquidation in OOS block
- max drawdown below configured strategy cap

## Phase 2: Candidate Dataset

### Purpose
Create point-in-time ML training rows from rule candidates. This converts "strategy logic" into "ML-selectable candidate events."

### Labeling Method
Use triple-barrier labeling for event-level decisions.

For event at time T:
- entry price: open at T+1
- max holding: candidate expected horizon
- upper barrier: `take_profit_atr_mult * ATR`
- lower barrier: `stop_atr_mult * ATR`
- time barrier: expected holding bars

Label:
```text
1 if upper barrier hit before lower barrier and gross_return - ex_ante_cost > hurdle
0 otherwise
```

Also store continuous labels:
```text
gross_fwd_bps
edge_after_hurdle_bps
mae_bps
mfe_bps
time_to_exit_bars
```

### Anti-Leakage Rules
- All rolling features are computed using data up to T.
- Entry is T+1 open.
- Labels may inspect T+1 to T+h because they are targets only.
- Fold split must purge and embargo by max horizon.
- Candidate feature normalization is fit on train fold only.
- No symbol-level hyperparameter is chosen from OOS.

### Phase 2 Contracts

Path: `src/domain/futures/strategy/candidate_labels.py`

```python
def label_candidate_events(
    *,
    events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Attach leak-free forward outcomes to candidate events."""
```

Path: `src/domain/futures/strategy/candidate_dataset.py`

```python
def build_candidate_dataset(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    split_start: int,
    split_end: int,
) -> CandidateDataset:
    """Build model matrix for candidate gate and edge models."""
```

Feature groups:
- candidate identity: family, variant, side
- signal strength: score, z-score, signal slope, signal age
- symbol state: realized vol, ATR pct, volume z, funding z, OI z
- market state: BTC trend, BTC vol, market breadth, dispersion, correlation regime
- cost/liquidity: execution_cost_bps, ADV, amihud, spread proxy
- risk: beta vs BTC, cluster id, downside vol, recent drawdown

Do not include:
- future returns
- realized labels
- OOS rank or final backtest metrics
- raw symbol ID as an unrestricted categorical feature in v1

## Phase 3: ML Gate v1

### Purpose
Reduce false-positive trades. The classifier answers:
```text
Should this rule candidate be traded now after realistic costs?
```

### Model
Use `lightgbm.LGBMClassifier`.

Base parameters:
```python
{
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 100,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 2.0,
    "reg_lambda": 20.0,
    "random_state": cfg.seed,
}
```

Theoretical basis:
- meta-labeling separates signal generation from bet sizing.
- candidate rules define a prior.
- classifier learns conditional validity under regime, cost, and liquidity.
- this lowers overfit relative to unconstrained alpha discovery.

### Calibration
Probability must be calibrated on validation folds only.

Allowed:
- Platt sigmoid calibration
- isotonic only if validation sample count is large enough

Disallowed:
- calibration on full dataset
- calibration using OOS test block

### Phase 3 Contracts

Path: `src/domain/futures/strategy/candidate_gate.py`

```python
@dataclass(slots=True, frozen=True)
class CandidateGateModel:
    model: Any
    calibrator: Any | None
    feature_names: tuple[str, ...]
    train_window: tuple[int, int]
    valid_window: tuple[int, int]
```

```python
def fit_candidate_gate(
    *,
    train: CandidateDataset,
    valid: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateGateModel:
    """Fit and calibrate candidate trade/no-trade classifier."""
```

```python
def predict_candidate_gate(
    *,
    model: CandidateGateModel,
    dataset: CandidateDataset,
) -> NDArray[np.float64]:
    """Return calibrated pass probability for candidate events."""
```

### Phase 3 Acceptance
Gate is useful only if it improves:
- false positive rate
- net bps after costs
- max drawdown
- mean log growth
- turnover

It is rejected if it only improves classification AUC without improving portfolio results.

## Phase 4: ML Edge v1

### Purpose
Estimate economic magnitude and downside after the gate. The model answers:
```text
If traded, how much edge remains and how bad is the downside?
```

### Model
Use LightGBM regressors:
- center model: `objective="huber"` or `regression_l1`
- downside model: `objective="quantile", alpha=0.10`
- upside model: `objective="quantile", alpha=0.90`

Output:
```text
mu_gross_bps
mu_net_decision_bps = mu_gross_bps - expected_cost_bps
q10_net_bps
q90_net_bps
```

Decision utility:
```text
utility = p_pass * mu_net_decision_bps
          - downside_penalty * abs(min(q10_net_bps, 0))
          - turnover_penalty * turnover_proxy
          - concentration_penalty
```

### Phase 4 Contracts

Path: `src/domain/futures/strategy/candidate_edge.py`

```python
@dataclass(slots=True, frozen=True)
class CandidateEdgeModels:
    center_model: Any
    q10_model: Any
    q90_model: Any
    feature_names: tuple[str, ...]
```

```python
def fit_candidate_edge_models(
    *,
    train: CandidateDataset,
    valid: CandidateDataset,
    cfg: CandidateStrategyConfig,
) -> CandidateEdgeModels:
    """Fit robust expected edge and quantile downside models."""
```

```python
def predict_candidate_edges(
    *,
    models: CandidateEdgeModels,
    dataset: CandidateDataset,
    p_pass: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> CandidateModelOutput:
    """Return expected edge, downside quantiles, and utility scores."""
```

### Phase 4 Acceptance
Reject edge model if:
- predicted edge is monotonic but not realized in OOS bins
- top utility decile does not outperform lower deciles after costs
- q10 underestimates downside in stress regimes
- output creates high turnover without net log-growth improvement

## Phase 5: Portfolio v1

### Purpose
Convert candidate-level predictions into target weights for the existing engine.

### Selection Rule
Per timestamp and symbol:
1. collect all candidate events.
2. filter:
   - `p_pass >= min_gate_probability`
   - `mu_net_decision_bps >= min_expected_net_bps`
   - `q10_net_bps >= -max_expected_shortfall_bps`
3. choose the highest utility candidate per symbol.
4. if long and short candidates both pass for one symbol, choose the side with higher utility and set the other side to zero.

This is not universe ranking. It is per-symbol candidate arbitration.

### Sizing Rule
Approximate expected log-growth with fractional Kelly:
```text
raw_weight_i = kelly_fraction * mu_i / variance_i
```

Where:
- `mu_i` is expected simple return per holding window converted to per-bar scale.
- `variance_i` uses trailing symbol variance or residual variance.
- sign comes from candidate side.

Then apply:
- per-symbol cap
- gross cap
- net cap
- BTC beta cap
- cluster cap
- target volatility cap
- min-notional quantization

### Phase 5 Contracts

Path: `src/domain/futures/strategy/candidate_portfolio.py`

```python
def select_candidate_events_for_portfolio(
    *,
    model_output: CandidateModelOutput,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Select at most one active candidate per symbol per timestamp."""
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
    """Build target_weights_2d for the backtest engine."""
```

```python
def build_candidate_alpha_panel(
    *,
    selected_events: pd.DataFrame,
    target_weights_2d: NDArray[np.float64],
    datetimes: NDArray[np.datetime64],
    symbols: tuple[str, ...],
) -> pd.DataFrame:
    """Build long-format panel for merge into data maps."""
```

### Backtest Input Rule
`merge_ml_output_into_data_maps()` must be replaced with:
```python
def merge_candidate_output_into_data_maps(
    output: CandidatePortfolioResult,
    is_maps: dict[str, dict[str, Any]],
    oos_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    """Merge target_weights and candidate diagnostics into data maps."""
```

Required dataframe columns after merge:
```text
target_weight
alpha_long
alpha_short
candidate_family
p_pass
mu_net_decision_bps
utility_score
```

`target_weight` is the source of truth for execution.

## Phase 6: OOS Evaluation

### Purpose
Promote only strategies that compound capital under realistic execution.

### Evaluation Protocol
Use nested walk-forward:
- inner train/validation for model and candidate thresholds
- outer OOS block for promotion
- OOS block length: 6 months
- blocks must be non-overlapping
- purge/embargo must cover max holding horizon

Single-symbol evaluation:
- run each symbol independently for rule-only strategy.
- store per-symbol equity curve and diagnostics.
- promote symbol-strategy candidates only if multiple OOS blocks pass.

Portfolio evaluation:
- combine promoted candidates through the portfolio constructor.
- run full multi-symbol engine using `target_weights`.

### Metrics
Primary:
```text
mean_log_growth
CAGR
max_drawdown
MAR
final_equity
liquidation_count
```

Execution:
```text
net_pnl
fees
funding
slippage
turnover
avg_hold_bars
capacity_decay
intrabar_decay
```

Robustness:
```text
block_pass_ratio
worst_block_return
DSR
PBO
stationary_bootstrap_p5
regime_breakdown
```

### Phase 6 Contracts

Path: `src/domain/futures/strategy/candidate_evaluation.py`

```python
@dataclass(slots=True, frozen=True)
class CompoundEvaluationReport:
    mean_log_growth: float
    cagr: float
    max_drawdown: float
    mar: float
    final_equity: float
    net_pnl: float
    fees: float
    funding: float
    turnover: float
    block_pass_ratio: float
    worst_block_return: float
    dsr: float
    pbo: float
    liquidation_count: int
    pass_compound_gate: bool
    fail_reasons: tuple[str, ...]
```

```python
def evaluate_compound_backtest(
    *,
    trades: pd.DataFrame,
    equity_curve: NDArray[np.float64],
    diag: NDArray[np.float64],
    cfg: CandidateStrategyConfig,
) -> CompoundEvaluationReport:
    """Evaluate geometric growth and execution realism."""
```

Promotion gate:
```text
pass if:
  mean_log_growth > 0
  CAGR > 0
  MAR >= 0.75
  max_drawdown <= configured cap
  liquidation_count == 0
  block_pass_ratio >= 0.70
  worst_block_return > -max_block_loss
  net_pnl after fees and funding > 0
```

## Phase 7: Ablation

### Purpose
Prove each complexity layer adds value.

### Required Comparisons
1. `rule_only_equal_size`
2. `rule_only_fractional_kelly`
3. `rule_plus_ml_gate`
4. `rule_plus_ml_gate_plus_edge`
5. `rule_plus_ml_gate_plus_edge_plus_portfolio_caps`
6. `candidate_ml_full`

Optional benchmark:
- old lambdamart output only during transition, then remove.

### Phase 7 Contract

Path: `src/domain/futures/strategy/ablation.py`

```python
@dataclass(slots=True, frozen=True)
class AblationRow:
    variant: str
    mean_log_growth: float
    cagr: float
    max_drawdown: float
    mar: float
    turnover: float
    final_equity: float
    pass_compound_gate: bool
```

```python
def run_candidate_ablation(
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: tuple[str, ...],
    tf: str,
    cfg: CandidateStrategyConfig,
) -> pd.DataFrame:
    """Run rule-only, ML-gated, edge, and portfolio ablations."""
```

Acceptance:
- ML gate must improve rule-only MAR or mean log growth.
- edge model must improve ML-gate-only portfolio.
- portfolio caps must reduce drawdown without destroying CAGR.
- no layer is accepted only because it improves IC.

## Runtime Flow

### New `run_candidate_strategy_for_universe`
Path: `src/domain/futures/strategy_runtime/bridge.py`

```python
@dataclass(slots=True)
class CandidatePipelineOutput:
    alpha_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    rule_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    gate_report: dict[str, Any] = field(default_factory=dict)
    evaluation_report: dict[str, Any] = field(default_factory=dict)
```

```python
def run_candidate_strategy_for_universe(
    symbols: list[str],
    tf: str,
    fetch_start: str | None,
    end_date: str | None,
    opt_config: dict[str, Any],
    *,
    strategy_cfg: StrategyConfig | None = None,
    preloaded_data_maps: dict[str, dict[str, Any]] | None = None,
    mode: Literal["rule_baseline", "candidate_ml"] = "candidate_ml",
) -> CandidatePipelineOutput:
```

### `build_strategy_alpha`
Path: `src/domain/futures/strategy/builder.py`

Replace `lambdamart` branch:
```python
if cfg.name in {"candidate_ml", "rule_baseline"}:
    from src.domain.futures.strategy.candidate_backtest import build_candidate_strategy_output
    return build_candidate_strategy_output(...)
```

Remove import of:
```python
build_ml_strategy_alpha
```

### `opt_main_futures.py`
Replace strategy-stage wording:
```text
>> STRATEGY: candidate_ml
```

Replace alpha evaluation:
- old `_run_alpha_evaluation_report()` should be deleted.
- add `_run_candidate_evaluation_report()`.
- report compound metrics and ablation table, not rank IC scoreboard.

## Universe Redesign

### Current Issue
Stage6 currently ranks symbols and selects `k_in`. That is useful for liquidity/risk narrowing but should not be the alpha selection mechanism.

### New Policy
Use universe stages as follows:
- Stage0 to Stage5: hard eligibility and tradability filters.
- Stage6: optional execution pool cap and metadata enrichment.
- Strategy candidate generation: per symbol across the eligible pool.
- ML selection: per candidate event.
- Portfolio selection: expected utility and risk caps.

### Single-Symbol Backtest Requirement
For rule baseline:
- every eligible symbol is evaluated independently.
- symbol backtest result is stored even if the final portfolio does not trade it.
- no symbol is rejected because it is low in alpha rank.
- rejection must be caused by data quality, liquidity, cost, risk event, or poor OOS strategy performance.

### Universe Config Changes
Path: `src/domain/futures/universe/config.py`

Add:
```python
strategy_pool_mode: Literal["stage5_all", "stage6_pool"] = "stage5_all"
stage6_is_alpha_rank: bool = False
```

When `stage6_is_alpha_rank=False`, Stage6 report should call its score:
```text
execution_pool_score
```
not alpha rank.

## Surgical Plan

### Phase 0: Removal Preparation

`src/domain/futures/strategy/config.py`
- `[ACTION: REPLACE]` `StrategyMLConfig` with `CandidateStrategyConfig`.
- `[ACTION: REPLACE]` `StrategyConfig.ml` with `StrategyConfig.candidate`.
- `[ACTION: REPLACE]` allowed strategy names.

`src/domain/futures/strategy/__init__.py`
- `[ACTION: REPLACE]` export `CandidateStrategyConfig`.
- `[ACTION: DELETE]` export `StrategyMLConfig`.

`src/domain/futures/strategy_runtime/bridge.py`
- `[ACTION: REPLACE]` `MLPipelineOutput` with `CandidatePipelineOutput`.
- `[ACTION: REPLACE]` `run_ml_pipeline_for_universe()` with `run_candidate_strategy_for_universe()`.
- `[ACTION: REPLACE]` merge function to merge `target_weight`.

`src/execution/opt_main_futures.py`
- `[ACTION: REPLACE]` ML bridge calls.
- `[ACTION: DELETE]` old rank alpha report path.
- `[ACTION: ADD]` candidate evaluation and ablation report path.

### Phase 1: Rule Baseline

`src/domain/futures/strategy/rule_signals.py`
- `[ACTION: ADD]` rule panel builders for all eight rule families.
- `[ACTION: ADD]` vectorized rolling helpers.
- `[ACTION: ADD]` candidate event conversion.

`src/domain/futures/strategy/candidate_backtest.py`
- `[ACTION: ADD]` single-symbol backtest runner.
- `[ACTION: ADD]` rule-only target weight builder.

### Phase 2: Dataset

`src/domain/futures/strategy/candidate_labels.py`
- `[ACTION: ADD]` triple-barrier label builder.
- `[ACTION: ADD]` forward gross return and edge-after-hurdle columns.

`src/domain/futures/strategy/candidate_dataset.py`
- `[ACTION: ADD]` feature matrix builder.
- `[ACTION: ADD]` fold-local normalizer and one-hot encoder.

### Phase 3: Gate

`src/domain/futures/strategy/candidate_gate.py`
- `[ACTION: ADD]` LightGBM classifier wrapper.
- `[ACTION: ADD]` fold-local probability calibration.
- `[ACTION: ADD]` reliability diagnostics.

### Phase 4: Edge

`src/domain/futures/strategy/candidate_edge.py`
- `[ACTION: ADD]` LightGBM edge and quantile models.
- `[ACTION: ADD]` utility score computation.

### Phase 5: Portfolio

`src/domain/futures/strategy/candidate_portfolio.py`
- `[ACTION: ADD]` event arbitration.
- `[ACTION: ADD]` target weight builder.
- `[ACTION: ADD]` cap projection and alpha panel conversion.

### Phase 6: Evaluation

`src/domain/futures/strategy/candidate_evaluation.py`
- `[ACTION: ADD]` compound report.
- `[ACTION: ADD]` block OOS pass/fail logic.
- `[ACTION: ADD]` bootstrap/DSR/PBO hooks.

### Phase 7: Ablation

`src/domain/futures/strategy/ablation.py`
- `[ACTION: ADD]` required ablation variants.
- `[ACTION: ADD]` summary table and promotion verdict.

### Move Old ML Alpha Files to Legacy
After new tests pass, move the following files to the `legacy/` directory to completely separate them:
- `[ACTION: MOVE]` `src/domain/futures/strategy/ml_builder.py` -> `legacy/strategy/ml_builder.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/ranker.py` -> `legacy/strategy/ranker.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/calibrator.py` -> `legacy/strategy/calibrator.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/rank_selection.py` -> `legacy/strategy/rank_selection.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/alpha_evaluation.py` -> `legacy/strategy/alpha_evaluation.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/features.py` -> `legacy/strategy/features.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/labels.py` -> `legacy/strategy/labels.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/dataset.py` -> `legacy/strategy/dataset.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/inference.py` -> `legacy/strategy/inference.py`
- `[ACTION: MOVE]` `src/domain/futures/strategy/cache.py` -> `legacy/strategy/cache.py`

Update or move tests that import these modules so they are also preserved in the legacy folder.

## Verification

### Phase 1
```bash
uv run ruff check --fix src/domain/futures/strategy/config.py src/domain/futures/strategy/rule_signals.py src/domain/futures/strategy/candidate_contracts.py src/domain/futures/strategy/candidate_backtest.py
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/rule_signals.py src/domain/futures/strategy/candidate_contracts.py src/domain/futures/strategy/candidate_backtest.py
uv run pytest tests/unit/domain/futures/strategy/test_rule_signals.py tests/unit/domain/futures/strategy/test_candidate_backtest.py --tb=short
```

Expected:
- no look-ahead tests pass.
- T signal maps to T+1 entry.
- each rule emits deterministic candidates.
- single-symbol backtest uses existing engine.

### Phase 2
```bash
uv run ruff check --fix src/domain/futures/strategy/candidate_labels.py src/domain/futures/strategy/candidate_dataset.py
uv run mypy src/domain/futures/strategy/candidate_labels.py src/domain/futures/strategy/candidate_dataset.py
uv run pytest tests/unit/domain/futures/strategy/test_candidate_labels.py tests/unit/domain/futures/strategy/test_candidate_dataset.py --tb=short
```

Expected:
- triple-barrier labels use only future target window.
- features use no future data.
- purge/embargo split removes overlapping labels.

### Phase 3 and 4
```bash
uv run ruff check --fix src/domain/futures/strategy/candidate_gate.py src/domain/futures/strategy/candidate_edge.py
uv run mypy src/domain/futures/strategy/candidate_gate.py src/domain/futures/strategy/candidate_edge.py
uv run pytest tests/unit/domain/futures/strategy/test_candidate_gate.py tests/unit/domain/futures/strategy/test_candidate_edge.py --tb=short
```

Expected:
- model fit is deterministic with fixed seed.
- validation calibration does not use test fold.
- top utility bucket outperforms bottom utility bucket in synthetic fixture.

### Phase 5
```bash
uv run ruff check --fix src/domain/futures/strategy/candidate_portfolio.py src/domain/futures/strategy_runtime/bridge.py src/execution/opt_main_futures.py
uv run mypy src/domain/futures/strategy/candidate_portfolio.py src/domain/futures/strategy_runtime/bridge.py src/execution/opt_main_futures.py
uv run pytest tests/unit/domain/futures/strategy/test_candidate_portfolio.py tests/unit/execution/test_opt_main_futures_strategy_mode.py --tb=short
```

Expected:
- target weights satisfy caps.
- contradictory long/short candidates collapse to one side.
- `target_weights` is merged into data maps.
- engine consumes target weights directly.

### Phase 6 and 7
```bash
uv run ruff check --fix src/domain/futures/strategy/candidate_evaluation.py src/domain/futures/strategy/ablation.py
uv run mypy src/domain/futures/strategy/candidate_evaluation.py src/domain/futures/strategy/ablation.py
uv run pytest tests/unit/domain/futures/strategy/test_candidate_evaluation.py tests/unit/domain/futures/strategy/test_ablation.py --tb=short
```

Expected:
- compound gate is based on log growth and drawdown, not IC.
- ablation rejects ML if it fails to beat rule-only baseline.

### Smoke
```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase strategy --trials 1 --timeframe 4h --date 2026-05-01 --sync skip
```

Expected:
- no import of old `ml_builder`, `ranker`, `rank_selection`, or `alpha_evaluation`.
- strategy stage logs candidate rule, gate, edge, portfolio, and compound evaluation summaries.
- backtest engine runs with `target_weights`.

## Critical Self-Review

### Potential Logical Weakness 1: Rule Candidates Can Still Overfit
Risk:
- many rule families, windows, thresholds, symbols, and regimes create a large multiple-testing surface.

Control:
- treat every rule variant as a hypothesis.
- require non-overlapping OOS block pass ratio.
- use DSR/PBO/bootstrap.
- promote families, not one lucky symbol-parameter pair.

### Potential Logical Weakness 2: Per-Symbol Backtests Can Fragment Evidence
Risk:
- each symbol has limited history and strong regime dependency.

Control:
- train ML gate as pooled candidate model across symbols.
- use symbol diagnostics for reporting, not unrestricted symbol-specific hyperparameters.
- allow symbol-level eligibility only after multiple OOS blocks.

### Potential Logical Weakness 3: ML Gate May Learn Rule Backtest Artifacts
Risk:
- model learns simulator quirks or liquidity artifacts, not market structure.

Control:
- features must be available before execution.
- capacity ladder must degrade expected returns.
- require intrabar decay and fee/funding decomposition.

### Potential Logical Weakness 4: Expected Kelly Inputs Are Noisy
Risk:
- small mu estimation errors cause oversized Kelly weights.

Control:
- cap Kelly fraction at 0.25.
- shrink mu by calibration error or q10 downside.
- enforce per-symbol, gross, beta, volatility, and drawdown caps.
- never scale up due to drawdown precompute.

### Potential Logical Weakness 5: Stage5 Universe Can Still Be Too Broad
Risk:
- running all Stage5 symbols can increase noise and compute cost.

Control:
- Stage6 remains as execution pool mode if needed.
- Stage6 must be framed as liquidity/capacity pool, not alpha rank.
- single-symbol benchmark can be run offline in batches.

### Potential Logical Weakness 6: Candidate Router May Miss Pure Cross-Sectional Alpha
Risk:
- removing ranker discards genuine cross-sectional effects.

Control:
- cross-sectional context is allowed as features.
- candidate selection is per symbol, but portfolio allocator still sees all symbols.
- later add "relative strength candidate" as a rule family, not a global ranker.

### Potential Logical Weakness 7: Objective Can Become Too Conservative
Risk:
- strict drawdown, cost, and OOS gates may reject all candidates.

Control:
- report frontier distance to pass.
- store near-pass candidates by family and regime.
- only relax research thresholds, never final execution costs or look-ahead rules.

## Final Architecture Verdict
This design is logically stronger than the current ML alpha ranker because it separates:
- hypothesis generation: rule candidates
- conditional validity: ML gate
- economic magnitude: edge/quantile model
- sizing: fractional Kelly and caps
- realism: existing futures backtest engine
- promotion: OOS compound-growth evaluation

The main cost is complexity and larger implementation surface. The control is strict phase-by-phase ablation. If rule-only baselines cannot produce positive post-cost OOS candidates, ML should not be added. If ML cannot beat rule-only under identical backtest semantics, ML must remain disabled.
