---
title: Candidate ML Compounding Improvement Blueprint
domain: futures-strategy
type: refactor
status: ready_for_implementation
priority: critical
created_at: 2026-06-03
related_paths:
  - docs/results/result.md
  - docs/specs/candidate_ml_strategy.md
  - src/domain/futures/strategy/candidate_dataset.py
  - src/domain/futures/strategy/candidate_gate.py
  - src/domain/futures/strategy/rule_signals.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/strategy/ablation.py
  - src/domain/futures/universe/config.py
  - src/domain/futures/universe/selection.py
---

# Candidate ML Compounding Improvement Blueprint

## 1. Diagnosis

Latest `mode alpha` result has 6 promoted rule variants, but live candidate execution remains blocked:

- Rule promotion is no longer the primary blocker: `KEEP=6`.
- ML output is inactive: `Active Signals: 0`, blocker reported as `ML gate p_pass < 0.40`.
- Ablation shows zero exposure for all ML variants, so compounding cannot be evaluated yet.

The user's model of the pipeline is directionally correct:

1. Build rule candidate panels.
2. Label events with future outcomes.
3. Diagnose variants and keep/flip/drop them.
4. Train ML gate and edge models on promoted events.
5. Select events and convert them into backtest target weights.

However, current code has higher-priority structural issues than adding more strategies.

## 2. Critical Weaknesses

### P0-1. OOS split is used for ML calibration and inference

Current production bridge flow:

- `train_set = [0, 80%)`
- `valid_set = [80%, 100%)`
- `fit_candidate_gate(train=train_set, valid=valid_set)`
- `fit_candidate_edge_models(train=train_set, valid=valid_set)`
- `predict_candidate_gate(dataset=valid_set)`
- `predict_candidate_edges(dataset=valid_set)`

This means the same OOS period is used for gate calibration / edge validation and then used as the inference target. Once the ML path starts trading, this becomes look-ahead leakage.

### P0-2. Gate target defaults to gross direction, not net profitable trade

`label_candidate_events` creates both:

- `gross_direction_label = gross_fwd_bps > 0`
- `profitable_after_hurdle_label = edge_after_hurdle_bps > 0`

`build_candidate_dataset` currently selects `gross_direction_label` when present. In production, it is always present after labeling. Therefore the gate learns "direction was correct before costs" rather than "trade was profitable after cost/hurdle."

This is not aligned with compounding. For leveraged futures, directionally correct but net-negative trades increase turnover and drawdown.

### P0-3. Candidate event generation ignores warm / entry block / inference masks

`build_rule_signal_panels` uses:

```python
valid_mask = aligned.active_mask & np.isfinite(close) & np.isfinite(high) & np.isfinite(low)
```

It does not apply:

- `aligned.warm_mask`
- `aligned.entry_block_mask`
- `aligned.inference_entry_warm_mask`
- `aligned.kill_mask`

This can generate candidate entries during universe warm-up or blocked entry windows. The backtest engine may later suppress execution, but ML labels and diagnostics are still polluted by candidates that could not realistically be traded.

### P0-4. Event cost is hard-coded instead of being modeled in two layers

`candidate_panels_to_events` writes:

```python
"cost_floor_bps": 24.0
```

The problem is not that `24bps` exists. The problem is that the value is injected as a literal, so the code cannot distinguish between:

- `decision_cost_floor_bps`: conservative policy floor used to reject weak edge.
- `physical_execution_cost_bps`: universe Stage4 estimate of actual execution cost.

This ignores `CandidateStrategyConfig.cost_floor_bps` and ignores `aligned.execution_cost_bps_2d`. As a result, rule diagnostics and ML labels can diverge from the universe Stage4 execution cost model, and the code cannot express the user's maker-vs-taker execution assumption explicitly.

Do not lower cost just to create trades. The fix is to make the label cost source explicit and realistic, while still preserving `24bps` as a configurable conservative floor if that remains the chosen policy.

### P1. Strategy expansion is secondary

Adding more candidate families may increase `KEEP`, but it will not fix:

- invalid ML validation protocol,
- gross-direction gate target,
- polluted warm-up events,
- cost mismatch,
- probability calibration collapse.

Only add new strategy families after P0 fixes produce nonzero, cost-realistic OOS selected events.

## 3. Target Files

Modify:

- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/candidate_dataset.py`
- `src/domain/futures/strategy/candidate_gate.py`
- `src/domain/futures/strategy/rule_signals.py`
- `src/domain/futures/strategy_runtime/bridge.py`
- `src/domain/futures/strategy/ablation.py`
- `tests/unit/domain/futures/strategy/test_candidate_dataset.py`
- `tests/unit/domain/futures/strategy/test_candidate_gate.py`
- `tests/unit/domain/futures/strategy/test_rule_signals.py`
- `tests/unit/domain/futures/strategy/test_ablation.py`

Reference only:

- `src/domain/futures/universe/config.py`
- `src/domain/futures/universe/selection.py`
- `src/domain/futures/universe/membership.py`
- `src/domain/futures/strategy/common/alignment.py`

## 4. Contracts

### 4.1 CandidateStrategyConfig

Add fields:

```python
gate_label_column: Literal[
    "profitable_after_hurdle_label",
    "barrier_first_label",
    "gross_direction_label",
] = "profitable_after_hurdle_label"
gate_calibration_method: Literal["sigmoid", "isotonic", "none"] = "sigmoid"
min_gate_calibration_obs: int = 100
min_gate_calibration_pos: int = 10
min_gate_probability_std: float = 0.03
ml_fit_fraction: float = 0.60
ml_calibration_fraction: float = 0.20
```

Validation:

- `gate_label_column` must be one of the listed values.
- `gate_calibration_method` must be one of the listed values.
- `min_gate_calibration_obs >= 1`.
- `min_gate_calibration_pos >= 1`.
- `min_gate_probability_std >= 0.0`.
- `0.1 <= ml_fit_fraction < 1.0`.
- `0.0 <= ml_calibration_fraction < 1.0`.
- `ml_fit_fraction + ml_calibration_fraction < 1.0`.

### 4.2 candidate_panels_to_events

Replace signature:

```python
def candidate_panels_to_events(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    min_abs_score: float,
    side_flip_variants: tuple[str, ...] = (),
) -> pd.DataFrame:
```

with:

```python
def candidate_panels_to_events(
    panels: tuple[CandidateSignalPanel, ...],
    *,
    min_abs_score: float,
    side_flip_variants: tuple[str, ...] = (),
    cost_floor_bps: float = 24.0,
    execution_cost_bps_2d: NDArray[np.float64] | None = None,
) -> pd.DataFrame:
```

Cost assignment:

```python
event_cost = np.full(t_idx.shape[0], float(cost_floor_bps), dtype=np.float64)
if execution_cost_bps_2d is not None:
    physical_cost = execution_cost_bps_2d[t_idx, s_idx].astype(np.float64, copy=False)
    physical_cost = np.nan_to_num(physical_cost, nan=0.0, posinf=0.0, neginf=0.0)
    event_cost = np.maximum(event_cost, physical_cost)
```

This keeps a strict floor, but the floor is now an explicit policy input rather than a buried constant.

### 4.3 build_candidate_dataset

Keep signature unchanged:

```python
def build_candidate_dataset(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    split_start: int,
    split_end: int,
) -> CandidateDataset:
```

Replace gate label selection with:

```python
gate_label_col = cfg.gate_label_column
if gate_label_col not in kept_events.columns:
    raise ValueError(f"missing configured gate label column: {gate_label_col}")
y_gate = kept_events[gate_label_col].to_numpy(dtype=np.int8, copy=False)
```

Do not silently fall back to gross labels in production.

### 4.4 CandidateGateModel

Replace:

```python
@dataclass(slots=True, frozen=True)
class CandidateGateModel:
    model: LGBMClassifier
    calibrator: CalibratedClassifierCV | None
    feature_names: tuple[str, ...]
    train_window: tuple[int, int]
    valid_window: tuple[int, int]
```

with:

```python
@dataclass(slots=True, frozen=True)
class CandidateGateModel:
    model: LGBMClassifier
    calibrator: CalibratedClassifierCV | None
    feature_names: tuple[str, ...]
    train_window: tuple[int, int]
    valid_window: tuple[int, int]
    calibration_method: str
    calibration_used: bool
    calibration_reason: str
```

`predict_candidate_gate` must use `calibrator` only when `calibration_used is True`.

### 4.5 ML temporal split helper

Add to `src/domain/futures/strategy_runtime/bridge.py` or a small local helper module if reuse is needed:

```python
def _candidate_ml_split_indices(
    *,
    n_bars: int,
    fit_fraction: float,
    calibration_fraction: float,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[int, int, int, int, int, int]:
    """Return fit, calibration, and oos split indices with purge/embargo gaps."""
```

Return:

- `fit_start`
- `fit_end`
- `calibration_start`
- `calibration_end`
- `oos_start`
- `oos_end`

Rules:

- `fit_start = 0`
- `fit_end = int(n_bars * fit_fraction)`
- `calibration_start = fit_end + purge_bars`
- `calibration_end = int(n_bars * (fit_fraction + calibration_fraction))`
- `oos_start = calibration_end + embargo_bars`
- `oos_end = n_bars`
- Raise `ValueError` if any split is empty after purge/embargo.

## 5. Step-by-Step Logic

### Step A. Make event eligibility realistic

In `build_rule_signal_panels`:

1. Compute `entry_warm_mask`:
   - if `aligned.inference_entry_warm_mask is not None`, use it.
   - else use `aligned.warm_mask`.
2. Compute `entry_allowed_mask`:
   - `aligned.active_mask`
   - `entry_warm_mask`
   - `~aligned.entry_block_mask`
   - `~aligned.kill_mask`
   - finite OHLC data
3. Use `entry_allowed_mask` as every panel's `valid_mask_2d`.

### Step B. Make cost source explicit

Update every call to `candidate_panels_to_events`:

```python
raw_events = candidate_panels_to_events(
    panels,
    min_abs_score=cfg.min_rule_net_bps * 1e-4,
    side_flip_variants=cfg.side_flip_candidate_variants,
    cost_floor_bps=cfg.cost_floor_bps,
    execution_cost_bps_2d=aligned.execution_cost_bps_2d,
)
```

Apply in:

- `src/domain/futures/strategy_runtime/bridge.py`
- `src/domain/futures/strategy/ablation.py`

### Step C. Train gate on net-profitable target

In `build_candidate_dataset`:

1. Use `cfg.gate_label_column`.
2. Default is `profitable_after_hurdle_label`.
3. Raise on missing column.

Expected effect:

- `p_pass` estimates `P(edge_after_hurdle_bps > 0 | features)`.
- Gate probability becomes aligned with compounding, not raw direction.

### Step D. Remove OOS calibration leakage

In production bridge:

1. Build `fit_set` from fit window.
2. Build `calibration_set` from calibration window.
3. Build `oos_set` from OOS window.
4. Fit gate and edge models using `fit_set` and `calibration_set`.
5. Predict only on `oos_set`.
6. Build target weights only from `oos_set.event_index`.

Do not fit calibration, early stopping, or thresholds on the same OOS rows used for inference.

### Step E. Use calibration only if it preserves useful probability dispersion

In `fit_candidate_gate`:

1. Always fit base LightGBM on `train`.
2. If `cfg.gate_calibration_method == "none"`, skip calibration.
3. Skip calibration if validation rows or positives are below config minimum.
4. Fit `CalibratedClassifierCV` with configured method.
5. Compare raw and calibrated probabilities on calibration set.
6. Use calibrator only if:
   - calibrated probabilities are finite,
   - `std(calibrated_p) >= cfg.min_gate_probability_std`,
   - calibrated Brier score is not worse than raw Brier by more than `1e-6`.
7. Store reason in `CandidateGateModel.calibration_reason`.

### Step F. Strategy expansion gate

Do not add new families until all conditions hold:

- OOS selected events > 0.
- Candidate ML final equity changes from 1,000,000 in OOS ablation.
- At least one ML variant passes or fails for real P/L / MaxDD reasons, not zero exposure.
- Net edge remains positive after `max(cfg.cost_floor_bps, execution_cost_bps_2d)`.

After P0 is fixed, strategy expansion should prioritize orthogonal, continuous-score families:

- funding term-structure / funding acceleration,
- basis dislocation mean reversion,
- OI impulse with volume confirmation,
- BTC residual momentum after beta adjustment.

Reject families that only duplicate existing F1/F2 exposure unless they improve OOS net log growth after costs.

## 6. Surgical Plan

### `src/domain/futures/strategy/config.py`

Action: ADD fields to `CandidateStrategyConfig`.

Action: REPLACE `__post_init__` validation to include new fields.

### `src/domain/futures/strategy/rule_signals.py`

Action: REPLACE `valid_mask` construction in `build_rule_signal_panels`.

Action: REPLACE `candidate_panels_to_events` signature and `cost_floor_bps` assignment.

### `src/domain/futures/strategy/candidate_dataset.py`

Action: REPLACE gate label fallback block with explicit `cfg.gate_label_column`.

Action: ADD debug logging if module already has logger; otherwise keep testable behavior via exceptions only.

### `src/domain/futures/strategy/candidate_gate.py`

Action: REPLACE `CandidateGateModel` dataclass fields.

Action: REPLACE calibration logic with guarded calibration selection.

Pseudo-code:

```python
raw_prob = model.predict_proba(valid.X)[:, 1]
raw_brier = np.average((raw_prob - valid.y_gate) ** 2, weights=valid.sample_weight)

if method == "none" or insufficient_valid:
    return CandidateGateModel(..., calibration_used=False, calibration_reason="...")

calibrator = CalibratedClassifierCV(...)
calibrator.fit(valid.X, valid.y_gate, sample_weight=valid.sample_weight)
cal_prob = calibrator.predict_proba(valid.X)[:, 1]
cal_brier = np.average((cal_prob - valid.y_gate) ** 2, weights=valid.sample_weight)

if np.std(cal_prob) < cfg.min_gate_probability_std:
    calibration_used = False
elif cal_brier > raw_brier + 1e-6:
    calibration_used = False
else:
    calibration_used = True
```

### `src/domain/futures/strategy_runtime/bridge.py`

Action: ADD `_candidate_ml_split_indices`.

Action: REPLACE current 80/20 train/valid split with fit/calibration/OOS split.

Action: REPLACE call to `candidate_panels_to_events` to pass cost config and execution cost matrix.

### `src/domain/futures/strategy/ablation.py`

Action: REPLACE event conversion call to pass cost config and execution cost matrix.

Action: Update ablation ML rows so production-equivalent candidate ML uses the same fit/calibration/OOS split as bridge.

## 7. Verification

Run L1 checks after implementation:

```bash
uv run ruff check --fix src/domain/futures/strategy/config.py src/domain/futures/strategy/candidate_dataset.py src/domain/futures/strategy/candidate_gate.py src/domain/futures/strategy/rule_signals.py src/domain/futures/strategy_runtime/bridge.py src/domain/futures/strategy/ablation.py
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/candidate_dataset.py src/domain/futures/strategy/candidate_gate.py src/domain/futures/strategy/rule_signals.py src/domain/futures/strategy_runtime/bridge.py src/domain/futures/strategy/ablation.py
```

Run focused tests:

```bash
uv run pytest tests/unit/domain/futures/strategy/test_candidate_dataset.py tests/unit/domain/futures/strategy/test_candidate_gate.py tests/unit/domain/futures/strategy/test_rule_signals.py tests/unit/domain/futures/strategy/test_candidate_portfolio.py tests/unit/domain/futures/strategy/test_ablation.py --tb=short
```

Run integration smoke:

```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode strategy-smoke --skip-universe --skip-data-sync --symbols BTCUSDT --trials 1 --tf 4h --reference-date 2026-05-01 --strategy candidate_ml
```

Expected outcomes:

- `build_candidate_dataset` uses `profitable_after_hurdle_label` by default.
- No candidate event is produced where warm / entry block / kill masks disallow entry.
- `cost_floor_bps` in candidate events equals `max(cfg.cost_floor_bps, physical_execution_cost_bps)` when physical cost exists, so the 24bps policy can remain conservative without being hard-coded.
- Gate calibration is skipped when it collapses probability dispersion.
- Bridge uses separate fit, calibration, and OOS inference windows.
- Candidate ML no longer reports zero exposure solely because of probability compression; if it still reports zero exposure, diagnostics must point to net edge / q10 / utility eligibility rather than OOS leakage.

## 8. Acceptance Criteria

- No OOS row used for calibration or edge validation is used for ML inference in the same run.
- Gate target is net-profit-after-hurdle by default.
- Event generation respects PIT universe membership and warm-up entry eligibility.
- Event labels use realistic cost from config and universe execution cost matrix.
- Strategy expansion is deferred until the ML path can produce valid nonzero OOS selections under realistic costs.
