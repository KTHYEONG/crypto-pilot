# Spec: ML-OOS Alpha/Cost Gate Refactor

## Type
bug-fix

## Target Files
- `src/domain/futures/strategy/diagnostics.py`
- `src/domain/futures/strategy/ml_builder.py`
- `src/domain/futures/optimization/objectives.py`
- `tests/unit/domain/futures/strategy/test_ml_diagnostics.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`
- `tests/unit/domain/futures/optimization/test_strategy_signal_path.py`

## Diagnosis
The failing run is not primarily a risk-forecast problem.

Observed log facts:
- `RAW-SIGNAL-DIAG var_retention=0.951`: beta residualization keeps most cross-sectional dispersion.
- `in_fold_valid_alpha_p95_bps=53.02` but `alpha_p95_bps=5.53`: validation magnitude does not survive OOS/live inference.
- `alpha_gate_pass=True` despite `alpha_gate_floor_bps=24.0`: the alpha gate uses validation p95 as the cost-wall metric.
- `xs_long_preservation_ratio=5.03`: preservation ratio compares different bases and can exceed 1.0, so it is not measuring post-cost survival.
- `spearman_rank_ic=-0.00298`: the quality gate fails correctly, but the report does not distinguish ranker score IC from final side-composed alpha IC.

Primary defects:
- `build_ml_strategy_alpha()` and `build_ml_strategy_alpha_anchored()` overwrite `assemble_alpha_panel()` output with raw side EV grids, bypassing the panel's clip and eligibility masking.
- `alpha_gate_diagnostics()` is called with `in_fold_valid_alpha_p95_bps` instead of actual OOS/live `alpha_p95_bps`.
- Preservation ratios in `ml_builder.py` compare side-specific panel alpha against a signed `ev_long - ev_short` split, so the ratio can exceed 1.0.
- `_build_strategy_compose_diag()` in `objectives.py` recomputes pre-hurdle mu without respecting `COST_GATE_AMORTIZE`, which can make diagnostics disagree with `compose_mu()`.

## Contract

### `src/domain/futures/strategy/diagnostics.py`

Add:

```python
def nonzero_ratio(arr: np.ndarray, *, eps: float = 1e-12) -> float:
    """Return finite non-zero ratio for a numeric array."""
```

Add:

```python
def preservation_ratio(
    before: np.ndarray,
    after: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Return non-zero survival ratio after gating.

    Raises:
        ValueError: If shapes differ.
    """
```

Modify:

```python
def build_quality_report(
    *,
    feature_values: np.ndarray,
    feature_valid_mask: np.ndarray,
    label_eligible_mask: np.ndarray,
    score_2d: np.ndarray,
    signed_ret_2d: np.ndarray,
    relevance_2d: np.ndarray,
    q10_2d: np.ndarray | None = None,
    q50_2d: np.ndarray | None = None,
    q90_2d: np.ndarray | None = None,
    alpha_long_2d: np.ndarray | None = None,
    alpha_short_2d: np.ndarray | None = None,
    cost_2d: np.ndarray | None = None,
    ic_score_2d: np.ndarray | None = None,
) -> dict[str, float]:
```

Rules:
- Keep backward compatibility by defaulting `ic_score_2d=None`.
- Use `score_2d` for `ranker_valid_ndcg_at_5`.
- Use `ic_score_2d if ic_score_2d is not None else score_2d` for `rolling_ic()`.
- Validate `ic_score_2d.shape == signed_ret_2d.shape` when provided.

Do not change `alpha_gate_diagnostics()` signature. The caller must pass actual OOS/live alpha p95.

### `src/domain/futures/strategy/ml_builder.py`

Within both `build_ml_strategy_alpha()` and `build_ml_strategy_alpha_anchored()`:

Add local side-finalization logic after `ev_long_grid` / `ev_short_grid` are complete:

```python
clip_lim = float(ml_cfg.alpha_clip_bps / 10000.0)
eligible_2d = labels.eligible_mask
alpha_long_final = np.where(
    eligible_2d,
    np.clip(np.maximum(ev_long_grid, 0.0), 0.0, clip_lim),
    0.0,
).astype(np.float32, copy=False)
alpha_short_final = np.where(
    eligible_2d,
    np.clip(np.maximum(ev_short_grid, 0.0), 0.0, clip_lim),
    0.0,
).astype(np.float32, copy=False)
alpha_ic_score = (alpha_long_final - alpha_short_final).astype(np.float64, copy=False)
```

Then:
- Call `assemble_alpha_panel()` with `ev_grid=alpha_ic_score` and the existing eligibility mask.
- Overwrite panel side columns with `alpha_long_final.reshape(-1)` and `alpha_short_final.reshape(-1)`, not raw EV grids.
- Call `build_quality_report(..., score_2d=score_grid, ic_score_2d=alpha_ic_score, alpha_long_2d=alpha_long_final, alpha_short_2d=alpha_short_final)`.
- Keep `in_fold_valid_alpha_p95_bps` as an overfit diagnostic only.
- Call `alpha_gate_diagnostics(alpha_p95_bps=float(quality_report.get("alpha_p95_bps", 0.0)), ...)`.
- Compute preservation ratios from a static cost-gated proxy:

```python
cost_floor = (round_trip_cost_bps() + float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG))) / 10000.0
xs_long_proxy = np.where(alpha_long_final >= cost_floor, alpha_long_final, 0.0)
xs_short_proxy = np.where(alpha_short_final >= cost_floor, alpha_short_final, 0.0)
xs_long_preservation = preservation_ratio(alpha_long_final, xs_long_proxy)
xs_short_preservation = preservation_ratio(alpha_short_final, xs_short_proxy)
```

Important:
- `xs_*_preservation_ratio` must be in `[0.0, 1.0]` for these proxies.
- Do not use `valid_alpha` p95 for the alpha gate.
- Retain `in_fold_valid_alpha_p95_bps` so logs can expose validation-to-OOS collapse.
- Keep purge/embargo unchanged.

### `src/domain/futures/optimization/objectives.py`

Modify:

```python
def _build_strategy_compose_diag(
    *,
    alpha_long: np.ndarray,
    alpha_short: np.ndarray,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    params: dict[str, Any],
    cost_snapshot: CostSnapshot,
    holding_bars: int | None = None,
) -> dict[str, float]:
```

Rules:
- Compute `effective_friction_2d` exactly like `compose_mu()`:

```python
effective_friction_2d = friction_2d
if params.get("COST_GATE_AMORTIZE", False) and holding_bars and int(holding_bars) > 1:
    effective_friction_2d = friction_2d / float(int(holding_bars))
```

- Use `effective_friction_2d` for `mu_l_pre` and `mu_s_pre`.
- Use `effective_friction_bps_2d` for `threshold_bps`.
- Add diagnostics:
  - `cost_gate_amortized`: `1.0` when amortized, else `0.0`
  - `holding_bars`: float holding bars used
  - `raw_friction_bps`: mean raw cost
  - `effective_friction_bps`: mean effective cost
- Pass `holding_bars=holding_bars` from `_compose_strategy_scores_inplace()`.
- Use `preservation_ratio(alpha_long, xs_long)` and `preservation_ratio(alpha_short, xs_short)` from `strategy.diagnostics`.

## Step-by-Step Logic

1. Add `nonzero_ratio()` and `preservation_ratio()` to `diagnostics.py`.
2. Extend `build_quality_report()` with `ic_score_2d`.
3. In `ml_builder.py`, finalize side-specific alpha grids with eligibility mask and clip before panel overwrite.
4. Use side-composed alpha (`alpha_long_final - alpha_short_final`) for IC.
5. Use actual OOS/live `alpha_p95_bps` for `alpha_gate_diagnostics()`.
6. Replace signed-grid preservation logic with post-cost proxy preservation.
7. Apply the same changes to the anchored path.
8. Align `_build_strategy_compose_diag()` with `compose_mu()` cost amortization.
9. Add regression tests for separated IC score, alpha gate p95 source, preservation ratio bounds, and amortized compose diagnostics.

## Surgical Plan

### `src/domain/futures/strategy/diagnostics.py`
[ACTION: ADD]
- Add `nonzero_ratio()` near `ml_alpha_metrics()`.
- Add `preservation_ratio()` near `passes_signal_preservation_gate()`.

[ACTION: REPLACE]
- In `build_quality_report()`, select `ic_input = ic_score_2d if ic_score_2d is not None else score_2d`.
- Compute `ic_series = rolling_ic(ic_input, signed_ret_2d, method="spearman")`.

### `src/domain/futures/strategy/ml_builder.py`
[ACTION: REPLACE]
- In both regular and anchored builders, replace raw side overwrite blocks:

```python
panel.loc[:, "alpha_long"] = ev_long_grid.reshape(-1)
panel.loc[:, "alpha_short"] = ev_short_grid.reshape(-1)
```

with side-finalized clipped/masked grids.

[ACTION: REPLACE]
- Replace `build_quality_report()` calls to pass final side grids and `ic_score_2d=alpha_ic_score`.

[ACTION: REPLACE]
- Replace `alpha_gate_diagnostics(alpha_p95_bps=in_fold_valid_alpha_p95_bps, ...)` with `alpha_p95_bps=alpha_p95_bps`.

[ACTION: REPLACE]
- Replace preservation ratio calculations with `preservation_ratio(alpha_*_final, xs_*_proxy)`.

### `src/domain/futures/optimization/objectives.py`
[ACTION: REPLACE]
- Extend `_build_strategy_compose_diag()` signature with `holding_bars`.
- Use amortized cost when `COST_GATE_AMORTIZE=True`.
- Replace local preservation math with imported `preservation_ratio()`.
- Pass `holding_bars` at the call site.

## Acceptance Criteria
- A run where `alpha_p95_bps < round_trip_cost_bps + EV_HURDLE_BPS` must fail `alpha_gate_pass` even if `in_fold_valid_alpha_p95_bps` is high.
- `xs_long_preservation_ratio` and `xs_short_preservation_ratio` from ML alpha gate proxy must not exceed `1.0`.
- `spearman_rank_ic` must be computed from final side-composed alpha when `ic_score_2d` is provided.
- `ranker_valid_ndcg_at_5` must remain computed from ranker score.
- `_strategy_compose_diag["mu_pre_hurdle_p95_*"]` must match `compose_mu()` output under `COST_GATE_AMORTIZE=True`.
- No look-ahead change: fold boundaries, purge bars, and embargo bars remain unchanged.

## Verification

Run:

```bash
uv run ruff check src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/ml_builder.py src/domain/futures/optimization/objectives.py tests/unit/domain/futures/strategy/test_ml_diagnostics.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/optimization/test_strategy_signal_path.py
```

Expected:
- Exit code 0.

Run:

```bash
uv run mypy src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/ml_builder.py src/domain/futures/optimization/objectives.py
```

Expected:
- Exit code 0.

Run:

```bash
uv run pytest tests/unit/domain/futures/strategy/test_ml_diagnostics.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/optimization/test_strategy_signal_path.py --tb=short
```

Expected:
- Exit code 0.

Optional smoke after implementation:

```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py --mode strategy-smoke --skip-universe --skip-data-sync --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT --trials 1 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1
```

Expected:
- If alpha remains weak, failure reason should be `alpha_p95_below_cost_wall` or negative final alpha IC with bounded preservation ratios.
- It must not report `alpha_gate_pass=True` when actual `alpha_p95_bps` is below the static cost wall.

---

# Addendum: 100-Trial Alpha Fail Follow-up

## Type
refactor

## New Diagnosis
The first refactor fixed the false-positive alpha gate. The latest 100-trial run now fails for the correct reason:

- `alpha_p95_bps=6.66` is below `floor_bps=24.00`.
- Fold-level OOS predictions are not dead: `alpha_nonzero=0.9793`, `0.9665`, and virtual refit `long_nonzero=0.9463`, `short_nonzero=1.0000`.
- Final side alpha is sparse after side split and eligibility masking: `long_nz=0.1196`, `short_nz=0.1239`.
- Static post-cost survival exists but is thin: `xs_long_preservation=0.2100`, `xs_short_preservation=0.1027`.
- `RAW-SIGNAL-DIAG var_retention=0.951` means beta residualization is not the main magnitude sink.

Primary conclusion:
- The current gate compares the cost wall against full-matrix `p95`, where zero-filled inactive cells dominate the percentile.
- Alpha magnitude should be judged on active eligible side predictions, while coverage and tradable density should be judged separately.
- Actual uplift should come from tail-aware EV composition and horizon selection based on OOS active tradable edge, not from relaxing the cost wall.

## Additional Target Files
- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/diagnostics.py`
- `src/domain/futures/strategy/calibrator.py`
- `src/domain/futures/strategy/ml_builder.py`
- `tests/unit/domain/futures/strategy/test_ml_config.py`
- `tests/unit/domain/futures/strategy/test_ml_diagnostics.py`
- `tests/unit/domain/futures/strategy/test_ml_calibrator.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`

## Additional Contract

### `src/domain/futures/strategy/config.py`

Extend `StrategyMLConfig`:

```python
alpha_gate_min_tradable_long_nz: float = 0.01
alpha_gate_min_tradable_short_nz: float = 0.01
ev_tail_blend_weight: float = 0.25
```

Validation:
- `0.0 <= alpha_gate_min_tradable_long_nz <= 1.0`
- `0.0 <= alpha_gate_min_tradable_short_nz <= 1.0`
- `0.0 <= ev_tail_blend_weight <= 1.0`

Quant constraint:
- Do not change `purge_bars`, `embargo_bars`, fold boundaries, or label indexing.
- `ev_tail_blend_weight` is a static config parameter only; it must not be fit on test rows.

### `src/domain/futures/strategy/diagnostics.py`

Add:

```python
def side_alpha_tail_metrics(
    alpha_long: np.ndarray,
    alpha_short: np.ndarray,
    *,
    cost_floor: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Measure active-side magnitude and tradable density separately from full matrix sparsity."""
```

Rules:
- Validate `alpha_long.shape == alpha_short.shape`.
- `alpha_full_matrix_p95_bps`: max side p95 over all finite cells, including zeros.
- `alpha_active_p95_bps`: max side p95 over finite cells where `abs(alpha) > eps`.
- `alpha_long_active_p95_bps`, `alpha_short_active_p95_bps`: side-specific active p95.
- `alpha_long_tradable_nz`, `alpha_short_tradable_nz`: ratio over all finite cells where side alpha is `>= cost_floor`.
- `alpha_long_active_count`, `alpha_short_active_count`: active counts as floats for logging.
- Empty active side returns `0.0` for that side's active p95.

Modify:

```python
def alpha_gate_diagnostics(
    *,
    alpha_p95_bps: float,
    friction_bps: float,
    hurdle_bps: float,
    long_nz: float,
    short_nz: float,
    xs_long_preservation_ratio: float,
    xs_short_preservation_ratio: float,
    min_long_nz: float,
    min_short_nz: float,
    min_xs_preservation: float,
    cost_wall_tolerance_bps: float = 0.0,
    active_alpha_p95_bps: float | None = None,
    tradable_long_nz: float = 0.0,
    tradable_short_nz: float = 0.0,
    min_tradable_long_nz: float = 0.0,
    min_tradable_short_nz: float = 0.0,
) -> dict[str, object]:
```

Rules:
- Use `active_alpha_p95_bps` for the cost-wall comparison when it is not `None`; otherwise use `alpha_p95_bps`.
- Return `alpha_gate_metric_bps` with the exact metric used for the cost-wall comparison.
- Return `alpha_gate_metric_source` as `"active_alpha_p95_bps"` or `"alpha_p95_bps"`.
- Add fail reasons `tradable_long_nz_below_threshold` and `tradable_short_nz_below_threshold`.
- Preserve existing defaults so current callers remain source-compatible.

### `src/domain/futures/strategy/calibrator.py`

Modify:

```python
def compute_conservative_ev(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    cfg: StrategyMLConfig,
) -> NDArray[np.float32]:
```

Rules:
- Keep current behavior when `cfg.ev_tail_blend_weight == 0.0`.
- For `ev_mode == "quantile"`, compute the existing conservative EV first.
- Then apply a bounded tail blend:

```python
tail_room = np.maximum(upper - np.maximum(ev, np.float32(0.0)), np.float32(0.0))
confidence = np.clip(np.abs(median) / uncertainty, np.float32(0.0), np.float32(1.0))
ev = ev + np.float32(cfg.ev_tail_blend_weight) * confidence * tail_room
```

- Clip final EV to `[-alpha_clip_bps, +alpha_clip_bps]`.
- Do not apply tail blend to negative side output after side-specific magnitude targets; side builders already train long and short magnitudes separately.
- For `ev_mode == "prob_x_magnitude"`, leave the existing formula unchanged in this addendum.

Rationale:
- The calibrator already predicts q10/q50/q90 from train/valid-only models.
- Blending toward q90 raises only confident tail predictions and remains bounded by the model's own predicted tail.
- This targets the observed `alpha_p95_below_cost_wall` without injecting future returns or bypassing friction.

### `src/domain/futures/strategy/ml_builder.py`

Within both `build_ml_strategy_alpha()` and `build_ml_strategy_alpha_anchored()`:

1. After `cost_floor` and `xs_*_proxy` are computed, call `side_alpha_tail_metrics()`.
2. Keep `alpha_p95_bps` as the full-matrix metric from `build_quality_report()` for backward-compatible diagnostics.
3. Add these fields to `quality_report`:
   - `alpha_full_matrix_p95_bps`
   - `alpha_active_p95_bps`
   - `alpha_long_active_p95_bps`
   - `alpha_short_active_p95_bps`
   - `alpha_long_tradable_nz`
   - `alpha_short_tradable_nz`
   - `alpha_long_active_count`
   - `alpha_short_active_count`
4. Call `alpha_gate_diagnostics()` with:

```python
active_alpha_p95_bps=float(quality_report.get("alpha_active_p95_bps", 0.0)),
tradable_long_nz=float(quality_report.get("alpha_long_tradable_nz", 0.0)),
tradable_short_nz=float(quality_report.get("alpha_short_tradable_nz", 0.0)),
min_tradable_long_nz=ml_cfg.alpha_gate_min_tradable_long_nz,
min_tradable_short_nz=ml_cfg.alpha_gate_min_tradable_short_nz,
```

5. Runtime failure message must print both:
   - `alpha_gate_metric_bps`
   - `alpha_full_matrix_p95_bps`

6. In horizon experiment mode, replace selection score:

Current:

```python
alpha_p95_bps = float(report.get("in_fold_valid_alpha_p95_bps", 0.0))
score_bps = alpha_p95_bps - floor_bps
```

New:

```python
active_p95_bps = float(report.get("alpha_active_p95_bps", report.get("alpha_p95_bps", 0.0)))
tradable_density = min(
    float(report.get("alpha_long_tradable_nz", 0.0)),
    float(report.get("alpha_short_tradable_nz", 0.0)),
)
score_bps = (active_p95_bps - floor_bps) + 10.0 * tradable_density
```

Rules:
- Record `active_p95_bps`, `full_matrix_p95_bps`, `tradable_density`, and `score_bps` in `horizon_experiment`.
- Do not use `in_fold_valid_alpha_p95_bps` for horizon selection except as an overfit diagnostic.

## Additional Step-by-Step Logic

1. Add config knobs with strict validation.
2. Add active-tail alpha diagnostics that separate magnitude from sparse coverage.
3. Extend alpha gate to use active p95 for cost-wall comparison and tradable density for coverage realism.
4. Add bounded q90 tail blend to conservative EV.
5. Wire active-tail diagnostics into regular and anchored ML builders.
6. Update horizon experiment scoring to use OOS active tradable edge.
7. Add tests for active p95, tradable density fail reasons, tail blend bounds, config validation, and horizon score fields.

## Additional Acceptance Criteria

- Given a matrix where only 12% of cells are non-zero but the active tail is above cost, `alpha_gate_metric_bps` must use `alpha_active_p95_bps`, not full-matrix `alpha_p95_bps`.
- Full-matrix p95 must remain logged as `alpha_full_matrix_p95_bps`.
- If active p95 passes but tradable density is below configured minimum, the gate must fail with `tradable_*_nz_below_threshold`.
- `compute_conservative_ev()` with `ev_tail_blend_weight=0.0` must be numerically identical to the current behavior.
- `compute_conservative_ev()` with `ev_tail_blend_weight>0.0` must not exceed q90 or `alpha_clip_bps`.
- Horizon experiment selection must not depend on `in_fold_valid_alpha_p95_bps`.
- No look-ahead change: all diagnostics and EV transforms use model outputs available at inference time plus static/ex-ante cost.

## Additional Verification

Run:

```bash
uv run ruff check src/domain/futures/strategy/config.py src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/calibrator.py src/domain/futures/strategy/ml_builder.py tests/unit/domain/futures/strategy/test_ml_config.py tests/unit/domain/futures/strategy/test_ml_diagnostics.py tests/unit/domain/futures/strategy/test_ml_calibrator.py tests/unit/domain/futures/strategy/test_ml_builder.py
```

Expected:
- Exit code 0.

Run:

```bash
uv run mypy src/domain/futures/strategy/config.py src/domain/futures/strategy/diagnostics.py src/domain/futures/strategy/calibrator.py src/domain/futures/strategy/ml_builder.py
```

Expected:
- Exit code 0.

Run:

```bash
uv run pytest tests/unit/domain/futures/strategy/test_ml_config.py tests/unit/domain/futures/strategy/test_ml_diagnostics.py tests/unit/domain/futures/strategy/test_ml_calibrator.py tests/unit/domain/futures/strategy/test_ml_builder.py --tb=short
```

Expected:
- Exit code 0.

Run 100-trial strategy smoke after implementation:

```bash
PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode strategy --skip-universe --skip-data-sync --symbols BTCUSDT --trials 100 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1
```

Expected:
- Gate failure, if any, must distinguish active magnitude failure from tradable-density failure.
- Passing alpha must show `alpha_gate_metric_bps >= alpha_gate_floor_bps`.
- Reported `alpha_full_matrix_p95_bps` may remain below floor when alpha is intentionally sparse.
