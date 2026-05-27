# Spec: Strategy OOS Alpha Readiness and Cost-Wall Diagnostics

## Type
bug-fix

## Target Files
- `src/application/futures/optimization/strategy_service.py`
- `src/domain/futures/strategy_runtime/bridge.py`
- `src/domain/futures/strategy/ml_builder.py`
- `tests/unit/application/futures/optimization/test_strategy_service.py`
- `tests/unit/domain/futures/strategy/test_ml_builder.py`

## Diagnosis
The latest 100-trial run still fails, but the observed failure is no longer the original alpha gate bug.

Observed facts:
- ML alpha generation is alive:
  - `ML-OOS alpha_nonzero=0.9983`, `0.9963`
  - virtual refit `long_nonzero=0.9848`, `short_nonzero=1.0000`
  - merged summary before readiness check: `ALPHA-MERGE alpha_long_nz=0.106 alpha_short_nz=0.106`
- `assert_strategy_alpha_ready()` then fails with:
  - `strategy merge produced zero-only alpha columns (nonzero long=0, short=0)`

Root cause:
- `pick_strategy_data_maps()` intentionally sends IS-only maps into the strategy bridge to avoid look-ahead.
- `merge_ml_output_into_is_and_oos()` merges that IS-trained alpha panel into both IS maps and full-history OOS maps.
- `filtered_oos_maps[sym][tf]` is not OOS-only. It is the full frame, and `oos_start_idx_{tf}` points to the target OOS start.
- `assert_strategy_alpha_ready()` currently counts only `df["alpha_*"][oos_start_idx:]`.
- Because the preflight alpha panel was built from IS maps, the target OOS tail can legitimately be zero at this stage. The readiness check is therefore slicing away the actual merged panel window and reporting a false zero-only failure.

Secondary issue:
- `[ML-COST-WALL]` still logs legacy `alpha_p95_bps` only. After the active-tail refactor, the actionable gate metric is `alpha_gate_metric_bps` with `alpha_gate_metric_source`. The current log makes a passing active-tail gate look like a full-matrix cost-wall failure.

Quant constraints:
- Do not train on target OOS labels during preflight readiness.
- Do not weaken the cost wall.
- Do not treat IS-generated alpha as valid target OOS alpha.
- Separate "merge contract is valid" from "target OOS tradable alpha is available".

## Contract

### `src/application/futures/optimization/strategy_service.py`

Add:

```python
from dataclasses import dataclass
```

Add:

```python
@dataclass(frozen=True, slots=True)
class StrategyAlphaReadinessReport:
    """Alpha merge readiness diagnostics for strategy preflight."""

    merged_symbols: int
    panel_long_non_zero: int
    panel_short_non_zero: int
    merged_panel_long_non_zero: int
    merged_panel_short_non_zero: int
    target_oos_long_non_zero: int
    target_oos_short_non_zero: int
    target_oos_rows: int
    panel_start: str
    panel_end: str
    warnings: tuple[str, ...] = ()
```

Add:

```python
def summarize_strategy_alpha_readiness(
    *,
    ml_out: MLPipelineOutput,
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> StrategyAlphaReadinessReport:
    """Summarize alpha-panel merge coverage without assuming target OOS coverage."""
```

Rules:
- Validate `ml_out.alpha_panel` is non-empty and has `alpha_long` / `alpha_short`.
- Call `validate_alpha_forecast_metadata(alpha_panel)`.
- Normalize alpha panel datetimes and frame datetimes with `pd.to_datetime(..., utc=True).dt.tz_localize(None)`.
- Count `panel_long_non_zero` and `panel_short_non_zero` directly from the panel.
- For each symbol:
  - confirm merged frame has `alpha_long` and `alpha_short`, else raise `RuntimeError("strategy merge missing alpha columns for symbol=...")`.
  - build `panel_window_mask` using symbol-specific panel datetimes:

```python
panel_dt_by_symbol = {
    sym: set(sym_rows["_merge_datetime"].to_numpy())
    for sym, sym_rows in panel_reset.groupby("symbol", sort=False)
}
panel_window_mask = frame_dt.isin(panel_dt_by_symbol.get(sym, set())).to_numpy()
```

  - count merged non-zero alpha on `panel_window_mask`, not on `oos_start_idx:`.
  - separately count target OOS non-zero alpha on `np.arange(len(df)) >= oos_start_idx`.
- If `target_oos_rows > 0` and target OOS long/short non-zero are zero while merged panel counts are positive, add warning:
  - `"target_oos_alpha_absent_preflight"`
- `panel_start` and `panel_end` are ISO-like strings from normalized panel datetime min/max.

Modify:

```python
def assert_strategy_alpha_ready(
    *,
    ml_out: MLPipelineOutput,
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
    require_target_oos_alpha: bool = False,
) -> StrategyAlphaReadinessReport:
```

Rules:
- Preserve backward compatibility with default `require_target_oos_alpha=False`.
- Return `StrategyAlphaReadinessReport` on success.
- Fail if no merged frames exist:
  - `RuntimeError("strategy mode has no merged symbol frames for selected timeframe")`
- Fail if panel itself is zero-only:
  - `RuntimeError("strategy alpha_panel is zero-only (nonzero long=..., short=...)")`
- Fail if merged panel window is zero-only:
  - `RuntimeError("strategy merge produced zero-only alpha columns in panel window (nonzero long=..., short=...)")`
- Only when `require_target_oos_alpha=True`, fail if target OOS slice is zero-only:
  - `RuntimeError("strategy target OOS alpha is zero-only (nonzero long=..., short=...)")`
- Log warning when `report.warnings` is non-empty.

### `src/domain/futures/strategy_runtime/bridge.py`

Modify `merge_ml_output_into_data_maps()` logging only.

Rules:
- Keep merge behavior unchanged.
- Add panel coverage fields to `[ALPHA-MERGE]` summary:
  - `panel_start`
  - `panel_end`
  - `target_oos_long_nz` and `target_oos_short_nz` when `oos_start_idx_{tf}` exists
- Do not change alpha values or datetime matching in this spec.

### `src/domain/futures/strategy/ml_builder.py`

Modify the regular path `[ML-COST-WALL]` log.

Current legacy log:

```python
"[ML-COST-WALL] alpha_p95=..."
```

Replace with:

```python
_gate_metric_bps = float(
    quality_report.get(
        "alpha_gate_metric_bps",
        quality_report.get("alpha_active_p95_bps", quality_report.get("alpha_p95_bps", 0.0)),
    )
)
_gate_metric_source = str(
    quality_report.get(
        "alpha_gate_metric_source",
        "alpha_active_p95_bps" if "alpha_active_p95_bps" in quality_report else "alpha_p95_bps",
    )
)
_full_matrix_p95 = float(
    quality_report.get("alpha_full_matrix_p95_bps", quality_report.get("alpha_p95_bps", 0.0))
)
_active_p95 = float(quality_report.get("alpha_active_p95_bps", _full_matrix_p95))
_logger.info(
    "[ML-COST-WALL] gate_metric=%.2fbps source=%s full_matrix_p95=%.2fbps "
    "active_p95=%.2fbps friction=%.1fbps hurdle_default=%.1fbps floor=%.1fbps "
    "gate_clears_floor=%s",
    _gate_metric_bps,
    _gate_metric_source,
    _full_matrix_p95,
    _active_p95,
    _friction_bps,
    _hurdle_default_bps,
    _floor_bps,
    str(_gate_metric_bps >= _floor_bps),
)
```

Rules:
- Do not change alpha gate logic.
- Do not change `quality_report` values.
- This is observability only.

## Step-by-Step Logic

1. Add `StrategyAlphaReadinessReport`.
2. Add `summarize_strategy_alpha_readiness()` that counts:
   - alpha panel non-zero counts
   - merged non-zero counts in panel-covered rows
   - target OOS non-zero counts separately
3. Modify `assert_strategy_alpha_ready()` to validate panel-window merge by default.
4. Keep target OOS zero-only as warning by default.
5. Add `require_target_oos_alpha=True` option for future callers that explicitly expect live/OOS alpha.
6. Enrich bridge merge logs with panel and target-OOS coverage.
7. Replace the legacy `ML-COST-WALL` log with gate metric source-aware logging.
8. Add regression tests covering:
   - full-history OOS map where panel ends before `oos_start_idx` passes by default
   - the same fixture fails with `require_target_oos_alpha=True`
   - zero-only panel still fails
   - merged panel window zero-only still fails
   - cost-wall log uses `alpha_gate_metric_bps` / source fields

## Surgical Plan

### `src/application/futures/optimization/strategy_service.py`
[ACTION: ADD]
- Add the dataclass and summary function above `assert_strategy_alpha_ready()`.

[ACTION: REPLACE]
- Replace `assert_strategy_alpha_ready()` body with:
  - call `summarize_strategy_alpha_readiness()`
  - apply panel-level and merged-panel checks
  - apply optional target-OOS check only when requested
  - return report

### `src/domain/futures/strategy_runtime/bridge.py`
[ACTION: REPLACE]
- Replace `[ALPHA-MERGE] summary log` block with a version that computes:
  - merged symbol count
  - mean alpha non-zero ratios
  - target OOS non-zero ratios if `oos_start_idx_{tf}` exists
  - panel start/end from `panel.index.get_level_values("datetime")`

### `src/domain/futures/strategy/ml_builder.py`
[ACTION: REPLACE]
- Replace only the `[ML-COST-WALL]` logging block.
- No model, gate, fold, label, or alpha mutation is allowed.

### `tests/unit/application/futures/optimization/test_strategy_service.py`
[ACTION: ADD]
- Add test where OOS frame has 4 rows, `oos_start_idx_4h=3`, alpha panel covers rows 0-1 only, merged alpha exists in rows 0-1, and `assert_strategy_alpha_ready(..., require_target_oos_alpha=False)` passes with warning `"target_oos_alpha_absent_preflight"`.
- Add same fixture with `require_target_oos_alpha=True` and expect `RuntimeError("strategy target OOS alpha is zero-only")`.
- Add zero-only panel regression.
- Add merged panel-window zero-only regression.

### `tests/unit/domain/futures/strategy/test_ml_builder.py`
[ACTION: ADD]
- Add or update a logging-focused test that injects a `quality_report` containing:
  - `alpha_gate_metric_bps=30.0`
  - `alpha_gate_metric_source="active_alpha_p95_bps"`
  - `alpha_full_matrix_p95_bps=8.0`
  - `alpha_active_p95_bps=30.0`
- Assert the cost-wall log or constructed logger call includes `gate_metric` and `source`.

## Acceptance Criteria

- A full-history OOS map whose `oos_start_idx_{tf}` is after the alpha panel end must not fail preflight readiness when panel-window merge has non-zero long and short alpha.
- The same fixture must fail when `require_target_oos_alpha=True`.
- Zero-only alpha panel must still fail.
- A merge that loses alpha inside the alpha panel datetime window must still fail.
- Readiness report must expose target OOS alpha absence as a warning, not as the default failure reason.
- `[ML-COST-WALL]` must show the actual gate metric and source, so full-matrix p95 below floor no longer masks active-tail gate state.
- No look-ahead change: preflight readiness must not train on, infer from, or relabel target OOS rows.

## Verification

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run ruff check src/application/futures/optimization/strategy_service.py src/domain/futures/strategy_runtime/bridge.py src/domain/futures/strategy/ml_builder.py tests/unit/application/futures/optimization/test_strategy_service.py tests/unit/domain/futures/strategy/test_ml_builder.py
```

Expected:
- Exit code 0.

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run mypy src/application/futures/optimization/strategy_service.py src/domain/futures/strategy_runtime/bridge.py src/domain/futures/strategy/ml_builder.py
```

Expected:
- Exit code 0.

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run pytest tests/unit/application/futures/optimization/test_strategy_service.py tests/unit/domain/futures/strategy/test_ml_builder.py --tb=short
```

Expected:
- Exit code 0.

Run strategy smoke:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode strategy-smoke --skip-universe --skip-data-sync --symbols BTCUSDT --trials 1 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1
```

Expected:
- Must not fail with `strategy merge produced zero-only alpha columns`.
- If target OOS alpha is absent during preflight, the log must expose `target_oos_alpha_absent_preflight`.

Run 100-trial strategy command:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode strategy --skip-universe --skip-data-sync --symbols BTCUSDT --trials 100 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1
```

Expected:
- Must not fail at preflight with `strategy merge produced zero-only alpha columns`.
- If it fails later, the failure reason must come from optimization/evaluation economics or explicit target OOS alpha requirements, not from default preflight readiness slicing.
