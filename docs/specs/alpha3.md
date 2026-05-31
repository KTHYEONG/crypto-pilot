---
title: Alpha signed-rank contract repair and robust OOS evaluation
domain: futures-alpha
type: bug-fix
status: ready
priority: critical
ai_read_policy: when_related
last_verified: 2026-05-31
related_paths:
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/execution/opt_main_futures.py
  - src/domain/futures/strategy/ml_builder.py
  - tests/unit/domain/futures/strategy/test_alpha_evaluation.py
dependencies:
  documents: [docs/specs/alpha1.md, docs/specs/alpha2.md, docs/results/re-alpha.md]
change_triggers:
  - src/execution/opt_main_futures.py
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/domain/futures/strategy/ml_builder.py
---

# Alpha signed-rank contract repair and robust OOS evaluation

## 0. Current Evidence

Phase 1-6 of `alpha2.md` fixed OOS date extraction:

```text
[OOS-DIAG] rank_cols=96 finite_rows=1417 oos_idx=1417 common_idx=1417
```

The remaining failure is not empty OOS coverage. The current smoke run shows a signal-contract collapse after ML scoring:

```text
[SCORE-IC] dense_ranker ic=0.0347 t=4.51 hit=0.570 breadth=3.7
[RESID-IC] raw=0.0347 resid=0.0361 resid_hit=0.564
[RANK-SCOREBOARD] net_signal applied: q=0.35 long_nz=0.400 short_nz=0.400
[IC-DECOMP] dense_c1_raw=0.0021 dense_c1_resid=0.0021 dense_c3_raw=0.0045 dense_c3_resid=0.0045
[RANK-QUALITY L1] ic=0.0000 t=0.00 hit=0.000 breadth=0.0
[RANK-IC C3] ic=0.0000 t=0.00 breadth=0.00
[L3-BASKET] ew_bps=nan net_bps=nan ir_t=nan hit=nan n=0
```

Interpretation:

- ML builder's dense ranker skill is alive (`ic=0.0347`, `t=4.51`).
- `opt_main_futures.py` reconstructs a signed dense signal as `rank_score_long - rank_score_short`.
- In current signed single-ranker mode, `rank_score_short` is the same raw signed ranker score evaluated on the short-side dataset. Lower values are better for shorts.
- Therefore `rank_score_long - rank_score_short` can become all-zero for cofinite cells, destroying L1/C3 rank IC, monotonicity, basket diagnostics, and `ALPHA_PASS`.

Root cause:

```text
rank_score_long semantic: higher score is better long
rank_score_short semantic: lower score is better short, not positive short attractiveness

current eval formula:
    signed = rank_score_long - rank_score_short

signed single-ranker case:
    rank_score_long ~= rank_score_short ~= raw signed score
    signed ~= 0
```

This is a contract mismatch between `ml_builder.py` score metadata and `opt_main_futures.py` evaluation reconstruction.

## 1. Target Files

- `src/domain/futures/strategy/alpha_evaluation.py`
- `src/execution/opt_main_futures.py`
- `src/domain/futures/strategy/ml_builder.py`
- `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py` if an existing lightweight hook is available
- `docs/results/re-alpha.md`

## 2. Contracts

### 2.1 Add rank score contract helper

File: `src/domain/futures/strategy/alpha_evaluation.py`

```python
def derive_signed_rank_signal(
    rank_score_long_2d: NDArray[np.float64],
    rank_score_short_2d: NDArray[np.float64],
    *,
    same_score_rtol: float = 1e-6,
    same_score_atol: float = 1e-8,
) -> NDArray[np.float64]:
    """Derive a signed dense rank signal from long/short rank score panels.

    Current single-ranker contract:
        rank_score_long: raw signed rank score; higher is better for long.
        rank_score_short: raw signed rank score; lower is better for short.

    If long and short score panels are effectively the same on cofinite cells,
    the signed signal is the raw score itself. Otherwise, fall back to the
    legacy dual-side interpretation: 0.5 * (long - short).
    """
```

Required behavior:

- Input shape must match. If not, raise `ValueError`.
- Preserve `NaN` only where both sides are non-finite.
- For signed single-ranker mode where cofinite values are effectively equal, return the raw signed score:

```python
signed = np.where(np.isfinite(long_arr), long_arr, short_arr)
```

- For legacy dual-side mode, return:

```python
signed = 0.5 * (long_arr - short_arr)
```

- Use only score arrays. Do not use realized returns to choose the contract branch. This prevents look-ahead leakage.

### 2.2 Optional metadata contract

File: `src/domain/futures/strategy/ml_builder.py`

Add metadata to both WF and AWF alpha panels:

```python
panel.attrs["rank_score_contract"] = {
    "version": 1,
    "mode": "signed_single_ranker",
    "long_higher_is_better": True,
    "short_lower_is_better": True,
    "signed_signal_formula": "derive_signed_rank_signal(rank_score_long, rank_score_short)",
}
```

This is diagnostic metadata only. Do not gate runtime behavior on this metadata in `alpha3`; the helper must remain robust if older artifacts lack attrs.

### 2.3 Forward return helper contract

File: `src/execution/opt_main_futures.py`

Add a private local helper near `_run_alpha_evaluation`:

```python
def _forward_log_return_on_index(
    close: pd.Series,
    target_index: pd.Index,
    horizon_bars: int,
) -> pd.Series:
    """Compute forward log return on the full series, then align to target_index."""
```

Required behavior:

- Convert `close.index` with `pd.to_datetime(..., utc=True).tz_localize(None)`.
- Sort by index before shifting.
- Compute `np.log(close.shift(-horizon_bars) / close)` on the full contiguous series.
- Reindex the result to `target_index`.
- Never reindex the close series before `shift(-horizon_bars)`.

## 3. Step-by-Step Logic

### Step A. Encode signed rank contract

1. Add `derive_signed_rank_signal()` to `alpha_evaluation.py`.
2. Detect same-score mode with cofinite cells only:

```python
finite = np.isfinite(long_arr) & np.isfinite(short_arr)
if not np.any(finite):
    return np.where(np.isfinite(long_arr), long_arr, short_arr)
scale = max(float(np.nanmedian(np.abs(long_arr[finite]))), 1.0)
same_score = bool(
    np.nanmedian(np.abs(long_arr[finite] - short_arr[finite]))
    <= same_score_atol + same_score_rtol * scale
)
```

3. If `same_score`, return the raw signed score.
4. Else return `0.5 * (long - short)`.

### Step B. Replace zeroing formulas in alpha evaluation orchestration

File: `src/execution/opt_main_futures.py`

Replace all rank dense signal reconstruction paths:

```python
_net_signal = _rs_l - _rs_s
```

with:

```python
_net_signal = derive_signed_rank_signal(_rs_l, _rs_s)
```

Replace:

```python
_rank_pred_c3 = _rl_c3 - _rs_c3
_rank_pred_c1 = _rank_l_c1 - _rank_s_c1
```

with:

```python
_rank_pred_c3 = derive_signed_rank_signal(_rl_c3, _rs_c3)
_rank_pred_c1 = derive_signed_rank_signal(_rank_l_c1, _rank_s_c1)
```

Add one concise diagnostic after C3 signal derivation:

```python
_logger.info(
    "[RANK-CONTRACT] c3_signed_nz=%.3f c3_signed_std=%.6f c3_long_short_absdiff_p50=%.6f",
    float(np.count_nonzero(np.isfinite(_rank_pred_c3) & (_rank_pred_c3 != 0.0)) / max(_rank_pred_c3.size, 1)),
    float(np.nanstd(_rank_pred_c3)),
    float(np.nanmedian(np.abs(_rl_c3 - _rs_c3))),
)
```

Expected change:

- `[RANK-QUALITY L1]` should no longer have `breadth=0.0` solely because long/short score panels are identical.
- `[RANK-IC C3]` should reflect the same signed rank skill family seen in `[SCORE-IC]`, subject to C3 filtering and beta residualization.
- `[MONOTONICITY]` and `[L3-BASKET]` should produce non-NaN observations when selected alpha emits long/short baskets.

### Step C. Consolidate forward return calculation

File: `src/execution/opt_main_futures.py`

Use `_forward_log_return_on_index()` in every alpha evaluation realized-return path:

- primary `realized_df`
- `_c1_real_rows_rs`
- `c1_real_rows`
- multi-horizon `realized_map`

This closes the remaining variants of the `alpha1.md` reindex-before-shift bug.

Required replacement pattern:

```python
fwd_ret = _forward_log_return_on_index(df["close"], pivot_long.index, horizon)
```

and:

```python
fwd = _forward_log_return_on_index(df["close"], common_idx, h)
```

### Step D. Preserve quant integrity

Check the implementation against:

- Anti-bias: branch selection for signed rank contract must not inspect realized returns.
- Time-series integrity: forward returns must be computed before target index filtering.
- Trading realism: `cost_floor_bps=24.0` and current breakeven gates remain unchanged.
- Statistical robustness: no pass criteria should be loosened to force `ALPHA_PASS=TRUE`.

## 4. Surgical Plan

### `src/domain/futures/strategy/alpha_evaluation.py`

[ACTION: ADD]

Add `derive_signed_rank_signal()` after `compute_net_ic()` or before it if tests prefer helper grouping.

[CODE_OR_INSTRUCTION]

Implement with strict shape validation, no realized-return inputs, and Google Style docstring.

### `src/execution/opt_main_futures.py`

[ACTION: REPLACE]

Import `derive_signed_rank_signal` together with existing alpha evaluation imports in `_run_alpha_evaluation`.

[ACTION: ADD]

Add `_forward_log_return_on_index()` near `_run_alpha_evaluation`.

[ACTION: REPLACE]

Replace all `rank_score_long - rank_score_short` evaluation formulas with `derive_signed_rank_signal()`.

[ACTION: REPLACE]

Replace all local alpha evaluation forward-return calculations with `_forward_log_return_on_index()`.

### `src/domain/futures/strategy/ml_builder.py`

[ACTION: ADD]

Add `panel.attrs["rank_score_contract"]` in both WF and AWF panel creation paths.

### `tests/unit/domain/futures/strategy/test_alpha_evaluation.py`

[ACTION: ADD]

Add tests:

```python
def test_derive_signed_rank_signal_same_score_returns_raw_score() -> None:
    rank = np.array([[-2.0, 0.0, 3.0], [1.0, -1.0, 0.5]], dtype=np.float64)
    signed = derive_signed_rank_signal(rank, rank.copy())
    np.testing.assert_allclose(signed, rank)
```

```python
def test_derive_signed_rank_signal_dual_side_uses_half_spread() -> None:
    rank_l = np.array([[2.0, 1.0, -1.0]], dtype=np.float64)
    rank_s = np.array([[-2.0, -1.0, 1.0]], dtype=np.float64)
    signed = derive_signed_rank_signal(rank_l, rank_s)
    np.testing.assert_allclose(signed, np.array([[2.0, 1.0, -1.0]], dtype=np.float64))
```

```python
def test_derive_signed_rank_signal_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        derive_signed_rank_signal(np.zeros((2, 2)), np.zeros((2, 3)))
```

### `tests/unit/execution/test_opt_main_futures_strategy_mode.py`

[ACTION: ADD IF LOW-COST]

Add a focused regression test only if existing fixtures allow it without running the full pipeline:

- Build two identical rank score pivots.
- Assert the evaluation reconstruction path produces non-zero signed signal via `derive_signed_rank_signal`.

If this requires heavy pipeline setup, do not add it. The helper unit tests are the minimum required guard.

### `docs/results/re-alpha.md`

[ACTION: REPLACE]

After implementation and smoke run, update:

- latest `RANK-CONTRACT`
- latest `RANK-QUALITY L1`
- latest `RANK-IC C3`
- latest `L3-BASKET`
- latest `ALPHA_PASS`
- whether failure remains signal quality/cost/regime, not contract collapse

## 5. Verification

### L1

```bash
uv run ruff check --fix src/domain/futures/strategy/alpha_evaluation.py src/execution/opt_main_futures.py src/domain/futures/strategy/ml_builder.py tests/unit/domain/futures/strategy/test_alpha_evaluation.py
```

Expected:

```text
All checks passed!
```

```bash
uv run mypy src/domain/futures/strategy/alpha_evaluation.py src/execution/opt_main_futures.py src/domain/futures/strategy/ml_builder.py
```

Expected:

```text
Success: no issues found
```

### L2

```bash
uv run pytest tests/unit/domain/futures/strategy/test_alpha_evaluation.py --tb=short
```

Expected:

```text
passed
```

```bash
uv run pytest tests/unit/execution/test_opt_main_futures_strategy_mode.py tests/e2e/test_cli_modes.py tests/integration/execution/test_opt_main_futures_bypass.py --tb=short
```

Expected:

```text
passed
```

### Smoke

```bash
uv run python src/execution/opt_main_futures.py \
  --mode alpha --sync-mode skip --trials 1 --tf 4h \
  --reference-date 2026-05-01
```

Expected minimum:

```text
[OOS-DIAG] ... finite_rows=1417 ... common_idx=1417
[RANK-CONTRACT] c3_signed_nz>0 c3_signed_std>0
[RANK-QUALITY L1] breadth>0
[RANK-IC C3] breadth>0
[L3-BASKET] n>0
```

`ALPHA_PASS` is not required to become `TRUE` in this spec. The required outcome is that failure, if any, is no longer caused by signed-rank reconstruction collapse (`breadth=0`, `ic=0`, `basket n=0`).

## 6. Acceptance Checklist

```text
[ ] Signed rank signal helper added and unit-tested.
[ ] `rank_score_long - rank_score_short` zero-collapse removed from alpha evaluation orchestration.
[ ] Forward returns in alpha evaluation are always computed before reindexing.
[ ] Smoke shows `RANK-CONTRACT` non-zero signed signal and non-zero C3 rank breadth.
[ ] `docs/results/re-alpha.md` updated with post-alpha3 evidence.
[ ] No gate threshold is relaxed to force pass.
```
