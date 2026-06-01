---
title: Alpha production uplift via EMA score-smoothing and robust validation calibration
domain: futures-alpha
type: spec
status: proposed
priority: critical
ai_read_policy: always
created: 2026-06-01
references:
  - docs/specs/alpha1.md
  - docs/specs/alpha2.md
  - docs/specs/alpha3.md
  - docs/results/re-alpha.md
target_phase: alpha4
---

# Alpha Production Uplift via EMA Score-Smoothing and Robust Validation Calibration (Phase 4)

## 0. Technical Context & Diagnostic Audit

The current documented status is `ALPHA_PASS=FALSE` with the blocker `policy_economics.validation_net_lcb_non_positive` (all candidate policies are rejected in `calibrate_rank_portfolio_policy()`, causing fold calibration to fall back to the dummy `no-trade` policy).

Our deep quantitative audit of `rank_selection.py` reveals two critical structural issues:
1. **Horizon Mismatch & Cost Accumulation**: While the portfolio selection models are calibrated to a 12h or 18h holding horizon, `build_signed_rank_weights()` builds and rebalances the entire cross-sectional portfolio from scratch on every 4h bar `t`. Because ML ranker scores fluctuate with high-frequency noise, this bar-by-bar rebalancing triggers massive turnover (e.g. 0.40 - 0.60 per bar). Accumulating 24bps (fallback + hurdle) cost per bar over 12 bars consumes ~144bps of edge, completely wiping out the ~15bps cross-sectional return.
2. **Pooled Monotonicity Contamination**: `_score_monotonicity()` pools cross-sectional bin scores and returns across all time steps `t` into one 1D array of size `5 * T`. Because raw realized returns suffer from massive time-series volatility (market beta and regime drift), pooling them directly contaminates the cross-sectional spearman correlation with time-series noise, leading to negative or near-zero `validation_monotonicity` which triggers strict candidate rejections.

To resolve these issues, we must introduce **EMA Score-Smoothing** along the time dimension and **Robust Monotonicity Evaluation**.

---

## 1. 아키텍처 설계 (EMA score-smoothing & Robust Validation)

### 1.1 Time-Series EMA score-smoothing
Apply a rolling Exponential Moving Average (EMA) to the 2D prediction panel `signed_score_2d` along axis 0 (time) before computing z-scores and weights.
- **Smoothing Span**: Equal to the candidate's `holding_bars` (e.g. 12 or 18 bars).
- **Rationale**: Filters high-frequency signal noise, aligns the trading signal speed with the target prediction horizon, and drastically reduces portfolio turnover/transaction cost.
- **Handling NaNs**: The EMA must propagate correctly, ignoring NaNs and warming up cleanly to prevent data leakage.

### 1.2 Cross-Sectional Monotonicity
Fix `_score_monotonicity` to correlate cross-sectionally standardized/de-meaned returns, or compute spearman correlation bar-by-bar and average them, preventing time-series regime shifts from contaminating the cross-sectional rank signal quality.

---

## 2. Target Files

- `src/domain/futures/strategy/rank_selection.py`
- `tests/unit/domain/futures/strategy/test_rank_selection.py`
- `docs/results/re-alpha.md`

---

## 3. Contracts & Surgical Plan

### 3.1 `src/domain/futures/strategy/rank_selection.py`

#### [ACTION: REPLACE] - Add `_ema_2d` helper and integrate EMA score-smoothing into `build_signed_rank_weights`

We will implement an efficient, leakage-free 2D EMA smoothing helper `_ema_2d()` and call it on `signed_score_2d` inside `build_signed_rank_weights()` using `span = policy.holding_bars`.

```python
def _ema_2d(arr: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    """Compute rolling Exponential Moving Average along axis 0 (time) with NaNs preservation."""
    if span <= 1 or arr.shape[0] == 0:
        return arr
    alpha = 2.0 / (span + 1.0)
    out = np.full_like(arr, np.nan)
    
    last_val = np.zeros(arr.shape[1], dtype=np.float64)
    initialized = np.zeros(arr.shape[1], dtype=bool)
    
    for t in range(arr.shape[0]):
        val = arr[t]
        mask = np.isfinite(val)
        
        # Update where already initialized
        to_update = mask & initialized
        last_val[to_update] = (1.0 - alpha) * last_val[to_update] + alpha * val[to_update]
        
        # Initialize where first valid
        to_init = mask & ~initialized
        last_val[to_init] = val[to_init]
        initialized[to_init] = True
        
        out[t, initialized] = last_val[initialized]
    return out
```

Inside `build_signed_rank_weights()`:
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
    """Convert rank scores to signed portfolio weights with EMA score-smoothing."""
    score_raw = np.asarray(signed_score_2d, dtype=np.float64)
    eligible = np.asarray(eligible_2d, dtype=bool)
    
    # Apply EMA score-smoothing along axis 0 (time) to align signal frequency with holding horizon
    score_2d = _ema_2d(score_raw, span=policy.holding_bars)
    
    z = _cs_zscore_2d(score_2d)
    signed = policy.polarity * z
    ...
```

#### [ACTION: REPLACE] - Fix `_score_monotonicity` pooled return bias

We will de-mean the realized returns cross-sectionally per bar before pooling them for the Spearman correlation, isolating pure cross-sectional rank skill from time-series regime shifts.

```python
def _score_monotonicity(
    score_2d: NDArray[np.float64],
    realized_2d: NDArray[np.float64],
    eligible_2d: NDArray[np.bool_],
) -> float:
    """Compute cross-sectional rank score monotonicity using de-meaned returns to avoid pooled beta bias."""
    bucket_scores: list[float] = []
    bucket_rets: list[float] = []
    for t in range(score_2d.shape[0]):
        score = score_2d[t]
        ret = realized_2d[t]
        mask = eligible_2d[t] & np.isfinite(score) & np.isfinite(ret)
        n_ok = int(np.count_nonzero(mask))
        if n_ok < 10:
            continue
        s = score[mask]
        # Cross-sectionally de-mean to isolate rank edge from market return
        r = (ret[mask] - np.mean(ret[mask])) * 1e4
        
        edges = np.quantile(s, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(s, edges, right=True)
        for b in range(5):
            bmask = bins == b
            if int(np.count_nonzero(bmask)) < 2:
                continue
            bucket_scores.append(float(np.mean(s[bmask])))
            bucket_rets.append(float(np.mean(r[bmask])))
            
    if len(bucket_scores) < 5:
        return float("nan")
    rho = scipy.stats.spearmanr(np.asarray(bucket_scores), np.asarray(bucket_rets)).statistic
    return float(rho) if np.isfinite(rho) else float("nan")
```

---

## 4. Verification & Target Metrics

- **Unit Tests**: Run `uv run pytest tests/unit/domain/futures/strategy/test_rank_selection.py` to confirm functional compatibility.
- **E2E Smoke Test**:
  ```bash
  UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
  ```
- **Target Outcome**: Validation LCB becomes highly positive due to massive transaction cost reduction, allowing the OOS panel to populate and pass the Alpha phase gates successfully.

---

## 5. Future Improvements

### 5.1 Dynamic Monotonicity Threshold
To prevent excessive bar omissions under small-universe regimes (e.g. extreme crypto bear markets), consider replacing the static `n_ok < 10` filter in `_score_monotonicity` with a dynamic threshold such as `n_ok < max(5, n_universe // 3)`. This will improve robustness in regime calibration without sacrificing time-series isolation purity.

