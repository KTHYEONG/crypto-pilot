"""MHS statistical evidence: Sharpe/IC/bootstrap/autocorrelation diagnostics.

Pure computation helpers used by the orchestrator's diagnostic and evidence
paths. No alpha, cost, or inventory arithmetic is introduced here; this module
composes the frozen ``src.mhs`` primitives (deflated_sharpe_ratio,
autocorrelation_adjusted_sharpe, rank_weight_book, phase_tranche_book).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.application.research.mhs.contracts import MhsBookReport, MhsFoldReport
from src.common.errors import DataIntegrityError
from src.mhs.books import phase_tranche_book, rank_weight_book
from src.mhs.contracts import BookSpec
from src.mhs.evaluation import autocorrelation_adjusted_sharpe, deflated_sharpe_ratio
from src.mhs.execution import SimulatedInventoryLedgerResult
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H

_logger = logging.getLogger(__name__)

_BOOTSTRAP_SEED = 20260807
_BOOTSTRAP_REPLICATES = 2000
_BOOTSTRAP_MEAN_BLOCK = 168

def _xs_rank_ic(
    signal: pd.DataFrame, opens: pd.DataFrame, forward_bars: int,
) -> dict[str, float]:
    """Cross-sectional rank IC of ``signal`` on a tradable forward window.

    The forward return is built internally as
    ``opens.pct_change(forward_bars).shift(-(forward_bars + 1))`` so the
    measured window starts at ``open_{t+1}`` and avoids overlapping lookbacks.
    """
    if forward_bars < 1:
        raise ValueError(f"forward_bars must be >= 1, got {forward_bars}")
    fwd = opens.pct_change(forward_bars).shift(-(forward_bars + 1))
    common_index = signal.index.intersection(fwd.index)
    common_columns = signal.columns.intersection(fwd.columns)
    if common_index.empty or common_columns.empty:
        return {}
    signal_common = signal.loc[common_index, common_columns]
    fwd_common = fwd.loc[common_index, common_columns]
    valid = signal_common.notna() & fwd_common.notna()
    signal_rank = signal_common.where(valid).rank(axis=1)
    fwd_rank = fwd_common.where(valid).rank(axis=1)
    signal_centered = signal_rank.sub(signal_rank.mean(axis=1), axis=0)
    fwd_centered = fwd_rank.sub(fwd_rank.mean(axis=1), axis=0)
    denominator = np.sqrt(
        signal_centered.pow(2).sum(axis=1) * fwd_centered.pow(2).sum(axis=1),
    )
    correlations = (
        (signal_centered * fwd_centered).sum(axis=1) / denominator
    ).where(valid.sum(axis=1).ge(5) & denominator.gt(0.0)).dropna()
    if correlations.empty:
        return {}
    series = correlations.astype("float64")
    n_dates = len(series)
    mean_ic = float(series.mean())
    sd = float(series.std(ddof=1)) if n_dates > 1 else 0.0
    t_stat = mean_ic / (sd / np.sqrt(n_dates)) if sd > 0 else float("nan")
    return {
        "n_dates": n_dates, "mean_ic": mean_ic, "t_stat": t_stat,
        "forward_bars": forward_bars,
    }
def _annualized_1h_sharpe(net: pd.Series) -> float | None:
    """Annualized Sharpe of an hourly net-return series, or None when undefinable.

    A missing/empty series, a zero standard deviation, or a non-finite result
    return ``None`` explicitly -- never NaN silently coerced to 0.0 (the
    trend-sleeve diagnostic contract requires every reported value to be finite
    or an explicit None).
    """
    net = net.dropna()
    if len(net) < 2:
        return None
    sd = float(net.std(ddof=1))
    if sd <= 0:
        return None
    value = float(net.mean() / sd * np.sqrt(_PERIODS_PER_YEAR_1H))
    return value if np.isfinite(value) else None
def _finite_or_none(value: float) -> float | None:
    """Coerce a metric to an explicit None when it is not finite (JSON-safe)."""
    return None if not np.isfinite(value) else float(value)
def _date_clustered_ols(
    opens: pd.DataFrame, past: pd.DataFrame, forward_bars: int,
) -> dict[str, float]:
    """Pooled panel regression of a tradable forward return on ``past``.

    Same causality fix as ``_xs_rank_ic`` (RC-3): the dependent variable is
    built internally from ``opens`` with the ``shift(-(forward_bars + 1))``
    convention, so the regression never regresses a return window that lies
    inside its own predictor's lookback. Standard errors are date-clustered.
    """
    if forward_bars < 1:
        raise ValueError(f"forward_bars must be >= 1, got {forward_bars}")
    fwd = opens.pct_change(forward_bars).shift(-(forward_bars + 1))
    common_index = past.index.intersection(fwd.index)
    common_columns = past.columns.intersection(fwd.columns)
    if common_index.empty or common_columns.empty:
        return {
            "n": 0, "n_dates": 0, "past_beta": float("nan"),
            "past_t": float("nan"), "forward_bars": forward_bars,
        }
    x = past.loc[common_index, common_columns].to_numpy(dtype="float64", copy=False)
    y = fwd.loc[common_index, common_columns].to_numpy(dtype="float64", copy=False)
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 10:
        return {
            "n": n, "n_dates": 0, "past_beta": float("nan"),
            "past_t": float("nan"), "forward_bars": forward_bars,
        }
    x_valid = np.where(valid, x, 0.0)
    y_valid = np.where(valid, y, 0.0)
    sum_x = float(x_valid.sum())
    sum_y = float(y_valid.sum())
    xtx = np.array([[n, sum_x], [sum_x, float(np.square(x_valid).sum())]])
    xty = np.array([sum_y, float((x_valid * y_valid).sum())])
    inv_xtx = np.linalg.inv(xtx)
    beta = inv_xtx @ xty
    residual = np.where(valid, y - beta[0] - beta[1] * x, 0.0)
    daily_scores = pd.DataFrame(
        {"intercept": residual.sum(axis=1), "slope": (x_valid * residual).sum(axis=1)},
        index=common_index,
    ).resample("1D").sum()
    scores = daily_scores.to_numpy(dtype="float64", copy=False)
    meat = scores.T @ scores
    cov = inv_xtx @ meat @ inv_xtx
    se = np.sqrt(np.diag(cov))
    t_beta = beta[1] / se[1] if se[1] > 0 else float("nan")
    return {
        "n": n, "n_dates": len(daily_scores), "past_beta": float(beta[1]),
        "past_t": float(t_beta), "forward_bars": forward_bars,
    }
def _block_bootstrap_replicate_mean(
    arr: np.ndarray, n: int, p_block: float, rng: np.random.Generator,
) -> float:
    """Mean of one block-bootstrap replicate (scalar fallback path).

    Mirrors the original geometric block composition: block starts are uniform,
    block lengths grow while ``rng.random() > p_block``, blocks are truncated at
    the array end and again to the remaining sample length.  Only used for the
    degenerate ``mean_block <= 0`` configuration and for the astronomically
    rare vectorized shortfall, where a replicate's drawn blocks did not reach
    length ``n``.
    """
    blocks: list[float] = []
    while len(blocks) < n:
        start = int(rng.integers(0, n))
        length = 1
        while length < n and rng.random() > p_block:
            length += 1
        length = min(length, n - len(blocks))
        blocks.extend(arr[start : start + length].tolist())
    return float(np.mean(blocks[:n]))
def _bootstrap_ci(net: pd.Series, n_replicates: int, mean_block: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = net.to_numpy(dtype="float64")
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        m = float(arr[0])
        return m, m
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    if p_block <= 0.0:
        means = np.empty(n_replicates, dtype=np.float64)
        for r in range(n_replicates):
            means[r] = _block_bootstrap_replicate_mean(arr, n, p_block, rng)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    # Vectorized block bootstrap: block lengths are ``geometric(p_block)`` --
    # the same length law as the scalar ``while`` loop -- and block starts are
    # uniform.  A 6x block-count safety margin makes running short effectively
    # impossible; any shortfall still falls back to the scalar replicate path.
    max_blocks = min(n, int(np.ceil(n * 6.0 / mean_block)) + 16)
    means = np.empty(n_replicates, dtype=np.float64)
    chunk = 128
    for r0 in range(0, n_replicates, chunk):
        r1 = min(r0 + chunk, n_replicates)
        k = r1 - r0
        lengths = rng.geometric(p_block, size=(k, max_blocks))
        starts = rng.integers(0, n, size=(k, max_blocks))
        ends = np.cumsum(lengths, axis=1)
        short = ends[:, -1] < n
        for r in np.flatnonzero(short).tolist():
            means[r0 + r] = _block_bootstrap_replicate_mean(arr, n, p_block, rng)
        valid = ~short
        if valid.any():
            ends_trunc = np.minimum(ends, n)
            used = ends_trunc - np.concatenate(
                [np.zeros((k, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1,
            )
            u = used[valid].ravel()
            s = starts[valid].ravel()
            keep = u > 0
            u = u[keep]
            s = s[keep]
            block_start = np.cumsum(u) - u
            offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
            arr_idx = (np.repeat(s, u) + offsets) % n
            sample = arr[arr_idx].reshape(int(valid.sum()), n)
            means[r0 + np.flatnonzero(valid)] = sample.mean(axis=1)

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
def _placebo_sharpe_percentile(
    signal: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    spec: BookSpec,
    observed_sharpe: float,
    n_placebos: int,
    seed: int,
) -> float | None:
    rng = np.random.default_rng(seed)
    ranks: list[float] = []
    cols = list(signal.columns)
    n_cols = len(cols)
    sig_step = signal.reindex(grid_1h)
    el_step = eligible.reindex(grid_1h)
    # The frozen ledger raises ``DataIntegrityError`` unless weights, opens, and
    # funding share an identical index and column set; preserve that contract
    # instead of silently aligning via ``reindex``.
    if not opens.index.equals(grid_1h) or not bar_funding.index.equals(grid_1h):
        raise DataIntegrityError("opens and bar_funding must share the placebo grid index")
    opens_arr = opens[cols].to_numpy(dtype="float64")
    funding_arr = bar_funding[cols].to_numpy(dtype="float64")

    # The placebo shuffle relabels the signal/eligible columns without moving
    # their values, so ``rank_weight_book`` on any shuffled copy returns the
    # identical weight matrix; only the price/funding columns are genuinely
    # permuted relative to those weights.  The whole weight pipeline is
    # therefore computed once as a 2D float64 matrix instead of re-materializing
    # pandas DataFrames inside the 500-step loop (spec §3, Optimization 1).
    weights = rank_weight_book(sig_step, el_step, spec.band.sign, spec.min_symbols)
    weights = phase_tranche_book(weights, spec.tranche_count())
    w_arr = weights.reindex(grid_1h).ffill().fillna(0.0).to_numpy(dtype="float64")

    n_rows = opens_arr.shape[0]
    lag = 1 + 1  # ``mhs_ledger_pnl`` uses ``execution_delay_bars=1``.
    lagged = np.zeros_like(w_arr)
    if lag < n_rows:
        lagged[lag:] = w_arr[: n_rows - lag]

    o2o = np.zeros_like(opens_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = opens_arr[1:] / opens_arr[:-1] - 1.0

    prev_lagged = np.zeros_like(lagged)
    prev_lagged[1:] = lagged[:-1]
    turnover = np.abs(lagged - prev_lagged).sum(axis=1)
    half = 8.0 / 2.0 * 1e-4
    cost_rate = half + half
    nonfinite = ~np.isfinite(o2o) | ~np.isfinite(funding_arr)
    safe_o2o = np.where(np.isfinite(o2o), o2o, 0.0)
    safe_funding = np.where(np.isfinite(funding_arr), funding_arr, 0.0)
    active = lagged != 0.0

    for _p in range(n_placebos):
        perm = rng.permutation(n_cols)
        # A shuffled placebo can pair a non-zero weight with a symbol outside
        # its lifecycle; such a placebo is invalid, not evidence that the
        # production ledger should relax its active-cell guard.
        if (active & nonfinite[:, perm]).any():
            continue
        book_return = (lagged * safe_o2o[:, perm]).sum(axis=1)
        funding_charge = (lagged * safe_funding[:, perm]).sum(axis=1)
        net_returns = book_return - turnover * cost_rate - funding_charge
        if np.any(net_returns <= -1.0):
            continue
        equity = 10000.0 * np.cumprod(1.0 + net_returns)
        net = equity[1:] / equity[:-1] - 1.0
        if len(net) <= 1:
            continue
        sd = float(np.std(net, ddof=1))
        if sd > 0:
            ranks.append(float(np.mean(net) / sd * np.sqrt(_PERIODS_PER_YEAR_1H)))
    if not ranks:
        return None
    return float(np.mean([1.0 if observed_sharpe >= r else 0.0 for r in ranks]))

def _log_autocorr_diagnostic(tag: str, returns: pd.Series, adjusted_sharpe: float) -> None:
    """Debug-only Lo(2002) decomposition: separates the denominator penalty
    (serial-correlation artifact) from raw-return decay, per fold/regime tag.
    """
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    sample_sharpe = mean / std * float(np.sqrt(365)) if std > 0 else float("nan")
    denom = (
        sample_sharpe / adjusted_sharpe
        if np.isfinite(adjusted_sharpe) and adjusted_sharpe != 0.0
        else float("nan")
    )
    n = len(returns)
    rho1 = float(returns.autocorr(1)) if n > 2 else float("nan")
    rho3 = float(returns.autocorr(3)) if n > 4 else float("nan")
    rho7 = float(returns.autocorr(7)) if n > 8 else float("nan")
    _logger.debug(
        "[EVAL] tag=%s n=%d mean=%.5f std=%.5f sample_sharpe=%.3f adjusted_sharpe=%.3f "
        "lo_denom=%.3f rho1=%.3f rho3=%.3f rho7=%.3f",
        tag, n, mean, std, sample_sharpe, adjusted_sharpe, denom, rho1, rho3, rho7,
    )
def _daily_autocorr_sharpe(
    ledger: SimulatedInventoryLedgerResult, *, debug_tag: str | None = None,
) -> float:
    if ledger.equity.empty:
        return float("nan")
    daily = ledger.equity.resample("1D").last().dropna()
    if len(daily) < 9:
        return float("nan")
    returns = daily.pct_change().dropna()
    adjusted = autocorrelation_adjusted_sharpe(returns, 365, 7)
    if debug_tag is not None and _logger.isEnabledFor(logging.DEBUG):
        _log_autocorr_diagnostic(debug_tag, returns, adjusted)
    return adjusted
def _hourly_ledger_series(
    equity: pd.Series, fill_turnover: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Resample a native execution-timeframe ledger to the annualization grid.

    The ``_PERIODS_PER_YEAR_1H`` annualization constant describes hourly bars,
    but the replay ledgers run on ``request.execution_timeframe`` (3m default),
    so every headline metric derived from them must first be resampled to 1h
    (the ``equity_1h`` pattern already present in ``_run_post_diag_deploy``).
    Turnover is a per-bar traded-notional fraction, so hourly aggregation is a
    sum (not a last-value sample, unlike equity). An already-hourly input passes
    through unchanged.
    """
    equity_1h = equity.resample("1h").last().dropna()
    net_returns_1h = equity_1h.pct_change().dropna()
    turnover_1h = (
        fill_turnover.resample("1h").sum()
        .reindex(net_returns_1h.index)
        .fillna(0.0)
    )
    return equity_1h, net_returns_1h, turnover_1h
def _naive_sharpe(ledger: SimulatedInventoryLedgerResult) -> float:
    net = ledger.equity.resample("1h").last().dropna().pct_change().dropna()
    if len(net) < 2:
        return float("nan")
    sd = float(net.std(ddof=1))
    if sd <= 0:
        return float("inf") if float(net.mean()) > 0 else float("-inf")
    return float(net.mean() / sd * np.sqrt(_PERIODS_PER_YEAR_1H))
def _mean_ann(series: pd.Series, periods_per_year: float) -> float:
    return float(series.mean()) * periods_per_year if len(series) else float("nan")
def _geometric_cagr(equity: pd.Series) -> float:
    if equity.empty or float(equity.iloc[0]) <= 0 or float(equity.iloc[-1]) <= 0:
        return float("nan")
    n = len(equity)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (_PERIODS_PER_YEAR_1H / n) - 1.0)

def _mdd(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())
def _causal_lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation of a rolling window (raw ndarray, for use
    inside ``Series.rolling(...).apply(..., raw=True)``).
    """
    if len(x) < 3:
        return float("nan")
    x0, x1 = x[:-1], x[1:]
    if np.std(x0) == 0.0 or np.std(x1) == 0.0:
        return float("nan")
    return float(np.corrcoef(x0, x1)[0, 1])

def _per_observation_sharpe(returns: pd.Series) -> float:
    """Per-observation (non-annualized) sample Sharpe of a return series.

    ``mean / std`` with no ``sqrt(periods_per_year)`` scaling, matching the
    per-observation input contract of ``probabilistic_sharpe_ratio``/
    ``deflated_sharpe_ratio``. Degenerate zero-variance returns NaN so a
    non-finite observed Sharpe never reaches the deflation statistic.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    sd = float(returns.std(ddof=1))
    if sd <= 0.0:
        return float("nan")
    return float(returns.mean() / sd)
def _deflated_sharpe_evidence(
    blend_report: MhsBookReport | None,
    folds: tuple[MhsFoldReport, ...],
    n_trials: int,
) -> float | None:
    """Per-observation deflated Sharpe of the blend primary against anchored-fold trial dispersion."""
    if blend_report is None or blend_report.primary is None or not folds:
        return None
    _equity_1h, net_returns_1h, _turnover = _hourly_ledger_series(
        blend_report.primary.ledger.equity,
        blend_report.primary.ledger.fill_turnover,
    )
    observed_sr = _per_observation_sharpe(net_returns_1h)
    if not np.isfinite(observed_sr):
        return None
    trial_sharpes: list[float] = []
    for fold in folds:
        if fold.strict is None or fold.failures:
            continue
        _fold_equity, fold_net, _fold_turnover = _hourly_ledger_series(
            fold.strict.ledger.equity, fold.strict.ledger.fill_turnover,
        )
        trial_sharpes.append(_per_observation_sharpe(fold_net))
    if not trial_sharpes:
        return None
    trial_variance = (
        float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) >= 2 else 0.0
    )
    returns = net_returns_1h.dropna()
    if len(returns) < 2:
        return None
    result = deflated_sharpe_ratio(
        observed_sr,
        trial_variance,
        n_trials,
        len(returns),
        float(returns.skew()),
        float(returns.kurt()) + 3.0,
    )
    # Fail closed: a degenerate skew/kurtosis can push the statistic to NaN,
    # which must never leak into the report payload as a real deflated value.
    return result if np.isfinite(result) else None
