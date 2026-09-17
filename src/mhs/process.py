"""Causal estimators that replace hand-set MHS knobs with per-refit decisions.

Every function reads only the rows it is given; the caller owns the train/apply
split, so a refit can never see data at or after its own ``train_end``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

from src.common.errors import DataIntegrityError
from src.mhs.params import (
    PROCESS_MIN_TRAIN_DAYS,
    PROCESS_PURGE_HOURS,
    PROCESS_REFIT_FREQUENCY,
    PROCESS_RISK_EWMA_LAMBDA,
    PROCESS_SMOOTHING_HALFLIFE_LADDER_DAYS,
)


@dataclass(frozen=True, slots=True)
class RefitPoint:
    """One refit: fitted on rows strictly before ``train_end``, governs ``[effective_from, effective_to)``."""

    effective_from: pd.Timestamp
    effective_to: pd.Timestamp
    train_end: pd.Timestamp


def monthly_refit_schedule(
    data_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    *,
    min_train_days: int = PROCESS_MIN_TRAIN_DAYS,
    purge_hours: int = PROCESS_PURGE_HOURS,
) -> tuple[RefitPoint, ...]:
    """Monthly refit points whose purged train span is at least ``min_train_days``.

    The first out-of-sample day is therefore derived from the data start and the
    sample requirement, never chosen by inspecting results. ``evaluation_end`` is
    the last decision day (inclusive).

    Raises:
        ValueError: naive timestamps, ``data_start >= evaluation_end``, non-positive
            ``min_train_days``/``purge_hours``, or no month start qualifies.
    """
    if data_start.tzinfo is None or evaluation_end.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    if data_start >= evaluation_end:
        raise ValueError("data_start must be before evaluation_end")
    if min_train_days <= 0:
        raise ValueError(f"min_train_days must be > 0, got {min_train_days}")
    if purge_hours <= 0:
        raise ValueError(f"purge_hours must be > 0, got {purge_hours}")
    candidates = pd.date_range(
        data_start.normalize() + pd.Timedelta(days=1),
        evaluation_end.normalize(),
        freq=PROCESS_REFIT_FREQUENCY,
        tz="UTC",
    )
    kept: list[pd.Timestamp] = []
    for candidate in candidates:
        train_end = candidate - pd.Timedelta(hours=purge_hours)
        if train_end - data_start >= pd.Timedelta(days=min_train_days):
            kept.append(candidate)
    if not kept:
        raise ValueError("no month start qualifies for the sample requirement")
    last_end = evaluation_end.normalize() + pd.Timedelta(days=1)
    points: list[RefitPoint] = []
    for i, start in enumerate(kept):
        end = kept[i + 1] if i + 1 < len(kept) else last_end
        points.append(
            RefitPoint(
                effective_from=start,
                effective_to=end,
                train_end=start - pd.Timedelta(hours=purge_hours),
            )
        )
    return tuple(points)


def estimation_adjusted_mean(returns: pd.DataFrame) -> pd.Series:
    """Per-column mean shrunk by ``max(0, 1 - 1/t^2)``.

    ``E[mean^2] = mu^2 + sigma^2/n``, so ``mean^2 (1 - 1/t^2)`` is unbiased for
    ``mu^2``; the factor zeroes an edge that is statistically indistinguishable
    from noise and needs no tuning constant.

    Raises:
        ValueError: non-finite values.
    """
    values = returns.to_numpy(dtype="float64")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("returns must be finite")
    out: dict[str, float] = {}
    for column in returns.columns:
        col = returns[column].to_numpy(dtype="float64")
        n = len(col)
        if n < 2:
            out[str(column)] = 0.0
            continue
        m = float(col.mean())
        s = float(col.std(ddof=1))
        if s == 0.0:
            out[str(column)] = 0.0
            continue
        t2 = n * m * m / (s * s)
        if t2 == 0.0:
            out[str(column)] = 0.0
            continue
        out[str(column)] = m * max(0.0, 1.0 - 1.0 / t2)
    return pd.Series(out, index=returns.columns)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf (2004) covariance shrunk toward a scaled identity.

    Candidate member books are strongly collinear (carry lookbacks, momentum
    horizons); the sample covariance is near-singular on a one-year train
    window, and the analytic shrinkage intensity removes that without a knob.

    Raises:
        ValueError: fewer than 2 rows or non-finite values.
    """
    values = returns.to_numpy(dtype="float64")
    if returns.shape[0] < 2:
        raise ValueError("returns must contain at least 2 rows")
    if not bool(np.isfinite(values).all()):
        raise ValueError("returns must be finite")
    centered = values - values.mean(axis=0)
    n_obs, n_dim = centered.shape
    sample = (centered.T @ centered) / n_obs
    mu = float(np.trace(sample) / n_dim)
    identity = np.eye(n_dim)
    diff = sample - mu * identity
    d2 = float((diff**2).sum() / n_dim)
    sample_norm2 = float((sample**2).sum())
    quad = centered @ sample
    row_norm2 = (centered**2).sum(axis=1)
    cross = (quad * centered).sum(axis=1)
    per_row = row_norm2**2 - 2.0 * cross + sample_norm2
    b2bar = float(per_row.sum() / (n_dim * n_obs * n_obs))
    b2 = min(b2bar, d2)
    delta = 1.0 if d2 == 0.0 else b2 / d2
    shrunk = delta * mu * identity + (1.0 - delta) * sample
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


def long_only_growth_weights(expected: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Non-negative weights maximizing ``w'mu - 0.5 w' Sigma w``.

    This is the second-order log-growth of a combination of member books; the
    long-only constraint holds because inverting a member book still pays its
    turnover cost. Collinear members share weight instead of stacking it.

    Raises:
        ValueError: ``expected`` index differs from ``covariance`` index/columns.
        numpy.linalg.LinAlgError: ``covariance`` is not positive definite while
            some expected value is positive.
    """
    if not expected.index.equals(covariance.index) or not expected.index.equals(covariance.columns):
        raise ValueError("expected index must match covariance index/columns")
    mu = expected.to_numpy(dtype="float64")
    if bool((mu <= 0).all()):
        return pd.Series(np.zeros_like(mu), index=expected.index)
    sigma = covariance.to_numpy(dtype="float64")
    lower = np.linalg.cholesky(sigma)
    target = solve_triangular(lower, mu, lower=True)
    weights, _ = nnls(lower.T, target)
    return pd.Series(weights, index=expected.index)


def ema_smoothing_rate(halflife_days: float) -> float:
    """Per-decision-day adjustment rate ``1 - 2**(-1/h)``; ``h == 0`` means full adjustment.

    Raises:
        ValueError: negative or non-finite ``halflife_days``.
    """
    if not np.isfinite(halflife_days):
        raise ValueError(f"halflife_days must be finite, got {halflife_days}")
    if halflife_days < 0:
        raise ValueError(f"halflife_days must be >= 0, got {halflife_days}")
    if halflife_days == 0.0:
        return 1.0
    return float(1.0 - 2.0 ** (-1.0 / halflife_days))


def smoothed_book_path(
    targets: pd.DataFrame,
    rates: pd.Series,
    initial: pd.Series | None = None,
) -> pd.DataFrame:
    """Partial-adjustment path ``s_t = s_{t-1} + a_t (target_t - s_{t-1})``.

    The state carries across refits, so a new fit changes the aim portfolio,
    not the holdings already on the book.

    Raises:
        ValueError: ``rates`` not aligned to ``targets.index``, a rate outside
            ``(0, 1]``, or ``initial`` columns differ from ``targets``.
    """
    if not rates.index.equals(targets.index):
        raise ValueError("rates must be aligned to targets.index")
    rate_values = rates.to_numpy(dtype="float64")
    if not bool(((rate_values > 0.0) & (rate_values <= 1.0)).all()):
        raise ValueError("rates must be in (0, 1]")
    if initial is not None and list(initial.index) != list(targets.columns):
        raise ValueError("initial columns must match targets columns")
    target_values = targets.to_numpy(dtype="float64")
    state = (
        initial.to_numpy(dtype="float64").copy()
        if initial is not None
        else np.zeros(target_values.shape[1], dtype="float64")
    )
    out = np.empty_like(target_values)
    for i in range(target_values.shape[0]):
        state = state + rate_values[i] * (target_values[i] - state)
        out[i] = state
    return pd.DataFrame(out, index=targets.index, columns=targets.columns)


def step_proxy_net_returns(
    weights: pd.DataFrame,
    log_close: pd.DataFrame,
    funding: pd.DataFrame,
    one_way_bps: float,
) -> pd.Series:
    """Decision-grid screening return of holding ``weights`` for one step.

    ``r_t = sum w_t (exp(dlog_{t->t+1}) - 1) - sum w_t f_t - bps * sum |w_t - w_{t-1}|``
    where ``f_t`` is the funding accrued over ``(t, t+1]``. It is used only for
    per-refit member evidence and smoothing selection; the evaluated path uses
    the hourly ledger. The last row has no forward step and is omitted.

    Raises:
        ValueError: misaligned index/columns or negative ``one_way_bps``.
    """
    if one_way_bps < 0:
        raise ValueError(f"one_way_bps must be >= 0, got {one_way_bps}")
    if not (weights.index.equals(log_close.index) and weights.index.equals(funding.index)):
        raise ValueError("weights, log_close, and funding must share an identical index")
    if not (list(weights.columns) == list(log_close.columns) == list(funding.columns)):
        raise ValueError("weights, log_close, and funding must share identical columns")
    if len(weights.index) < 2:
        return pd.Series(dtype="float64")
    w = weights.to_numpy(dtype="float64")
    forward = log_close.shift(-1).to_numpy(dtype="float64") - log_close.to_numpy(dtype="float64")
    price_move = np.where(np.isfinite(forward), np.exp(forward) - 1.0, 0.0)
    fund = np.where(np.isfinite(funding.to_numpy(dtype="float64")), funding.to_numpy(dtype="float64"), 0.0)
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    turnover = np.abs(w - prev).sum(axis=1)
    cost_rate = one_way_bps * 1e-4
    gross_price = (w * price_move).sum(axis=1)
    gross_funding = (w * fund).sum(axis=1)
    rets = gross_price - gross_funding - cost_rate * turnover
    return pd.Series(rets[:-1], index=weights.index[:-1])


def select_smoothing_halflife(
    targets: pd.DataFrame,
    log_close: pd.DataFrame,
    funding: pd.DataFrame,
    one_way_bps: float,
    train_end: pd.Timestamp,
    ladder: tuple[float, ...] = PROCESS_SMOOTHING_HALFLIFE_LADDER_DAYS,
) -> float:
    """Ladder half-life with the highest mean train log-growth of the smoothed book.

    Cost and signal decay trade off jointly inside the train window; ties go to
    the longer half-life (fewer trades).

    Raises:
        ValueError: empty ladder or no train row whose forward step ends by ``train_end``.
        DataIntegrityError: a train step return at or below -1.
    """
    if len(ladder) == 0:
        raise ValueError("ladder must not be empty")
    if len(targets.index) < 2:
        raise ValueError("no train row whose forward step ends by train_end")
    step = targets.index[1] - targets.index[0]
    positions = [i for i, t in enumerate(targets.index) if t + step <= train_end]
    if not positions:
        raise ValueError("no train row whose forward step ends by train_end")
    last = positions[-1]
    extended = list(range(last + 2)) if last + 1 < len(targets.index) else list(range(last + 1))
    sub_targets = targets.iloc[extended]
    sub_log = log_close.reindex(sub_targets.index)
    sub_fund = funding.reindex(sub_targets.index)
    train_rows = targets.index[positions]
    best: float = ladder[0]
    best_score = float("-inf")
    for halflife in ladder:
        rate = ema_smoothing_rate(halflife)
        rates = pd.Series(rate, index=sub_targets.index)
        smoothed = smoothed_book_path(sub_targets, rates)
        rets = step_proxy_net_returns(smoothed, sub_log, sub_fund, one_way_bps)
        train_rets = rets.loc[train_rows]
        if bool((train_rets.to_numpy(dtype="float64") <= -1.0).any()):
            raise DataIntegrityError("train step return at or below -1")
        score = float(np.log1p(train_rets.to_numpy(dtype="float64")).mean())
        if score > best_score or (score == best_score and halflife > best):
            best_score = score
            best = halflife
    return best


def estimation_adjusted_kelly_exposure(
    unit_returns: pd.Series,
    *,
    active_from: pd.Timestamp,
    cap: float,
    ewma_lambda: float = PROCESS_RISK_EWMA_LAMBDA,
) -> pd.Series:
    """Daily exposure multiple from the process's own out-of-sample unit returns.

    ``f_t = clip(mean / ewma_var * max(0, 1 - 1/t^2), 0, cap)`` over returns in
    ``[active_from, t)``: the mean is the whole out-of-sample record (edge moves
    slowly), the variance is RiskMetrics EWMA (risk clusters), and the
    estimation factor withholds leverage until the record itself is significant.
    ``cap`` is the user's risk envelope, the only non-estimated input.

    Raises:
        ValueError: naive ``active_from``, non-increasing index, non-finite returns,
            ``cap <= 0``, or ``ewma_lambda`` outside ``(0, 1)``.
    """
    if active_from.tzinfo is None:
        raise ValueError("active_from must be tz-aware")
    if not unit_returns.index.is_monotonic_increasing:
        raise ValueError("unit_returns index must be strictly increasing")
    values = unit_returns.to_numpy(dtype="float64")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("unit_returns must be finite")
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")
    if not (0.0 < ewma_lambda < 1.0):
        raise ValueError(f"ewma_lambda must be in (0, 1), got {ewma_lambda}")
    active = unit_returns.loc[unit_returns.index >= active_from]
    # shift(1): t일 노출은 t 이전 수익만 읽는다.
    n = pd.Series(np.arange(len(active), dtype="float64"), index=active.index)
    mean = active.expanding().mean().shift(1)
    std = active.expanding().std(ddof=1).shift(1)
    ewma_var = (active**2).ewm(alpha=1.0 - ewma_lambda, adjust=True).mean().shift(1)
    valid = (n >= 2) & (mean > 0) & (std > 0) & (ewma_var > 0)
    t2 = (n * mean * mean / (std * std)).where(valid)
    factor = (1.0 - 1.0 / t2).clip(lower=0.0)
    raw = (mean / ewma_var * factor).where(valid, 0.0).fillna(0.0).clip(lower=0.0, upper=cap)
    return raw.reindex(unit_returns.index).fillna(0.0)
