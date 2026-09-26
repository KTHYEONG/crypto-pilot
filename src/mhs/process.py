"""Causal estimators that replace hand-set MHS knobs with per-refit decisions.

Every function reads only the rows it is given; the caller owns the train/apply
split, so a refit can never see data at or after its own ``train_end``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

from src.common.errors import DataIntegrityError
from src.mhs.books import portfolio_rebalance_trigger
from src.mhs.params import (
    PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
    PROCESS_MIN_TRAIN_DAYS,
    PROCESS_PURGE_HOURS,
    PROCESS_REFIT_FREQUENCY,
)

if TYPE_CHECKING:
    from src.mhs.backtest.labels import MaturedMemberReturns


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


def matured_monthly_refit_schedule(
    labels: MaturedMemberReturns,
    evaluation_end: pd.Timestamp,
    *,
    min_train_days: int,
    fit_latency: pd.Timedelta,
) -> tuple[RefitPoint, ...]:
    """Schedule monthly applications after enough complete historical labels exist.

    Args:
        labels: Member evidence with explicit maturity and knowledge masks.
        evaluation_end: Last requested UTC decision date, inclusive.
        min_train_days: Registered minimum count of complete daily training labels.
        fit_latency: Registered nonnegative time between fitting and application.
    Returns:
        Contiguous monthly application intervals with information cutoffs derived
        from label maturity, not backward feature lookbacks.
    Raises:
        ValueError: Controls are invalid or no refit has sufficient completed evidence.
        DataIntegrityError: Evidence labels or the resulting intervals are inconsistent.
    """
    if evaluation_end.tzinfo is None or str(evaluation_end.tz) != "UTC":
        raise ValueError("timestamps must be tz-aware UTC")
    if isinstance(min_train_days, bool) or not isinstance(min_train_days, int) or min_train_days < 1:
        raise ValueError(f"min_train_days must be a positive integer, got {min_train_days}")
    if not isinstance(fit_latency, pd.Timedelta) or fit_latency < pd.Timedelta(0):
        raise ValueError(f"fit_latency must be a nonnegative timedelta, got {fit_latency}")
    if len(labels.returns) == 0:
        raise ValueError("no refit has sufficient completed evidence")
    if not (
        len(labels.label_start)
        == len(labels.label_end)
        == len(labels.available_at)
        == len(labels.returns)
    ):
        raise DataIntegrityError("label intervals must align exactly with return rows")
    known_all = labels.known.to_numpy(dtype=bool).all(axis=1)
    first_day = labels.returns.index[0].normalize() + pd.Timedelta(days=1)
    candidates = pd.date_range(first_day, evaluation_end.normalize(), freq=PROCESS_REFIT_FREQUENCY, tz="UTC")
    kept: list[pd.Timestamp] = []
    for candidate in candidates:
        cutoff = candidate - fit_latency
        timely = np.asarray(labels.available_at <= cutoff, dtype=bool) & np.asarray(
            labels.label_end <= cutoff, dtype=bool
        )
        if int((known_all & timely).sum()) >= min_train_days:
            kept.append(candidate)
    if not kept:
        raise ValueError("no refit has sufficient completed evidence")
    last_end = evaluation_end.normalize() + pd.Timedelta(days=1)
    points: list[RefitPoint] = []
    for i, start in enumerate(kept):
        end = kept[i + 1] if i + 1 < len(kept) else last_end
        points.append(
            RefitPoint(
                effective_from=start,
                effective_to=end,
                train_end=start - fit_latency,
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
    fwd_ret = log_close.shift(-1) - log_close
    forward = fwd_ret.to_numpy(dtype="float64")
    price_move = np.where(np.isfinite(forward), np.exp(forward) - 1.0, 0.0)
    fund = np.where(np.isfinite(funding.to_numpy(dtype="float64")), funding.to_numpy(dtype="float64"), 0.0)
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    turnover = np.abs(w - prev).sum(axis=1)
    cost_rate = one_way_bps * 1e-4
    gross_price = (w * price_move).sum(axis=1)
    gross_funding = (w * fund).sum(axis=1)
    rets = gross_price - gross_funding - cost_rate * turnover
    return pd.Series(rets[:-1], index=weights.index[:-1])


def volatility_scaled_exposure(
    unit_returns: pd.Series,
    *,
    cap: float,
    halflife_days: int = PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS,
) -> pd.Series:
    """Daily exposure multiple ``cap * median(vol) / vol`` clipped to ``[0, cap]``.

    Realized Kelly leverage of the unit book is several times any registered cap,
    so growth is maximized at the cap in ordinary regimes; the ratio of the
    expanding median forecast volatility to the current forecast only de-risks
    when risk is elevated relative to the strategy's own history. The forecast
    for day ``t`` uses returns strictly before ``t``; days without a positive
    forecast carry zero exposure (unobservable risk is never levered).

    Raises:
        ValueError: ``cap <= 0``, ``halflife_days < 1``, non-increasing index, or
            non-finite returns.
    """
    if cap <= 0:
        raise ValueError(f"cap must be > 0, got {cap}")
    if halflife_days < 1:
        raise ValueError(f"halflife_days must be >= 1, got {halflife_days}")
    if not unit_returns.index.is_monotonic_increasing:
        raise ValueError("unit_returns index must be non-decreasing")
    values = unit_returns.to_numpy(dtype="float64")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("unit_returns must be finite")
    vol = unit_returns.ewm(halflife=halflife_days, min_periods=halflife_days).std().shift(1)
    median = vol.expanding().median()
    exposure = (cap * median / vol.where(vol > 0)).clip(lower=0.0, upper=cap).fillna(0.0)
    exposure.index = unit_returns.index
    return exposure


@dataclass(frozen=True, slots=True)
class ProcessRiskSizingSpec:
    """Causal portfolio-volatility sizing contract for a unit process book.

    The specification controls exposure only after unit-book composition is
    known. It prevents future returns, missing risk observations, and
    unbounded leverage from being translated into historical or live exposure.

    Args:
        annual_volatility_target: Positive annualized arithmetic-return target.
        ewma_halflife_days: Positive integral EWMA half-life in daily bars.
        minimum_observations: Positive count of prior daily returns required.
        leverage_cap: Positive finite maximum gross exposure multiple.
    """

    annual_volatility_target: float
    ewma_halflife_days: int
    minimum_observations: int
    leverage_cap: float

    def __post_init__(self) -> None:
        """Validate every control before any market data is read."""
        target = self.annual_volatility_target
        if isinstance(target, bool) or not isinstance(target, (int, float, np.integer, np.floating)):
            raise ValueError(f"annual_volatility_target must be finite and > 0, got {target!r}")
        if not np.isfinite(float(target)) or float(target) <= 0.0:
            raise ValueError(f"annual_volatility_target must be finite and > 0, got {target!r}")
        for name in ("ewma_halflife_days", "minimum_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            if int(value) < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        cap = self.leverage_cap
        if isinstance(cap, bool) or not isinstance(cap, (int, float, np.integer, np.floating)):
            raise ValueError(f"leverage_cap must be finite and > 0, got {cap!r}")
        if not np.isfinite(float(cap)) or float(cap) <= 0.0:
            raise ValueError(f"leverage_cap must be finite and > 0, got {cap!r}")
        object.__setattr__(self, "annual_volatility_target", float(target))
        object.__setattr__(self, "ewma_halflife_days", int(self.ewma_halflife_days))
        object.__setattr__(self, "minimum_observations", int(self.minimum_observations))
        object.__setattr__(self, "leverage_cap", float(cap))


def causal_volatility_scaled_exposure(
    unit_returns: pd.Series,
    spec: ProcessRiskSizingSpec,
) -> pd.Series:
    """Return bounded next-day exposure from strictly preceding unit returns.

    Args:
        unit_returns: Finite chronological daily net returns before sizing.
        spec: Validated risk-sizing parameters for this evaluation.

    Returns:
        Float64 exposure indexed exactly as `unit_returns` and bounded by cap.

    Raises:
        ValueError: Returns or sizing controls violate temporal or numeric
            requirements.
    """
    if not isinstance(spec, ProcessRiskSizingSpec):
        raise ValueError(f"spec must be a ProcessRiskSizingSpec, got {type(spec).__name__}")
    if not isinstance(unit_returns, pd.Series):
        raise ValueError("unit_returns must be a Series")
    index = unit_returns.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("unit_returns must have a DatetimeIndex")
    if index.hasnans:
        raise ValueError("unit_returns index must not contain NaT")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("unit_returns index must be unique and increasing")
    if index.tz is None:
        raise ValueError("unit_returns index must be timezone-aware UTC")
    if not index.tz_convert("UTC").equals(index):
        raise ValueError("unit_returns index must be UTC")
    values = unit_returns.to_numpy(dtype="float64")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("unit_returns must be finite")
    if values.size and bool((values <= -1.0).any()):
        raise ValueError("unit_returns must exceed minus one")
    prior_vol = unit_returns.ewm(
        halflife=spec.ewma_halflife_days, min_periods=spec.minimum_observations
    ).std().shift(1)
    annualized = prior_vol * float(np.sqrt(365.0))
    exposure = (
        (spec.annual_volatility_target / annualized.where(annualized > 0.0))
        .clip(lower=0.0, upper=spec.leverage_cap)
        .fillna(0.0)
    )
    exposure = exposure.astype("float64")
    exposure.index = unit_returns.index
    return exposure


@dataclass(frozen=True, slots=True)
class ProcessExecutionPolicy:
    """Explicit research control for adopting a smoothed unit book.

    The threshold measures the portfolio L1 distance to the last adopted
    unit book, before volatility sizing. It is not a cash fraction, a
    per-symbol limit, or a forecast of trading profitability. No discovered
    threshold is selected implicitly.

    Args:
        tracking_error_threshold: Non-negative finite L1 distance. None
            disables trade deferral and preserves the baseline path.

    Raises:
        ValueError: A supplied threshold is non-finite or negative.
    """

    tracking_error_threshold: float | None = None

    def __post_init__(self) -> None:
        """Reject invalid research controls before any market data is read."""
        threshold = self.tracking_error_threshold
        if threshold is None:
            return
        if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"tracking_error_threshold must be finite, got {threshold}") from None
        value = float(threshold)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"tracking_error_threshold must be finite and >= 0, got {threshold}")
        object.__setattr__(self, "tracking_error_threshold", value)


def _validate_unit_targets(unit_targets: pd.DataFrame) -> None:
    if not isinstance(unit_targets.index, pd.DatetimeIndex):
        raise ValueError("unit_targets must have a DatetimeIndex")
    index = unit_targets.index
    if len(index) == 0 and len(unit_targets.columns) == 0:
        return
    if len(unit_targets.columns) == 0:
        raise ValueError("unit_targets must have at least one symbol column")
    if index.hasnans:
        raise ValueError("unit_targets index must not contain NaT")
    if index.has_duplicates:
        raise ValueError("unit_targets index must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError("unit_targets index must be increasing")
    if index.tz is None:
        raise ValueError("unit_targets index must be timezone-aware UTC")
    utc_index = index.tz_convert("UTC")
    if not utc_index.equals(index):
        raise ValueError("unit_targets index must be UTC")
    if len(unit_targets.columns) != len(set(unit_targets.columns)):
        raise ValueError("unit_targets columns must be unique")
    if len(unit_targets) == 0:
        return
    try:
        values = unit_targets.to_numpy(dtype="float64")
    except (TypeError, ValueError):
        raise ValueError("unit_targets must be numeric") from None
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("unit_targets must be finite")


def apply_process_execution_policy(
    unit_targets: pd.DataFrame,
    policy: ProcessExecutionPolicy,
) -> pd.DataFrame:
    """Adopt complete unit-book rows only when portfolio change warrants it.

    A complete-row adoption preserves the input book's neutrality and
    relative weights. Volatility sizing and execution safeguards remain
    downstream so this control cannot freeze a risk reduction or substitute
    an adopted target for actual held inventory.

    Args:
        unit_targets: Finite decision-time books with a unique, increasing
            UTC DatetimeIndex and unique ordered symbol columns.
        policy: Explicit control; None or a zero threshold is the identity.

    Returns:
        A float64 frame with identical labels. Every output row is an
        unchanged copy of a row observed at or before that decision.

    Raises:
        ValueError: Targets are non-finite, non-numeric, or have invalid
            decision-time or symbol labels.
    """
    _validate_unit_targets(unit_targets)
    threshold = policy.tracking_error_threshold
    if threshold is None or threshold == 0.0:
        return unit_targets.copy().astype("float64")
    return portfolio_rebalance_trigger(unit_targets.copy().astype("float64"), threshold)
