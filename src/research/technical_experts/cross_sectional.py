"""Cross-sectional dollar-neutral composite construction for the XS screen.

The 450-cell TS screen cannot survive top-k selection because its alpha is
broad and thin (the tail is thinner than noise while the mean is significantly
positive). The structure that survives a selection screen is beta, not alpha.
This module rebuilds the same 30 frozen identity targets as one per-symbol
composite score, EWMA-smoothes it, demeans it across symbols into a unit-gross
dollar-neutral book, and holds every name inside a no-trade band so the book
does not trade itself to zero at round-trip costs. All construction parameters
are frozen by :class:`XsCompositeSpec`; the execution contract (t+1+delay open
fills, turnover costs, funding on the held book) is applied by
:func:`run_xs_composite_ledger` and the scale-invariant admission gate by
:func:`evaluate_xs_admission`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.risk.growth_sizing import (
    GrowthSizingConfig,
    GrowthSizingResult,
    apply_realised_risk_overlay,
    solve_growth_optimal_risk,
)

_BARS_PER_YEAR = 2190
_INITIAL_EQUITY = 10_000.0


def apply_no_trade_band(target_weights: np.ndarray, band: float) -> np.ndarray:
    """Carry each name's held weight until its target drifts past ``band``.

    Starting from an all-zero held book, row ``i`` snaps a name to its current
    target exactly when ``abs(target - held) > band``; names inside the band are
    carried unchanged. Only the drifted name snaps -- the rest of the row is
    untouched -- and row ``i`` depends only on rows ``0..i`` (strictly causal).
    ``band <= 0`` is a pass-through and returns the input unchanged.
    """
    arr = np.asarray(target_weights)
    if arr.ndim != 2:
        raise ValueError(f"target_weights must be a 2-D array, got {arr.ndim}-D")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"target_weights must be a float array, got dtype {arr.dtype}")
    if not np.isfinite(band):
        raise ValueError(f"band must be finite, got {band}")
    if band <= 0.0:
        return arr

    held = np.zeros(arr.shape[1], dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for t in range(arr.shape[0]):
        snap = np.abs(arr[t] - held) > band
        held = np.where(snap, arr[t], held)
        out[t] = held
    return out


def build_xs_neutral_weights(
    score: pd.DataFrame,
    halflife: int,
    band: float,
) -> pd.DataFrame:
    """EWMA-smooth, demean, and unit-gross-normalize a cross-sectional score.

    Every invested row of the *pre-band* normalized weights satisfies
    ``sum(w) == 0`` and ``sum(abs(w)) == 1``; a row with zero cross-sectional
    dispersion is all zeros rather than NaN. ``halflife`` is in bars and
    ``halflife == 0`` skips smoothing. The no-trade band is applied on the
    normalized weights (never on the raw score) and is the returned value as
    is: re-normalizing after the band would rescale every name -- including
    ones the band held flat -- on every bar, defeating the band's purpose and
    materially inflating turnover. Because names snap asynchronously (a banded
    row mixes freshly-snapped entries with entries carried from a different,
    earlier row), neither invariant is exact on the banded output any more:
    ``sum(abs(w))`` and ``sum(w)`` both drift near, but not exactly at, 1 and
    0 respectively (both hold exactly only on the pre-band normalized
    weights). Realized portfolio-level neutrality and gross exposure are
    instead verified empirically -- via realized beta and annualized turnover
    -- in :func:`evaluate_xs_admission`. The frame is never shifted here --
    execution lag is the ledger's contract.
    """
    if halflife < 0:
        raise ValueError(f"halflife must be >= 0, got {halflife}")
    if not isinstance(score, pd.DataFrame):
        raise ValueError("score must be a DataFrame")

    smoothed = score.ewm(halflife=halflife, min_periods=1).mean() if halflife > 0 else score
    values = smoothed.to_numpy(dtype=np.float64)
    demeaned = values - values.mean(axis=1, keepdims=True)
    abs_sum = np.abs(demeaned).sum(axis=1, keepdims=True)
    normalized = np.divide(
        demeaned, abs_sum, out=np.zeros_like(demeaned), where=(abs_sum > 0),
    )
    weights = apply_no_trade_band(normalized, band)
    return pd.DataFrame(weights, index=score.index, columns=score.columns)


@dataclass(frozen=True, slots=True)
class XsCompositeSpec:
    """Frozen construction and execution contract of the XS composite profile.

    ``halflife_bars`` and ``no_trade_band`` are the only fitted parameters in
    the whole construction and were frozen on discovery data alone, so the
    qualification result stays an honest out-of-sample test. The remaining
    fields are the production execution convention shared with the TS screen.
    """

    halflife_bars: int = 6
    no_trade_band: float = 0.05
    execution_delay_bars: int = 1
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0003

    def __post_init__(self) -> None:
        if self.halflife_bars < 0:
            raise ValueError(
                f"halflife_bars must be >= 0, got {self.halflife_bars}"
            )
        if not 0.0 <= self.no_trade_band < 1.0:
            raise ValueError(
                f"no_trade_band must be in [0.0, 1.0), got {self.no_trade_band}"
            )
        if self.execution_delay_bars < 0:
            raise ValueError(
                f"execution_delay_bars must be >= 0, got {self.execution_delay_bars}"
            )
        if self.fee_rate < 0:
            raise ValueError(f"fee_rate must be >= 0, got {self.fee_rate}")
        if self.slippage_rate < 0:
            raise ValueError(f"slippage_rate must be >= 0, got {self.slippage_rate}")

    def round_trip_cost_rate(self) -> float:
        """Per-unit-turnover charge ``fee_rate + slippage_rate``."""
        return self.fee_rate + self.slippage_rate


@dataclass(frozen=True, slots=True)
class XsAlphaCompositeSpec:
    """Frozen identity of the multi-horizon cross-sectional alpha profile.

    ``signal_windows`` fixes the one/two/four-calendar-week 4h-bar horizons
    (42/84/168) and ``components`` names the three economically diversified
    alpha families. The class contains no fitted coefficients and no return,
    cost, or holdout input: every horizon and component enters the score with
    equal weight by construction.
    """

    signal_windows: tuple[int, int, int] = (42, 84, 168)
    components: tuple[str, str, str] = (
        "trend", "funding_contrarian", "taker_imbalance",
    )

    def __post_init__(self) -> None:
        if self.signal_windows != (42, 84, 168):
            raise ValueError(
                f"signal_windows must be (42, 84, 168), got {self.signal_windows}"
            )
        if self.components != ("trend", "funding_contrarian", "taker_imbalance"):
            raise ValueError(
                f"components must be trend/funding_contrarian/taker_imbalance, "
                f"got {self.components}"
            )


def _validate_alpha_panels(
    closes: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    bar_funding: pd.DataFrame,
) -> None:
    """Fail closed on any malformed or misaligned alpha input panel.

    Every matrix must be a tz-aware UTC DataFrame with an identical unique,
    monotonic index and an identical ordered column set. Closes must be finite
    and strictly positive, taker ratios finite and in ``[0, 1]``, and funding
    finite. Malformed input raises :class:`DataIntegrityError` -- never a
    zero-filling fallback.
    """
    for name, frame in (
        ("closes", closes),
        ("taker_buy_ratio", taker_buy_ratio),
        ("bar_funding", bar_funding),
    ):
        if not isinstance(frame, pd.DataFrame):
            raise DataIntegrityError(f"{name} must be a DataFrame")
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex) or getattr(index, "tz", None) is None:
            raise DataIntegrityError(f"{name} index must be a tz-aware UTC DatetimeIndex")
        if not index.is_unique:
            raise DataIntegrityError(f"{name} index must be unique")
        if not index.is_monotonic_increasing:
            raise DataIntegrityError(f"{name} index must be monotonic increasing")
    if not (
        closes.index.equals(taker_buy_ratio.index)
        and closes.index.equals(bar_funding.index)
    ):
        raise DataIntegrityError(
            "closes, taker_buy_ratio, and bar_funding must share an identical index"
        )
    if not (
        list(closes.columns) == list(taker_buy_ratio.columns)
        and list(closes.columns) == list(bar_funding.columns)
    ):
        raise DataIntegrityError(
            "closes, taker_buy_ratio, and bar_funding must share an identical "
            "ordered column set"
        )

    closes_values = closes.to_numpy(dtype=np.float64)
    taker_values = taker_buy_ratio.to_numpy(dtype=np.float64)
    funding_values = bar_funding.to_numpy(dtype=np.float64)
    if not np.isfinite(closes_values).all() or (closes_values <= 0.0).any():
        raise DataIntegrityError("closes must be finite and strictly positive")
    if not np.isfinite(taker_values).all() or (taker_values < 0.0).any() or (taker_values > 1.0).any():
        raise DataIntegrityError("taker_buy_ratio must be finite and in [0, 1]")
    if not np.isfinite(funding_values).all():
        raise DataIntegrityError("bar_funding must be finite")


def _cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Finite-only cross-sectional z-score with explicit output buffers.

    Rows with fewer than two finite observations (or zero cross-sectional
    dispersion) produce an all-zero row rather than NaN or +/-inf. The common
    index and ordered columns are preserved by the caller.
    """
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    safe_count = np.maximum(count, 1)
    mean = np.divide(
        np.where(finite, values, 0.0).sum(axis=1, keepdims=True),
        safe_count[:, None],
    )
    demeaned = np.where(finite, values - mean, 0.0)
    var = np.divide(
        (demeaned ** 2).sum(axis=1, keepdims=True),
        np.maximum(count - 1, 1)[:, None],
    )
    std = np.sqrt(np.maximum(var, 0.0))
    out = np.zeros_like(values, dtype=np.float64)
    np.divide(
        demeaned, std, out=out,
        where=(count[:, None] >= 2) & (std > 0.0),
    )
    return out


def build_xs_alpha_family_scores(
    closes: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    bar_funding: pd.DataFrame,
    spec: XsAlphaCompositeSpec,
) -> dict[str, pd.DataFrame]:
    """Build the three economically distinct multi-horizon alpha family scores.

    For every horizon ``L`` the trend component is the volatility-adjusted log
    return ``log(close / close.shift(L)) / rolling_std(dlog, L)``, the funding
    component is the settled, causally aligned contrarian carry
    ``-bar_funding.shift(1).rolling(L).sum()``, and the taker component is
    ``taker_buy_ratio.rolling(L).mean() - 0.5``. Each of the nine component
    panels is cross-sectionally z-scored (rows with fewer than two finite
    observations are all zero). The returned mapping holds ``trend``,
    ``funding_contrarian``, and ``taker_imbalance`` in that immutable order,
    each being the equal-weight sum of its three z-scored horizon panels on the
    common index and ordered columns.
    """
    _validate_alpha_panels(closes, taker_buy_ratio, bar_funding)
    log_close = np.log(closes)
    dlog = log_close.diff()

    family_scores = {
        name: np.zeros((len(closes.index), len(closes.columns)), dtype=np.float64)
        for name in spec.components
    }
    for window in spec.signal_windows:
        trend = np.log(closes / closes.shift(window)) / dlog.rolling(window).std()
        carry = -bar_funding.shift(1).rolling(window).sum()
        taker = taker_buy_ratio.rolling(window).mean() - 0.5
        family_scores["trend"] += _cross_sectional_zscore(
            trend.to_numpy(dtype=np.float64),
        )
        family_scores["funding_contrarian"] += _cross_sectional_zscore(
            carry.to_numpy(dtype=np.float64),
        )
        family_scores["taker_imbalance"] += _cross_sectional_zscore(
            taker.to_numpy(dtype=np.float64),
        )
    return {
        name: pd.DataFrame(frame, index=closes.index, columns=list(closes.columns))
        for name, frame in family_scores.items()
    }


def build_xs_alpha_composite_score(
    closes: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    bar_funding: pd.DataFrame,
    spec: XsAlphaCompositeSpec,
) -> pd.DataFrame:
    """Build the nine-component multi-horizon cross-sectional alpha score.

    The composite is defined as the exact sum of the three family score frames
    from :func:`build_xs_alpha_family_scores`; it therefore preserves the v2
    panel formulas, causal prefix invariance, and the common index and ordered
    columns exactly (within the existing floating-point tolerance).
    """
    family_scores = build_xs_alpha_family_scores(
        closes, taker_buy_ratio, bar_funding, spec,
    )
    return (
        family_scores["trend"]
        + family_scores["funding_contrarian"]
        + family_scores["taker_imbalance"]
    )


def build_xs_alpha_weights(
    closes: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    bar_funding: pd.DataFrame,
    alpha_spec: XsAlphaCompositeSpec,
    execution_spec: XsCompositeSpec,
) -> pd.DataFrame:
    """Shared EWMA/demean/unit-gross/no-trade-band construction for the alpha score.

    Delegates to :func:`build_xs_alpha_composite_score` and then applies
    :func:`build_xs_neutral_weights` exactly once with the execution spec's
    half-life and no-trade band. The weights are never re-normalized after the
    band and never shifted here -- execution lag is the ledger's contract.
    """
    score = build_xs_alpha_composite_score(closes, taker_buy_ratio, bar_funding, alpha_spec)
    return build_xs_neutral_weights(
        score, execution_spec.halflife_bars, execution_spec.no_trade_band,
    )


def build_xs_alpha_family_weights(
    closes: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    bar_funding: pd.DataFrame,
    alpha_spec: XsAlphaCompositeSpec,
    execution_spec: XsCompositeSpec,
) -> dict[str, pd.DataFrame]:
    """Build per-family EWMA/demean/unit-gross/no-trade-band sleeve weights.

    Applies :func:`build_xs_neutral_weights` exactly once to every family
    score frame from :func:`build_xs_alpha_family_scores` with the execution
    spec's half-life and no-trade band. Weights are never re-normalized after
    the band and never shifted here -- execution lag is the ledger's contract.
    """
    family_scores = build_xs_alpha_family_scores(
        closes, taker_buy_ratio, bar_funding, alpha_spec,
    )
    return {
        name: build_xs_neutral_weights(
            score, execution_spec.halflife_bars, execution_spec.no_trade_band,
        )
        for name, score in family_scores.items()
    }


def build_xs_alpha_dual_family_weights(
    closes: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    bar_funding: pd.DataFrame,
    alpha_spec: XsAlphaCompositeSpec,
    execution_spec: XsCompositeSpec,
) -> pd.DataFrame:
    """Build the equal-weight trend + taker_imbalance composite sleeve.

    Drops the standalone-failing ``funding_contrarian`` family: reuses
    :func:`build_xs_alpha_family_scores` verbatim (the frozen panel
    construction) but sums only the ``trend`` and ``taker_imbalance`` keys and
    then applies :func:`build_xs_neutral_weights` exactly once -- the same
    smooth-once-after-selection invariant as the score-router relocation.
    """
    family_scores = build_xs_alpha_family_scores(
        closes, taker_buy_ratio, bar_funding, alpha_spec,
    )
    combined = family_scores["trend"] + family_scores["taker_imbalance"]
    return build_xs_neutral_weights(
        combined, execution_spec.halflife_bars, execution_spec.no_trade_band,
    )


def _causal_family_inverse_vol_weights(
    sleeve_returns: pd.DataFrame,
    lookback_bars: int,
) -> pd.DataFrame:
    """Causal trailing inverse-realized-vol weights across the alpha families.

    For each bar ``t`` and family column the weight is ``1/std`` over the
    strictly-prior window ``[t - lookback_bars, t)`` of the family's standalone
    net-of-cost realized returns; the current bar's own return never enters its
    own weight. A family whose trailing window is incomplete (fewer than
    ``lookback_bars`` prior completed bars) or whose trailing std is
    zero/non-finite falls back to the shared default ``1/3`` (never CASH, never
    NaN), and every row is renormalized to sum to exactly 1. Vectorized via
    cumulative sums -- no per-row Python loop.
    """
    x = sleeve_returns.to_numpy(dtype=np.float64)
    filled = np.where(np.isnan(x), 0.0, x)
    cum = np.cumsum(filled, axis=0)
    cum_sq = np.cumsum(filled * filled, axis=0)

    n = len(sleeve_returns)
    bar_pos = np.arange(n)
    starts = np.maximum(bar_pos - lookback_bars, 0)
    counts = bar_pos - starts

    up_to_prev = np.where((bar_pos == 0)[:, None], 0.0, cum[bar_pos - 1])
    before_start = np.where((starts == 0)[:, None], 0.0, cum[starts - 1])
    sums = up_to_prev - before_start
    up_to_prev_sq = np.where((bar_pos == 0)[:, None], 0.0, cum_sq[bar_pos - 1])
    before_start_sq = np.where((starts == 0)[:, None], 0.0, cum_sq[starts - 1])
    sums_sq = up_to_prev_sq - before_start_sq

    count_col = np.maximum(counts, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / count_col
        var = (sums_sq - count_col * mean * mean) / np.maximum(counts - 1, 1)[:, None]
    std = np.sqrt(np.clip(var, 0.0, None))

    invalid = (
        (counts < lookback_bars)[:, None]
        | (std <= 0.0)
        | ~np.isfinite(std)
    )
    std = np.where(invalid, np.nan, std)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / std
    weights = np.where(np.isfinite(inv), inv, 1.0 / 3.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return pd.DataFrame(
        weights, index=sleeve_returns.index, columns=list(sleeve_returns.columns),
    )


def build_xs_alpha_vol_weighted_weights(
    closes: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    bar_funding: pd.DataFrame,
    opens: pd.DataFrame,
    alpha_spec: XsAlphaCompositeSpec,
    execution_spec: XsCompositeSpec,
) -> pd.DataFrame:
    """Build the causal inverse-realized-vol tilted multi-family alpha sleeve.

    Replaces the fixed 1/3 family blend: every family score is built with
    :func:`build_xs_alpha_family_scores` and its standalone net-of-cost ledger
    replayed via :func:`run_xs_composite_ledger`, then :func:`_causal_family_inverse_vol_weights`
    turns the per-family realized returns into strictly-causal inverse-vol
    weights over ``alpha_spec.signal_windows[0]`` bars. The weighted family
    scores are summed and passed through :func:`build_xs_neutral_weights`
    exactly once (the smooth-once invariant shared with every profile in this
    family). ``opens`` is required because realized-return vol estimation needs
    the full execution-cost ledger replay. Malformed input fails closed via the
    existing ``DataIntegrityError`` paths.
    """
    family_scores = build_xs_alpha_family_scores(
        closes, taker_buy_ratio, bar_funding, alpha_spec,
    )
    family_weights = build_xs_alpha_family_weights(
        closes, taker_buy_ratio, bar_funding, alpha_spec, execution_spec,
    )
    sleeve_returns: dict[str, pd.Series] = {}
    for name, family_w in family_weights.items():
        equity, _turnover = run_xs_composite_ledger(
            family_w, opens, bar_funding, execution_spec,
        )
        sleeve_returns[name] = equity.pct_change()
    sleeve_returns_frame = pd.DataFrame(sleeve_returns, index=closes.index)
    vol_weights = _causal_family_inverse_vol_weights(
        sleeve_returns_frame, alpha_spec.signal_windows[0],
    )
    combined = sum(
        vol_weights[name].to_numpy()[:, None] * family_scores[name].to_numpy()
        for name in family_scores
    )
    return build_xs_neutral_weights(
        pd.DataFrame(combined, index=closes.index, columns=list(closes.columns)),
        execution_spec.halflife_bars,
        execution_spec.no_trade_band,
    )


def run_xs_composite_ledger(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    spec: XsCompositeSpec,
) -> tuple[pd.Series, pd.Series]:
    """Compound the composite book into a total-equity ledger and turnover.

    Weights formed at close[t] are lagged by ``1 + execution_delay_bars`` bars
    so they are only held against the ``open[t+1+delay] -> open[t+2+delay]``
    return, matching ``run_technical_expert_backtest``. Each bar's net return is
    ``sum(w_lagged * open-to-open) - turnover * round_trip_cost_rate() -
    sum(w_lagged * bar_funding)`` where turnover is the row sum of absolute
    lagged-weight changes. Returns the strictly-positive equity ledger and the
    per-bar turnover series, both sharing the input index.
    """
    if not (
        weights.index.equals(opens.index)
        and weights.index.equals(bar_funding.index)
    ):
        raise DataIntegrityError(
            "weights, opens, and bar_funding must share an identical index"
        )
    if not (
        list(weights.columns) == list(opens.columns)
        and list(weights.columns) == list(bar_funding.columns)
    ):
        raise DataIntegrityError(
            "weights, opens, and bar_funding must share an identical column set"
        )

    w = weights.to_numpy(dtype=np.float64)
    o = opens.to_numpy(dtype=np.float64)
    f = bar_funding.to_numpy(dtype=np.float64)

    lag = 1 + spec.execution_delay_bars
    lagged = np.zeros_like(w)
    if lag < w.shape[0]:
        lagged[lag:] = w[: w.shape[0] - lag]

    o2o = np.zeros_like(o)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = o[1:] / o[:-1] - 1.0

    prev_lagged = np.zeros_like(lagged)
    prev_lagged[1:] = lagged[:-1]
    turnover = np.abs(lagged - prev_lagged).sum(axis=1)

    book_return = (lagged * o2o).sum(axis=1)
    funding = (lagged * f).sum(axis=1)
    net_returns = book_return - turnover * spec.round_trip_cost_rate() - funding

    equity_values = _INITIAL_EQUITY * np.cumprod(1.0 + net_returns)
    if not np.isfinite(equity_values).all() or (equity_values <= 0.0).any():
        raise DataIntegrityError("xs composite equity would reach zero")

    equity = pd.Series(equity_values, index=weights.index, name="equity", dtype=np.float64)
    turnover_series = pd.Series(turnover, index=weights.index, name="turnover", dtype=np.float64)
    return equity, turnover_series


@dataclass(frozen=True, slots=True)
class XsAdmissionConfig:
    """Scale-invariant, structure-only admission gates for the XS profile.

    ``sharpe_floor`` / ``beta_abs_max`` / ``turnover_max`` /
    ``cost_breakeven_min`` are the spec section 3.1 gates. ``annual_bars_min``
    is the minimum bars a calendar year needs to count toward the
    per-year-sub-sharpe gate, and ``round_trip_cost_rate`` is the per-unit-
    turnover charge used to reconstruct the gross return in the breakeven-cost
    computation. No absolute CAGR hurdle and no t-stat floor are applied.
    """

    sharpe_floor: float = 0.80
    beta_abs_max: float = 0.15
    annual_bars_min: int = 60
    turnover_max: float = 150.0
    cost_breakeven_min: float = 0.0024
    round_trip_cost_rate: float = 0.0008


@dataclass(frozen=True, slots=True)
class XsAdmissionResult:
    """Admission verdict, every failing gate, and the diagnostic metrics.

    ``binding_constraint`` names the sorted, stable set of failing gates and is
    ``None`` exactly when ``admitted`` is ``True``. ``cagr`` and ``t_stat`` are
    diagnostics only -- the profile never applies an absolute CAGR hurdle or a
    t-stat floor.
    """

    admitted: bool
    binding_constraint: str | None
    sharpe: float
    beta: float
    cagr: float
    mdd: float
    t_stat: float
    annual_sharpe: dict[str, float]
    annualized_turnover: float
    breakeven_cost: float


def _annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std())
    if std <= 0.0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(_BARS_PER_YEAR))


def _realized_beta(equity_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    var_bm = float(np.var(benchmark_returns, ddof=1))
    if var_bm <= 0.0:
        return 0.0
    cov = float(np.cov(equity_returns, benchmark_returns, ddof=1)[0, 1])
    return cov / var_bm


def evaluate_xs_admission(
    equity: pd.Series,
    turnover: pd.Series,
    benchmark: pd.Series,
    config: XsAdmissionConfig,
) -> XsAdmissionResult:
    """Evaluate the scale-invariant, structure-only admission gates.

    The book is admitted only when every gate passes: annualized Sharpe at or
    above ``sharpe_floor``, realized absolute beta against the benchmark return
    stream within ``beta_abs_max``, every calendar year with at least
    ``annual_bars_min`` bars has positive Sharpe, annualized turnover within
    ``turnover_max``, and the breakeven cost rate at or above
    ``cost_breakeven_min``. Every failing gate is recorded in the sorted
    ``binding_constraint``; the verdict is scale-invariant to a constant
    leverage rescaling.
    """
    if len(equity) < 2:
        raise ValueError("equity must have at least 2 marks")
    if not equity.index.equals(benchmark.index):
        raise ValueError("equity and benchmark must share an identical index")
    if not equity.index.equals(turnover.index):
        raise ValueError("equity and turnover must share an identical index")

    equity_returns = equity.pct_change().dropna()
    if len(equity_returns) < 2:
        raise ValueError("equity must have at least 2 usable return marks")
    eq_r_full = equity_returns.to_numpy(dtype=np.float64)
    bm_r_full = benchmark.reindex(equity_returns.index).to_numpy(dtype=np.float64)
    valid = np.isfinite(eq_r_full) & np.isfinite(bm_r_full)
    eq_r = eq_r_full[valid]
    bm_r = bm_r_full[valid]

    sharpe = _annualized_sharpe(equity_returns)
    beta = _realized_beta(eq_r, bm_r)

    years = float(
        (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    )
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1.0).min())

    t_stat = sharpe * np.sqrt(len(equity_returns) / _BARS_PER_YEAR)

    year_bars = equity_returns.groupby(equity_returns.index.year).size()
    annual_sharpe: dict[str, float] = {}
    for year, group in equity_returns.groupby(equity_returns.index.year):
        annual_sharpe[str(year)] = _annualized_sharpe(group)

    annualized_turnover = (
        float(turnover.sum() * _BARS_PER_YEAR / len(turnover))
        if len(turnover) > 0
        else 0.0
    )

    total_turnover = float(turnover.sum())
    if total_turnover > 0.0:
        gross_return = float(eq_r.sum()) + config.round_trip_cost_rate * total_turnover
        breakeven_cost = gross_return / total_turnover
    else:
        breakeven_cost = 0.0

    failed: list[str] = []
    if sharpe < config.sharpe_floor:
        failed.append("sharpe_floor")
    if abs(beta) > config.beta_abs_max:
        failed.append("beta_abs_max")
    bad_year = any(
        annual_sharpe[str(year)] <= 0.0 and int(year_bars[year]) >= config.annual_bars_min
        for year in year_bars.index
    )
    if bad_year:
        failed.append("annual_sub_sharpe")
    if annualized_turnover > config.turnover_max:
        failed.append("turnover_max")
    if breakeven_cost < config.cost_breakeven_min:
        failed.append("cost_breakeven_min")

    failed_sorted = sorted(set(failed))
    binding_constraint = ";".join(failed_sorted) if failed_sorted else None
    return XsAdmissionResult(
        admitted=not failed_sorted,
        binding_constraint=binding_constraint,
        sharpe=float(sharpe),
        beta=float(beta),
        cagr=float(cagr),
        mdd=float(mdd),
        t_stat=float(t_stat),
        annual_sharpe=annual_sharpe,
        annualized_turnover=annualized_turnover,
        breakeven_cost=breakeven_cost,
    )


def size_xs_alpha_growth_optimal(
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    spec: XsCompositeSpec,
    discovery_start: pd.Timestamp,
    discovery_end: pd.Timestamp,
    sizing_config: GrowthSizingConfig,
) -> tuple[pd.Series, pd.DataFrame, GrowthSizingResult]:
    """Select a discovery-only growth-optimal gross-leverage overlay for an XS book.

    The base ledger is replayed verbatim through :func:`run_xs_composite_ledger`
    (unchanged signature -- the frozen contract shared by v1..v6). Sizing is
    selected strictly from the ``[discovery_start, discovery_end]`` slice of
    the realized net returns via :func:`solve_growth_optimal_risk` --
    qualification and holdout bars never influence the chosen risk. When the
    constraints are infeasible the original net returns and the original input
    weights are returned unchanged (fail closed, never a default scale). When
    feasible, :func:`apply_realised_risk_overlay` scales the net returns and
    the lag-reconstructed realized weights
    (``weights.shift(1 + execution_delay_bars).fillna(0.0)`` -- the ledger's own
    lag convention, recomputed here rather than changing the frozen ledger
    signature) over the full available history, so the path-dependent drawdown
    ladder sees one continuous deployed history.
    """
    equity, _turnover = run_xs_composite_ledger(weights, opens, bar_funding, spec)
    net = equity.pct_change().dropna()
    discovery_net = net[(net.index >= discovery_start) & (net.index <= discovery_end)]
    sizing = solve_growth_optimal_risk(discovery_net.to_numpy(), sizing_config)
    if sizing.selected_risk is None:
        return net, weights, sizing

    net_full = equity.pct_change().fillna(0.0)
    lag = 1 + spec.execution_delay_bars
    realized_weights = weights.shift(lag).fillna(0.0)
    scaled_net, scaled_weights = apply_realised_risk_overlay(
        net_full, realized_weights, sizing.selected_risk, sizing_config.reference_risk,
    )
    return scaled_net, scaled_weights, sizing


def _check_contract() -> None:
    """Executable assertions locking the frozen cross-sectional surface."""
    from inspect import signature

    assert list(signature(apply_no_trade_band).parameters) == [
        "target_weights", "band",
    ]
    assert list(signature(build_xs_neutral_weights).parameters) == [
        "score", "halflife", "band",
    ]
    assert list(signature(run_xs_composite_ledger).parameters) == [
        "weights", "opens", "bar_funding", "spec",
    ]
    assert list(signature(evaluate_xs_admission).parameters) == [
        "equity", "turnover", "benchmark", "config",
    ]
    assert list(signature(size_xs_alpha_growth_optimal).parameters) == [
        "weights", "opens", "bar_funding", "spec", "discovery_start",
        "discovery_end", "sizing_config",
    ]
    spec = XsCompositeSpec()
    assert (spec.halflife_bars, spec.no_trade_band, spec.execution_delay_bars) == (6, 0.05, 1)
    assert abs(spec.round_trip_cost_rate() - 0.0008) < 1e-12
    assert list(signature(build_xs_alpha_composite_score).parameters) == [
        "closes", "taker_buy_ratio", "bar_funding", "spec",
    ]
    assert list(signature(build_xs_alpha_family_scores).parameters) == [
        "closes", "taker_buy_ratio", "bar_funding", "spec",
    ]
    assert list(signature(build_xs_alpha_weights).parameters) == [
        "closes", "taker_buy_ratio", "bar_funding", "alpha_spec", "execution_spec",
    ]
    assert list(signature(build_xs_alpha_family_weights).parameters) == [
        "closes", "taker_buy_ratio", "bar_funding", "alpha_spec", "execution_spec",
    ]
    assert list(signature(build_xs_alpha_dual_family_weights).parameters) == [
        "closes", "taker_buy_ratio", "bar_funding", "alpha_spec", "execution_spec",
    ]
    assert list(signature(_causal_family_inverse_vol_weights).parameters) == [
        "sleeve_returns", "lookback_bars",
    ]
    assert list(signature(build_xs_alpha_vol_weighted_weights).parameters) == [
        "closes", "taker_buy_ratio", "bar_funding", "opens", "alpha_spec", "execution_spec",
    ]
    assert XsAlphaCompositeSpec().signal_windows == (42, 84, 168)


_check_contract()
