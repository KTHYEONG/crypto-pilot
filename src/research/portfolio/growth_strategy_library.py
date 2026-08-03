"""Source-controlled cross-sectional growth strategy library.

The engine promotes exactly one frozen ``v1`` strategy family after the
unchanged falsification and sizing gates, or publishes a reproducible
scorecard and holds CASH.  Every family carries three calendar-horizon
variants, so the family-wide multiplicity size is exactly ``FAMILY_SIZE``.
All numerical work is vectorized NumPy/pandas; no ``DataFrame.apply`` and no
per-row Python loops are used for scores or weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError

XS_QUANTILE = 0.30


@dataclass(frozen=True, slots=True)
class GrowthStrategyDefinition:
    """Immutable identity of one source-controlled strategy family."""

    strategy_id: str
    windows: tuple[int, ...]
    requires_funding: bool = False

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must not be empty")
        if not self.windows:
            raise ValueError("windows must not be empty")
        if any(w < 2 for w in self.windows):
            raise ValueError(f"windows must be >= 2, got {self.windows}")


@dataclass(frozen=True, slots=True)
class GrowthStrategyScreen:
    """Immutable per-candidate result: screened weights or a fail-closed screen.

    ``status`` is ``"SCREENED"`` when every scheduled symbol satisfied the
    strategy's data contract, ``"DATA_INVALID"`` when a funding-dependent
    candidate lacks finite, causally alignable funding history.
    """

    strategy_id: str
    parameter: int
    status: Literal["SCREENED", "DATA_INVALID"]
    weights: pd.DataFrame
    reason: str | None = None


#: Exactly four independent, source-controlled v1 families.  The direct
#: reversal, market-residual trend, and legacy ``xs_momentum`` identities are
#: retired and cannot be reintroduced under a renamed identifier.
STRATEGY_REGISTRY: tuple[GrowthStrategyDefinition, ...] = (
    GrowthStrategyDefinition("funding_contrarian_v1", (42, 84, 168), requires_funding=True),
    GrowthStrategyDefinition("taker_imbalance_v1", (42, 84, 168)),
    GrowthStrategyDefinition("vol_adjusted_trend_v1", (42, 84, 180)),
    GrowthStrategyDefinition("donchian_channel_position_v1", (42, 84, 168)),
)

RETIRED_STRATEGY_IDS: tuple[str, ...] = (
    "xs_momentum",
    "short_horizon_reversal_v1",
    "market_residual_trend_v1",
)

FAMILY_SIZE: int = sum(len(definition.windows) for definition in STRATEGY_REGISTRY)


def registry_definition(strategy_id: str) -> GrowthStrategyDefinition:
    """Return the frozen definition for a v1 identity, failing closed otherwise.

    Unknown identities raise ``ValueError``; retired identities are explicitly
    named so they cannot be reintroduced under a renamed identifier.
    """
    if strategy_id in RETIRED_STRATEGY_IDS:
        raise ValueError(
            f"strategy {strategy_id!r} is retired and cannot be re-screened"
        )
    for definition in STRATEGY_REGISTRY:
        if definition.strategy_id == strategy_id:
            return definition
    raise ValueError(f"unknown growth strategy identity: {strategy_id!r}")


def _validate_frame(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataIntegrityError(f"{name} must have a DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz) != "UTC":
        raise DataIntegrityError(f"{name} must have a tz-aware UTC index")
    if not frame.index.is_monotonic_increasing:
        raise DataIntegrityError(f"{name} must have a monotonic increasing index")


def align_funding_bars(
    funding: pd.DataFrame,
    grid: pd.DatetimeIndex,
    *,
    forward: bool = False,
) -> pd.DataFrame:
    """Aggregate raw funding events into per-bar sums on the decision grid.

    ``settled`` buckets (``forward=False``) cover ``[bar[t], bar[t+1])`` and
    ``forward`` buckets (``forward=True``) cover ``(bar[t], bar[t+1]]``, so a
    settlement inside the post-decision interval contributes to realised PnL at
    ``t`` while the decision's score at ``t`` only ever sees funding up to the
    previous completed settlement.  Missing, non-finite, non-UTC,
    non-monotonic, or unalignable funding raises ``DataIntegrityError`` and is
    never zero-filled.
    """
    if not isinstance(funding.index, pd.DatetimeIndex):
        raise DataIntegrityError("funding must have a DatetimeIndex")
    if funding.index.tz is None:
        raise DataIntegrityError("funding must have a tz-aware index")
    if str(funding.index.tz) != "UTC":
        raise DataIntegrityError("funding index must be UTC")
    if funding.index.hasnans:
        raise DataIntegrityError("funding index must not contain NaT")
    if not funding.index.is_monotonic_increasing:
        raise DataIntegrityError("funding index must be monotonic increasing")
    if len(grid) < 2:
        raise DataIntegrityError("grid must have at least two bars")
    bar_period = grid[1] - grid[0]
    window_end = grid[-1] + bar_period
    out = pd.DataFrame(0.0, index=grid, columns=list(funding.columns), dtype="float64")
    side = "left" if forward else "right"
    for column in funding.columns:
        rates = pd.to_numeric(funding[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(rates).all():
            raise DataIntegrityError(f"funding for {column} must be finite")
        ts = funding.index
        if len(ts):
            inside = (ts >= grid[0]) & (ts <= window_end)
            if not inside.all():
                raise DataIntegrityError(
                    f"funding for {column} is not causally alignable to the bar window"
                )
        pos = grid.searchsorted(ts, side=side) - 1
        aligned = np.zeros(len(grid), dtype=np.float64)
        keep = pos >= 0
        np.add.at(aligned, pos[keep], rates[keep])
        out[column] = aligned
    return out


def _score_panel(
    strategy_id: str,
    parameter: int,
    prices: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    funding_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    """Causal score at completed decision bar ``t``, all in vectorized form."""
    if strategy_id == "funding_contrarian_v1":
        if funding_panel is None:
            raise DataIntegrityError(
                "funding_contrarian_v1 requires aligned settled funding"
            )
        return -funding_panel.shift(1).rolling(parameter).mean()
    if strategy_id == "taker_imbalance_v1":
        return taker_buy_ratio.rolling(parameter).mean() - 0.5
    if strategy_id == "vol_adjusted_trend_v1":
        ret_w = prices.pct_change(parameter)
        std_w = prices.pct_change().rolling(parameter).std()
        score = ret_w / std_w
        return score.mask(~np.isfinite(score))
    if strategy_id == "donchian_channel_position_v1":
        rolling_high = prices.rolling(parameter).max()
        rolling_low = prices.rolling(parameter).min()
        width = rolling_high - rolling_low
        score = (prices - rolling_low) / width
        return score.mask(~np.isfinite(score))
    raise ValueError(f"unsupported strategy identity: {strategy_id!r}")


def _cross_sectional_weights(
    score: pd.DataFrame,
    schedule: Mapping[pd.Timestamp, tuple[str, ...]],
    grid: pd.DatetimeIndex,
    columns: Sequence[str],
    quantile: float = XS_QUANTILE,
) -> pd.DataFrame:
    """PIT-roster-only, dollar-neutral long/short weights for every grid bar.

    Longs and shorts are the top/bottom ``quantile`` of the roster by score
    with equal weight inside each leg; a missing window, a zero channel width,
    or fewer than one valid long/short member yields zero weight for that bar.
    """
    weights = pd.DataFrame(0.0, index=grid, columns=list(columns), dtype="float64")
    month_key = grid.normalize() - pd.to_timedelta(grid.day - 1, unit="D")
    for date, roster in schedule.items():
        if not roster:
            continue
        bars = grid[month_key == date]
        sub = score.loc[bars, list(roster)]
        valid = sub.notna()
        cnt = valid.sum(axis=1)
        rank = sub.rank(axis=1, ascending=False)
        k = (cnt * quantile).round().astype(int).clip(lower=1)
        longs = rank.le(k, axis=0) & valid
        shorts = rank.gt(cnt - k, axis=0) & valid
        w_long = longs.astype(float).div(longs.sum(axis=1).replace(0, np.nan), axis=0)
        w_short = shorts.astype(float).div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
        w = (w_long - w_short) / 2
        weights.loc[bars, list(roster)] = w.fillna(0.0).to_numpy()
    return weights


def _schedule_symbols(schedule: Mapping[pd.Timestamp, tuple[str, ...]]) -> tuple[str, ...]:
    return tuple(sorted({sym for roster in schedule.values() for sym in roster}))


def build_growth_strategy_weights(
    strategy_id: str,
    parameter: int,
    schedule: Mapping[pd.Timestamp, tuple[str, ...]],
    prices: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    settled_funding: pd.DataFrame,
) -> pd.DataFrame:
    """Build UTC-indexed, roster-only, dollar-neutral weights for one candidate.

    ``parameter`` must be a registered window of the strategy's family.  A
    funding-dependent candidate raises ``DataIntegrityError`` (recorded as
    ``DATA_INVALID`` by :func:`screen_growth_strategy_weights`) when any
    scheduled symbol lacks a finite, causally alignable funding history; it is
    never silently zero-filled and never invalidates price-only candidates.
    """
    definition = registry_definition(strategy_id)
    if parameter not in definition.windows:
        raise ValueError(
            f"parameter {parameter} is not a registered window for "
            f"{strategy_id} (windows={definition.windows})"
        )
    _validate_frame(prices, "prices")
    _validate_frame(taker_buy_ratio, "taker_buy_ratio")
    if not taker_buy_ratio.index.equals(prices.index):
        raise DataIntegrityError("taker_buy_ratio must share the identical price index")
    roster_symbols = _schedule_symbols(schedule)

    funding_panel: pd.DataFrame | None = None
    if definition.requires_funding:
        if settled_funding.empty:
            raise DataIntegrityError(f"{strategy_id} requires settled funding data")
        missing = [sym for sym in roster_symbols if sym not in settled_funding.columns]
        if missing:
            raise DataIntegrityError(f"{strategy_id} is missing funding for {missing}")
        funding_panel = align_funding_bars(
            settled_funding[list(roster_symbols)], prices.index, forward=False,
        )
        if not np.isfinite(funding_panel.to_numpy(dtype=np.float64)).all():
            raise DataIntegrityError(
                f"{strategy_id} has non-finite funding for a scheduled symbol"
            )

    score = _score_panel(strategy_id, parameter, prices, taker_buy_ratio, funding_panel)
    return _cross_sectional_weights(score, schedule, prices.index, prices.columns)


def screen_growth_strategy_weights(
    strategy_id: str,
    parameter: int,
    schedule: Mapping[pd.Timestamp, tuple[str, ...]],
    prices: pd.DataFrame,
    taker_buy_ratio: pd.DataFrame,
    settled_funding: pd.DataFrame,
) -> GrowthStrategyScreen:
    """Screen one candidate, recording funding-integrity failures as DATA_INVALID."""
    try:
        weights = build_growth_strategy_weights(
            strategy_id, parameter, schedule, prices, taker_buy_ratio, settled_funding,
        )
    except DataIntegrityError as exc:
        empty_weights = (
            pd.DataFrame(index=prices.index, columns=prices.columns, dtype="float64")
            if prices is not None
            else pd.DataFrame()
        )
        return GrowthStrategyScreen(
            strategy_id=strategy_id,
            parameter=parameter,
            status="DATA_INVALID",
            weights=empty_weights,
            reason=f"{type(exc).__name__}: {exc}",
        )
    return GrowthStrategyScreen(
        strategy_id=strategy_id,
        parameter=parameter,
        status="SCREENED",
        weights=weights,
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen v1 strategy library surface."""
    assert FAMILY_SIZE == 12
    assert tuple(d.strategy_id for d in STRATEGY_REGISTRY) == (
        "funding_contrarian_v1",
        "taker_imbalance_v1",
        "vol_adjusted_trend_v1",
        "donchian_channel_position_v1",
    )
    assert tuple(d.windows for d in STRATEGY_REGISTRY) == (
        (42, 84, 168),
        (42, 84, 168),
        (42, 84, 180),
        (42, 84, 168),
    )
    assert all(d.requires_funding == (d.strategy_id == "funding_contrarian_v1") for d in STRATEGY_REGISTRY)
    for retired in RETIRED_STRATEGY_IDS:
        try:
            registry_definition(retired)
            raise AssertionError(f"{retired} must be retired")
        except ValueError:
            pass


_check_contract()
