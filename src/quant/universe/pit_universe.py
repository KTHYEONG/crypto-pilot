from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Literal

import pandas as pd


@dataclass(frozen=True, slots=True)
class PitUniverseSpec:
    """Immutable, pre-registered point-in-time universe construction parameters.

    ``universe_size`` is the liquidity-ranked roster width and is deliberately
    reused as the only cutoff by :func:`derive_backfill_candidates`, so no
    secondary threshold or margin constant is ever invented for backfill
    prioritisation.
    """

    universe_size: int = 20
    max_positions: int = 5
    seasoning_days: int = 365
    liquidity_lookback_days: int = 30
    min_bar_coverage: float = 0.99
    dev_fraction: float = 0.80

    def __post_init__(self) -> None:
        if not self.universe_size >= self.max_positions >= 1:
            raise ValueError(
                f"universe_size >= max_positions >= 1 must hold, got "
                f"universe_size={self.universe_size} max_positions={self.max_positions}"
            )
        if self.seasoning_days < 1:
            raise ValueError(f"seasoning_days must be >= 1, got {self.seasoning_days}")
        if self.liquidity_lookback_days < 1:
            raise ValueError(
                f"liquidity_lookback_days must be >= 1, got {self.liquidity_lookback_days}"
            )
        if not 0 < self.min_bar_coverage <= 1:
            raise ValueError(
                f"min_bar_coverage must be in (0, 1], got {self.min_bar_coverage}"
            )
        if not 0 < self.dev_fraction < 1:
            raise ValueError(f"dev_fraction must be in (0, 1), got {self.dev_fraction}")


@dataclass(frozen=True, slots=True)
class SymbolCoverage:
    """Per-symbol bar-coverage metadata derived from the archive, never from prices."""

    symbol: str
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    bar_coverage: float

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.first_bar.tzinfo is None or self.last_bar.tzinfo is None:
            raise ValueError("first_bar and last_bar must be tz-aware UTC")
        if self.last_bar < self.first_bar:
            raise ValueError("last_bar must not precede first_bar")
        if not 0 <= self.bar_coverage <= 1:
            raise ValueError(f"bar_coverage must be in [0, 1], got {self.bar_coverage}")


def _validate_rebalance_dates(rebalance_dates: Sequence[pd.Timestamp]) -> list[pd.Timestamp]:
    dates = list(pd.DatetimeIndex(list(rebalance_dates)))
    if not dates:
        return dates
    if any(ts.tzinfo is None for ts in dates):
        raise ValueError("rebalance_dates must be tz-aware UTC")
    if len(dates) > 1 and not all(
        dates[i] < dates[i + 1] for i in range(len(dates) - 1)
    ):
        raise ValueError("rebalance_dates must be strictly monotonic increasing")
    return dates


def symbol_partition(symbol: str, dev_fraction: float = 0.80) -> Literal["dev", "holdout"]:
    """Deterministic pre-registered symbol split driven only by the symbol string.

    The bucket is ``int(sha256(symbol)[:8], 16) % 100``; a symbol lands in
    ``"dev"`` exactly when its bucket is below ``dev_fraction * 100``.  No
    returns, performance, or market data is ever consulted, so the split cannot
    be tuned to hide a failing signal.
    """
    if not symbol:
        raise ValueError("symbol must not be empty")
    if not 0 < dev_fraction < 1:
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
    bucket = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 100
    return "dev" if bucket < dev_fraction * 100 else "holdout"


def _eligible_at(cov: SymbolCoverage, date: pd.Timestamp, spec: PitUniverseSpec) -> bool:
    return (
        cov.first_bar <= date - pd.Timedelta(days=spec.seasoning_days)
        and cov.last_bar >= date
        and cov.bar_coverage >= spec.min_bar_coverage
    )


def _eligible_count(
    coverage: Sequence[SymbolCoverage],
    date: pd.Timestamp,
    spec: PitUniverseSpec,
) -> int:
    return sum(1 for cov in coverage if _eligible_at(cov, date, spec))


def earliest_admissible_start(
    coverage: Sequence[SymbolCoverage],
    rebalance_dates: Sequence[pd.Timestamp],
    spec: PitUniverseSpec,
) -> pd.Timestamp | None:
    """Derive the evaluation start from data instead of a hardcoded calendar date.

    Returns the earliest rebalance date ``d`` such that the eligible pool is
    ``>= spec.universe_size`` at ``d`` and at every later rebalance date;
    ``None`` (fail-closed) when no such date exists.
    """
    dates = _validate_rebalance_dates(rebalance_dates)
    for i, date in enumerate(dates):
        if _eligible_count(coverage, date, spec) < spec.universe_size:
            continue
        if all(
            _eligible_count(coverage, later, spec) >= spec.universe_size
            for later in dates[i + 1 :]
        ):
            return date
    return None


def build_universe_schedule(
    coverage: Sequence[SymbolCoverage],
    liquidity: Mapping[str, pd.Series],
    rebalance_dates: Sequence[pd.Timestamp],
    spec: PitUniverseSpec,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    """Build a strictly causal liquidity-ranked PIT universe per rebalance date.

    Only bars with ``index < date`` (completed before the decision) within the
    trailing ``liquidity_lookback_days`` window contribute to the quote-volume
    ranking, so a future bar can never enter the roster.  Ranking is descending
    trailing quote volume with a lexicographic symbol tie-break; symbols with an
    empty window or any NaN/negative quote volume are excluded.  Realized
    returns are never consulted.
    """
    dates = _validate_rebalance_dates(rebalance_dates)
    schedule: dict[pd.Timestamp, tuple[str, ...]] = {}
    for date in dates:
        ranked: list[tuple[float, str]] = []
        for cov in coverage:
            if not _eligible_at(cov, date, spec):
                continue
            series = liquidity.get(cov.symbol)
            if series is None:
                continue
            window = series.loc[
                (series.index < date)
                & (series.index >= date - pd.Timedelta(days=spec.liquidity_lookback_days))
            ]
            if window.empty:
                continue
            quote_vol = pd.to_numeric(window, errors="coerce")
            if quote_vol.isna().any() or (quote_vol < 0).any():
                continue
            ranked.append((float(quote_vol.sum()), cov.symbol))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        schedule[date] = tuple(symbol for _, symbol in ranked[: spec.universe_size])
    return schedule


def derive_backfill_candidates(
    coverage: Sequence[SymbolCoverage],
    liquidity: Mapping[str, pd.Series],
    rebalance_dates: Sequence[pd.Timestamp],
    spec: PitUniverseSpec,
) -> tuple[str, ...]:
    """Minimal backfill filter: the sorted union of every scheduled roster.

    Reuses ``spec.universe_size`` as the sole cutoff inside
    :func:`build_universe_schedule`; no new threshold, percentile, or margin
    constant is introduced.  The result is order-independent with respect to
    ``rebalance_dates``.
    """
    schedule = build_universe_schedule(coverage, liquidity, rebalance_dates, spec)
    union: set[str] = set()
    for roster in schedule.values():
        union.update(roster)
    return tuple(sorted(union))


def _check_contract() -> None:
    """Executable assertions locking the frozen PIT universe contract surface."""
    spec = PitUniverseSpec()
    assert (spec.universe_size, spec.max_positions, spec.seasoning_days,
            spec.liquidity_lookback_days) == (20, 5, 365, 30)
    assert {f.name for f in fields(PitUniverseSpec)} == {
        "universe_size", "max_positions", "seasoning_days", "liquidity_lookback_days",
        "min_bar_coverage", "dev_fraction",
    }
    assert {f.name for f in fields(SymbolCoverage)} == {
        "symbol", "first_bar", "last_bar", "bar_coverage",
    }
    assert symbol_partition("BTCUSDT") == "dev"
    assert symbol_partition("ETHUSDT") == "holdout"
    assert symbol_partition("SOLUSDT") == "dev"
    assert symbol_partition("BTCUSDT") == symbol_partition("BTCUSDT")


_check_contract()
