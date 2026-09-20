"""Frozen PIT target policy and causal candidate builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.books import rank_weight_book
from src.mhs.features import FEATURE_REGISTRY
from src.mhs.frozen_research_universe import build_frozen_pit_roster

_REQUIRED_PANELS = ("close", "quote_vol", "taker_buy_quote")
_HISTORY_BARS = 720


@dataclass(frozen=True, slots=True)
class FrozenFeatureMember:
    """Declare one registered cross-sectional feature and its fixed rank direction."""

    name: str
    sign: Literal[-1, 1]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("member name must be a non-empty string")
        if self.sign not in (-1, 1):
            raise ValueError(f"member sign must be -1 or +1, got {self.sign!r}")


@dataclass(frozen=True, slots=True)
class FrozenMhsStrategySpec:
    """Declare an immutable PIT target policy independently of execution mechanics.

    The definition binds universe breadth, registered feature identities,
    ranking population, and the UTC observation-to-entry clock.  It lets new
    research variants create the same executable target contract without
    changing the inventory accounting engine.
    """

    strategy_id: str
    breadth: int
    members: tuple[FrozenFeatureMember, ...]
    min_rank_symbols: int
    snapshot_hour_utc: int
    release_hour_utc: int
    entry_hour_utc: int

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_id, str) or not self.strategy_id:
            raise ValueError("strategy_id must be a non-empty string")
        if isinstance(self.breadth, bool) or not isinstance(self.breadth, int) or self.breadth <= 0:
            raise ValueError(f"breadth must be a positive integer, got {self.breadth!r}")
        if not isinstance(self.members, tuple) or len(self.members) == 0:
            raise ValueError("members must be a non-empty tuple of FrozenFeatureMember")
        names = [m.name for m in self.members]
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("member names must be unique non-empty strings")
        if (
            isinstance(self.min_rank_symbols, bool)
            or not isinstance(self.min_rank_symbols, int)
            or self.min_rank_symbols < 2
        ):
            raise ValueError(f"min_rank_symbols must be an integer >= 2, got {self.min_rank_symbols!r}")
        for label, hour in (
            ("snapshot_hour_utc", self.snapshot_hour_utc),
            ("release_hour_utc", self.release_hour_utc),
            ("entry_hour_utc", self.entry_hour_utc),
        ):
            if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
                raise ValueError(f"{label} must be an integer hour in [0, 23], got {hour!r}")
        if not self.snapshot_hour_utc < self.release_hour_utc:
            raise ValueError("snapshot_hour_utc must be strictly earlier than release_hour_utc")


FROZEN_MHS_TOP20_V1 = FrozenMhsStrategySpec(
    strategy_id="frozen_mhs_top20_v1",
    breadth=20,
    members=(
        FrozenFeatureMember(name="flow_imb_168h", sign=1),
        FrozenFeatureMember(name="flow_imb_720h", sign=1),
        FrozenFeatureMember(name="xs_mom_336h", sign=1),
        FrozenFeatureMember(name="xs_idio_mom_336h", sign=1),
        FrozenFeatureMember(name="mom3_skew_168h", sign=1),
    ),
    min_rank_symbols=8,
    snapshot_hour_utc=22,
    release_hour_utc=23,
    entry_hour_utc=0,
)

FROZEN_MHS_TOP40_CONTROL_V1 = FrozenMhsStrategySpec(
    strategy_id="frozen_mhs_top40_control_v1",
    breadth=40,
    members=(
        FrozenFeatureMember(name="flow_imb_168h", sign=1),
        FrozenFeatureMember(name="flow_imb_720h", sign=1),
        FrozenFeatureMember(name="xs_mom_336h", sign=1),
        FrozenFeatureMember(name="xs_idio_mom_336h", sign=1),
        FrozenFeatureMember(name="mom3_skew_168h", sign=1),
    ),
    min_rank_symbols=8,
    snapshot_hour_utc=22,
    release_hour_utc=23,
    entry_hour_utc=0,
)


@dataclass(frozen=True, slots=True)
class FrozenMhsCandidate:
    """Bind exact PIT target weights to their release and entry timestamps.

    ``target_weights.index`` is the UTC entry time, not the source-observation
    time.  Every row is actionable only after its matching release timestamp;
    this distinction is retained through the three-minute execution stream.
    """

    target_weights: pd.DataFrame
    signal_available_at: pd.DatetimeIndex
    strategy: FrozenMhsStrategySpec

    def __post_init__(self) -> None:
        if len(self.target_weights) != len(self.signal_available_at):
            raise DataIntegrityError("target_weights and signal_available_at must have matching rows")

    @property
    def breadth(self) -> int:
        return self.strategy.breadth


def build_frozen_mhs_candidate(
    hourly_panels: Mapping[str, pd.DataFrame],
    hourly_available_at: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_quote_volume: pd.DataFrame,
    census_symbols: tuple[str, ...],
    *,
    strategy: FrozenMhsStrategySpec = FROZEN_MHS_TOP20_V1,
    blocked_decisions: pd.DataFrame | None = None,
) -> FrozenMhsCandidate:
    """Build one immutable, causal target plan from complete hourly sources.

    The builder evaluates only registered features and source bars known by each strategy
    release timestamp. It emits dollar-neutral target rows at the declared entry time and
    emits zero rather than a retrospective proxy whenever required history, population, or
    recoverable execution evidence is unavailable for that specific decision.

    Args:
        hourly_panels: Aligned close, quote-volume, and taker-buy-quote planes.
        hourly_available_at: Per-symbol publication timestamps for those bars.
        daily_close: Complete historical daily close census for PIT membership.
        daily_quote_volume: Complete historical daily turnover census.
        census_symbols: Canonical source-symbol order, including retired names.
        strategy: Frozen target definition; Top-20 v1 is the primary default.
        blocked_decisions: Boolean decision-day frame withdrawing a symbol from both roster
            eligibility and target emission for exactly those days.
    Returns:
        Exact entry targets and matching signal-release timestamps.
    Raises:
        DataIntegrityError: Source planes, census, or roster evidence is inconsistent.
    """
    if not isinstance(strategy, FrozenMhsStrategySpec):
        raise ValueError("strategy must be a FrozenMhsStrategySpec")
    census = list(census_symbols)
    registry = {spec.name: spec for spec in FEATURE_REGISTRY}
    if any(m.name not in registry for m in strategy.members):
        raise ValueError("strategy references an unregistered feature")
    roster = build_frozen_pit_roster(
        daily_close,
        daily_quote_volume,
        census_symbols,
        breadth=strategy.breadth,
        blocked_decisions=blocked_decisions,
    )
    if any(k not in hourly_panels for k in _REQUIRED_PANELS):
        raise DataIntegrityError("hourly_panels must contain close, quote_vol, and taker_buy_quote")
    panels = {k: hourly_panels[k] for k in _REQUIRED_PANELS}
    first = panels["close"]
    if any(not isinstance(p.index, pd.DatetimeIndex) or not _is_utc_hourly_grid(p.index) for p in panels.values()):
        raise DataIntegrityError("hourly indexes must be unique increasing UTC hour-open labels on a 1h grid")
    if any(not p.index.equals(first.index) or list(p.columns) != list(first.columns) for p in panels.values()):
        raise DataIntegrityError("hourly panels must share an identical index and column order")
    hourly_symbols = list(first.columns)
    ever_selected = list(roster.columns[roster.any(axis=0)])
    if any(s not in hourly_symbols for s in ever_selected):
        raise DataIntegrityError("every historically selected symbol must have an hourly source archive")
    if not hourly_available_at.index.equals(first.index) or list(hourly_available_at.columns) != hourly_symbols:
        raise DataIntegrityError("hourly_available_at must align with the hourly panel")
    if hourly_available_at.isna().any().any():
        raise DataIntegrityError("hourly_available_at must not contain missing publication timestamps")
    if not all(
        isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"
        for dtype in hourly_available_at.dtypes
    ):
        raise DataIntegrityError("hourly_available_at must contain timezone-aware timestamps")
    available_values = hourly_available_at.to_numpy(dtype="datetime64[ns]")
    grid_values = first.index.to_numpy(dtype="datetime64[ns]")[:, None]
    if bool((available_values < grid_values).any()):
        raise DataIntegrityError("hourly publication cannot precede the bar open")
    available_utc = hourly_available_at.apply(lambda col: pd.to_datetime(col, utc=True))
    census = list(census_symbols)
    features = [(m, registry[m.name].builder(panels)) for m in strategy.members]
    daily_idx = roster.index
    decisions = daily_idx[:-1]
    roster_dec = roster.loc[decisions]
    hourly_index = first.index
    books = []
    for member, feature in features:
        aligned = feature.reindex(columns=census)
        snaps = pd.DataFrame(float("nan"), index=decisions, columns=census, dtype="float64")
        for stamp in decisions:
            bar = stamp + pd.Timedelta(hours=int(strategy.snapshot_hour_utc))
            release = stamp + pd.Timedelta(hours=int(strategy.release_hour_utc))
            if bar in hourly_index:
                bar_pos = hourly_index.get_loc(bar)
                history_start = max(0, int(bar_pos) - (_HISTORY_BARS - 1))
                publication = available_utc.iloc[history_start : int(bar_pos) + 1]
                source_ready = publication.le(release).all(axis=0)
                values = aligned.loc[bar].reindex(census)
                values = values.where(source_ready.reindex(census), other=float("nan"))
                snaps.loc[stamp] = values.to_numpy()
        eligible = roster_dec & snaps.notna()
        books.append(rank_weight_book(snaps, eligible, int(member.sign), int(strategy.min_rank_symbols)))
    ensemble = books[0]
    for book in books[1:]:
        ensemble = ensemble.add(book)
    ensemble = ensemble / float(len(books))
    entries = pd.DatetimeIndex(
        [d + pd.Timedelta(days=1, hours=int(strategy.entry_hour_utc)) for d in decisions], tz="UTC"
    )
    target = pd.DataFrame(ensemble.to_numpy(dtype="float64"), index=entries, columns=census, dtype="float64")
    available = pd.DatetimeIndex(
        [d + pd.Timedelta(hours=int(strategy.release_hour_utc)) for d in decisions], tz="UTC"
    )
    return FrozenMhsCandidate(target_weights=target, signal_available_at=available, strategy=strategy)


def _is_utc_hourly_grid(idx: pd.DatetimeIndex) -> bool:
    """Check unique increasing UTC hour-open labels on an exact 1-hour grid."""
    if idx.tz is None:
        return False
    utc = idx.tz_convert("UTC")
    if not idx.equals(utc):
        return False
    if bool((idx.minute != 0).any()) or bool((idx.second != 0).any()) or bool((idx.microsecond != 0).any()):
        return False
    if bool(idx.duplicated().any()) or not bool(idx.is_monotonic_increasing):
        return False
    return not bool(((idx[1:] - idx[:-1]) != pd.Timedelta(hours=1)).any())
