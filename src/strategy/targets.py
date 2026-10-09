"""Strategy PIT target policy and causal candidate builder."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.params import GROWTH_EXPOSURE_MULTIPLIER, STRATEGY_NAME_CLIP
from src.strategy.books import clip_names_preserving_gross, rank_weight_book
from src.strategy.features import FEATURE_REGISTRY, MARKET_CLOSE_PANEL
from src.strategy.universe import build_pit_roster

_REQUIRED_PANELS = ("close", "quote_vol", "taker_buy_quote")
_HISTORY_BARS = 720


@dataclass(frozen=True, slots=True)
class FeatureMember:
    """Declare one registered cross-sectional feature and its fixed rank direction."""

    name: str
    sign: Literal[-1, 1]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("member name must be a non-empty string")
        if self.sign not in (-1, 1):
            raise ValueError(f"member sign must be -1 or +1, got {self.sign!r}")


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Declare an immutable PIT target policy independently of execution mechanics."""

    strategy_id: str
    breadth: int
    members: tuple[FeatureMember, ...]
    min_rank_symbols: int
    snapshot_hour_utc: int
    release_hour_utc: int
    entry_hour_utc: int
    design_data_cutoff: pd.Timestamp
    exposure_multiplier: float = 1.0
    name_clip: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.design_data_cutoff, pd.Timestamp) or pd.isna(self.design_data_cutoff):
            raise ValueError("design_data_cutoff must be a valid timestamp")
        if self.design_data_cutoff.tzinfo is None or self.design_data_cutoff.utcoffset() is None:
            raise ValueError("design_data_cutoff must be timezone-aware UTC")
        if self.design_data_cutoff.utcoffset().total_seconds() != 0:
            raise ValueError("design_data_cutoff must be timezone-aware UTC")
        if not isinstance(self.strategy_id, str) or not self.strategy_id:
            raise ValueError("strategy_id must be a non-empty string")
        if isinstance(self.breadth, bool) or not isinstance(self.breadth, int) or self.breadth <= 0:
            raise ValueError(f"breadth must be a positive integer, got {self.breadth!r}")
        if not isinstance(self.members, tuple) or len(self.members) == 0:
            raise ValueError("members must be a non-empty tuple of FeatureMember")
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
        if (
            isinstance(self.exposure_multiplier, bool)
            or not isinstance(self.exposure_multiplier, (int, float))
            or not (math.isfinite(float(self.exposure_multiplier)) and float(self.exposure_multiplier) > 0.0)
        ):
            raise ValueError(f"exposure_multiplier must be a finite float > 0, got {self.exposure_multiplier!r}")
        if self.name_clip is not None and (
            isinstance(self.name_clip, bool)
            or not isinstance(self.name_clip, (int, float))
            or not (math.isfinite(float(self.name_clip)) and 0.0 < float(self.name_clip) <= 1.0)
        ):
            raise ValueError(f"name_clip must be None or a finite float in (0, 1], got {self.name_clip!r}")


FLOW_MOM_TOP20 = StrategySpec(
    strategy_id="flow_mom_top20",
    breadth=20,
    members=(
        FeatureMember(name="flow_imb_168h", sign=1),
        FeatureMember(name="flow_imb_720h", sign=1),
        FeatureMember(name="xs_mom_336h", sign=1),
        FeatureMember(name="xs_idio_mom_336h", sign=1),
        FeatureMember(name="mom3_skew_168h", sign=1),
    ),
    min_rank_symbols=8,
    snapshot_hour_utc=22,
    release_hour_utc=23,
    entry_hour_utc=0,
    design_data_cutoff=pd.Timestamp("2026-07-01T00:00:00Z"),
)

FLOW_MOM_TOP40_CONTROL = StrategySpec(
    strategy_id="flow_mom_top40_control",
    breadth=40,
    members=(
        FeatureMember(name="flow_imb_168h", sign=1),
        FeatureMember(name="flow_imb_720h", sign=1),
        FeatureMember(name="xs_mom_336h", sign=1),
        FeatureMember(name="xs_idio_mom_336h", sign=1),
        FeatureMember(name="mom3_skew_168h", sign=1),
    ),
    min_rank_symbols=8,
    snapshot_hour_utc=22,
    release_hour_utc=23,
    entry_hour_utc=0,
    design_data_cutoff=pd.Timestamp("2026-07-01T00:00:00Z"),
)

FLOW_MOM_TOP20_GROWTH = StrategySpec(
    strategy_id="flow_mom_top20_growth",
    breadth=20,
    members=(
        FeatureMember(name="flow_imb_168h", sign=1),
        FeatureMember(name="flow_imb_720h", sign=1),
        FeatureMember(name="xs_mom_336h", sign=1),
        FeatureMember(name="xs_idio_mom_336h", sign=1),
        FeatureMember(name="mom3_skew_168h", sign=1),
    ),
    min_rank_symbols=8,
    snapshot_hour_utc=22,
    release_hour_utc=23,
    entry_hour_utc=0,
    design_data_cutoff=pd.Timestamp("2026-07-01T00:00:00Z"),
    exposure_multiplier=GROWTH_EXPOSURE_MULTIPLIER,
    name_clip=STRATEGY_NAME_CLIP,
)

# Unleveraged clip book with exposure chosen by account policy; signals match FLOW_MOM_TOP20.
FLOW_MOM_TOP20_ACCOUNT_UNIT = StrategySpec(
    strategy_id="flow_mom_top20",
    breadth=20,
    members=(
        FeatureMember(name="flow_imb_168h", sign=1),
        FeatureMember(name="flow_imb_720h", sign=1),
        FeatureMember(name="xs_mom_336h", sign=1),
        FeatureMember(name="xs_idio_mom_336h", sign=1),
        FeatureMember(name="mom3_skew_168h", sign=1),
    ),
    min_rank_symbols=8,
    snapshot_hour_utc=22,
    release_hour_utc=23,
    entry_hour_utc=0,
    design_data_cutoff=pd.Timestamp("2026-07-01T00:00:00Z"),
    exposure_multiplier=1.0,
    name_clip=STRATEGY_NAME_CLIP,
)


LEGACY_STRATEGY_IDS: Mapping[str, str] = MappingProxyType(
    {
        "frozen_mhs_top20_v2": "flow_mom_top20",
        "frozen_mhs_top40_control_v2": "flow_mom_top40_control",
        "frozen_mhs_top20_growth_v2": "flow_mom_top20_growth",
    }
)
"""Pre-rename strategy ids. Read-only: persisted evidence keeps resolving through them."""


_AD_HOC_CONTROL_PATTERN = re.compile(r"flow_mom_b[1-9][0-9]*_control")
"""Writer-known ad-hoc control family minted by the backtest CLI for non-registered breadths."""


def resolve_strategy_id(raw: str) -> str:
    """Canonical strategy id for an id read from persisted evidence.

    Run manifests, backtest index rows and result envelopes written before the
    rename carry legacy ids; they denote the same strategy definition and must
    keep matching it. Unknown ids raise DataIntegrityError rather than silently
    matching nothing.
    """
    if not isinstance(raw, str) or not raw:
        raise DataIntegrityError(f"strategy id must be a non-empty string, got {raw!r}")
    if raw in LEGACY_STRATEGY_IDS:
        return LEGACY_STRATEGY_IDS[raw]
    if _AD_HOC_CONTROL_PATTERN.fullmatch(raw):
        return raw
    known = {
        spec.strategy_id
        for spec in (
            FLOW_MOM_TOP20,
            FLOW_MOM_TOP40_CONTROL,
            FLOW_MOM_TOP20_GROWTH,
            FLOW_MOM_TOP20_ACCOUNT_UNIT,
        )
    }
    if raw in known:
        return raw
    raise DataIntegrityError(f"unknown strategy id: {raw!r}")


def strategy_id_matches(stored: object, canonical: object) -> bool:
    """True when a persisted strategy id denotes the canonical strategy."""
    if not isinstance(stored, str):
        return False
    try:
        return resolve_strategy_id(stored) == canonical
    except DataIntegrityError:
        return False


@dataclass(frozen=True, slots=True)
class StrategyTargets:
    """Bind exact PIT target weights to their release and entry timestamps.

    ``target_weights.index`` is the UTC entry time, not the source-observation
    time.  Every row is actionable only after its matching release timestamp;
    this distinction is retained through the three-minute execution stream.
    """

    target_weights: pd.DataFrame
    signal_available_at: pd.DatetimeIndex
    strategy: StrategySpec

    def __post_init__(self) -> None:
        if len(self.target_weights) != len(self.signal_available_at):
            raise DataIntegrityError("target_weights and signal_available_at must have matching rows")

    @property
    def breadth(self) -> int:
        return self.strategy.breadth


@dataclass(frozen=True, slots=True)
class SourceReadiness:
    """Per decision day and hourly symbol, whether every bar in the trailing 720-bar history was published at or before the release instant."""

    decisions: pd.DatetimeIndex
    bar_positions: np.ndarray
    ready: np.ndarray


_READINESS_CHUNK_BARS = 4096


def _sliding_window_max(a: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(a)
    n = a.shape[0]
    if n == 0:
        return out
    prefix = np.empty_like(a)
    prefix[0] = a[0]
    for i in range(1, n):
        if i % window == 0:
            prefix[i] = a[i]
        else:
            np.maximum(a[i], prefix[i - 1], out=prefix[i])
    suffix = np.empty_like(a)
    suffix[-1] = a[-1]
    for i in range(n - 2, -1, -1):
        if (i + 1) % window == 0:
            suffix[i] = a[i]
        else:
            np.maximum(a[i], suffix[i + 1], out=suffix[i])
    out[: min(window - 1, n)] = prefix[: min(window - 1, n)]
    for i in range(window - 1, n):
        np.maximum(suffix[i - window + 1], prefix[i], out=out[i])
    return out


def compute_source_readiness(
    hourly_available_at: pd.DataFrame,
    decisions: pd.DatetimeIndex,
    strategy: StrategySpec,
) -> SourceReadiness:
    """Per decision day and hourly symbol, whether every bar in the trailing 720-bar history was published at or before the release instant."""
    hourly_index = hourly_available_at.index
    bars = decisions + pd.Timedelta(hours=int(strategy.snapshot_hour_utc))
    bar_positions = np.asarray(hourly_index.get_indexer(bars), dtype=np.int64)
    n_bars = len(hourly_index)
    n_sym = hourly_available_at.shape[1]
    n_dec = len(decisions)
    ready = np.zeros((n_dec, n_sym), dtype=bool)
    if n_dec == 0 or n_sym == 0 or n_bars == 0:
        return SourceReadiness(decisions=decisions, bar_positions=bar_positions, ready=ready)
    release_ns = (decisions + pd.Timedelta(hours=int(strategy.release_hour_utc))).to_numpy(
        dtype="datetime64[ns]"
    ).astype(np.int64)
    window = _HISTORY_BARS
    for chunk_start in range(0, n_bars, _READINESS_CHUNK_BARS):
        chunk_end = min(chunk_start + _READINESS_CHUNK_BARS, n_bars)
        slice_start = max(0, chunk_start - (window - 1))
        mask = (bar_positions >= chunk_start) & (bar_positions < chunk_end)
        if not bool(mask.any()):
            continue
        block = hourly_available_at.iloc[slice_start:chunk_end].to_numpy(dtype="datetime64[ns]").view(np.int64)
        win = _sliding_window_max(block, window)
        local = win[(chunk_start - slice_start):]
        hit = np.flatnonzero(mask)
        ready[hit] = local[bar_positions[hit] - chunk_start] <= release_ns[hit, None]
    return SourceReadiness(decisions=decisions, bar_positions=bar_positions, ready=ready)


def _snapshot_inputs_fingerprint(
    hourly_panels: Mapping[str, pd.DataFrame],
    hourly_available_at: pd.DataFrame,
) -> tuple[object, ...]:
    frames = {**hourly_panels, "available_at": hourly_available_at}
    return tuple(
        (
            key, frame.shape, tuple(frame.columns),
            hashlib.sha256(pd.util.hash_pandas_object(frame.index).to_numpy().tobytes()).digest(),
            hashlib.sha256(pd.util.hash_pandas_object(frame.iloc[-1:]).to_numpy().tobytes()).digest(),
        )
        for key, frame in sorted(frames.items())
    )


class MemberSnapshotCache:
    """Member snapshot frame (decisions x census), built once per feature name and reused across strategy variants."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, int | None, int | None], pd.DataFrame] = {}
        self._fingerprint: tuple[object, ...] | None = None
        self._readiness: dict[tuple[int, int], SourceReadiness] = {}

    def bind(self, fingerprint: tuple[object, ...]) -> None:
        """Reject reuse with a different source or decision grid."""
        if self._fingerprint is None:
            self._fingerprint = fingerprint
        elif fingerprint != self._fingerprint:
            raise DataIntegrityError("member snapshot cache built for different source inputs")

    def source_readiness(
        self, available: pd.DataFrame, decisions: pd.DatetimeIndex, strategy: StrategySpec,
    ) -> SourceReadiness:
        """Reuse publication readiness across members and variants on the same clock."""
        key = (strategy.snapshot_hour_utc, strategy.release_hour_utc)
        if key not in self._readiness:
            self._readiness[key] = compute_source_readiness(available, decisions, strategy)
        return self._readiness[key]

    def snapshots(
        self,
        member: FeatureMember,
        build: Callable[[], pd.DataFrame],
        *,
        snapshot_hour: int | None = None,
        release_hour: int | None = None,
        fingerprint: tuple[object, ...] | None = None,
    ) -> pd.DataFrame:
        key = (str(member.name), snapshot_hour, release_hour)
        if fingerprint is not None:
            self.bind(fingerprint)
        elif self._fingerprint is not None:
            raise DataIntegrityError("member snapshot cache built for different source inputs")
        if key in self._store:
            return self._store[key]
        frame = build()
        self._store[key] = frame
        return frame


def build_strategy_targets(
    hourly_panels: Mapping[str, pd.DataFrame],
    hourly_available_at: pd.DataFrame,
    daily_close: pd.DataFrame,
    daily_quote_volume: pd.DataFrame,
    census_symbols: tuple[str, ...],
    *,
    market_close: pd.DataFrame,
    strategy: StrategySpec = FLOW_MOM_TOP20,
    blocked_decisions: pd.DataFrame | None = None,
    readiness: SourceReadiness | None = None,
    snapshot_cache: MemberSnapshotCache | None = None,
) -> StrategyTargets:
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
        market_close: Full-census hourly close plane (columns in ``census_symbols`` order, same hourly
            index as ``hourly_panels``) defining the contemporaneous market cross-section for
            market-relative features; required so no feature can fall back to the hindsight-selected
            ``hourly_panels`` columns.
        strategy: Strategy target definition; the primary Top-20 is the default.
        blocked_decisions: Boolean decision-day frame withdrawing a symbol from both roster
            eligibility and target emission for exactly those days.
    Returns:
        Exact entry targets and matching signal-release timestamps. Target gross may exceed 1.0
        (levered) under growth specs.
    Raises:
        DataIntegrityError: Source planes, census, or roster evidence is inconsistent.
    """
    if not isinstance(strategy, StrategySpec):
        raise ValueError("strategy must be a StrategySpec")
    census = list(census_symbols)
    registry = {spec.name: spec for spec in FEATURE_REGISTRY}
    if any(m.name not in registry for m in strategy.members):
        raise ValueError("strategy references an unregistered feature")
    roster = build_pit_roster(
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
        isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC" for dtype in hourly_available_at.dtypes
    ):
        raise DataIntegrityError("hourly_available_at must contain timezone-aware timestamps")
    available_values = hourly_available_at.to_numpy(dtype="datetime64[ns]")
    grid_values = first.index.to_numpy(dtype="datetime64[ns]")[:, None]
    if bool((available_values < grid_values).any()):
        raise DataIntegrityError("hourly publication cannot precede the bar open")
    if not market_close.index.equals(first.index):
        raise DataIntegrityError("market_close must share the hourly close panel index exactly")
    if list(market_close.columns) != list(census_symbols):
        raise DataIntegrityError("market_close columns must match census_symbols order")
    panels[MARKET_CLOSE_PANEL] = market_close
    census = list(census_symbols)
    daily_idx = roster.index
    decisions = daily_idx[:-1]
    roster_dec = roster.loc[decisions]
    hourly_symbols = list(first.columns)
    column_of = {s: i for i, s in enumerate(hourly_symbols)}
    census_hourly = np.array([column_of.get(s, -1) for s in census], dtype=np.int64)
    fingerprint: tuple[object, ...] | None = None
    if snapshot_cache is not None:
        fingerprint = (_snapshot_inputs_fingerprint(panels, hourly_available_at), tuple(decisions), tuple(census))
        snapshot_cache.bind(fingerprint)
    if readiness is None:
        readiness = (
            compute_source_readiness(hourly_available_at, decisions, strategy)
            if snapshot_cache is None else snapshot_cache.source_readiness(hourly_available_at, decisions, strategy)
        )
    bar_positions = np.asarray(readiness.bar_positions, dtype=np.int64)
    ready = np.asarray(readiness.ready, dtype=bool)
    expected_positions = first.index.get_indexer(decisions + pd.Timedelta(hours=strategy.snapshot_hour_utc))
    if (
        not readiness.decisions.equals(decisions)
        or ready.shape != (len(decisions), len(hourly_symbols))
        or not np.array_equal(bar_positions, expected_positions)
    ):
        raise DataIntegrityError("source readiness does not match decision grid or hourly sources")

    def _snapshot_frame(member: FeatureMember) -> pd.DataFrame:
        feature = registry[member.name].builder(panels)
        aligned_values = feature.reindex(index=first.index, columns=census).to_numpy(dtype="float64")
        n_dec = len(decisions)
        n_cen = len(census)
        census_ready = np.zeros((n_dec, n_cen), dtype=bool)
        valid_cols = census_hourly >= 0
        if bool(valid_cols.any()) and n_dec:
            census_ready[:, valid_cols] = ready[:, census_hourly[valid_cols]]
        snaps_values = np.full((n_dec, n_cen), np.nan, dtype=np.float64)
        hit = np.flatnonzero(bar_positions >= 0)
        if hit.size:
            rows = aligned_values[bar_positions[hit]]
            snaps_values[hit] = np.where(census_ready[hit], rows, np.nan)
        return pd.DataFrame(snaps_values, index=decisions, columns=census, dtype="float64")

    def _snapshot_builder(member: FeatureMember) -> Callable[[], pd.DataFrame]:
        def _build() -> pd.DataFrame:
            return _snapshot_frame(member)

        return _build

    books = []
    for member in strategy.members:
        if snapshot_cache is not None:
            assert fingerprint is not None
            snaps = snapshot_cache.snapshots(
                member,
                _snapshot_builder(member),
                snapshot_hour=int(strategy.snapshot_hour_utc),
                release_hour=int(strategy.release_hour_utc),
                fingerprint=fingerprint,
            )
        else:
            snaps = _snapshot_frame(member)
        eligible = roster_dec & snaps.notna()
        books.append(rank_weight_book(snaps, eligible, int(member.sign), int(strategy.min_rank_symbols)))
    ensemble = books[0]
    for book in books[1:]:
        ensemble = ensemble.add(book)
    ensemble = ensemble / float(len(books))
    if strategy.name_clip is not None:
        ensemble = clip_names_preserving_gross(ensemble, strategy.name_clip)
    ensemble = ensemble * strategy.exposure_multiplier
    entries = pd.DatetimeIndex(
        [d + pd.Timedelta(days=1, hours=int(strategy.entry_hour_utc)) for d in decisions], tz="UTC"
    )
    target = pd.DataFrame(ensemble.to_numpy(dtype="float64"), index=entries, columns=census, dtype="float64")
    available = pd.DatetimeIndex([d + pd.Timedelta(hours=int(strategy.release_hour_utc)) for d in decisions], tz="UTC")
    return StrategyTargets(target_weights=target, signal_available_at=available, strategy=strategy)


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
