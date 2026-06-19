from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.core.settings import LOG_DIR
from src.core.utils.utils import PERF
from src.domain.futures.optimization.opt_config import get_quarterly_window
from src.domain.futures.universe import (
    UniverseSnapshot,
    UniverseStateCube,
    build_universe,
    load_or_build_universe_snapshot,
    load_universe_snapshot,
)
from src.domain.futures.universe.config import UniverseConfig
from src.domain.futures.universe.models import RejectCode, SymbolMeta

_logger = logging.getLogger("futures_universe_service")
_UNIVERSE_AUDIT_DIR = LOG_DIR / "futures/universe"


@dataclass(slots=True, frozen=True)
class UniverseMembershipWindow:
    """Quarterly membership window used by optimizer/backtest gating."""

    effective_from: pd.Timestamp
    effective_to: pd.Timestamp | None
    snapshot_as_of: str
    active_symbols: tuple[str, ...]
    entry_symbols: tuple[str, ...]
    exit_symbols: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class UniverseMembershipTimeline:
    """Universe membership timeline across quarterly windows."""

    tf: str
    windows: tuple[UniverseMembershipWindow, ...]

    def as_mapping(self) -> dict[date, frozenset[str]]:
        """Return quarter_start → active_symbols mapping for membership mask injection.

        Returns:
            Dict mapping each window's effective_from date to its active symbol frozenset.

        """
        result: dict[date, frozenset[str]] = {}
        for window in self.windows:
            q_date: date = window.effective_from.date()
            result[q_date] = frozenset(window.active_symbols)
        return result


@dataclass(slots=True, frozen=True)
class UniverseTimelineResult:
    """Quarterly universe timeline and selected snapshot bundle."""

    symbols: tuple[str, ...]
    timeline: UniverseMembershipTimeline
    state_cube: UniverseStateCube
    snapshots: tuple[UniverseSnapshot, ...]
    snapshot: UniverseSnapshot
    report: pd.DataFrame
    audit: pd.DataFrame
    # 신규 필드 (기존 report 필드 뒤에 추가)
    inference_symbols: tuple[str, ...] = field(default_factory=tuple)
    inference_timeline: object | None = None  # UniverseMembershipTimeline | None (순환 임포트 회피)
    inference_panel_quarter_membership: dict[date, frozenset[str]] = field(default_factory=dict)

    @property
    def instrument_ids(self) -> tuple[str, ...]:
        """Alias for the active instrument axis."""
        return self.symbols


def _quarter_start(dt: date) -> date:
    q_month = ((dt.month - 1) // 3) * 3 + 1
    return date(dt.year, q_month, 1)


def _discover_symbols_via_universe(
    *,
    tf: str,
    reference_date: str | None,
    force_rebuild: bool = False,
    previous_selection: tuple[str, ...] | None = None,
    cfg: UniverseConfig | None = None,
) -> tuple[tuple[str, ...], UniverseSnapshot, pd.DataFrame]:
    _fetch_start, _start, as_of_date, _end = get_quarterly_window(reference_date)
    if force_rebuild:
        snapshot, selected_frame, report = build_universe(
            as_of=as_of_date,
            tf=tf,
            cfg=cfg,
            previous_selection=previous_selection,
        )
    else:
        snapshot, selected_frame, report = load_or_build_universe_snapshot(
            as_of=as_of_date,
            tf=tf,
            cfg=cfg,
            previous_selection=previous_selection,
        )

    discovered_symbols: list[str] = []
    if (
        selected_frame is not None
        and not selected_frame.empty
        and "symbol" in selected_frame.columns
    ):
        discovered_symbols = [
            str(symbol).strip()
            for symbol in selected_frame["symbol"].astype(str).tolist()
            if str(symbol).strip()
        ]
    if not discovered_symbols:
        discovered_symbols = [
            str(meta.symbol).strip() for meta in snapshot.selected if str(meta.symbol).strip()
        ]
    return tuple(dict.fromkeys(discovered_symbols)), snapshot, report


def _snapshot_selected_meta_map(snapshot: UniverseSnapshot) -> dict[str, SymbolMeta]:
    return {
        str(meta.symbol).strip(): meta
        for meta in snapshot.selected
        if str(meta.symbol).strip()
    }


def _snapshot_quality_symbols(snapshot: UniverseSnapshot) -> tuple[str, ...]:
    selected = tuple(
        str(meta.symbol).strip()
        for meta in snapshot.selected
        if str(meta.symbol).strip()
    )
    return tuple(dict.fromkeys(selected))


def _discover_universe_timeline_pit(
    *,
    tf: str,
    is_start: date,
    oos_start: date,
    end_date: date,
    force_rebuild: bool,
    cfg: UniverseConfig,
) -> UniverseTimelineResult:
    """PIT path: quarterly loop with pit build_universe, no panel fallback, no k_in=20 limit.

    Time Complexity: O(Q * T * N) where Q=quarters, T=bars/quarter, N=instruments.
    Space Complexity: O(T * N) for merged state arrays.

    Args:
        tf: Bar timeframe string (e.g. "4h").
        is_start: IS window start date.
        oos_start: OOS window start date; the quarterly snapshot at this quarter is used as
            the canonical ``oos_snapshot``.
        end_date: Timeline end date (inclusive).
        force_rebuild: When True, skip cache and rebuild each quarterly snapshot.
        cfg: UniverseConfig with ``universe_engine == "pit"``.

    Returns:
        UniverseTimelineResult with merged UniverseStateCube and per-quarter windows.

    Raises:
        ValueError: When the quarterly loop does not include a snapshot for ``oos_start``.

    """
    current_dt = _quarter_start(is_start)
    all_symbols: set[str] = set()
    oos_snapshot: UniverseSnapshot | None = None
    oos_report: pd.DataFrame = pd.DataFrame()
    previous_selection: tuple[str, ...] | None = None
    snapshots_by_quarter: list[tuple[date, UniverseSnapshot, pd.DataFrame]] = []
    # (quarter_date, pit_cube) — collected to forward-fill into merged cube
    pit_cubes_by_quarter: list[tuple[date, UniverseStateCube]] = []

    while current_dt <= end_date:
        ref_dt = current_dt + relativedelta(months=3)
        t_quarter = time.perf_counter()
        symbols, snapshot, report = _discover_symbols_via_universe(
            tf=tf,
            reference_date=ref_dt.isoformat(),
            force_rebuild=force_rebuild,
            previous_selection=previous_selection,
            cfg=cfg,
        )
        _logger.log(
            PERF,
            "[perf-universe-pit] quarter=%s symbols=%d took %.4fs",
            current_dt.isoformat(),
            len(symbols),
            time.perf_counter() - t_quarter,
        )
        # PIT mode: use ALL eligible symbols — no panel fallback, no k_in cap
        current_set: frozenset[str] = frozenset(
            str(s).strip() for s in symbols if str(s).strip()
        )
        all_symbols.update(current_set)
        previous_selection = tuple(sorted(current_set))

        if current_dt == oos_start:
            oos_snapshot = snapshot
            oos_report = report

        snapshots_by_quarter.append((current_dt, snapshot, report))

        pit_cube: UniverseStateCube | None = getattr(snapshot, "pit_state_cube", None)
        if pit_cube is not None:
            pit_cubes_by_quarter.append((current_dt, pit_cube))
        else:
            _logger.warning(
                "_discover_universe_timeline_pit: quarter=%s returned pit_state_cube=None; "
                "eligible will be all-False for this quarter",
                current_dt.isoformat(),
            )

        current_dt += relativedelta(months=3)

    if oos_snapshot is None:
        raise ValueError(
            f"Universe PIT timeline did not include oos_start={oos_start.isoformat()} snapshot."
        )

    # ── Build merged state cube ───────────────────────────────────────────────
    # Shape: eligible[T, N], instrument_ids sorted for deterministic axis
    instrument_ids: tuple[str, ...] = tuple(sorted(all_symbols))
    n_inst = len(instrument_ids)
    inst_idx: dict[str, int] = {sym: i for i, sym in enumerate(instrument_ids)}

    calendar: pd.DatetimeIndex = pd.date_range(
        start=pd.Timestamp(is_start, tz="UTC"),
        end=pd.Timestamp(end_date, tz="UTC"),
        freq=tf,
    )
    n_bars = len(calendar)

    # eligible[T,N], bool_ — default False (fail-closed)
    eligible: np.ndarray = np.zeros((n_bars, n_inst), dtype=np.bool_)
    entry_block: np.ndarray = np.zeros((n_bars, n_inst), dtype=np.bool_)
    exit_required: np.ndarray = np.zeros((n_bars, n_inst), dtype=np.bool_)
    capacity_usdt: np.ndarray = np.zeros((n_bars, n_inst), dtype=np.float64)
    risk_scale: np.ndarray = np.ones((n_bars, n_inst), dtype=np.float64)
    cost_bps: np.ndarray = np.zeros((n_bars, n_inst), dtype=np.float64)

    # Forward-fill quarterly pit cubes into merged arrays
    # For each quarter's cube, determine bar range [start_pos, end_pos)
    # and copy last-bar state (src_t = -1) into that range — O(Q * N)
    n_pit_cubes = len(pit_cubes_by_quarter)
    for pit_idx, (q_date, pit_cube) in enumerate(pit_cubes_by_quarter):
        q_ts = pd.Timestamp(q_date, tz="UTC")
        next_q_ts: pd.Timestamp | None = (
            pd.Timestamp(pit_cubes_by_quarter[pit_idx + 1][0], tz="UTC")
            if pit_idx + 1 < n_pit_cubes
            else None
        )
        start_pos = int(calendar.searchsorted(q_ts, side="left"))
        end_pos = (
            int(calendar.searchsorted(next_q_ts, side="left"))
            if next_q_ts is not None
            else n_bars
        )
        if end_pos <= start_pos:
            continue

        # pit_cube.instrument_ids uses "binance_usdt_perpetual:SYMBOL" format
        # Extract symbol portion for mapping to merged instrument_ids axis
        for pit_n, pit_iid in enumerate(pit_cube.instrument_ids):
            sym = pit_iid.split(":")[-1] if ":" in pit_iid else pit_iid
            merged_n = inst_idx.get(sym)
            if merged_n is None:
                continue
            # Take the last bar of the pit cube as the current-quarter state
            src_t = min(pit_cube.eligible.shape[0] - 1, pit_cube.eligible.shape[0] - 1)
            eligible[start_pos:end_pos, merged_n] = pit_cube.eligible[src_t, pit_n]
            entry_block[start_pos:end_pos, merged_n] = pit_cube.entry_block[src_t, pit_n]
            capacity_usdt[start_pos:end_pos, merged_n] = pit_cube.capacity_usdt[src_t, pit_n]
            risk_scale[start_pos:end_pos, merged_n] = pit_cube.risk_scale[src_t, pit_n]
            cost_bps[start_pos:end_pos, merged_n] = pit_cube.cost_bps[src_t, pit_n]

    # Detect eligible→ineligible transitions for exit_required — O(T * N)
    if n_bars > 1:
        transition_mask: np.ndarray = eligible[:-1] & ~eligible[1:]
        exit_required[1:] = transition_mask

    state_cube = UniverseStateCube(
        calendar=calendar,
        instrument_ids=instrument_ids,
        eligible=eligible,
        entry_block=entry_block,
        exit_required=exit_required,
        capacity_usdt=capacity_usdt,
        risk_scale=risk_scale,
        cost_bps=cost_bps,
    )

    # ── Build membership windows ──────────────────────────────────────────────
    windows: list[UniverseMembershipWindow] = []
    previous_symbols: frozenset[str] = frozenset()
    audit_rows: list[dict[str, object]] = []

    n_snapshots = len(snapshots_by_quarter)
    for idx, (quarter_start, snapshot, _) in enumerate(snapshots_by_quarter):
        next_q_ts_w: pd.Timestamp | None = (
            pd.Timestamp(snapshots_by_quarter[idx + 1][0], tz="UTC")
            if idx + 1 < n_snapshots
            else None
        )
        q_ts_w = pd.Timestamp(quarter_start, tz="UTC")
        # Derive active symbols from snapshot.selected (SymbolMeta objects)
        q_sym: frozenset[str] = frozenset(
            str(meta.symbol).strip()
            for meta in snapshot.selected
            if str(meta.symbol).strip()
        )
        windows.append(
            UniverseMembershipWindow(
                effective_from=q_ts_w,
                effective_to=next_q_ts_w,
                snapshot_as_of=snapshot.as_of,
                active_symbols=tuple(sorted(q_sym)),
                entry_symbols=tuple(sorted(q_sym - previous_symbols)),
                exit_symbols=tuple(sorted(previous_symbols - q_sym)),
            )
        )
        start_pos_a = int(calendar.searchsorted(q_ts_w, side="left"))
        audit_rows.extend(
            {
                "quarter_start": quarter_start.isoformat(),
                "symbol": sym,
                "eligible": sym in q_sym,
                "snapshot_as_of": snapshot.as_of,
                "entry": sym in q_sym - previous_symbols,
                "exit": sym in previous_symbols - q_sym,
                "cost_bps": (
                    float(cost_bps[start_pos_a, inst_idx[sym]])
                    if start_pos_a < n_bars and sym in inst_idx
                    else 0.0
                ),
                "capacity_usdt": (
                    float(capacity_usdt[start_pos_a, inst_idx[sym]])
                    if start_pos_a < n_bars and sym in inst_idx
                    else 0.0
                ),
            }
            for sym in instrument_ids
        )
        previous_symbols = q_sym

    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "quarter_start",
            "symbol",
            "eligible",
            "snapshot_as_of",
            "entry",
            "exit",
            "cost_bps",
            "capacity_usdt",
        ],
    ) if audit_rows else pd.DataFrame(
        columns=[
            "quarter_start",
            "symbol",
            "eligible",
            "snapshot_as_of",
            "entry",
            "exit",
            "cost_bps",
            "capacity_usdt",
        ],
    )

    _write_universe_audit_parquet(
        snapshots_by_quarter=snapshots_by_quarter,
        windows=tuple(windows),
    )

    return UniverseTimelineResult(
        symbols=instrument_ids,
        timeline=UniverseMembershipTimeline(tf=tf, windows=tuple(windows)),
        state_cube=state_cube,
        snapshots=tuple(snap for _, snap, _ in snapshots_by_quarter),
        snapshot=oos_snapshot,
        report=oos_report,
        audit=audit,
        inference_symbols=instrument_ids,
        inference_timeline=UniverseMembershipTimeline(tf=tf, windows=tuple(windows)),
        inference_panel_quarter_membership={
            quarter: frozenset(sorted(all_symbols))
            for quarter, _, _ in snapshots_by_quarter
        },
    )


def discover_universe_timeline(
    *,
    tf: str,
    is_start: date,
    oos_start: date,
    end_date: date,
    force_rebuild: bool = False,
    l2_start: date | None = None,
    min_history_bars: int = 0,
    cfg: UniverseConfig | None = None,
) -> UniverseTimelineResult:
    if l2_start is not None and l2_start < oos_start:
        _logger.warning(
            "discover_universe_timeline: l2_start(%s) < oos_start(%s), "
            "current universe timeline is still 2-way and does not treat l2_start as a separate boundary",
            l2_start.isoformat(),
            oos_start.isoformat(),
        )
        l2_start = None

    # ── PIT-only path (Stage6 legacy path removed) ──
    if cfg is None:
        raise ValueError("universe_engine=pit required; stage6 path removed")
    return _discover_universe_timeline_pit(
        tf=tf,
        is_start=is_start,
        oos_start=oos_start,
        end_date=end_date,
        force_rebuild=force_rebuild,
        cfg=cfg,
    )


def _write_universe_audit_parquet(
    *,
    snapshots_by_quarter: list[tuple[date, UniverseSnapshot, pd.DataFrame]],
    windows: tuple[UniverseMembershipWindow, ...],
) -> None:
    _UNIVERSE_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    timeline_rows: list[dict[str, object]] = [
        {
            "quarter_start": window.effective_from.date().isoformat(),
            "effective_from": window.effective_from,
            "effective_to": window.effective_to,
            "snapshot_as_of": window.snapshot_as_of,
            "active_symbols": ",".join(window.active_symbols),
            "entry_symbols": ",".join(window.entry_symbols),
            "exit_symbols": ",".join(window.exit_symbols),
            "n_active": len(window.active_symbols),
        }
        for window in windows
    ]
    pd.DataFrame(timeline_rows).to_parquet(
        _UNIVERSE_AUDIT_DIR / "universe_timeline.parquet",
        index=False,
    )

    membership_rows: list[dict[str, object]] = []
    previous_members: set[str] = set()
    for quarter_start, snapshot, _report in snapshots_by_quarter:
        selected_map = {meta.symbol: meta for meta in snapshot.selected}
        selected_symbols = set(selected_map.keys())
        all_snapshot_symbols = set(selected_symbols) | set(snapshot.rejected.keys())
        for symbol in sorted(all_snapshot_symbols):
            meta = selected_map.get(symbol)
            rejected = snapshot.rejected.get(symbol)
            stage_reason = None
            if rejected is not None:
                stage_reason = (
                    rejected.stage1_reason
                    or rejected.stage2_reason
                    or rejected.stage3_reason
                    or rejected.stage4_reason
                    or rejected.stage5_reason
                    or rejected.stage6_reason
                )
            stage6_metrics = rejected.stage6_metrics if rejected is not None else {}
            dwell_days = stage6_metrics.get("dwell_days")
            if dwell_days is None:
                dwell_days = stage6_metrics.get("membership_days")
            membership_rows.append(
                {
                    "quarter_start": quarter_start.isoformat(),
                    "symbol": symbol,
                    "is_selected": symbol in selected_symbols,
                    "selection_reason": (
                        "SELECTED" if meta is not None else str(stage_reason or "REJECTED")
                    ),
                    "rank": (
                        meta.tradeable_rank
                        if meta is not None
                        else (rejected.final_rank if rejected is not None else None)
                    ),
                    "dwell_days": dwell_days,
                    "was_prev_member": symbol in previous_members,
                }
            )
        previous_members = selected_symbols

    pd.DataFrame(membership_rows).to_parquet(
        _UNIVERSE_AUDIT_DIR / "membership_state.parquet",
        index=False,
    )


def validate_universe_quality(
    *,
    snapshot: UniverseSnapshot,
    report: pd.DataFrame,
    reference_date: str | None,
    tf: str,
) -> bool:
    _ = report
    quality_symbols = _snapshot_quality_symbols(snapshot)
    if not quality_symbols:
        _logger.error("[UNIVERSE-QUALITY] no_symbols_selected pass=false")
        return False

    selected_meta_map = _snapshot_selected_meta_map(snapshot)
    quality_meta = [
        selected_meta_map[symbol]
        for symbol in quality_symbols
        if symbol in selected_meta_map
    ]
    if not quality_meta:
        _logger.error("[UNIVERSE-QUALITY] no_quality_metadata pass=false")
        return False

    costs = [float(meta.execution_cost_bps) for meta in quality_meta]
    advs = [float(meta.adv_usdt) for meta in quality_meta]
    median_cost = float(np.median(costs))
    median_adv = float(np.median(advs))

    ref_dt = datetime.now().date()
    if reference_date:
        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d").date()
    prev_quarter_dt = ref_dt - relativedelta(months=3)
    _, _, prev_as_of, _ = get_quarterly_window(prev_quarter_dt.isoformat())
    previous_snapshot_frame = load_universe_snapshot(as_of=prev_as_of, tf=tf)

    dropout_pass = True
    dropout_rate = 0.0
    if previous_snapshot_frame is not None and not previous_snapshot_frame.empty:
        prev_symbols = set(previous_snapshot_frame["symbol"].astype(str).tolist())
        prev_universe_size = len(prev_symbols)
        if prev_universe_size >= 10:
            curr_symbols = set(quality_symbols)
            dropped_symbols = prev_symbols - curr_symbols
            forced_dropouts = 0
            for sym in dropped_symbols:
                filt_report = snapshot.rejected.get(sym)
                if filt_report is None:
                    continue
                is_forced = any(
                    [
                        filt_report.stage1_reason,
                        filt_report.stage2_reason,
                        filt_report.stage3_reason,
                        filt_report.stage4_reason,
                        filt_report.stage5_reason,
                    ]
                )
                if not is_forced and filt_report.stage6_reason != RejectCode.RANKED_OUT:
                    is_forced = True
                if is_forced:
                    forced_dropouts += 1
            dropout_rate = forced_dropouts / prev_universe_size
            dropout_pass = dropout_rate <= 0.10

    quality_pass = median_cost <= 50.0 and median_adv >= 25_000_000.0 and dropout_pass
    _logger.debug(
        ".. UNIVERSE_Q: med_cost=%.1fbps med_adv=%.1fm drop_rate=%.2f pass=%s",
        median_cost,
        median_adv / 1_000_000,
        dropout_rate,
        str(bool(quality_pass)).lower(),
    )
    return quality_pass
