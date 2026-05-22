from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.domain.futures.optimization.opt_config import get_quarterly_window
from src.domain.futures.universe import (
    UniverseSnapshot,
    build_universe,
    load_or_build_universe_snapshot,
    load_universe_snapshot,
)
from src.domain.futures.universe.contracts import RejectCode

_logger = logging.getLogger("futures_universe_service")
_UNIVERSE_AUDIT_DIR = Path("logs/futures/universe")


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


@dataclass(slots=True, frozen=True)
class UniverseTimelineResult:
    """Quarterly universe timeline and selected snapshot bundle."""

    symbols: tuple[str, ...]
    timeline: UniverseMembershipTimeline
    snapshots: tuple[UniverseSnapshot, ...]
    snapshot: UniverseSnapshot
    report: pd.DataFrame


def _quarter_start(dt: date) -> date:
    q_month = ((dt.month - 1) // 3) * 3 + 1
    return date(dt.year, q_month, 1)


def _discover_symbols_via_universe(
    *,
    tf: str,
    reference_date: str | None,
    force_rebuild: bool = False,
    previous_selection: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], UniverseSnapshot, pd.DataFrame]:
    _fetch_start, _start, as_of_date, _end = get_quarterly_window(reference_date)
    if force_rebuild:
        snapshot, selected_frame, report = build_universe(
            as_of=as_of_date,
            tf=tf,
            previous_selection=previous_selection,
        )
    else:
        snapshot, selected_frame, report = load_or_build_universe_snapshot(
            as_of=as_of_date,
            tf=tf,
            previous_selection=previous_selection,
        )

    selected_symbols: list[str] = []
    if (
        selected_frame is not None
        and not selected_frame.empty
        and "symbol" in selected_frame.columns
    ):
        selected_symbols = [
            str(symbol).strip()
            for symbol in selected_frame["symbol"].astype(str).tolist()
            if str(symbol).strip()
        ]
    if not selected_symbols:
        selected_symbols = [
            str(meta.symbol).strip() for meta in snapshot.selected if str(meta.symbol).strip()
        ]
    return tuple(dict.fromkeys(selected_symbols)), snapshot, report


def discover_universe_timeline(
    *,
    tf: str,
    is_start: date,
    oos_start: date,
    end_date: date,
    force_rebuild: bool = False,
) -> UniverseTimelineResult:
    current_dt = _quarter_start(is_start)
    all_symbols: set[str] = set()
    oos_snapshot: UniverseSnapshot | None = None
    oos_report = pd.DataFrame()
    previous_selection: tuple[str, ...] | None = None
    timeline_by_quarter: dict[date, frozenset[str]] = {}
    snapshots_by_quarter: list[tuple[date, UniverseSnapshot, pd.DataFrame]] = []

    while current_dt <= end_date:
        ref_dt = current_dt + relativedelta(months=3)
        symbols, snapshot, report = _discover_symbols_via_universe(
            tf=tf,
            reference_date=ref_dt.isoformat(),
            force_rebuild=force_rebuild,
            previous_selection=previous_selection,
        )
        current_set = frozenset(symbols)
        timeline_by_quarter[current_dt] = current_set
        all_symbols.update(current_set)
        prev_set = set(previous_selection or ())
        new_symbols = sorted(current_set - prev_set)
        dropped_symbols = sorted(prev_set - set(current_set))
        retained_symbols = sorted(prev_set & set(current_set))
        _logger.info(
            "[UNIVERSE-TIMELINE] quarter=%s selected=%d new=%d dropped=%d retained=%d",
            current_dt.isoformat(),
            len(current_set),
            len(new_symbols),
            len(dropped_symbols),
            len(retained_symbols),
        )
        previous_selection = tuple(sorted(current_set))
        if current_dt == oos_start:
            oos_snapshot = snapshot
            oos_report = report
        snapshots_by_quarter.append((current_dt, snapshot, report))
        current_dt += relativedelta(months=3)

    if oos_snapshot is None:
        raise ValueError("Universe timeline did not include oos_start snapshot.")
    windows: list[UniverseMembershipWindow] = []
    previous_symbols: frozenset[str] = frozenset()
    for idx, (quarter_start, snapshot, _report) in enumerate(snapshots_by_quarter):
        symbols = timeline_by_quarter[quarter_start]
        next_start = (
            pd.Timestamp(snapshots_by_quarter[idx + 1][0])
            if idx + 1 < len(snapshots_by_quarter)
            else None
        )
        windows.append(
            UniverseMembershipWindow(
                effective_from=pd.Timestamp(quarter_start),
                effective_to=next_start,
                snapshot_as_of=snapshot.as_of,
                active_symbols=tuple(sorted(symbols)),
                entry_symbols=tuple(sorted(symbols - previous_symbols)),
                exit_symbols=tuple(sorted(previous_symbols - symbols)),
            )
        )
        previous_symbols = symbols

    _write_universe_audit_parquet(
        snapshots_by_quarter=snapshots_by_quarter,
        windows=tuple(windows),
    )
    return UniverseTimelineResult(
        symbols=tuple(sorted(all_symbols)),
        timeline=UniverseMembershipTimeline(tf=tf, windows=tuple(windows)),
        snapshots=tuple(snapshot for _, snapshot, _ in snapshots_by_quarter),
        snapshot=oos_snapshot,
        report=oos_report,
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
    if not snapshot.selected:
        _logger.error("[UNIVERSE-QUALITY] no_symbols_selected pass=false")
        return False

    costs = [m.execution_cost_bps for m in snapshot.selected]
    advs = [m.adv_usdt for m in snapshot.selected]
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
            curr_symbols = {m.symbol for m in snapshot.selected}
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
    _logger.info(
        "[UNIVERSE-QUALITY] median_cost_bps=%.3f median_adv=%.3f dropout_rate=%.4f pass=%s",
        median_cost,
        median_adv,
        dropout_rate,
        str(bool(quality_pass)).lower(),
    )
    return quality_pass
