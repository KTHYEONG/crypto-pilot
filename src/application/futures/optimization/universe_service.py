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
    build_universe,
    load_or_build_universe_snapshot,
    load_universe_snapshot,
)
from src.domain.futures.universe.models import RejectCode

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
    snapshots: tuple[UniverseSnapshot, ...]
    snapshot: UniverseSnapshot
    report: pd.DataFrame
    # 신규 필드 (기존 report 필드 뒤에 추가)
    inference_symbols: tuple[str, ...] = field(default_factory=tuple)
    inference_timeline: object | None = None  # UniverseMembershipTimeline | None (순환 임포트 회피)
    inference_panel_quarter_membership: dict[date, frozenset[str]] = field(default_factory=dict)


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
    l2_start: date | None = None,          # 신규: 3-way window L2 시작일 (None=기존 2-way 동작)
    min_history_bars: int = 0,             # 신규: 심볼 최소 bar 수 게이트 (0=비활성)
) -> UniverseTimelineResult:
    # 3-way window: l2_start이 제공되면 oos_start 이후를 L2/holdout으로 분리 (확장 예약)
    # 현재는 메타데이터 전달만 수행, timeline 분기는 추후 구현
    if l2_start is not None and l2_start < oos_start:
        _logger.warning(
            "discover_universe_timeline: l2_start(%s) < oos_start(%s), "
            "current universe timeline is still 2-way and does not treat l2_start as a separate boundary",
            l2_start.isoformat(),
            oos_start.isoformat(),
        )
        l2_start = None

    # min_history_bars: is_start 이전 충분한 데이터를 가진 심볼만 포함 (0=비활성)
    # 실제 필터링은 _discover_symbols_via_universe 반환 후 적용
    _min_history_bars = max(0, min_history_bars)

    from dataclasses import replace as _dataclass_replace

    current_dt = _quarter_start(is_start)
    all_symbols: set[str] = set()
    oos_snapshot: UniverseSnapshot | None = None
    oos_report = pd.DataFrame()
    previous_selection: tuple[str, ...] | None = None
    timeline_by_quarter: dict[date, frozenset[str]] = {}
    snapshots_by_quarter: list[tuple[date, UniverseSnapshot, pd.DataFrame]] = []
    inference_panel_quarter_membership: dict[date, frozenset[str]] = {}
    inference_symbols_set: set[str] = set()

    while current_dt <= end_date:
        ref_dt = current_dt + relativedelta(months=3)
        t_quarter = time.perf_counter()
        symbols, snapshot, report = _discover_symbols_via_universe(
            tf=tf,
            reference_date=ref_dt.isoformat(),
            force_rebuild=force_rebuild,
            previous_selection=previous_selection,
        )
        _logger.log(
            PERF,
            "[perf-universe] _discover_symbols_via_universe for quarter=%s took %.4fs",
            current_dt.isoformat(),
            time.perf_counter() - t_quarter,
        )
        current_set = frozenset(symbols)
        timeline_by_quarter[current_dt] = current_set
        all_symbols.update(current_set)
        ml_panel = current_set
        inference_panel_quarter_membership[current_dt] = ml_panel
        inference_symbols_set.update(ml_panel)
        prev_set = set(previous_selection or ())
        new_symbols = sorted(current_set - prev_set)
        dropped_symbols = sorted(prev_set - set(current_set))
        retained_symbols = sorted(prev_set & set(current_set))
        _logger.debug(
            ".. UNIVERSE_T: quarter=%s sel=%d new=%d drop=%d ret=%d ml=%d",
            current_dt.isoformat(),
            len(current_set),
            len(new_symbols),
            len(dropped_symbols),
            len(retained_symbols),
            len(ml_panel),
        )
        previous_selection = tuple(sorted(current_set))
        if current_dt == oos_start:
            oos_snapshot = snapshot
            oos_report = report
        snapshots_by_quarter.append((current_dt, snapshot, report))
        current_dt += relativedelta(months=3)

    # min_history_bars 게이트: 충분한 히스토리가 없는 심볼 제거
    if _min_history_bars > 0 and oos_snapshot is not None:
        _logger.info(
            "discover_universe_timeline: applying min_history_bars=%d gate, "
            "symbols_before=%d",
            _min_history_bars,
            len(all_symbols),
        )
        # 근사: 분기 윈도우수 * bars_per_quarter 로 추정
        # 정확한 bar 수 검증은 data_loader 통합 시 구현 예정
        # 현재는 파라미터 수용 및 로깅만 수행 (호환성 유지)
        _logger.debug(
            "discover_universe_timeline: min_history_bars gate (stub) — "
            "full implementation deferred to data_loader integration"
        )

    if oos_snapshot is None:
        raise ValueError("Universe timeline did not include oos_start snapshot.")
    windows: list[UniverseMembershipWindow] = []
    previous_symbols: frozenset[str] = frozenset()
    for idx, (quarter_start, snapshot, _report) in enumerate(snapshots_by_quarter):
        quarter_symbols = timeline_by_quarter[quarter_start]
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
                active_symbols=tuple(sorted(quarter_symbols)),
                entry_symbols=tuple(sorted(quarter_symbols - previous_symbols)),
                exit_symbols=tuple(sorted(previous_symbols - quarter_symbols)),
            )
        )
        previous_symbols = quarter_symbols

    # Candidate ML inference timeline uses Stage6 membership.
    inference_windows: list[UniverseMembershipWindow] = []
    prev_ml_panel: frozenset[str] = frozenset()
    for idx, (q_start, snap_q, _) in enumerate(snapshots_by_quarter):
        members = inference_panel_quarter_membership.get(q_start, frozenset())
        next_q = (
            pd.Timestamp(snapshots_by_quarter[idx + 1][0])
            if idx + 1 < len(snapshots_by_quarter)
            else None
        )
        inference_windows.append(
            UniverseMembershipWindow(
                effective_from=pd.Timestamp(q_start),
                effective_to=next_q,
                snapshot_as_of=snap_q.as_of,
                active_symbols=tuple(sorted(members)),
                entry_symbols=tuple(sorted(members - prev_ml_panel)),
                exit_symbols=tuple(sorted(prev_ml_panel - members)),
            )
        )
        prev_ml_panel = members

    inference_timeline_obj = UniverseMembershipTimeline(
        tf=tf, windows=tuple(inference_windows)
    )

    # OOS snapshot에 C1 union 주입 (frozen → replace)
    oos_snapshot = _dataclass_replace(
        oos_snapshot,
        inference_panel=tuple(sorted(inference_symbols_set)),
        historical_trading_panel=tuple(sorted(all_symbols)),
        inference_panel_quarter_membership={
            k: tuple(sorted(v)) for k, v in inference_panel_quarter_membership.items()
        },
    )

    _write_universe_audit_parquet(
        snapshots_by_quarter=snapshots_by_quarter,
        windows=tuple(windows),
    )
    return UniverseTimelineResult(
        symbols=tuple(sorted(all_symbols)),
        timeline=UniverseMembershipTimeline(tf=tf, windows=tuple(windows)),
        snapshots=tuple(snap for _, snap, _ in snapshots_by_quarter),
        snapshot=oos_snapshot,
        report=oos_report,
        inference_symbols=tuple(sorted(inference_symbols_set)),
        inference_timeline=inference_timeline_obj,
        inference_panel_quarter_membership={
            k: frozenset(v) for k, v in inference_panel_quarter_membership.items()
        },
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
    _logger.debug(
        ".. UNIVERSE_Q: med_cost=%.1fbps med_adv=%.1fm drop_rate=%.2f pass=%s",
        median_cost,
        median_adv / 1_000_000,
        dropout_rate,
        str(bool(quality_pass)).lower(),
    )
    return quality_pass
