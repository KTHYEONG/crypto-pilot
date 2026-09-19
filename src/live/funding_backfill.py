"""Operator-invoked one-shot PAPER funding backfill.

Restores funding fees the PAPER ledger never accrued by replaying the
position path reconstructed from the fills sidecar against the on-disk
funding files and completed 1h trade closes. Dry-run by default; persists
only with ``--apply``.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.audit import AUDIT_LOG_ROOT, AuditLog, default_audit_log_path
from src.live.fills import default_fills_dir, load_fills
from src.live.ledger import (
    POSITION_HISTORY_MAX,
    LedgerState,
    PositionSnapshot,
    default_ledger_path,
    load_ledger,
    position_at,
    save_ledger,
)
from src.live.runner import _load_paper_funding, _load_paper_trade_closes
from src.live.scheduler import _resolve_heartbeat_path
from src.live.settings import LiveSettings
from src.market_data.services.futures_collection import FUNDING_GAP_THRESHOLD_MS

logger = logging.getLogger("LiveFundingBackfill")

BACKFILL_AUDIT_NAME: str = "paper_funding_backfill"
# 운영 의미는 src/application/ops/daemon_idle_gate.py 의 BUSY_STAGES/DEFAULT_STALE_AFTER_S 와 동일(도구 스크립트 import 금지라 값 복제).
BACKFILL_BUSY_STAGES: frozenset[str] = frozenset({"refresh", "signal", "execute"})
BACKFILL_BUSY_STALE_S: float = 2700.0


@dataclass(frozen=True, slots=True)
class FundingBackfillResult:
    start: pd.Timestamp
    end: pd.Timestamp
    cash_delta: Decimal
    by_symbol: dict[str, Decimal]
    epochs: int


@dataclass(frozen=True, slots=True)
class BackfillPlan:
    start: pd.Timestamp
    end: pd.Timestamp
    cash_delta: Decimal
    by_symbol: dict[str, Decimal]
    epochs: int
    seeds_accrual_start: bool


def compute_funding_backfill(
    history: Sequence[PositionSnapshot],
    funding_by_symbol: Mapping[str, pd.Series],
    trade_close_by_symbol: Mapping[str, pd.Series],
    *,
    end: pd.Timestamp,
) -> FundingBackfillResult:
    """Reconstruct a paper funding estimate from actual settled rates, historical held units and completed trade prices only. Missing held events or prices fail without partial ledger mutation."""
    if not history:
        raise DataIntegrityError("paper funding backfill requires position history")
    start = history[0].effective_from
    if end <= start:
        raise DataIntegrityError(f"nothing to backfill start={start.isoformat()} end={end.isoformat()}")
    symbols = sorted({symbol for snap in history for symbol in snap.positions})
    # 보유 구간 커버리지 게이트: 공백은 부분 적립 없이 fail-closed.
    gap = pd.Timedelta(milliseconds=FUNDING_GAP_THRESHOLD_MS)
    bad: set[str] = set()
    for symbol in symbols:
        series = funding_by_symbol.get(symbol)
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for i, snap in enumerate(history):
            if snap.positions.get(symbol, Decimal(0)) == 0:
                continue
            s = snap.effective_from
            nxt = history[i + 1].effective_from if i + 1 < len(history) else end
            e = nxt if nxt < end else end
            if s < e:
                windows.append((s, e))
        if series is None:
            if windows:
                bad.add(symbol)
            continue
        idx = series.index
        for s, e in windows:
            points = [s, *sorted(idx[(idx > s) & (idx <= e)]), e]
            for a, b in itertools.pairwise(points):
                if b - a > gap:
                    bad.add(symbol)
                    break
    if bad:
        raise DataIntegrityError(f"paper funding backfill funding coverage gap symbols={','.join(sorted(bad))}")
    # 엔진과 동일 공식으로 적립: -(rate * qty * completed trade close).
    # 이후 바의 종가로 이전 펀딩 이벤트를 평가하지 않는다(정확한 완료-시각 매칭만 사용).
    by_symbol: dict[str, Decimal] = {}
    epochs = 0
    for symbol, series in funding_by_symbol.items():
        window = series[(series.index > start) & (series.index <= end)].sort_index()
        for epoch, rate in window.items():
            if not math.isfinite(float(rate)):
                raise DataIntegrityError(f"non-finite funding rate symbol={symbol} epoch={epoch.isoformat()}")
            qty = position_at(history, symbol, epoch)
            if qty == 0:
                continue
            bar = epoch.floor("h")
            trade_closes = trade_close_by_symbol.get(symbol)
            if trade_closes is None or bar not in trade_closes.index:
                raise DataIntegrityError(f"paper funding backfill trade-price missing symbol={symbol} bar={bar.isoformat()}")
            try:
                px_f = float(trade_closes.loc[bar])
            except (TypeError, ValueError, KeyError) as exc:
                raise DataIntegrityError(f"paper funding backfill trade-price missing symbol={symbol} bar={bar.isoformat()}") from exc
            if not math.isfinite(px_f) or px_f <= 0:
                raise DataIntegrityError(f"paper funding backfill trade-price missing symbol={symbol} bar={bar.isoformat()}")
            by_symbol[symbol] = by_symbol.get(symbol, Decimal(0)) - (
                Decimal(str(rate)) * qty * Decimal(str(px_f))
            )
            epochs += 1
    return FundingBackfillResult(
        start=start,
        end=end,
        cash_delta=sum(by_symbol.values(), Decimal(0)),
        by_symbol=by_symbol,
        epochs=epochs,
    )


def reconstruct_position_history(
    fills: pd.DataFrame,
    effective_by_decision: Mapping[pd.Timestamp, pd.Timestamp],
) -> tuple[PositionSnapshot, ...]:
    """Rebuild the position path from the fills sidecar with exact decimals."""
    from src.live.ledger import PositionSnapshot

    required = {"decision_time", "symbol", "quantity_delta"}
    missing = required - set(fills.columns)
    if missing:
        raise DataIntegrityError(f"fills missing columns {sorted(missing)}")
    if len(fills) == 0:
        raise DataIntegrityError("paper funding backfill found no fills")
    normalized = pd.to_datetime(fills["decision_time"], utc=True)
    decision_times = sorted({pd.Timestamp(dt) for dt in normalized.unique()})
    # 체결 누적을 Decimal(str(float)) 로 복원해 float64 드리프트를 제거.
    cumulative: dict[str, Decimal] = {}
    snaps: list[PositionSnapshot] = []
    previous: pd.Timestamp | None = None
    for dt in decision_times:
        effective = effective_by_decision.get(dt)
        if effective is None:
            raise DataIntegrityError(f"fill effective time missing decision_time={dt.isoformat()}")
        if previous is not None and effective <= previous:
            raise DataIntegrityError("fill effective times must be strictly increasing")
        previous = effective
        group = fills[normalized == dt]
        for _, row in group.iterrows():
            qty_f = float(row["quantity_delta"])
            if not math.isfinite(qty_f):
                raise DataIntegrityError(f"non-finite quantity_delta symbol={row['symbol']}")
            symbol = str(row["symbol"])
            cumulative[symbol] = cumulative.get(symbol, Decimal(0)) + Decimal(str(qty_f))
        snaps.append(
            PositionSnapshot(
                effective_from=pd.Timestamp(effective).tz_convert("UTC"),
                positions={s: q for s, q in cumulative.items() if q != 0},
            )
        )
    return tuple(snaps)


def fill_effective_times(
    audit_dir: Path, decision_times: Iterable[pd.Timestamp]
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each decision day to its first intent_outcome ts in the shadow audit."""
    resolved: dict[pd.Timestamp, pd.Timestamp] = {}
    for dt in decision_times:
        day = pd.Timestamp(dt).tz_convert("UTC")
        path = audit_dir / f"{day.strftime('%Y-%m-%d')}.jsonl"
        run_id = day.strftime("%Y%m%d")
        if not path.exists():
            raise DataIntegrityError(f"shadow_cycle audit missing path={path}")
        earliest: pd.Timestamp | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"shadow_cycle audit malformed path={path}") from exc
            if record.get("event") == "intent_outcome" and record.get("run_id") == run_id:
                ts = pd.Timestamp(record["ts"]).tz_convert("UTC")
                if earliest is None or ts < earliest:
                    earliest = ts
        if earliest is None:
            raise DataIntegrityError(f"shadow_cycle audit has no intent_outcome run_id={run_id}")
        resolved[pd.Timestamp(dt).tz_convert("UTC")] = earliest
    return resolved


def resolve_backfill_end(
    state: LedgerState, now: pd.Timestamp, accrual_start: pd.Timestamp | None
) -> tuple[pd.Timestamp, bool]:
    """Derive the backfill window end (engine bootstrap time) in fixed order."""
    if state.cash_usdt is None:
        raise DataIntegrityError("paper funding backfill requires cash_usdt")
    if state.funding_backfilled_through is not None:
        raise DataIntegrityError(
            f"paper funding backfill already applied through={state.funding_backfilled_through.isoformat()}"
        )
    # 엔진 마커 > 빈 이력(now 시드) > trim 전 history[0] > 운영자 지정.
    if state.funding_accrual_started_at is not None:
        derived: tuple[pd.Timestamp, bool] | None = (state.funding_accrual_started_at, False)
    elif not state.position_history:
        if state.funding_accrued_through is not None or state.funding_watermarks:
            raise DataIntegrityError("ambiguous accrual start: ledger has funding state without position history")
        derived = (now, True)
    elif len(state.position_history) < POSITION_HISTORY_MAX:
        derived = (state.position_history[0].effective_from, False)
    else:
        derived = None
    if accrual_start is not None:
        if accrual_start.tzinfo is None:
            raise DataIntegrityError("accrual_start must be tz-aware UTC")
        if accrual_start > now:
            raise DataIntegrityError("accrual_start must not be after now")
        if derived is not None and derived[0] != accrual_start:
            raise DataIntegrityError(
                f"accrual_start {accrual_start.isoformat()} does not match ledger-derived {derived[0].isoformat()}"
            )
        return (accrual_start.tz_convert("UTC"), derived[1] if derived is not None else False)
    if derived is None:
        raise DataIntegrityError("cannot derive accrual start from trimmed position history; pass --accrual-start")
    return derived


def plan_funding_backfill(
    state: LedgerState,
    history: Sequence[PositionSnapshot],
    funding_by_symbol: Mapping[str, pd.Series],
    trade_close_by_symbol: Mapping[str, pd.Series],
    *,
    now: pd.Timestamp,
    accrual_start: pd.Timestamp | None = None,
) -> BackfillPlan:
    """Reconcile fills against the ledger, then compute the backfill plan."""
    end, seeds = resolve_backfill_end(state, now, accrual_start)
    if history:
        # 체결 누적과 원장 포지션이 다르면 부분 적립 없이 fail-closed.
        expected = {symbol: qty for symbol, qty in state.positions.items() if qty != 0}
        if history[-1].positions != expected:
            divergent = sorted(
                set(expected) ^ set(history[-1].positions)
                | {s for s in expected if s in history[-1].positions and history[-1].positions[s] != expected[s]}
            )
            raise DataIntegrityError(
                f"fills do not reconcile with ledger positions symbols={','.join(divergent)}"
            )
    result = compute_funding_backfill(history, funding_by_symbol, trade_close_by_symbol, end=end)
    return BackfillPlan(
        start=result.start,
        end=result.end,
        cash_delta=result.cash_delta,
        by_symbol=result.by_symbol,
        epochs=result.epochs,
        seeds_accrual_start=seeds,
    )


def apply_funding_backfill(state: LedgerState, plan: BackfillPlan) -> LedgerState:
    """Apply the plan to cash and idempotency markers; positions untouched."""
    if state.cash_usdt is None:
        raise DataIntegrityError("paper funding backfill requires cash_usdt")
    return dataclasses.replace(
        state,
        cash_usdt=state.cash_usdt + plan.cash_delta,
        funding_backfilled_through=plan.end,
        funding_accrual_started_at=(
            state.funding_accrual_started_at if state.funding_accrual_started_at is not None else plan.end
        ),
        funding_accrued_through=plan.end if plan.seeds_accrual_start else state.funding_accrued_through,
    )


def _assert_daemon_idle(heartbeat_path: Path, now: pd.Timestamp) -> None:
    """Reject --apply while a cycle holds the ledger; stale busy is a crash remnant."""
    if not heartbeat_path.exists():
        return
    try:
        raw = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"heartbeat unreadable path={heartbeat_path}") from exc
    if not isinstance(raw, dict):
        raise DataIntegrityError(f"heartbeat unreadable path={heartbeat_path}")
    stage = raw.get("stage")
    ts_raw = raw.get("ts")
    if stage in BACKFILL_BUSY_STAGES and ts_raw is not None:
        age_s = (now - pd.Timestamp(ts_raw)).total_seconds()
        if age_s <= BACKFILL_BUSY_STALE_S:
            raise DataIntegrityError(f"daemon busy stage={stage}; retry when idle")


def run_paper_funding_backfill(
    settings: LiveSettings,
    *,
    apply: bool,
    now: pd.Timestamp,
    accrual_start: pd.Timestamp | None = None,
    funding_loader: Callable[[Sequence[str]], Mapping[str, pd.Series]] = _load_paper_funding,
    trade_close_loader: Callable[[Sequence[str]], Mapping[str, pd.Series]] | None = None,
    shadow_audit_dir: Path | None = None,
    backfill_audit_path: Path | None = None,
    heartbeat_path: Path | None = None,
    **kwargs: Any,
) -> BackfillPlan:
    """Reconstruct a paper funding estimate from actual settled rates, historical held units and completed trade prices only. Missing held events or prices fail without partial ledger mutation.

    Args:
        settings: Live settings (mutation-suppressed mode only).
        apply: Persist to the ledger when True; otherwise read-only plan.
        now: Backfill window end when the ledger cannot derive one.
        accrual_start: Operator override for the accrual start.
        funding_loader: On-disk funding series loader.
        trade_close_loader: Completed 1h trade-close series loader (same source
            as routine shadow accrual).
        shadow_audit_dir: shadow_cycle audit directory override.
        backfill_audit_path: Backfill audit JSONL override.
        heartbeat_path: Daemon heartbeat override.

    Returns:
        The computed :class:`BackfillPlan`.

    Raises:
        DataIntegrityError: On mode, idempotency, reconciliation, or data gaps.
    """
    if trade_close_loader is None:
        legacy = kwargs.pop("mark_loader", None)
        trade_close_loader = legacy if legacy is not None else _load_paper_trade_closes
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    if not settings.mode.suppresses_mutations:
        raise DataIntegrityError("paper funding backfill requires a mutation-suppressed mode")
    if apply:
        _assert_daemon_idle(
            heartbeat_path if heartbeat_path is not None else _resolve_heartbeat_path(settings), now
        )
    ledger_path = Path(settings.ledger_path) if settings.ledger_path else default_ledger_path()
    state = load_ledger(ledger_path)
    fills = load_fills(Path(settings.fills_dir) if settings.fills_dir else default_fills_dir())
    fills = fills[fills["mode"] == settings.mode.value] if "mode" in fills.columns else fills.iloc[0:0]
    if fills.empty:
        reconstruct_position_history(fills, {})
    decision_times = sorted(pd.Timestamp(dt) for dt in pd.to_datetime(fills["decision_time"], utc=True).unique())
    effective = fill_effective_times(
        shadow_audit_dir if shadow_audit_dir is not None else AUDIT_LOG_ROOT / "live" / "shadow_cycle",
        decision_times,
    )
    history = reconstruct_position_history(fills, effective)
    symbols = sorted({symbol for snap in history for symbol in snap.positions})
    plan = plan_funding_backfill(
        state, history, funding_loader(symbols), trade_close_loader(symbols), now=now, accrual_start=accrual_start
    )
    if apply:
        save_ledger(ledger_path, apply_funding_backfill(state, plan))
    audit = AuditLog(
        backfill_audit_path
        if backfill_audit_path is not None
        else default_audit_log_path(BACKFILL_AUDIT_NAME, for_date=now)
    )
    try:
        audit.record(
            "paper_funding_backfill",
            applied=apply,
            mode=settings.mode.value,
            start=plan.start.isoformat(),
            end=plan.end.isoformat(),
            cash_delta=str(plan.cash_delta),
            epochs=plan.epochs,
            symbols=len(plan.by_symbol),
            by_symbol={k: str(v) for k, v in sorted(plan.by_symbol.items())},
            seeds_accrual_start=plan.seeds_accrual_start,
        )
    finally:
        audit.close()
    logger.info(
        "[PORTFOLIO] paper_funding_backfill applied=%s start=%s end=%s cash_delta=%s epochs=%d symbols=%d",
        apply,
        plan.start.isoformat(),
        plan.end.isoformat(),
        plan.cash_delta,
        plan.epochs,
        len(plan.by_symbol),
    )
    return plan
