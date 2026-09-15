from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.mhs.research_go import (
    GO_REASON_CAPITAL_BREACH,
    GO_REASON_EXECUTION_GAP,
    GO_REASON_INVALID_PRIMARY,
    GO_REASON_NONFINITE_EQUITY,
    GO_REASON_RESOURCE_BREACH,
)
from src.common.errors import DataIntegrityError
from src.mhs.execution import ExecutionDataGap, StrategyExecutionReplayResult, laddered_fill_schedule
from src.mhs.types import ExecutionSpec

# 2026-09-15 mhs_symbol_lifespan_pit_roster: the other 50 previously-listed symbols
# (end-of-life: funding permanently ends before backtest end with zero internal gaps,
# exchangeInfo status=SETTLING) are now handled dynamically by `ledger_terminal_only`
# at finalize time instead of blanket exclusion.
#
# 2026-09-15 후속 재수집 스윕: 이전 Block1(OHLCV 캐시 없음/미미) 21개 심볼을
# ensure_ohlcv_data로 재수집한 결과 전부 복구됨(수집 누락이었음, 소스 공백 아님).
# 19개(ALPHAUSDT/BADGERUSDT/BSWUSDT/FLMUSDT/FTTUSDT/IDEXUSDT/KLAYUSDT/MKRUSDT/
# NULSUSDT/OBOLUSDT/OCEANUSDT/OMGUSDT/SLERFUSDT/STRAXUSDT/TROYUSDT/UNFIUSDT/
# VIDTUSDT/WAVESUSDT/XEMUSDT)는 backtest 구간 내부공백 0건으로 완전히 배제 해제됨
# (말기 종료는 ledger_terminal_only가 처리). CVXUSDT/SLPUSDT는 재수집 후에도
# 2025-06-19~2025-07-23(34일) 펀딩 공백이 재개되는 진짜 불확실성 구간이 드러나
# Block3(MID_LIFE_GAP)로 재분류.
#
# 2026-09-15 mhs_time_scoped_roster_mask: MISSING_ACTIVE_FUNDING(신규 진입 시도
# 차단, 보유 리스크 없음)을 KNOWN_ZERO_VOLUME과 동일하게 무조건 미체결 처리하고,
# ledger_terminal_only가 MISSING_HELD_MARK도 MISSING_HELD_FUNDING과 동일 규칙으로
# 인증하도록 확장한 뒤 실측 리플레이로 재검증 중 -- ICPUSDT(조기시작)는 이 확장만
# 으로 안전하게 해소되어 제외.
SOURCE_GAP_EXCLUDED_SYMBOLS = frozenset({
    # Block2: 펀딩 정상, 단일 영구 OHLCV 공백 8-17h, REST 확인으로 복구 불가.
    # 실측 검증 대상(MISSING_HELD_MARK 확장으로 해소되는지 리플레이로 확인 중).
    "AERGOUSDT", "CTKUSDT", "CVCUSDT", "MAVIAUSDT",
    # Block3: 백테스트 중간에 펀딩 공백이 발생했다가 재개되는 진짜 불확실성 구간, 제외 유지
    "LITUSDT", "PUMPUSDT", "CVXUSDT", "SLPUSDT",
    # Block4: BNXUSDT는 조기시작 외에도 자체 영구 OHLCV 공백을 보유, 실측 검증 대상
    "BNXUSDT",
})




#: Held-position gap codes eligible for the "later fill proves recovery"
#: terminal-equivalence check. MISSING_HELD_FUNDING and MISSING_HELD_MARK
#: share the same economics: valuation/funding during the gap never fabricates
#: a number (carried at the last known value / zero-charged), so the gap is a
#: bounded, disclosed limitation rather than a P&L-fabrication risk.
_RECOVERABLE_HELD_GAP_CODES = frozenset({"MISSING_HELD_FUNDING", "MISSING_HELD_MARK"})


def _funding_gap_terminal_symbols(
    data_gaps: Sequence[ExecutionDataGap],
    simulated_fills: pd.DataFrame,
) -> frozenset[str]:
    """Post-hoc classification of terminal held-position gaps.

    This is a finalize-time classification only and is never fed back into any
    trading decision (INV-PIT-RESUME-CAUSAL). A later fill for the same symbol
    proves the position resumed normal trading, so that symbol's gap is NOT
    terminal-equivalent. A ``delist_settlement`` fill (the causal idle-holdings
    settlement, ``_settle_idle_holdings``) is excluded from that "later fill"
    evidence: it is itself the terminal disclosure closing out a position that
    could never resume normal trading, not proof that trading recovered.
    """
    missing_last: dict[str, pd.Timestamp] = {}
    for g in data_gaps:
        if g.code in _RECOVERABLE_HELD_GAP_CODES:
            prev = missing_last.get(g.symbol)
            if prev is None or g.timestamp > prev:
                missing_last[g.symbol] = g.timestamp
    if not missing_last:
        return frozenset()
    if "reason" in simulated_fills.columns:
        resumable_fills = simulated_fills[simulated_fills["reason"] != "delist_settlement"]
    else:
        resumable_fills = simulated_fills
    fill_symbols = resumable_fills["symbol"] if "symbol" in resumable_fills.columns else pd.Series(dtype="object")
    fill_ts = pd.to_datetime(resumable_fills["timestamp"], utc=True) if "timestamp" in resumable_fills.columns else pd.Series(dtype="datetime64[ns, UTC]")
    terminal: set[str] = set()
    for sym, last_ts in missing_last.items():
        if not bool(((fill_symbols == sym) & (fill_ts > last_ts)).any()):
            terminal.add(sym)
    return frozenset(terminal)


def ledger_terminal_only(
    data_gaps: Sequence[ExecutionDataGap],
    simulated_fills: pd.DataFrame,
) -> bool:
    """Certify a ledger whose gaps are all disclosed terminal inventory.

    Generalizes the existing UNKNOWN_TERMINATION-only exception to also accept
    a MISSING_HELD_FUNDING or MISSING_HELD_MARK episode that never recovers
    before the replay's own grid end -- symmetric with the pre-existing 'held
    to backtest end is disclosed evidence, not a crash' precedent; no
    fabricated settlement, no change to funding/mark accounting
    (INV-NO-FABRICATED-SETTLEMENT).
    """
    if not data_gaps:
        return False
    terminal_funding_symbols = _funding_gap_terminal_symbols(data_gaps, simulated_fills)
    return all(
        g.code == "UNKNOWN_TERMINATION"
        or (g.code in _RECOVERABLE_HELD_GAP_CODES and g.symbol in terminal_funding_symbols)
        for g in data_gaps
    )


def _assert_cache_required_ledger_valid(
    name: str,
    primary: StrategyExecutionReplayResult,
) -> None:
    """Fail closed when a cache-required strict primary ledger is invalid.

    ``cache_required_stale_carry`` and ``ohlcv_close_fallback`` are explicit
    diagnostic modes and never call this gate. Disclosed terminal inventory
    (every gap is ``UNKNOWN_TERMINATION`` or a non-recovering
    ``MISSING_HELD_FUNDING`` episode, see ``ledger_terminal_only``) is
    evidence, not a crash: the position stays open, the ledger stays invalid,
    and deployment is blocked downstream by backtest-reliability
    certification instead of failing the book here.
    """
    gaps = primary.ledger.data_gaps
    terminal_only = bool(primary.ledger.primary_valid) or ledger_terminal_only(gaps, primary.simulated_fills)
    if not terminal_only:
        raise DataIntegrityError(
            f"cache_required strict primary ledger invalid for {name}: "
            f"{', '.join(primary.ledger.invalid_reasons)}"
        )


def _classify_execution_failure(exc: BaseException) -> str:
    """Stable fail-closed reason code for an expected strict replay error.

    The classifier is intentionally conservative: any unrecognized message maps
    to ``INVALID_PRIMARY_LEDGER`` so an unanticipated integrity error is never
    relabeled as a policy or Sharpe gate. Resource-budget breaches keep their
    own stable code so a fixed-RSS regression can be proven end to end.
    """
    message = str(exc).lower()
    if "pre-trade equity" in message or "capital" in message or "equity must be" in message:
        return GO_REASON_CAPITAL_BREACH
    if "rss budget" in message:
        return GO_REASON_RESOURCE_BREACH
    if "finite" in message:
        return GO_REASON_NONFINITE_EQUITY
    if "gap" in message or "mark" in message or "missing" in message:
        return GO_REASON_EXECUTION_GAP
    return GO_REASON_INVALID_PRIMARY




























def _validate_ladder_schedule_contract() -> None:
    """Runtime guard for the frozen ladder schedule contract (spec §1.4).

    Runs once per ``--ladder-diagnostic`` book pass, before the expensive
    windowed replay: a single tranche must reproduce the strict single-fill
    schedule and the ladder's ``qty_fraction`` values must conserve notional.
    """
    one = laddered_fill_schedule(
        100.0, 1, np.array([101.0]), np.array([101.0, 101.0]),
        1, ExecutionSpec(), True,
    )
    assert one == [(1, 101.0, ExecutionSpec().one_way_taker_bps(), 1.0)]
    ladder = laddered_fill_schedule(
        100.0, 1, np.array([101.0] * 4), np.array([101.0] * 5),
        4, ExecutionSpec(), True,
    )
    assert abs(sum(f[3] for f in ladder) - 1.0) < 1e-12







def _truncate_replayable_decisions(
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    execution_grid: pd.DatetimeIndex,
    spec: ExecutionSpec,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, int]:
    """Censor terminal-window decisions that can never be executed on the grid.

    A decision is retained only when its first post-signal submit bar exists and
    the exact strict ``passive_timeout_minutes`` bar exists on the execution
    grid. The strict boundary is applied to both strict and immediate-taker
    outputs so they cover the same decision population. Dropped rows are
    terminal telemetry, never ``MISSING_DATA``: the returned count records them
    so the report can distinguish censored terminal decisions from real source
    gaps. Retained targets are byte-for-byte unchanged.
    """
    if len(target_weights) != len(signal_available_at):
        raise DataIntegrityError("signal_available_at must align with target_weights")
    grid_ns = np.asarray(execution_grid, dtype="datetime64[ns]").astype("int64")
    n_grid = len(grid_ns)
    if n_grid == 0:
        return target_weights.iloc[0:0], signal_available_at[0:0], len(target_weights)
    timeout_ns_delta = int(spec.passive_timeout_minutes) * 60_000_000_000
    signal_ns = np.asarray(signal_available_at, dtype="datetime64[ns]").astype("int64")
    spos = np.searchsorted(grid_ns, signal_ns, side="right")
    spos_clipped = np.minimum(spos, n_grid - 1)
    timeout_pos = np.searchsorted(grid_ns, grid_ns[spos_clipped] + timeout_ns_delta, side="left")
    timeout_pos_clipped = np.minimum(timeout_pos, n_grid - 1)
    replayable = (
        (spos < n_grid)
        & (timeout_pos < n_grid)
        & (grid_ns[timeout_pos_clipped] == grid_ns[spos_clipped] + timeout_ns_delta)
    )
    censored = int((~replayable).sum())
    if censored == 0:
        return target_weights, signal_available_at, 0
    return target_weights.loc[replayable], signal_available_at[replayable], censored


def _assert_cache_required_marks(
    name: str,
    target_replay: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    minute_marks: pd.DataFrame,
) -> None:
    """Fail closed when a replay symbol lacks a finite positive mark at a decision point.

    A mark is required at every decision time where the target weight is
    non-zero. ``minute_marks`` is exactly aligned to the minute closes; a
    missing/non-positive mark at a required decision point raises
    ``DataIntegrityError`` carrying the stable provenance rather than silently
    falling back to OHLCV closes.
    """
    grid_set = set(minute_marks.index)
    for i, decision_time in enumerate(target_replay.index):
        signal_time = signal_available_at[i]
        for sym in target_replay.columns:
            weight = float(target_replay.loc[decision_time, sym])
            if not np.isfinite(weight) or weight == 0.0:
                continue
            mark = float("nan")
            if decision_time in grid_set:
                mark = float(minute_marks.loc[decision_time, sym])
            if not (np.isfinite(mark) and mark > 0):
                prior = minute_marks.index[(minute_marks.index <= signal_time)]
                if len(prior):
                    mark = float(minute_marks.loc[prior[-1], sym])
            if not (np.isfinite(mark) and mark > 0):
                after = minute_marks.index[minute_marks.index > signal_time]
                if len(after):
                    mark = float(minute_marks.loc[after[0], sym])
            if not (np.isfinite(mark) and mark > 0):
                raise DataIntegrityError(
                    "cache_required: no finite positive mark (MISSING_DECISION_MARK) "
                    f"symbol={sym} decision={decision_time} signal={signal_time} "
                    f"for {name}"
                )


