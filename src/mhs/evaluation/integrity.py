from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.data_policy import SOURCE_GAP_EXCLUDED_SYMBOLS as SOURCE_GAP_EXCLUDED_SYMBOLS
from src.mhs.execution import StrategyExecutionReplayResult, laddered_fill_schedule
from src.mhs.execution.integrity import (
    _funding_gap_terminal_symbols as _funding_gap_terminal_symbols,
)
from src.mhs.execution.integrity import ledger_terminal_only as ledger_terminal_only
from src.mhs.execution.integrity import replay_ledger_certified as replay_ledger_certified
from src.mhs.params import PNL_VOL_TARGET_BURN_IN_DAYS
from src.mhs.research_go import (
    GO_REASON_CAPITAL_BREACH,
    GO_REASON_EXECUTION_GAP,
    GO_REASON_INVALID_PRIMARY,
    GO_REASON_NONFINITE_EQUITY,
    GO_REASON_RESOURCE_BREACH,
)
from src.mhs.resources import MhsResourceAdmissionError
from src.mhs.types import ExecutionSpec


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
    if isinstance(exc, MhsResourceAdmissionError):
        return GO_REASON_RESOURCE_BREACH
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


def _assert_train_reference_ledger_certified(
    replay: StrategyExecutionReplayResult,
    fold_index: int,
) -> None:
    """Fail closed unless a fold's train-reference replay ledger is certified accounting.

    The reference equity sets the validation fold's target volatility and exposure
    warm-up, so a curve accrued over unknown held funding, missing held marks or
    blocked orders would size real exposure from fictional risk. Certification is
    the shared ``replay_ledger_certified`` verdict applied to the validation
    primary; priced, funded open inventory at the train cutoff is accepted because
    the reference ends at an observation cutoff, not an exit. Terminal-only gap
    classifications never certify.

    Args:
        replay: Train-reference replay result (ledger, gaps, terminal evidence).
        fold_index: Anchored fold index, reported in the error.
    Raises:
        DataIntegrityError: ``"fold <fold_index>: train reference ledger not
            certified: primary_valid=<bool> invalid_reasons=<r,...|none>
            gap_codes=<CODE:count,...|none> unresolved_terminal=<n>
            funding_incomplete_terminal=<n>"``; maps to
            ``RELEVANT_EXECUTION_DATA_GAP`` via ``_classify_execution_failure``.
    """
    if replay_ledger_certified(replay) is True:
        return None
    ledger = getattr(replay, "ledger", None)
    primary_valid = bool(getattr(ledger, "primary_valid", False)) if ledger is not None else False
    raw_reasons = getattr(ledger, "invalid_reasons", ()) if ledger is not None else ()
    reasons = tuple(raw_reasons) if raw_reasons else ()
    invalid_text = ",".join(str(r) for r in reasons) if reasons else "none"
    raw_gaps = getattr(ledger, "data_gaps", ()) if ledger is not None else ()
    gaps = list(raw_gaps) if raw_gaps else []
    counts: dict[str, int] = {}
    for gap in gaps:
        code = str(getattr(gap, "code", gap))
        counts[code] = counts.get(code, 0) + 1
    gap_text = ",".join(f"{code}:{counts[code]}" for code in sorted(counts)) if counts else "none"
    positions = getattr(replay, "terminal_positions", ()) if replay is not None else ()
    items = list(positions) if positions else []
    unresolved = sum(1 for p in items if getattr(p, "status", None) == "unresolved")
    funding_incomplete = sum(1 for p in items if getattr(p, "funding_complete", None) is not True)
    raise DataIntegrityError(
        f"fold {fold_index}: train reference ledger not certified: "
        f"primary_valid={primary_valid} invalid_reasons={invalid_text} "
        f"gap_codes={gap_text} unresolved_terminal={unresolved} "
        f"funding_incomplete_terminal={funding_incomplete}"
    )


def _assert_train_reference_returns_valid(daily: pd.Series, train_end: pd.Timestamp, fold_index: int) -> None:
    """Fail closed unless the fold's train-only sizing reference is causally usable.

    The reference sizes the validation fold, so it must be finite, strictly
    chronological, UTC, entirely before ``train_end`` (no validation leakage) and
    at least ``PNL_VOL_TARGET_BURN_IN_DAYS`` rows long. Raises DataIntegrityError
    naming the fold on the first violated condition.
    """
    if not bool(np.isfinite(daily.to_numpy(dtype="float64")).all()):
        raise DataIntegrityError(f"fold {fold_index}: train reference returns must be finite")
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise DataIntegrityError(f"fold {fold_index}: train reference index must be unique and monotonic")
    if not str(getattr(daily.index, "tz", None)) == "UTC":
        raise DataIntegrityError(f"fold {fold_index}: train reference index must be UTC")
    if not (daily.index < train_end).all():
        raise DataIntegrityError(f"fold {fold_index}: train reference extends into validation")
    if len(daily.dropna()) < PNL_VOL_TARGET_BURN_IN_DAYS:
        raise DataIntegrityError(f"fold {fold_index}: train reference has {len(daily.dropna())} rows, require >= {PNL_VOL_TARGET_BURN_IN_DAYS}")
