"""Paired intraday inventory evidence for the frozen MHS research candidate."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.backtest.certification import DailyPortfolioEvidence, inventory_daily_evidence
from src.mhs.execution import (
    ExecutionReplayWindow,
    StrategyExecutionReplayResult,
    _ExecutionBound,
    replay_execution_window_batch,
)
from src.mhs.execution.batch import _LiveAccumulatorSets
from src.mhs.frozen_research_candidate import FrozenMhsCandidate
from src.mhs.frozen_research_windows import validated_frozen_research_windows
from src.mhs.types import ExecutionSpec

_METRIC_COLUMNS: tuple[str, ...] = (
    "base_cagr", "stress_cagr",
    "base_max_drawdown", "stress_max_drawdown",
    "base_annual_turnover", "stress_annual_turnover",
    "base_target_gross", "stress_target_gross",
    "base_funding_contribution", "stress_funding_contribution",
    "base_fill_shortfall_bps", "stress_fill_shortfall_bps",
    "base_forced_exits", "stress_forced_exits",
    "base_source_gaps", "stress_source_gaps",
    "base_coverage", "stress_coverage",
)
_HELD_GAP_CODES = ("MISSING_HELD_MARK", "MISSING_HELD_FUNDING")
_LIMITATIONS: tuple[str, ...] = (
    "CANDLE_FILLS_NO_ORDER_BOOK_DEPTH",
    "CANDLE_FILLS_NO_QUEUE_POSITION",
    "CANDLE_FILLS_NO_PARTICIPATION_CAPACITY",
    "DISCOVERY_YEARS_RESEARCH_ONLY_NO_FORWARD_CLAIM",
    "NO_DEPLOYMENT_VERDICT",
)


@dataclass(frozen=True, slots=True)
class FrozenMhsReportPeriod:
    """Name one explicitly bounded historical reporting interval."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise DataIntegrityError("report period label must be a non-empty string")
        for name in ("start", "end"):
            value = getattr(self, name)
            if not isinstance(value, pd.Timestamp) or pd.isna(value):
                raise DataIntegrityError(f"report period {name} must be a valid timestamp")
            if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
                raise DataIntegrityError(f"report period {name} must be timezone-aware UTC")
        if not self.start < self.end:
            raise DataIntegrityError("report period must satisfy start < end")


@dataclass(frozen=True, slots=True)
class FrozenMhsResearchEvidence:
    """Hold comparable inventory results without implying forward success.

    Args:
        base: Exact frozen targets under the six-basis-point taker case.
        stress: Same market windows and targets at threefold crossing cost.
        base_daily: Completed observed daily inventory intervals.
        stress_daily: Matching completed daily stress intervals.
        period_metrics: Explicit historical-window economics and validity
            diagnostics, with unavailable intervals left unavailable.
        limitations: Source and model limitations requiring separate review.
    """

    base: StrategyExecutionReplayResult
    stress: StrategyExecutionReplayResult
    base_daily: DailyPortfolioEvidence
    stress_daily: DailyPortfolioEvidence
    period_metrics: pd.DataFrame
    limitations: tuple[str, ...]


def evaluate_frozen_mhs_research(
    candidate: FrozenMhsCandidate,
    windows: Iterable[ExecutionReplayWindow],
    *,
    initial_equity: float,
    base_spec: ExecutionSpec,
    stress_spec: ExecutionSpec,
    report_periods: tuple[FrozenMhsReportPeriod, ...],
    live_accumulators: _LiveAccumulatorSets | None = None,
) -> FrozenMhsResearchEvidence:
    """Replay a frozen target plan through one shared, paired 3m ledger stream.

    The result separates completed historical accounting from any claim of
    forward validity.  Base and stress consume the same target and market
    windows; only registered crossing cost differs.

    Args:
        candidate: Exact frozen target plan.
        windows: Validated one-pass 3m execution source.
        initial_equity: Positive finite research capital.
        base_spec: Registered six-basis-point immediate-taker cost specification.
        stress_spec: Same mechanics with registered eighteen-basis-point cost.
        report_periods: Non-overlapping UTC intervals to report when complete.
        live_accumulators: Optional accumulator registry shared with the window
            source so carried holdings remain in later local rosters.
    Returns:
        Paired ledger evidence, daily intervals, transparent metrics, and limits.
    Raises:
        DataIntegrityError: Paired economics, source provenance, or reporting
            intervals are inconsistent, incomplete, or unsafe to score.
    """
    if not (np.isfinite(initial_equity) and initial_equity > 0.0):
        raise DataIntegrityError("initial_equity must be a positive finite capital")
    if base_spec.one_way_taker_bps() != 6.0 or stress_spec.one_way_taker_bps() != 18.0:
        raise DataIntegrityError("base cost must be 6 bps and stress cost 18 bps one-way")
    if dataclasses.replace(
        stress_spec, taker_fee_bps=base_spec.taker_fee_bps, taker_slippage_bps=base_spec.taker_slippage_bps
    ) != base_spec:
        raise DataIntegrityError("stress spec must match base mechanics except crossing cost")
    _require_report_periods(report_periods)
    checked_windows = validated_frozen_research_windows(
        candidate,
        windows,
        settlement_bars=-(-int(base_spec.passive_timeout_minutes) // 3),
    )
    paired_bounds: list[tuple[_ExecutionBound, ExecutionSpec]] = [
        ("OHLCV_IMMEDIATE_TAKER", base_spec), ("OHLCV_IMMEDIATE_TAKER", stress_spec)
    ]
    results = replay_execution_window_batch(
        checked_windows, initial_equity, paired_bounds, live_accumulators=live_accumulators
    )
    base, stress = results[0], results[1]
    if (
        not base.ledger.primary_valid
        or not stress.ledger.primary_valid
        or any(gap.code in _HELD_GAP_CODES for gap in (*base.data_gaps, *stress.data_gaps))
        or any(pos.status == "unresolved" for pos in (*base.terminal_positions, *stress.terminal_positions))
    ):
        raise DataIntegrityError("replay held unpriced marks, unknown funding, or unresolved terminals")
    base_daily = inventory_daily_evidence(base)
    stress_daily = inventory_daily_evidence(stress)
    if (
        not base_daily.returns.index.equals(stress_daily.returns.index)
        or not base_daily.label_start.equals(stress_daily.label_start)
        or not base_daily.label_end.equals(stress_daily.label_end)
        or not base_daily.available_at.equals(stress_daily.available_at)
    ):
        raise DataIntegrityError("paired daily intervals must match by label and availability")
    metrics = _period_metrics(candidate, base, stress, base_daily, stress_daily, initial_equity, report_periods)
    gap_note = ("FUNDING_COVERAGE_GAPS_RECORDED",) if (base.funding_coverage_gaps or stress.funding_coverage_gaps) else ()
    return FrozenMhsResearchEvidence(
        base=base, stress=stress, base_daily=base_daily, stress_daily=stress_daily,
        period_metrics=metrics, limitations=_LIMITATIONS + gap_note,
    )


def _require_report_periods(report_periods: tuple[FrozenMhsReportPeriod, ...]) -> None:
    """Reject invalid or overlapping caller-supplied reporting intervals."""
    if not isinstance(report_periods, tuple) or len(report_periods) == 0:
        raise DataIntegrityError("report_periods must be a non-empty tuple of FrozenMhsReportPeriod")
    labels = [p.label for p in report_periods]
    if any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) != len(labels):
        raise DataIntegrityError("report period labels must be unique non-empty strings")
    ordered = sorted(report_periods, key=lambda p: (p.start.value, p.end.value))
    for prev, cur in itertools.pairwise(ordered):
        if not prev.end < cur.start:
            raise DataIntegrityError("report periods must not overlap")


def _period_metrics(
    candidate: FrozenMhsCandidate,
    base: StrategyExecutionReplayResult,
    stress: StrategyExecutionReplayResult,
    base_daily: DailyPortfolioEvidence,
    stress_daily: DailyPortfolioEvidence,
    initial_equity: float,
    report_periods: tuple[FrozenMhsReportPeriod, ...],
) -> pd.DataFrame:
    """Report per-period economics with unavailable intervals left as NaN."""
    rows: dict[str, dict[str, float]] = {}
    for period in report_periods:
        expected = pd.date_range(period.start, period.end, freq="D", tz="UTC")
        have = set(base_daily.returns.index)
        coverage = sum(day in have for day in expected) / len(expected) if len(expected) else 0.0
        if coverage != 1.0:
            rows[period.label] = {"base_coverage": coverage, "stress_coverage": coverage}
            continue
        rows[period.label] = _covered_row(
            candidate, base, stress, base_daily, stress_daily, initial_equity, period.start, period.end
        )
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=_METRIC_COLUMNS)
    frame.index.name = "period"
    return frame


def _covered_row(
    candidate: FrozenMhsCandidate,
    base: StrategyExecutionReplayResult,
    stress: StrategyExecutionReplayResult,
    base_daily: DailyPortfolioEvidence,
    stress_daily: DailyPortfolioEvidence,
    initial_equity: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    """Compute economics over one fully observed calendar period."""
    days = pd.date_range(start, end, freq="D", tz="UTC")
    horizon = end + pd.Timedelta(days=1)
    base_rets = base_daily.returns.loc[days].to_numpy(dtype="float64")
    stress_rets = stress_daily.returns.loc[days].to_numpy(dtype="float64")
    span = (base.ledger.equity.index >= start) & (base.ledger.equity.index < horizon)
    base_eq = base.ledger.equity.loc[span]
    stress_eq = stress.ledger.equity.loc[span]
    turnover_span = (base.ledger.fill_turnover.index >= start) & (base.ledger.fill_turnover.index < horizon)
    base_turnover = base.ledger.fill_turnover.loc[turnover_span].to_numpy(dtype="float64")
    stress_turnover = stress.ledger.fill_turnover.loc[turnover_span].to_numpy(dtype="float64")
    funding_span = (base.ledger.funding_charge.index >= start) & (base.ledger.funding_charge.index < horizon)
    base_funding = base.ledger.funding_charge.loc[funding_span].to_numpy(dtype="float64")
    stress_funding = stress.ledger.funding_charge.loc[funding_span].to_numpy(dtype="float64")
    gross_slice = candidate.target_weights.loc[
        (candidate.target_weights.index >= start) & (candidate.target_weights.index <= end)
    ]
    gross = gross_slice.abs().sum(axis=1).to_numpy(dtype="float64")
    years = len(days) / 365.0
    return {
        "base_cagr": float(np.prod(1.0 + base_rets) ** (1.0 / years) - 1.0),
        "stress_cagr": float(np.prod(1.0 + stress_rets) ** (1.0 / years) - 1.0),
        "base_max_drawdown": float(((base_eq.cummax() - base_eq) / base_eq.cummax()).max()),
        "stress_max_drawdown": float(((stress_eq.cummax() - stress_eq) / stress_eq.cummax()).max()),
        "base_annual_turnover": float(base_turnover.sum() / years),
        "stress_annual_turnover": float(stress_turnover.sum() / years),
        "base_target_gross": float(np.mean(gross)),
        "stress_target_gross": float(np.mean(gross)),
        "base_funding_contribution": float(base_funding.sum() / initial_equity),
        "stress_funding_contribution": float(stress_funding.sum() / initial_equity),
        "base_fill_shortfall_bps": float(base.all_intent_shortfall_bps),
        "stress_fill_shortfall_bps": float(stress.all_intent_shortfall_bps),
        "base_forced_exits": float(base.forced_exit_count),
        "stress_forced_exits": float(stress.forced_exit_count),
        "base_source_gaps": float(len(base.data_gaps)),
        "stress_source_gaps": float(len(stress.data_gaps)),
        "base_coverage": 1.0,
        "stress_coverage": 1.0,
    }
