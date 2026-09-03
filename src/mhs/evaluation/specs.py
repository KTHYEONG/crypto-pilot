from __future__ import annotations

from dataclasses import replace as dataclass_replace
from typing import Any

from src.mhs.execution import StrategyExecutionReplayResult
from src.mhs.params import SIGNAL_EMA_HORIZON_SPAN, STRESS_COST_MULTIPLIER
from src.mhs.types import ExecutionSpec


def _resolved_base_execution_spec(request: Any) -> ExecutionSpec:
    """Single owner of the request-driven base execution spec (S6).

    Threads the configured execution window (``passive_timeout_minutes``) into
    every replay bound; the frozen default 30 reproduces legacy specs exactly.
    """
    return dataclass_replace(
        ExecutionSpec(), passive_timeout_minutes=int(request.passive_timeout_minutes)
    )


def _stress_cost_execution_spec(base: ExecutionSpec | None = None) -> ExecutionSpec:
    """SPREAD_AND_COST_X3: the same realistic fill mechanic at 3x cost."""
    resolved = ExecutionSpec() if base is None else base
    return ExecutionSpec(
        maker_fee_bps=resolved.maker_fee_bps * STRESS_COST_MULTIPLIER,
        taker_fee_bps=resolved.taker_fee_bps * STRESS_COST_MULTIPLIER,
        taker_slippage_bps=resolved.taker_slippage_bps * STRESS_COST_MULTIPLIER,
        passive_timeout_minutes=resolved.passive_timeout_minutes,
    )




def _signal_ema_span(band_sign: int, horizon_hours: int, step_hours: int) -> int | None:
    """Whipsaw-suppressing EMA span, or None for a reversal band (sign=-1)."""
    if band_sign != 1:
        return None
    return max(1, round(horizon_hours / step_hours * SIGNAL_EMA_HORIZON_SPAN))


# Diagnostic reference-only execution bounds. OHLCV_IMMEDIATE_TAKER (primary and
# cost-stress) is deliberately absent: it carries capital and keeps fail-closed
# propagation.
REFERENCE_ONLY_EXECUTION_BOUNDS: frozenset[str] = frozenset(
    {"OHLCV_STRICT_PROXY", "OHLCV_TOUCH_PROXY", "OHLCV_LADDERED_PROXY", "OHLCV_PEG_CHASE_PROXY"}
)


def _peg_chase_fill_rate(report: StrategyExecutionReplayResult) -> float | None:
    """Share of peg-chase intents that terminated in a fill (maker or taker)."""
    denom = report.fill_count + report.fallback_count + report.residual_count
    if not denom:
        return None
    return (report.fill_count + report.fallback_count) / denom


def _peg_chase_maker_share(report: StrategyExecutionReplayResult) -> float | None:
    """Share of filled peg-chase notional intents that executed as maker."""
    denom = report.fill_count + report.fallback_count
    if not denom:
        return None
    return report.fill_count / denom


# Sentinel distinguishing the registered default exposure from an explicit
# committee_target_gross value: a bare MhsDiagnosticRequest() resolves to the
# registered constant without triggering the committee_capital requirement,
# while an explicit non-None value keeps requiring committee_capital=True.


# Unrecoverable source gap exclusions (Binance REST API & Vision archives have >4h gaps):
# SLPUSDT, CTKUSDT, LITUSDT, AERGOUSDT, PUMPUSDT, CVXUSDT, CVCUSDT
# BNXUSDT re-evaluated 2026-08-23 (ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING
# follow-up): confirmed via BOTH the live REST klines endpoint AND the Binance
# Vision monthly archive that FOUR windows are absent from every current
# Binance data source, not just the previously-documented 2023 one --
# 2022-04-16 23:57..2022-04-18 00:00, 2022-06-08 23:57..2022-06-10 00:00,
# 2022-08-09 23:57..2022-08-13 00:00, and 2023-01-31 23:57..2023-02-22 14:45
# UTC. exchangeInfo reports this symbol's current contract onboardDate as
# 2023-02-22 (a delist/relist), so the pre-relist history is not just
# uncollected but no longer served by the exchange at all -- no backfill path
# exists. The prior ~30% relative Calmar cost estimate was measured before
# committee_kelly_sizing/growth_extreme/breadth-60 became the defaults; see
# fold0's RELEVANT_EXECUTION_DATA_GAP measurement for the current cost.
