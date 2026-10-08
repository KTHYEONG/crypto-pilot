"""Durable research-only payload for one strategy inventory run."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.types import JsonValue
from src.engine.strategy_backtest import StrategyBacktestRun

_logger = logging.getLogger(__name__)


def _jsonable(value: object) -> JsonValue:
    """Convert one ledger-derived scalar to a JSON-safe value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return cast(JsonValue, value)
    if isinstance(value, (int, np.integer)):
        return cast(JsonValue, int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return cast(JsonValue, number)
    if isinstance(value, str):
        return cast(JsonValue, value)
    if isinstance(value, pd.Timestamp):
        return cast(JsonValue, value.isoformat())
    raise DataIntegrityError(f"evidence value {value!r} has no JSON representation")


def strategy_backtest_payload(
    run: StrategyBacktestRun,
    *,
    statistics: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """Serialize research evidence without discarding provenance or limitations.

    The payload is a durable human- and machine-readable summary of one
    historical run.  It exposes ledger-derived metrics and source constraints
    while keeping raw target and fill tables as separately referenced artifacts.

    Args:
        run: Completed strategy historical replay result.
    Returns:
        JSON-safe provenance, strategy, base/stress, and limitation fields.
    Raises:
        DataIntegrityError: Required evidence fields cannot be represented
        faithfully as a completed result.
    """
    request = run.request
    candidate = run.candidate
    evidence = run.evidence
    if not evidence.base.ledger.primary_valid or not evidence.stress.ledger.primary_valid:
        raise DataIntegrityError("payload requires a completed valid base/stress ledger pair")
    if list(candidate.target_weights.columns) != list(run.source_symbols):
        raise DataIntegrityError("candidate columns must follow the source symbol census")
    if candidate.strategy.strategy_id != request.strategy.strategy_id:
        raise DataIntegrityError("candidate strategy must match the request strategy")
    periods: dict[str, JsonValue] = {}
    for period in request.report_periods:
        try:
            row = evidence.period_metrics.loc[period.label]
        except KeyError as exc:
            raise DataIntegrityError(f"report period {period.label!r} has no metrics row") from exc
        coverage = float(row["base_coverage"])
        if coverage != 1.0 or float(row["stress_coverage"]) != 1.0:
            periods[period.label] = cast(
                JsonValue,
                {
                    "status": "unavailable",
                    "base_coverage": coverage,
                    "stress_coverage": float(row["stress_coverage"]),
                    "start": period.start.isoformat(),
                    "end": period.end.isoformat(),
                },
            )
            continue
        metrics = {column: _jsonable(row[column]) for column in evidence.period_metrics.columns}
        metrics["start"] = period.start.isoformat()
        metrics["end"] = period.end.isoformat()
        metrics["status"] = "complete"
        per_symbol = evidence.funding_by_symbol.get(period.label) if isinstance(evidence.funding_by_symbol, Mapping) else None
        if isinstance(per_symbol, Mapping):
            metrics["funding_by_symbol"] = {
                case: {sym: _jsonable(val) for sym, val in (vals.items() if isinstance(vals, Mapping) else {})}
                for case, vals in per_symbol.items()
            }
        periods[period.label] = cast(JsonValue, metrics)
    limitations = list(evidence.limitations)
    if getattr(run, "data_availability_withdrawals", ()) and "DATA_AVAILABILITY_SELECTION" not in limitations:
        limitations.append("DATA_AVAILABILITY_SELECTION")
    payload: dict[str, JsonValue] = {
        "strategy_id": candidate.strategy.strategy_id,
        "breadth": candidate.strategy.breadth,
        "exposure_multiplier": _jsonable(candidate.strategy.exposure_multiplier),
        "name_clip": _jsonable(candidate.strategy.name_clip),
        "daily_artifact": "daily.parquet",
        "members": cast(
            JsonValue,
            [{"name": member.name, "sign": member.sign} for member in candidate.strategy.members],
        ),
        "min_rank_symbols": candidate.strategy.min_rank_symbols,
        "source_start": request.source_start.isoformat(),
        "evaluation_start": request.evaluation_start.isoformat(),
        "evaluation_end": request.evaluation_end.isoformat(),
        "execution_start": run.execution_start.isoformat(),
        "execution_end": run.execution_end.isoformat(),
        "canonical_symbols": len(run.source_symbols),
        "source_symbols": cast(JsonValue, list(run.source_symbols)),
        "source_gap_excluded_symbols": cast(JsonValue, sorted(run.source_gap_excluded_symbols)),
        "source_gap_excluded_count": len(run.source_gap_excluded_symbols),
        "source_gap_blocked_decisions": run.source_gap_blocked_decisions,
        "delisting_blocked_decisions": run.delisting_blocked_decisions,
        "delisting_announcement_policy": (
            "Registry announced_at controls roster withdrawal; proxy_lead announcements are "
            "last_trade_at minus the registered lead, not evidenced public announcement times."
        ),
        "base_one_way_taker_bps": request.base_spec.one_way_taker_bps(),
        "stress_one_way_taker_bps": request.stress_spec.one_way_taker_bps(),
        "execution_bound": request.execution_bound,
        "decision_anchor": request.base_spec.decision_anchor,
        "maker_fee_bps": request.base_spec.maker_fee_bps,
        "base_valid": evidence.base.ledger.primary_valid,
        "stress_valid": evidence.stress.ledger.primary_valid,
        "base_source_gaps": len(evidence.base.data_gaps),
        "stress_source_gaps": len(evidence.stress.data_gaps),
        "base_terminal": cast(
            JsonValue,
            [{"symbol": pos.symbol, "status": pos.status} for pos in evidence.base.terminal_positions],
        ),
        "stress_terminal": cast(
            JsonValue,
            [{"symbol": pos.symbol, "status": pos.status} for pos in evidence.stress.terminal_positions],
        ),
        "limitations": cast(JsonValue, limitations),
        "data_availability_withdrawals": cast(
            JsonValue,
            [dict(entry) for entry in getattr(run, "data_availability_withdrawals", ())],
        ),
        "design_data_cutoff": candidate.strategy.design_data_cutoff.isoformat(),
        "report_periods": cast(JsonValue, periods),
        "research_only": True,
    }
    if statistics is not None:
        payload["statistics"] = cast(JsonValue, dict(statistics))
    json.dumps(payload)
    return payload


def strategy_daily_frame(run: StrategyBacktestRun) -> pd.DataFrame:
    """Daily ledger evidence needed to audit and re-derive the exposure policy.

    Summary metrics cannot answer how a drawdown budget binds, how intraday marks deepen
    drawdown relative to daily marks, or where funding and turnover concentrate; the drawdown
    budget for the growth rung is solved on exactly these daily series, so they are persisted
    with the run instead of being reconstructed by ad-hoc probes.

    Args:
        run: Completed strategy replay with valid paired ledgers.
    Returns:
        UTC-daily indexed frame with float64 columns ``base_return``, ``stress_return``,
        ``base_equity_close``, ``stress_equity_close``, ``base_equity_low``,
        ``stress_equity_low``, ``base_turnover``, ``stress_turnover``, ``base_funding``,
        ``stress_funding``, ``target_gross``, ``max_name_weight``; the largest single-name
        weight bounds the loss of an unhedgeable single-name gap, which the exposure
        solver prices.
    Raises:
        DataIntegrityError: Paired daily indexes disagree or any value is non-finite.
    """
    evidence = run.evidence
    days = evidence.base_daily.returns.index
    if not days.equals(evidence.stress_daily.returns.index):
        raise DataIntegrityError("paired daily indexes disagree")
    frame = pd.DataFrame(
        {
            "base_return": evidence.base_daily.returns,
            "stress_return": evidence.stress_daily.returns,
        },
        index=days,
        dtype="float64",
    )
    for prefix, replay in (("base", evidence.base), ("stress", evidence.stress)):
        ledger = replay.ledger
        day_labels = ledger.equity.index.normalize()
        frame[f"{prefix}_equity_close"] = ledger.equity.groupby(day_labels).last().reindex(days)
        frame[f"{prefix}_equity_low"] = ledger.equity.groupby(day_labels).min().reindex(days)
        frame[f"{prefix}_turnover"] = ledger.fill_turnover.groupby(
            ledger.fill_turnover.index.normalize()
        ).sum().reindex(days)
        frame[f"{prefix}_funding"] = ledger.funding_charge.groupby(
            ledger.funding_charge.index.normalize()
        ).sum().reindex(days)
    gross = run.candidate.target_weights.abs().sum(axis=1)
    frame["target_gross"] = gross.reindex(days, fill_value=0.0)
    max_name = run.candidate.target_weights.abs().max(axis=1)
    frame["max_name_weight"] = max_name.reindex(days, fill_value=0.0)
    frame = frame[
        [
            "base_return", "stress_return",
            "base_equity_close", "stress_equity_close",
            "base_equity_low", "stress_equity_low",
            "base_turnover", "stress_turnover",
            "base_funding", "stress_funding", "target_gross", "max_name_weight",
        ]
    ].astype("float64")
    if not bool(np.isfinite(frame.to_numpy(dtype="float64")).all()):
        raise DataIntegrityError("daily frame values must be finite")
    return frame


def persist_strategy_backtest(
    run: StrategyBacktestRun,
    output: Path,
    *,
    statistics: Mapping[str, JsonValue] | None = None,
) -> Path:
    """Atomically persist one fresh strategy research result envelope.

    Args:
        run: Completed research replay to persist.
        output: A non-existing JSON destination.
        statistics: Optional JSON-safe ``{"base": ..., "stress": ...}`` decision-grade
            statistics computed outside the engine (the engine never imports evaluation).
    Returns:
        The finalized JSON path.
    Raises:
        DataIntegrityError: The output is unsafe, occupied, or incomplete.
        OSError: Atomic persistence cannot complete.
    """
    if not isinstance(output, Path):
        raise DataIntegrityError("output must be a Path")
    if output.suffix != ".json":
        raise DataIntegrityError(f"output must be a JSON path, got {str(output)!r}")
    if os.path.lexists(output):
        raise DataIntegrityError(f"output must be fresh: {output} already exists")
    daily_path = output.parent / "daily.parquet"
    if os.path.lexists(daily_path):
        raise DataIntegrityError(f"daily artifact must be fresh: {daily_path} already exists")
    payload = strategy_backtest_payload(run, statistics=statistics)
    daily = strategy_daily_frame(run)
    daily_tmp = output.parent / "daily.parquet.tmp"
    tmp = output.parent / f"{output.name}.tmp"
    try:
        daily.to_parquet(daily_tmp)
        os.replace(daily_tmp, daily_path)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        os.replace(tmp, output)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(daily_tmp)
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        with contextlib.suppress(OSError):
            os.unlink(daily_path)
        with contextlib.suppress(OSError):
            os.unlink(output)
        raise
    _logger.debug(
        "[EVAL] strategy daily artifact rows=%d start=%s end=%s worst_base_day=%.6f min_base_equity_low=%.2f",
        len(daily), daily.index[0], daily.index[-1],
        float(daily["base_return"].min()), float(daily["base_equity_low"].min()),
    )
    return output
