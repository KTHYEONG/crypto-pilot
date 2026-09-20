"""Durable research-only payload for one frozen-MHS inventory run."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.backtests.contracts import JsonValue
from src.common.errors import DataIntegrityError
from src.mhs.frozen_research_run import FrozenMhsBacktestRun


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


def frozen_mhs_backtest_payload(run: FrozenMhsBacktestRun) -> dict[str, JsonValue]:
    """Serialize research evidence without discarding provenance or limitations.

    The payload is a durable human- and machine-readable summary of one
    historical run.  It exposes ledger-derived metrics and source constraints
    while keeping raw target and fill tables as separately referenced artifacts.

    Args:
        run: Completed frozen-MHS historical replay result.
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
        periods[period.label] = cast(JsonValue, metrics)
    payload: dict[str, JsonValue] = {
        "strategy_id": candidate.strategy.strategy_id,
        "breadth": candidate.strategy.breadth,
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
        "base_one_way_taker_bps": request.base_spec.one_way_taker_bps(),
        "stress_one_way_taker_bps": request.stress_spec.one_way_taker_bps(),
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
        "limitations": cast(JsonValue, list(evidence.limitations)),
        "report_periods": cast(JsonValue, periods),
        "research_only": True,
    }
    json.dumps(payload)
    return payload


def persist_frozen_mhs_backtest(run: FrozenMhsBacktestRun, output: Path) -> Path:
    """Atomically persist one fresh frozen-MHS research result envelope.

    Args:
        run: Completed research replay to persist.
        output: A non-existing JSON destination.
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
    payload = frozen_mhs_backtest_payload(run)
    tmp = output.parent / f"{output.name}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
        os.replace(tmp, output)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return output
