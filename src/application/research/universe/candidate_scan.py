"""Diagnostic orchestrator: run the principled universe-candidate scan over the lake.

Read-only measurement over already-collected futures files. Mirrors
``diagnose_growth_headroom``'s philosophy: passes or fails nothing, mutates no
production universe constant, and consults no realized returns. Results are
persisted into the consolidated reliability ledger under the caller's report
directory.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from src.application.research.technical.reliability_ledger import (
    persist_reliability_ledger_entry,
)
from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import load_ohlcv_1h_as
from src.research.universe.candidate_scan import (
    UniverseCandidateResult,
    UniverseCandidateSpec,
    evaluate_universe_candidate,
)

_logger = logging.getLogger("UniverseCandidateScan")


def _round_finite(value: float) -> float:
    """Round a diagnostic float to 8 decimals, coercing non-finite to 0.0."""
    rounded = round(value, 8)
    return rounded if math.isfinite(rounded) else 0.0


def run_universe_candidate_scan(
    symbols: tuple[str, ...],
    discovery_start: pd.Timestamp,
    end: pd.Timestamp,
    spec: UniverseCandidateSpec = UniverseCandidateSpec(),  # noqa: B008
) -> tuple[UniverseCandidateResult, ...]:
    """Evaluate every requested symbol against the qualification filters.

    Each symbol's 4h frame is loaded via ``load_ohlcv_1h_as`` (the same loader
    the XS screens use) and funding presence is checked via ``funding_path``.
    A symbol whose OHLCV file is missing or fails to load is skipped (never
    raises) with a WARNING log -- this scan is diagnostic-only over whatever
    the local data lake happens to contain. Deterministic for a fixed lake
    state.
    """
    results: list[UniverseCandidateResult] = []
    for symbol in symbols:
        path = ohlcv_path(symbol, "1h")
        if not path.exists():
            _logger.warning(
                "[UNIVERSE-SCAN] excluding %s: no 1h OHLCV file", symbol,
            )
            continue
        try:
            frame = load_ohlcv_1h_as(path, "4h", start=discovery_start, end=end)
            result = evaluate_universe_candidate(
                symbol,
                frame,
                funding_path(symbol).exists(),
                discovery_start,
                end,
                spec,
            )
        except (DataIntegrityError, FileNotFoundError, OSError, ValueError) as exc:
            _logger.warning(
                "[UNIVERSE-SCAN] excluding %s: load/evaluate failed: %s", symbol, exc,
            )
            continue
        results.append(result)
    return tuple(results)


def persist_universe_candidate_scan(
    results: tuple[UniverseCandidateResult, ...], path: Path,
) -> None:
    """Upsert the per-symbol qualification payload into the consolidated ledger.

    Keyed by ``path.stem``; with no ``reliability`` key the entry lands in
    ``xs_alpha_reliability_fail.json`` per the ledger's documented fallback
    rule for non-gated diagnostic reports (identical precedent to the baseline
    leg-selection persistence). Timestamps are ISO strings and floats are
    rounded to 8 places (the project's ``_round_finite`` convention), so the
    payload is deterministic.
    """
    payload = {
        result.symbol: {
            "symbol": result.symbol,
            "first_bar": result.first_bar.isoformat(),
            "last_bar": result.last_bar.isoformat(),
            "coverage": _round_finite(result.coverage),
            "has_funding": result.has_funding,
            "taker_ratio_valid": result.taker_ratio_valid,
            "avg_daily_quote_vol_recent": _round_finite(
                result.avg_daily_quote_vol_recent
            ),
            "qualifies": result.qualifies,
        }
        for result in results
    }
    persist_reliability_ledger_entry(path.stem, payload, path.parent)


def xs_universe_candidate_scan_report_path() -> Path:
    """Logical report key for the universe candidate scan (ledger entry name, not a literal write target)."""
    return Path("docs/results") / "xs_alpha_universe_candidate_scan.json"
