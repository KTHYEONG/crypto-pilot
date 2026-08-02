"""Sealed library admission diagnostic over the frozen technical candidates.

The application materializes the requested candidate universe into
``ExpertDefinition``s, executes each symbol's frozen candidates through one
coarse process worker (when parallelism is enabled), assembles the exact common
completed-return panel, builds the router context once, and delegates the
diagnostic to the pure evaluator. It never opens the holdout, never mutates the
ledger, registry, or catalog, and its worker count and wall time are telemetry
only.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.application.research.technical.evaluation import _load_technical_market_data
from src.common.config import ohlcv_path
from src.common.errors import DataIntegrityError
from src.core.settings import effective_worker_count
from src.market_data.storage.loaders import load_ohlcv_4h
from src.research.contracts import CostModel
from src.research.evaluation.policy import resolve_evaluation_end
from src.research.expert_portfolio.admission import evaluate_library_admission
from src.research.expert_portfolio.admission_reports import LibraryAdmissionReport
from src.research.expert_portfolio.admission_types import TechnicalLibraryAdmissionRequest
from src.research.expert_portfolio.contextual_router import build_causal_context_labels
from src.research.expert_portfolio.models import ContextualRouterSpec, ExpertDefinition
from src.research.provenance.code_manifest import TECHNICAL_CODE_UNITS, compute_code_hash
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.catalog import resolve_technical_candidate
from src.research.technical_experts.provenance import technical_data_hashes

_logger = logging.getLogger("TechnicalLibraryAdmission")

_BASE_DELAY_BARS = 1


def _symbol_admission_worker(
    symbol: str,
    sources: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
) -> dict[str, tuple[pd.Series, int]]:
    """Load one symbol's causal market data once and run its requested candidates.

    The worker returns only the per-source ``pct_change`` return series and the
    closed-trade count; large OHLCV/funding frames are never shipped between
    parent and worker. A failing candidate re-raises the original exception with
    the ``symbol`` and ``return_source`` note, so a partial panel is never
    returned.
    """
    frame, funding = _load_technical_market_data(symbol, start, end)
    costs = CostModel()
    evidence: dict[str, tuple[pd.Series, int]] = {}
    for source in sources:
        try:
            candidate = resolve_technical_candidate(source)
            result = run_technical_expert_backtest(
                frame, candidate, costs, funding,
                signal_delay_bars=_BASE_DELAY_BARS,
            )
            evidence[source] = (result.equity.pct_change(), len(result.trades))
        except Exception as exc:  # noqa: BLE001
            exc.add_note(f"symbol={symbol} return_source={source}")
            raise
    return evidence


def _materialize_definitions(
    request: TechnicalLibraryAdmissionRequest,
    code_hash: str,
) -> tuple[ExpertDefinition, ...]:
    definitions: list[ExpertDefinition] = []
    for symbol in request.symbols:
        for source in request.candidate_sources:
            candidate = resolve_technical_candidate(source)
            definitions.append(
                ExpertDefinition(
                    expert_id=f"{source}:{symbol}",
                    return_source=source,
                    family=candidate.family,
                    symbols=(symbol,),
                    runner="run_technical_expert",
                    code_hash=code_hash,
                )
            )
    return tuple(sorted(definitions, key=lambda d: d.expert_id))


def _run_symbol_tasks(
    request: TechnicalLibraryAdmissionRequest,
    end: str | pd.Timestamp | None,
    effective_workers: int,
    sources: tuple[str, ...],
) -> dict[str, dict[str, tuple[pd.Series, int]]]:
    """Run exactly one coarse task per distinct symbol, in declared order.

    ``effective_workers == 1`` forces the identical sequential code path. When a
    worker task fails, every future is cancelled before the original exception
    is re-raised.
    """
    if effective_workers == 1:
        return {
            symbol: _symbol_admission_worker(symbol, sources, request.start, end)
            for symbol in request.symbols
        }
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(
                _symbol_admission_worker, symbol, sources, request.start, end,
            )
            for symbol in request.symbols
        ]
        try:
            ordered = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return dict(
        zip(request.symbols, ordered, strict=True)
    )


def _assemble_panel(
    evidence_by_symbol: dict[str, dict[str, tuple[pd.Series, int]]],
    definitions: tuple[ExpertDefinition, ...],
    symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    series_by_id: dict[str, pd.Series] = {}
    counts_by_id: dict[str, int] = {}
    for symbol in symbols:
        for source, (returns, count) in evidence_by_symbol[symbol].items():
            expert_id = f"{source}:{symbol}"
            series_by_id[expert_id] = returns
            counts_by_id[expert_id] = count

    common = functools.reduce(
        lambda idx1, idx2: idx1.intersection(idx2),
        (series.index for series in series_by_id.values()),
    )
    if len(common) < 2:
        raise DataIntegrityError(
            "library admission components share fewer than 2 common bars, "
            f"got {len(common)}"
        )
    common_index = pd.DatetimeIndex(common)
    panel = pd.DataFrame(
        {
            definition.expert_id: series_by_id[definition.expert_id].loc[common_index]
            for definition in definitions
        },
        index=common_index,
    )
    trade_counts = {
        definition.expert_id: counts_by_id[definition.expert_id]
        for definition in definitions
    }
    return panel, trade_counts


def _build_admission_context(
    router: ContextualRouterSpec,
    index: pd.DatetimeIndex,
    start: str | None,
    end: str | pd.Timestamp | None,
) -> pd.Series:
    ohlcv = load_ohlcv_4h(
        ohlcv_path(router.context_symbol, "1h"), start=start, end=end,
    )
    labels = build_causal_context_labels(ohlcv["close"], router)
    if not labels.index.equals(index):
        raise DataIntegrityError(
            f"decision-context OHLCV for {router.context_symbol} does not align "
            "exactly to the component panel index; a recomputed or reindexed "
            "context series is rejected"
        )
    return labels


def run_technical_library_admission(
    request: TechnicalLibraryAdmissionRequest,
) -> LibraryAdmissionReport:
    """Execute one sealed library admission diagnostic over the requested universe.

    Every requested frozen candidate runs on the same sealed causal window under
    the identical ``CostModel`` and base signal delay. The report fingerprint
    binds the candidate definitions, sealed window, code hash, and symbol data
    hashes; worker count and wall time stay out of it. A worker failure or a
    panel/catalog integrity failure propagates as an exception and no partial
    report is produced.
    """
    end = resolve_evaluation_end(request.end, unseal_holdout=False)
    started = time.perf_counter()
    code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)

    definitions = _materialize_definitions(request, code_hash)
    sources = tuple(sorted(request.candidate_sources))
    effective = effective_worker_count(
        len(request.symbols), requested=request.admission.max_workers,
    )
    _logger.info(
        "[SYS] stage=symbol_workers workers=%d symbols=%d",
        effective, len(request.symbols),
    )
    evidence = _run_symbol_tasks(request, end, effective, sources)
    panel, trade_counts = _assemble_panel(evidence, definitions, request.symbols)
    decision_context = _build_admission_context(
        request.router, panel.index, request.start, end,
    )
    report = evaluate_library_admission(
        panel,
        trade_counts,
        definitions,
        decision_context,
        request.router,
        request.admission,
    )
    report = dataclasses.replace(
        report,
        code_hash=code_hash,
        data_hashes={
            symbol: technical_data_hashes(symbol) for symbol in request.symbols
        },
        execution_workers=effective,
        wall_seconds=round(time.perf_counter() - started, 6),
    )
    _logger.info(
        "[EVAL] library_admission status=%s workers=%d proposals=%d covered_states=%d",
        report.status, report.execution_workers, len(report.proposals),
        report.covered_states,
    )
    return report


def _check_contract() -> None:
    """Executable assertions locking the library admission application surface."""
    assert run_technical_library_admission.__name__ == "run_technical_library_admission"


_check_contract()
