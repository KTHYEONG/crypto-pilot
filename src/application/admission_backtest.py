"""In-memory base/stress backtests for selected library-admission proposals."""

from __future__ import annotations

import dataclasses
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import TypedDict

import pandas as pd

from src.application.expert_evaluation import _load_technical_market_data
from src.application.library_admission import _build_admission_context
from src.common.errors import DataIntegrityError
from src.core.settings import effective_worker_count
from src.research.baseline.backtest import BacktestResult
from src.research.contracts import CostModel
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.policy import resolve_evaluation_end
from src.research.evaluation.promotion import compose_promotion_verdict
from src.research.evaluation.reliability import (
    ReliabilityGateConfig,
    compute_equity_reliability_gate,
    compute_fold_distribution,
)
from src.research.expert_portfolio.admission_reports import LibraryAdmissionBacktestReport
from src.research.expert_portfolio.admission_types import (
    TechnicalLibraryAdmissionBacktestRequest,
    admission_proposal_id,
)
from src.research.expert_portfolio.backtest import (
    ExpertPortfolioBacktestResult,
    run_expert_portfolio,
)
from src.research.expert_portfolio.models import ExpertDefinition, ExpertPortfolioSpec
from src.research.provenance.code_manifest import TECHNICAL_CODE_UNITS, compute_code_hash
from src.research.provenance.results import record_library_admission_backtest_run
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.catalog import resolve_technical_candidate
from src.research.technical_experts.provenance import technical_data_hashes

_logger = logging.getLogger("TechnicalLibraryAdmissionBacktest")
_BASE_DELAY_BARS = 1
_STRESS_FEE_MULT = 1.5
_STRESS_SLIPPAGE_MULT = 2.0


class _SelectedEvidence(TypedDict):
    returns: pd.Series
    trades: pd.DataFrame


def _selected_symbol_worker(
    symbol: str,
    sources: tuple[str, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    signal_delay_bars: int,
) -> dict[str, _SelectedEvidence]:
    """Run all selected candidates for one symbol after one causal data load."""
    frame, funding = _load_technical_market_data(symbol, start, end)
    evidence: dict[str, _SelectedEvidence] = {}
    for source in sources:
        try:
            result = run_technical_expert_backtest(
                frame,
                resolve_technical_candidate(source),
                costs,
                funding,
                signal_delay_bars=signal_delay_bars,
            )
            trades = result.trades.copy()
            if len(trades) > 0:
                trades["expert_id"] = f"{source}:{symbol}"
            evidence[source] = {
                "returns": result.equity.pct_change(),
                "trades": trades,
            }
        except Exception as exc:  # noqa: BLE001
            exc.add_note(f"symbol={symbol} return_source={source}")
            raise
    return evidence


def _materialize_definitions(
    expert_ids: tuple[str, ...],
    code_hash: str,
) -> tuple[ExpertDefinition, ...]:
    if not expert_ids:
        raise ValueError("expert_ids must not be empty")
    definitions: list[ExpertDefinition] = []
    families: set[str] = set()
    symbols: set[str] = set()
    for expert_id in sorted(expert_ids):
        try:
            source, symbol = expert_id.rsplit(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"expert id must be '<return_source>:<symbol>', got {expert_id!r}"
            ) from exc
        if not source or not symbol:
            raise ValueError(f"expert id has an empty source or symbol: {expert_id!r}")
        candidate = resolve_technical_candidate(source)
        if candidate.family in families:
            raise ValueError(
                f"proposal admits duplicate family '{candidate.family}'"
            )
        if symbol in symbols:
            raise ValueError(f"proposal admits duplicate symbol '{symbol}'")
        families.add(candidate.family)
        symbols.add(symbol)
        definitions.append(
            ExpertDefinition(
                expert_id=expert_id,
                return_source=source,
                family=candidate.family,
                symbols=(symbol,),
                runner="run_technical_expert",
                code_hash=code_hash,
            )
        )
    return tuple(definitions)


def _run_selected_tasks(
    definitions: tuple[ExpertDefinition, ...],
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
    signal_delay_bars: int,
    max_workers: int | None,
) -> tuple[dict[str, dict[str, _SelectedEvidence]], int]:
    by_symbol: dict[str, list[str]] = {}
    for definition in definitions:
        symbol = definition.symbols[0]
        by_symbol.setdefault(symbol, []).append(definition.return_source)
    symbols = tuple(sorted(by_symbol))
    sources = {symbol: tuple(sorted(values)) for symbol, values in by_symbol.items()}
    workers = effective_worker_count(len(symbols), requested=max_workers)
    if workers == 1:
        return {
            symbol: _selected_symbol_worker(
                symbol, sources[symbol], start, end, costs, signal_delay_bars,
            )
            for symbol in symbols
        }, workers

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _selected_symbol_worker,
                symbol,
                sources[symbol],
                start,
                end,
                costs,
                signal_delay_bars,
            )
            for symbol in symbols
        ]
        try:
            results = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return dict(zip(symbols, results, strict=True)), workers


def _assemble_selected_panel(
    evidence_by_symbol: dict[str, dict[str, _SelectedEvidence]],
    definitions: tuple[ExpertDefinition, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_by_id: dict[str, pd.Series] = {}
    trade_frames: list[pd.DataFrame] = []
    for definition in definitions:
        symbol = definition.symbols[0]
        evidence = evidence_by_symbol[symbol][definition.return_source]
        series_by_id[definition.expert_id] = evidence["returns"]
        if len(evidence["trades"]) > 0:
            trade_frames.append(evidence["trades"])
    common = sorted(set.intersection(*(set(series.index) for series in series_by_id.values())))
    if len(common) < 2:
        raise DataIntegrityError(
            f"selected components share fewer than 2 common bars, got {len(common)}"
        )
    index = pd.DatetimeIndex(common)
    panel = pd.DataFrame(
        {
            definition.expert_id: series_by_id[definition.expert_id].loc[index]
            for definition in definitions
        },
        index=index,
    )
    trades = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames else pd.DataFrame()
    )
    return panel, trades


def _master_result(
    base: ExpertPortfolioBacktestResult,
    trades: pd.DataFrame,
) -> BacktestResult:
    return BacktestResult(
        equity=base.backtest_result.equity,
        trades=trades,
        signals=pd.DataFrame(),
    )


def run_technical_library_admission_backtest(
    request: TechnicalLibraryAdmissionBacktestRequest,
) -> LibraryAdmissionBacktestReport:
    """Backtest one proposal without catalog or registry mutation."""
    end = resolve_evaluation_end(request.end, unseal_holdout=False)
    code_hash = compute_code_hash(TECHNICAL_CODE_UNITS)
    definitions = _materialize_definitions(request.expert_ids, code_hash)
    spec = ExpertPortfolioSpec(experts=definitions, router=request.router)
    costs = CostModel()

    base_evidence, workers = _run_selected_tasks(
        definitions, request.start, end, costs, 0, request.max_workers,
    )
    panel, component_trades = _assemble_selected_panel(base_evidence, definitions)
    context = _build_admission_context(
        request.router, panel.index, request.start, end,
    )
    base = run_expert_portfolio(
        panel,
        spec,
        costs,
        initial_equity=request.initial_equity,
        decision_context=context,
    )
    base_result = _master_result(base, component_trades)
    observation_metrics = compute_metrics(base_result.equity, base_result.trades)
    observation_gate = compute_equity_reliability_gate(
        base_result.equity, len(base_result.trades),
    )
    observation_folds = compute_fold_distribution(base_result)

    stress_costs = CostModel(
        fee_rate=costs.fee_rate * _STRESS_FEE_MULT,
        slippage_rate=costs.slippage_rate * _STRESS_SLIPPAGE_MULT,
    )
    stress_evidence, stress_workers = _run_selected_tasks(
        definitions, request.start, end, stress_costs, 1, request.max_workers,
    )
    stress_panel, stress_trades = _assemble_selected_panel(
        stress_evidence, definitions,
    )
    stress = run_expert_portfolio(
        stress_panel,
        spec,
        stress_costs,
        initial_equity=request.initial_equity,
        fixed_weights=base.target_weights,
        signal_delay_bars=1,
    )
    stress_result = _master_result(stress, stress_trades)
    stress_metrics = compute_metrics(stress_result.equity, stress_result.trades)
    stress_gate = compute_equity_reliability_gate(
        stress_result.equity,
        len(stress_result.trades),
        config=dataclasses.replace(ReliabilityGateConfig(), hurdle_rate=0.0),
    )
    stress_folds = compute_fold_distribution(stress_result)
    promotion = compose_promotion_verdict(
        observation_gate, observation_folds, stress_gate, None,
    )
    _logger.info(
        "[EVAL] proposal=%s status=%s workers=%d/%d observation_cagr=%.4f stress_cagr=%.4f",
        admission_proposal_id(request.expert_ids), promotion.status,
        workers, stress_workers, observation_metrics.cagr, stress_metrics.cagr,
    )
    symbols = sorted({definition.symbols[0] for definition in definitions})
    report = LibraryAdmissionBacktestReport(
        status="COMPLETE",
        proposal_id=admission_proposal_id(request.expert_ids),
        expert_ids=tuple(definition.expert_id for definition in definitions),
        router=request.router,
        window_start=str(panel.index[0]),
        window_end=str(panel.index[-1]),
        observation_metrics=observation_metrics,
        observation_gate=observation_gate,
        observation_folds=observation_folds,
        stress_metrics=stress_metrics,
        stress_gate=stress_gate,
        stress_folds=stress_folds,
        promotion=promotion,
        allocation_cost_total=float(base.allocation_cost.sum()),
        stress_allocation_cost_total=float(stress.allocation_cost.sum()),
        execution_workers=max(workers, stress_workers),
        code_hash=code_hash,
        data_hashes={symbol: technical_data_hashes(symbol) for symbol in symbols},
    )
    if request.log_run:
        record_library_admission_backtest_run(report)
    return report


def _check_contract() -> None:
    assert (
        run_technical_library_admission_backtest.__name__
        == "run_technical_library_admission_backtest"
    )


_check_contract()
