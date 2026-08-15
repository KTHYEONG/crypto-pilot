"""In-process batched exit-mechanism sweep worker and runner (no portfolio/router/stress path)."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from statistics import mean, median

import pandas as pd

from src.application.research.technical.evaluation import _load_technical_market_data
from src.core.settings import effective_worker_count
from src.market_data.storage.loaders import timeframe_scale_factor
from src.research.contracts import CostModel
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.policy import resolve_evaluation_end
from src.research.evaluation.reliability import compute_equity_reliability_gate
from src.research.expert_portfolio.admission_reports import (
    ExitSweepCellResult,
    ExitSweepFamilySummary,
    TechnicalExpertExitSweepReport,
)
from src.research.expert_portfolio.admission_types import (
    ExitSweepSetting,
    TechnicalExpertExitSweepRequest,
)
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.catalog import resolve_technical_candidate


def _sweep_symbol_timeframe_worker(
    symbol: str,
    timeframe: str,
    candidate_sources: tuple[str, ...],
    settings: tuple[ExitSweepSetting, ...],
    atr_period: int,
    start: str | None,
    end: str | pd.Timestamp | None,
    costs: CostModel,
) -> list[ExitSweepCellResult]:
    """Run every (candidate, setting) cell for one (symbol, timeframe) pair.

    The causal OHLCV/funding data is loaded exactly once per pair and shared
    by every cell below; this is the batching fix that eliminates the repeated
    per-cell reloads of the previous CLI-per-cell sweep. Only the observation
    side is produced: no stress pass, no portfolio/router, no promotion.
    """
    frame, funding = _load_technical_market_data(
        symbol, start, end, timeframe=timeframe,
    )
    scaled_atr_period = max(1, round(atr_period * timeframe_scale_factor(timeframe)))
    cells: list[ExitSweepCellResult] = []
    for source in candidate_sources:
        candidate = resolve_technical_candidate(source, timeframe=timeframe)
        for setting in settings:
            result = run_technical_expert_backtest(
                frame, candidate, costs, funding,
                signal_delay_bars=0,
                stop_loss_mode=setting.stop_loss_mode,
                stop_loss_value=setting.stop_loss_value,
                atr_period=scaled_atr_period,
                trailing_stop=setting.trailing_stop,
            )
            metrics = compute_metrics(result.equity, result.trades)
            gate = compute_equity_reliability_gate(result.equity, len(result.trades))
            cells.append(
                ExitSweepCellResult(
                    candidate=source,
                    symbol=symbol,
                    timeframe=timeframe,
                    setting=setting,
                    cagr=metrics.cagr,
                    lcb90_cagr=gate.lcb90_cagr,
                    gate_pass=gate.verdict == "PASS",
                    trade_count=metrics.trade_count,
                )
            )
    return cells


def _aggregate_family_summary(
    cells: tuple[ExitSweepCellResult, ...],
    settings: tuple[ExitSweepSetting, ...],
) -> tuple[ExitSweepFamilySummary, ...]:
    """Aggregate per-(candidate, timeframe, setting) summaries across symbols.

    Groups are emitted in a deterministic order: candidate, then timeframe,
    then the setting's position in the request's settings grid.
    """
    order = {setting: index for index, setting in enumerate(settings)}
    groups: dict[tuple[str, str, ExitSweepSetting], list[ExitSweepCellResult]] = {}
    for cell in cells:
        groups.setdefault((cell.candidate, cell.timeframe, cell.setting), []).append(cell)

    return tuple(
        ExitSweepFamilySummary(
            candidate=candidate,
            timeframe=timeframe,
            setting=setting,
            symbol_count=len(group),
            mean_cagr=float(mean(cell.cagr for cell in group)),
            median_lcb90_cagr=float(median(cell.lcb90_cagr for cell in group)),
            gate_pass_count=sum(1 for cell in group if cell.gate_pass),
        )
        for (candidate, timeframe, setting), group in sorted(
            groups.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                order[item[0][2]],
            ),
        )
    )


def run_technical_expert_exit_sweep(
    request: TechnicalExpertExitSweepRequest,
) -> TechnicalExpertExitSweepReport:
    """Evaluate the full (candidate, symbol, timeframe, setting) exit grid.

    The unit of parallel work is one ``(symbol, timeframe)`` pair: each worker
    loads its pair once and loops over every candidate x setting cell in
    memory. Worker count follows the existing ``effective_worker_count``
    helper, so ``max_workers=1`` takes the identical sequential code path.
    """
    end = resolve_evaluation_end(request.end, unseal_holdout=False)
    costs = CostModel()
    settings = request.settings()
    pairs = tuple(
        (symbol, timeframe)
        for timeframe in request.timeframes
        for symbol in request.symbols
    )
    workers = effective_worker_count(len(pairs), requested=request.max_workers)
    started = time.monotonic()
    if workers == 1:
        per_pair = [
            _sweep_symbol_timeframe_worker(
                symbol, timeframe, request.candidate_sources, settings,
                request.atr_period, request.start, end, costs,
            )
            for symbol, timeframe in pairs
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _sweep_symbol_timeframe_worker,
                    symbol, timeframe, request.candidate_sources, settings,
                    request.atr_period, request.start, end, costs,
                )
                for symbol, timeframe in pairs
            ]
            try:
                per_pair = [future.result() for future in futures]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    wall_seconds = time.monotonic() - started
    cells = tuple(cell for batch in per_pair for cell in batch)
    family_summary = _aggregate_family_summary(cells, settings)
    return TechnicalExpertExitSweepReport(
        cells=cells,
        family_summary=family_summary,
        execution_workers=workers,
        wall_seconds=wall_seconds,
    )
